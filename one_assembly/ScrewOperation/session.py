"""ScrewSession: one screw, one correction phase, one prescrew handoff.

A session corresponds to one (work, action_idx, phase) tuple where the
phase is either ``'pick'`` (load a screw onto the SD bit at the pickup
station) or ``'place'`` (drive the screw into a hole on the workpiece).
Both phases use the same correction loop machinery; only the prescrew
TCP target differs.

The session-string grammar extends the assembly grammar used by
:mod:`one_assembly.assembly_planning`:

    "<target_token>[:<history>]"

where ``<target_token>`` is an assembly action token (e.g.
``rly_scrw``) **with a mandatory trailing phase suffix**, one of
``_pick`` or ``_place``; and ``<history>`` is a hyphen-separated list
of completed assembly tokens (no phase suffix — completed screws have
finished both phases). Examples::

    rly_scrw_pick                               # no history
    rly_scrw_place:wrkbnch-brckt-cpctr-rly      # pick already done above
    blt_fld_scrw_pick:wrkbnch-brckt-cpctr-rly-rly_scrw

The pickup pose is read from ``WorkList.get_screw_pickup_pose()`` and
the place pose from ``WorkList.get_screw_pose()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from one_assembly.assembly_data import DualRobotState
from one_assembly.assembly_planning import (
    AssemblyAction,
    parse_sequence_string,
    reset_work_state,
    _apply_symbolic_action,
    _label_maps,
)
from one_assembly.worklist import WorkList

from one_assembly.ScrewOperation.prescrew import (
    PrescrewSolution,
    prescrew_qs_from_screw_pose,
)


SCREW_PHASES: Tuple[str, ...] = ('pick', 'place')


@dataclass(frozen=True)
class ScrewSessionSpec:
    """Identifies one correction session.

    - ``target_action``: the AssemblyAction being targeted (always a screw
      action; ``target_action.action_type == 'screw'``).
    - ``phase``: 'pick' or 'place'.
    - ``history_actions``: assembly actions already completed before this
      session starts (used to restore the worklist state).
    - ``layout_name``: layout key used to bootstrap the WorkList.
    - ``target_token`` / ``history_string``: the verbatim tokens parsed
      from the input; helpful for logging and CSV-keying.
    """

    target_action: AssemblyAction
    phase: str
    history_actions: Tuple[AssemblyAction, ...] = field(default_factory=tuple)
    layout_name: str = 'home'
    target_token: str = ''
    history_string: str = ''

    def __post_init__(self):
        if self.phase not in SCREW_PHASES:
            raise ValueError(f"phase must be one of {SCREW_PHASES}, got {self.phase!r}")
        if self.target_action.action_type != 'screw':
            raise ValueError(
                f"target_action must be a screw action, got "
                f"action_type={self.target_action.action_type!r} (label={self.target_action.label})"
            )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _split_target_phase(target_token: str) -> Tuple[str, str]:
    """Strip a trailing _pick / _place suffix from a target token.

    Returns ``(base_token, phase)``. Raises ValueError when the suffix
    is missing or unrecognised.
    """
    for phase in SCREW_PHASES:
        suffix = f'_{phase}'
        if target_token.endswith(suffix):
            return target_token[: -len(suffix)], phase
    raise ValueError(
        f"target token {target_token!r} is missing a phase suffix; "
        f"expected one of {tuple(f'_{p}' for p in SCREW_PHASES)}"
    )


def _split_target_optional_phase(target_token: str) -> Tuple[str, Optional[str]]:
    """Like :func:`_split_target_phase` but returns ``phase=None`` when the
    suffix is absent. Used in full-screw mode where both phases run."""
    for phase in SCREW_PHASES:
        suffix = f'_{phase}'
        if target_token.endswith(suffix):
            return target_token[: -len(suffix)], phase
    return target_token, None


def _resolve_target_action(worklist: WorkList, base_token: str, completed_counts: List[int]) -> AssemblyAction:
    """Look up the screw AssemblyAction matching ``base_token``.

    Prefers the candidate whose ``action_idx`` matches the next-to-execute
    index given ``completed_counts`` (one per work); falls back to the
    first screw candidate. We deliberately allow targets that don't sit at
    the natural "next" boundary so callers can test a screw phase in
    isolation (no preceding history), at the cost of a possibly-invalid
    physical scene.
    """
    _, token_to_actions, _ = _label_maps(worklist)
    candidates = token_to_actions.get(base_token, [])
    screw_candidates = [c for c in candidates if c.action_type == 'screw']
    if not candidates:
        raise ValueError(f"Unknown assembly token in target: {base_token!r}")
    if not screw_candidates:
        raise ValueError(
            f"Target token {base_token!r} resolves to action_type="
            f"{candidates[0].action_type!r}; expected 'screw'."
        )
    # Prefer the screw candidate at the next-to-execute boundary.
    for cand in screw_candidates:
        if completed_counts[cand.work_idx] == cand.action_idx:
            return cand
    return screw_candidates[0]


def parse_screw_session_string(
    worklist: WorkList,
    sequence_str: str,
    *,
    layout_name: str = 'home',
) -> ScrewSessionSpec:
    """Parse '<target>[:<history>]'. Worklist is read-only during parsing."""
    raw = sequence_str.strip()
    if not raw:
        raise ValueError("Empty session string.")
    if ':' in raw:
        target_token, history_string = raw.split(':', 1)
        target_token = target_token.strip()
        history_string = history_string.strip()
    else:
        target_token = raw
        history_string = ''

    if not target_token:
        raise ValueError(f"Missing target token in session string: {sequence_str!r}")

    base_token, phase = _split_target_phase(target_token)

    # Parse history first so we know which screws are already done.
    history_actions: Tuple[AssemblyAction, ...] = ()
    if history_string:
        history_actions = tuple(
            parse_sequence_string(worklist, history_string, layout_name=layout_name)
        )

    # Determine completed-action counts AFTER applying the history.
    counts = [0] * len(worklist)
    for action in history_actions:
        counts[action.work_idx] = action.action_idx + 1

    target_action = _resolve_target_action(worklist, base_token, counts)

    return ScrewSessionSpec(
        target_action=target_action,
        phase=phase,
        history_actions=history_actions,
        layout_name=layout_name,
        target_token=target_token,
        history_string=history_string,
    )


def parse_screw_session_specs(
    worklist: WorkList,
    sequence_str: str,
    *,
    layout_name: str = 'home',
) -> List[ScrewSessionSpec]:
    """Like :func:`parse_screw_session_string` but accepts either:

    - a phase-tagged token (``rly_scrw_pick`` / ``rly_scrw_place``) — yields
      one spec, OR
    - a bare base token (``rly_scrw``) — yields TWO specs, one for each
      phase, in the natural execution order (pick before place).

    Returns ``list[ScrewSessionSpec]``. The history portion of the input is
    shared across all returned specs (both phases share the same upstream
    assembly state).
    """
    raw = sequence_str.strip()
    if not raw:
        raise ValueError("Empty session string.")
    if ':' in raw:
        target_token, history_string = raw.split(':', 1)
        target_token = target_token.strip()
        history_string = history_string.strip()
    else:
        target_token = raw
        history_string = ''
    if not target_token:
        raise ValueError(f"Missing target token in session string: {sequence_str!r}")

    base_token, phase = _split_target_optional_phase(target_token)

    history_actions: Tuple[AssemblyAction, ...] = ()
    if history_string:
        history_actions = tuple(
            parse_sequence_string(worklist, history_string, layout_name=layout_name)
        )
    counts = [0] * len(worklist)
    for action in history_actions:
        counts[action.work_idx] = action.action_idx + 1
    target_action = _resolve_target_action(worklist, base_token, counts)

    phases = (phase,) if phase is not None else SCREW_PHASES
    specs: List[ScrewSessionSpec] = []
    for ph in phases:
        token_for_phase = f'{base_token}_{ph}'
        specs.append(ScrewSessionSpec(
            target_action=target_action,
            phase=ph,
            history_actions=history_actions,
            layout_name=layout_name,
            target_token=token_for_phase,
            history_string=history_string,
        ))
    return specs


# ---------------------------------------------------------------------------
# Worklist state application
# ---------------------------------------------------------------------------

def apply_history(worklist: WorkList, spec: ScrewSessionSpec) -> None:
    """Mutate ``worklist`` so its state reflects all of spec.history_actions
    being completed. Resets to layout home before replaying — safe to call
    multiple times.
    """
    reset_work_state(worklist, layout_name=spec.layout_name)
    for action in spec.history_actions:
        _apply_symbolic_action(worklist, action)


# ---------------------------------------------------------------------------
# Target pose lookup
# ---------------------------------------------------------------------------

def screw_target_pose(worklist: WorkList, spec: ScrewSessionSpec):
    """Return (pos, rotmat) of the TCP target for this session's phase.

    - ``'pick'``  -> ``worklist.get_screw_pose()`` — the SD pickup station
      slot for the next unconsumed screw in the rack (layout.screw.origin +
      layout.screw.pitch * screw_counter). Assumes the worklist's
      ``screw_counter`` has already been advanced by :func:`apply_history`
      to the slot of the screw this session targets. The counter is
      snapshot/restored so the worklist is left untouched.
    - ``'place'`` -> ``worklist[target.work_idx].pose_after_action(
                       target.action_idx,
                       start_pose=work.current_pose)`` — the screw's
      destination on the workpiece.

    Raises ``RuntimeError`` when the worklist can't produce a place pose.
    """
    if spec.phase == 'pick':
        saved_counter = int(worklist.screw_counter)
        try:
            return worklist.get_screw_pose()
        finally:
            worklist.screw_counter = saved_counter

    if spec.phase == 'place':
        target = spec.target_action
        work = worklist[target.work_idx]
        pose = work.pose_after_action(target.action_idx, start_pose=work.current_pose)
        if pose is None:
            raise RuntimeError(
                f"pose_after_action returned None for {target.label!r}; "
                f"action_type={target.action_type!r}"
            )
        return pose

    raise ValueError(f"unknown phase {spec.phase!r}")


# ---------------------------------------------------------------------------
# End-to-end session resolution
# ---------------------------------------------------------------------------

def prescrew_qs_for_session(
    rgt_arm,
    worklist: WorkList,
    spec: ScrewSessionSpec,
    *,
    prescrew_offset: float = 0.005,
    ref_qs: Optional[np.ndarray] = None,
    flip_axis: bool = False,
    apply_history_first: bool = True,
) -> PrescrewSolution:
    """Resolve the prescrew rgt_qs for the session.

    Steps:
      1. (optional) apply the session's history to the worklist
      2. read the target pose (pick station or hole)
      3. solve IK to obtain rgt_qs at the prescrew TCP

    Returns a PrescrewSolution. Raises RuntimeError when IK fails.
    """
    if apply_history_first:
        apply_history(worklist, spec)
    pos, rotmat = screw_target_pose(worklist, spec)
    sol = prescrew_qs_from_screw_pose(
        rgt_arm, pos, rotmat,
        prescrew_offset=prescrew_offset,
        ref_qs=ref_qs,
        flip_axis=flip_axis,
    )
    if sol is None:
        raise RuntimeError(
            f"IK failed for session phase={spec.phase!r} target={spec.target_token!r}; "
            f"target pose pos={pos}, rotmat[:,2]={rotmat[:, 2]}"
        )
    return sol


# ---------------------------------------------------------------------------
# Helpers for downstream CSV labelling / debugging
# ---------------------------------------------------------------------------

def session_log_key(spec: ScrewSessionSpec) -> str:
    """A stable string key used to tag CSV rows / output directories.

    Format: ``<target_token>__<history_safe>`` where history is
    underscore-separated. Empty history becomes ``__home``.
    """
    history = spec.history_string.replace('-', '_') if spec.history_string else 'home'
    return f"{spec.target_token}__{history}"
