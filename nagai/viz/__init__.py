"""Reusable dual-arm (KHIBunri) visualization/playback toolkit.

This package factors the per-test playback code (originally duplicated across
``test_assembly_sequence_visual.py`` / ``test_draft_motion.py``) into composable
parts:

    timeline.py    Frame / Clip data model + draft->segment->frame expansion
    robotview.py   KHIBunriView: dual-arm state codec (apply/capture/tcp/home)
    ee_control.py  EEController (event-driven grasp/driver) + SceneResetter
    decorators.py  optional per-frame scene decorators (TCP frames, traces)
    player.py      Player: scheduled playback engine + standard key controls

Scope v1: dual-arm, EEEvent-driven grasping only (no index-threshold grasp).
Single-arm support and static-grasp decorators are intentionally left for v2.
"""

from .timeline import (
    Clip,
    Frame,
    build_clip_frames,
    build_sync_segment_from_draft,
    expand_segment_states,
    normalize_action_draft,
)
from .robotview import KHIBunriView
from .ee_control import EEController, SceneResetter
from .decorators import TcpFrameDecorator, TrajectoryTrace
from .player import Player, PlaybackMode

__all__ = [
    'Clip',
    'Frame',
    'build_clip_frames',
    'build_sync_segment_from_draft',
    'expand_segment_states',
    'normalize_action_draft',
    'KHIBunriView',
    'EEController',
    'SceneResetter',
    'TcpFrameDecorator',
    'TrajectoryTrace',
    'Player',
    'PlaybackMode',
]
