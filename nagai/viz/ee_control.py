"""EE event interpretation (scene side) and per-clip scene reset.

EEController turns EEEvents into actual scene side-effects: mounting a held part
onto the gripper, releasing it, and moving the screwdriver shank.  This is the
*scene* counterpart to timeline.apply_ee_event_to_state (which only bakes joint
values).

SceneResetter restores the world to the exact pre-clip condition so any clip --
including a mid-draft segment that inherits a grasp from an earlier draft -- can
be replayed in isolation.
"""

import numpy as np

from one_assembly.assembly_data import EEEvent
from one_assembly.assembly_planning import reset_work_state

from .robotview import KHIBunriView
from .timeline import Clip


class EEController:
    """Apply EEEvents to the live scene (mount/release part, move driver)."""

    def __init__(self, robot, worklist, view: KHIBunriView):
        self.robot = robot
        self.worklist = worklist
        self.view = view
        self.held = {'name': None}

    def clear_left_mountings(self):
        for child in list(self.robot.lft_gripper._mountings.keys()):
            self.robot.lft_gripper.release(child)
        self.held['name'] = None

    def apply_event(self, event: EEEvent):
        work = self.worklist.get_work(event.work_name) if event.work_name is not None else None
        if event.actor == 'left_gripper':
            self._apply_gripper_event(event, work)
        elif event.actor == 'right_driver' and event.action in ('extend', 'retract') and event.value is not None:
            self.robot.rgt_screwdriver.fk(np.array([float(event.value)], dtype=np.float32))

    def _apply_gripper_event(self, event: EEEvent, work):
        gripper = self.robot.lft_gripper
        if event.action == 'open':
            gripper.fk(self.view.open_gripper_qs())
        elif event.action == 'attach' and work is not None:
            if self.held['name'] != event.work_name:
                if hasattr(work.model, 'is_free'):
                    work.model.is_free = True
                jaw_width = event.value
                engage_tf = event.engage_tf
                if jaw_width is None or engage_tf is None:
                    jaw_width = float(np.sum(gripper.qs[:gripper.ndof]))
                    gripper.grasp(work.model, jaw_width=jaw_width)
                else:
                    gripper.set_jaw_width(float(jaw_width))
                    gripper.mount(work.model, gripper.runtime_root_lnk, engage_tf)
                    gripper._update_mounting(gripper._mountings[work.model])
                self.held['name'] = event.work_name
        elif event.action == 'release' and work is not None:
            if work.model in gripper._mountings:
                gripper.release(work.model)
            gripper.fk(self.view.open_gripper_qs())
            self.held['name'] = None


class SceneResetter:
    """Restore the world to a clip's pre-playback state, then re-establish grasps."""

    def __init__(self, robot, worklist, view: KHIBunriView, ee: EEController, layout_name: str):
        self.robot = robot
        self.worklist = worklist
        self.view = view
        self.ee = ee
        self.layout_name = layout_name
        self.root_work_state = reset_work_state(worklist, layout_name=layout_name)
        self.root_free_states = {
            work.name: bool(getattr(work.model, 'is_free', False))
            for work in worklist.work
        }

    def reset_for(self, clip: Clip):
        self.ee.clear_left_mountings()
        reset_work_state(self.worklist, state=self.root_work_state)
        reset_work_state(self.worklist, layout_name=self.layout_name, actions=clip.prefix_actions)
        for work in self.worklist.work:
            root_is_free = self.root_free_states.get(work.name)
            if root_is_free is not None and hasattr(work.model, 'is_free'):
                work.model.is_free = root_is_free
        self.view.reset_driver_shank()
        self.view.apply(clip.initial_state)
        # Re-establish scene side-effects (held part, driver extension) produced
        # by earlier segments/drafts so a mid-draft segment replays correctly.
        for event in clip.setup_events:
            self.ee.apply_event(event)
