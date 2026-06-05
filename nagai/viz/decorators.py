"""Optional per-frame scene decorators.

A decorator is anything with an ``on_frame(view)`` method (called after each
frame's state is applied) and an optional ``on_clip(clip)`` method (called when a
new clip starts).  The Player invokes whatever it is given; decorators are how
optional overlays (TCP frames, trajectory traces) attach without bloating the
core loop.
"""

import numpy as np

from one import ossop
import one.utils.math as oum

from .robotview import KHIBunriView


class TcpFrameDecorator:
    """Live coordinate frames tracking both arms' TCPs, updated every frame."""

    def __init__(self, base, view: KHIBunriView, length_scale=0.18, radius_scale=0.8):
        lft_tf, rgt_tf = view.tcp_tfs()
        self.lft_frame = ossop.frame(
            pos=lft_tf[:3, 3], rotmat=lft_tf[:3, :3],
            length_scale=length_scale, radius_scale=radius_scale,
        )
        self.rgt_frame = ossop.frame(
            pos=rgt_tf[:3, 3], rotmat=rgt_tf[:3, :3],
            length_scale=length_scale, radius_scale=radius_scale,
        )
        self.lft_frame.attach_to(base.scene)
        self.rgt_frame.attach_to(base.scene)

    def on_clip(self, clip):
        pass

    def on_frame(self, view: KHIBunriView):
        lft_tf, rgt_tf = view.tcp_tfs()
        self.lft_frame.set_rotmat_pos(rotmat=lft_tf[:3, :3], pos=lft_tf[:3, 3])
        self.rgt_frame.set_rotmat_pos(rotmat=rgt_tf[:3, :3], pos=rgt_tf[:3, 3])


class TrajectoryTrace:
    """Draw the left/right TCP polyline for whichever clip is currently playing.

    The trace is recomputed per clip from the clip's frames so only the active
    motion is shown.  Pass actor='right' to trace the screwdriver instead.
    """

    def __init__(self, base, view: KHIBunriView, actor='left',
                 radius=0.002, srgbs=(0.1, 0.2, 0.9), alpha=0.45):
        self.base = base
        self.view = view
        self.actor = actor
        self.radius = radius
        self.srgbs = oum.vec(*srgbs).astype(np.float32)
        self.alpha = alpha
        self._node = None

    def _tcp_pos(self):
        lft_tf, rgt_tf = self.view.tcp_tfs()
        tf = lft_tf if self.actor == 'left' else rgt_tf
        return tf[:3, 3].copy()

    def on_clip(self, clip):
        if self._node is not None:
            self._node.detach_from(self.base.scene)
            self._node = None
        # Walking the frames to sample TCP positions moves the live robot, so
        # snapshot and restore around it -- otherwise the robot is left parked at
        # the segment's last pose and flashes there for one frame before playback
        # snaps it back to the start.
        saved = self.view.capture()
        trace = []
        for frame in clip.frames:
            self.view.apply(frame.state)
            trace.append(self._tcp_pos())
        self.view.apply(saved)
        if len(trace) >= 2:
            segs = np.stack([np.asarray(trace[:-1]), np.asarray(trace[1:])], axis=1)
            self._node = ossop.linsegs(segs, radius=self.radius, srgbs=self.srgbs, alpha=self.alpha)
            self._node.attach_to(self.base.scene)

    def on_frame(self, view: KHIBunriView):
        pass
