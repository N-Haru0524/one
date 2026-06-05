"""Player: scheduled playback engine with standard keyboard controls.

Plays a list of Clips through a KHIBunriView, firing each frame's EE events via
an EEController and resetting the scene per clip via a SceneResetter.  Keyboard
controls (next / prev / replay / pause / quit) are standard in *every* mode --
that is the whole point of the consolidation.

Modes:
    step   play a clip once, hold on its last frame, wait for `next`  (default)
    loop   auto-advance clip->clip and wrap; keys still interrupt
    once   play every clip through once, then hold at the very end
"""

from pyglet.window import key

from .ee_control import EEController, SceneResetter
from .robotview import KHIBunriView
from .timeline import Clip


class PlaybackMode:
    STEP = 'step'
    LOOP = 'loop'
    ONCE = 'once'


_NEXT_KEYS = (key.SPACE, key.N, key.RIGHT)
_PREV_KEYS = (key.B, key.LEFT)
_REPLAY_KEYS = (key.R,)
_PAUSE_KEYS = (key.P,)
_QUIT_KEYS = (key.ESCAPE, key.Q)


class Player:
    def __init__(self,
                 base,
                 view: KHIBunriView,
                 ee: EEController,
                 resetter: SceneResetter,
                 clips: list[Clip],
                 mode: str = PlaybackMode.STEP,
                 interval: float = 0.05,
                 decorators=()):
        if not clips:
            raise ValueError('Player requires at least one clip.')
        self.base = base
        self.view = view
        self.ee = ee
        self.resetter = resetter
        self.clips = clips
        self.mode = mode
        self.interval = interval
        self.decorators = list(decorators)
        self.clip_idx = 0
        self.frame_idx = 0
        self.waiting = False
        self.paused = False

    # --- controls ----------------------------------------------------------
    @staticmethod
    def controls_help() -> str:
        return ('SPACE / N / RIGHT : next   B / LEFT : prev   '
                'R : replay   P : pause   ESC / Q : quit')

    def _announce(self):
        clip = self.clips[self.clip_idx]
        print(f'[clip {self.clip_idx + 1}/{len(self.clips)}] {clip.action_type} '
              f'{clip.group} | seg {clip.seg_index + 1}/{clip.seg_count}: '
              f'"{clip.label}" ({len(clip.frames)} frames)')

    def _goto(self, clip_idx: int):
        self.clip_idx = clip_idx % len(self.clips)
        self.frame_idx = 0
        self.waiting = False
        self.resetter.reset_for(self.clips[self.clip_idx])
        for decorator in self.decorators:
            if hasattr(decorator, 'on_clip'):
                decorator.on_clip(self.clips[self.clip_idx])
        self._announce()

    def _on_clip_end(self):
        if self.mode == PlaybackMode.LOOP:
            self._goto(self.clip_idx + 1)
        elif self.mode == PlaybackMode.ONCE and self.clip_idx + 1 < len(self.clips):
            self._goto(self.clip_idx + 1)
        else:
            self.waiting = True
            print('   done. ' + self.controls_help())

    # --- main loop ---------------------------------------------------------
    def _update(self, _dt):
        im = self.base.input_manager
        if any(im.is_key_pressed_edge(k) for k in _QUIT_KEYS):
            self.base.close()
            return
        if any(im.is_key_pressed_edge(k) for k in _NEXT_KEYS):
            self._goto(self.clip_idx + 1)
            return
        if any(im.is_key_pressed_edge(k) for k in _PREV_KEYS):
            self._goto(self.clip_idx - 1)
            return
        if any(im.is_key_pressed_edge(k) for k in _REPLAY_KEYS):
            self._goto(self.clip_idx)
            return
        if any(im.is_key_pressed_edge(k) for k in _PAUSE_KEYS):
            self.paused = not self.paused
            print(f'   {"paused" if self.paused else "resumed"}')
            return

        if self.paused or self.waiting:
            return

        clip = self.clips[self.clip_idx]
        frame = clip.frames[self.frame_idx]
        self.view.apply(frame.state)
        for event in frame.events:
            self.ee.apply_event(event)
        for decorator in self.decorators:
            if hasattr(decorator, 'on_frame'):
                decorator.on_frame(self.view)

        self.frame_idx += 1
        if self.frame_idx >= len(clip.frames):
            self.frame_idx = len(clip.frames) - 1   # hold on the last frame
            self._on_clip_end()

    def run(self):
        print('\n' + '=' * 64)
        print(f'{len(self.clips)} clip(s) | mode={self.mode} | controls (focus the 3D window):')
        print('  ' + self.controls_help())
        print('=' * 64 + '\n')
        self._goto(0)
        self.base.schedule_interval(self._update, interval=self.interval)
        self.base.run()
