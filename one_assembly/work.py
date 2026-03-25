import os
import pickle
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import yaml

import one.geom.loader as ogl
import one.scene.render_model as osrm
import one.scene.scene_object as osso
import one.utils.constant as ouc
import one.utils.math as oum

from .motion_planner.foldplanner import interpolate_fold


Pose = Tuple[np.ndarray, np.ndarray]


def _default_asset_root() -> str:
    asset_root = os.path.join(os.path.dirname(__file__), 'worklists', 'move_object')
    if not os.path.isdir(asset_root):
        raise FileNotFoundError(f'Assembly asset root not found: {asset_root}')
    return asset_root


@dataclass(frozen=True)
class WorkStep:
    action_type: str
    rel_pos: np.ndarray
    rel_rotmat: np.ndarray
    immediate: bool = False


@dataclass(frozen=True)
class WorkSpec:
    name: str
    steps: Tuple[WorkStep, ...]
    yaml_path: str
    mesh_path: str
    grasp_path: str
    mesh_file: str
    mesh_files: Tuple[str, ...]
    grasp_file: str


class Work:
    def __init__(self,
                 pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                 rotmat=np.eye(3, dtype=np.float32),
                 collision_type=ouc.CollisionType.AABB,
                 rgb=ouc.ExtendedColor.STEEL_GRAY,
                 alpha=1.0,
                 yaml_path=None,
                 mesh_path=None,
                 grasp_path=None,
                 obj_num=0):
        default_root = _default_asset_root()
        yaml_path = yaml_path or os.path.join(default_root, 'yamls')
        mesh_path = mesh_path or os.path.join(default_root, 'meshes')
        if grasp_path is None:
            default_grasp_path = os.path.join(default_root, 'pickles')
            if not os.path.isdir(default_grasp_path):
                default_grasp_path = os.path.join(default_root, 'grasps')
            grasp_path = default_grasp_path

        self.obj_num = int(obj_num)
        self.home_pos = np.asarray(pos, dtype=np.float32)
        self.home_rotmat = np.asarray(rotmat, dtype=np.float32)
        self.spec = self._load_spec(
            yaml_path=yaml_path,
            mesh_path=mesh_path,
            grasp_path=grasp_path,
            obj_num=self.obj_num)

        self.yaml_path = self.spec.yaml_path
        self.mesh_path = self.spec.mesh_path
        self.grasp_path = self.spec.grasp_path
        self.name = self.spec.name
        self.steps = list(self.spec.steps)

        # Compatibility fields for the current callers.
        self.immediate = [step.immediate for step in self.steps]
        self.type = [step.action_type for step in self.steps]
        self.pos = [step.rel_pos.copy() for step in self.steps]
        self.rotmat = [step.rel_rotmat.copy() for step in self.steps]

        self.grasp_collection = self._load_grasps(self.spec.grasp_file)
        self.model = self._build_model(
            mesh_files=self.spec.mesh_files,
            collision_type=collision_type,
            rgb=rgb,
            alpha=alpha)
        self.reset_pose()

    @staticmethod
    def _load_spec(yaml_path: str,
                   mesh_path: str,
                   grasp_path: str,
                   obj_num: int) -> WorkSpec:
        yaml_file = os.path.join(yaml_path, f'object{obj_num}.yaml')
        with open(yaml_file, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}

        name = raw.get('name')
        if not name:
            raise ValueError(f'Missing name in {yaml_file}')

        step_pattern = re.compile(r'^time(\d+)$')
        ordered_keys = sorted(
            (key for key in raw.keys() if step_pattern.match(key)),
            key=lambda key: int(step_pattern.match(key).group(1)))

        steps = []
        for key in ordered_keys:
            step_data = raw[key] or {}
            action_type = step_data.get('type')
            if action_type not in {'place', 'fold', 'screw'}:
                raise ValueError(f'Unsupported action type "{action_type}" in {yaml_file}:{key}')
            rel_pos = np.asarray(step_data.get('pos', [0.0, 0.0, 0.0]), dtype=np.float32)
            rel_rotmat = np.asarray(step_data.get('rotmat', np.eye(3)), dtype=np.float32)
            if rel_pos.shape != (3,):
                raise ValueError(f'Expected pos shape (3,) in {yaml_file}:{key}, got {rel_pos.shape}')
            if rel_rotmat.shape != (3, 3):
                raise ValueError(f'Expected rotmat shape (3, 3) in {yaml_file}:{key}, got {rel_rotmat.shape}')
            steps.append(WorkStep(
                action_type=action_type,
                rel_pos=rel_pos,
                rel_rotmat=rel_rotmat,
                immediate=bool(step_data.get('immediate', False))))

        mesh_files_raw = raw.get('mesh_files')
        if mesh_files_raw is None:
            mesh_files = (os.path.join(mesh_path, f'{name}.stl'),)
        else:
            if not isinstance(mesh_files_raw, (list, tuple)) or len(mesh_files_raw) == 0:
                raise ValueError(f'mesh_files must be a non-empty list in {yaml_file}')
            mesh_files = tuple(os.path.join(mesh_path, str(mesh_name)) for mesh_name in mesh_files_raw)
        mesh_file = mesh_files[0]
        grasp_file = os.path.join(grasp_path, f'{name}.pickle')
        npz_grasp_file = os.path.join(grasp_path, f'{name}.npz')
        if os.path.exists(npz_grasp_file):
            grasp_file = npz_grasp_file
        return WorkSpec(
            name=name,
            steps=tuple(steps),
            yaml_path=yaml_path,
            mesh_path=mesh_path,
            grasp_path=grasp_path,
            mesh_file=mesh_file,
            mesh_files=mesh_files,
            grasp_file=grasp_file)

    @staticmethod
    def _build_model(mesh_files, collision_type, rgb, alpha):
        if len(mesh_files) == 1:
            return osso.SceneObject.from_file(
                mesh_files[0],
                collision_type=collision_type,
                rgb=rgb,
                alpha=alpha)
        model = osso.SceneObject(collision_type=collision_type)
        for mesh_file in mesh_files:
            model.file_path = mesh_file
            model.add_visual(
                osrm.RenderModel(
                    geom=ogl.load_geometry(mesh_file),
                    rgb=rgb,
                    alpha=alpha),
                auto_make_collision=True)
        return model

    @staticmethod
    def _load_grasps(grasp_file: str):
        if not os.path.exists(grasp_file):
            return None
        if grasp_file.endswith('.npz'):
            try:
                grasp_data = np.load(grasp_file, allow_pickle=True)
            except OSError:
                return None
            required = {'pose', 'pre_pose', 'jaw_width', 'score'}
            if not required.issubset(grasp_data.files):
                return None
            pose_arr = np.asarray(grasp_data['pose'], dtype=np.float32)
            pre_pose_arr = np.asarray(grasp_data['pre_pose'], dtype=np.float32)
            jaw_width_arr = np.asarray(grasp_data['jaw_width'], dtype=np.float32).reshape(-1)
            score_arr = np.asarray(grasp_data['score'], dtype=np.float32).reshape(-1)
            if not (len(pose_arr) == len(pre_pose_arr) == len(jaw_width_arr) == len(score_arr)):
                return None
            grasp_collection = []
            for pose_tf, pre_pose_tf, jaw_width, score in zip(
                    pose_arr, pre_pose_arr, jaw_width_arr, score_arr):
                grasp_collection.append((
                    np.asarray(pose_tf, dtype=np.float32),
                    np.asarray(pre_pose_tf, dtype=np.float32),
                    float(jaw_width),
                    float(score),
                ))
            return grasp_collection
        try:
            with open(grasp_file, 'rb') as f:
                return pickle.load(f)
        except (OSError, pickle.PickleError, EOFError):
            return None

    def __len__(self) -> int:
        return len(self.steps)

    def _require_step(self, step_idx: int) -> WorkStep:
        if step_idx < 0 or step_idx >= len(self.steps):
            raise IndexError(f'step_idx {step_idx} out of range for {self.name}')
        return self.steps[step_idx]

    def _home_pose(self) -> Pose:
        if not self.steps:
            return self.home_pos.copy(), self.home_rotmat.copy()
        step0 = self.steps[0]
        if self.obj_num == 0:
            pos = step0.rel_pos + self.home_pos
            rotmat = step0.rel_rotmat @ self.home_rotmat
        else:
            pos = self.home_rotmat @ step0.rel_pos + self.home_pos
            rotmat = self.home_rotmat @ step0.rel_rotmat
        return pos.astype(np.float32), rotmat.astype(np.float32)

    @property
    def current_pose(self) -> Pose:
        return self.model.pos.copy(), self.model.rotmat.copy()

    def set_pose(self, pos, rotmat):
        self.model.pos = np.asarray(pos, dtype=np.float32)
        self.model.rotmat = np.asarray(rotmat, dtype=np.float32)
        return self.current_pose

    def reset_pose(self) -> Pose:
        return self.set_pose(*self._home_pose())

    def step(self, step_idx: int) -> WorkStep:
        return self._require_step(step_idx)

    def pose_after_action(self,
                          step_idx: int,
                          start_pose: Optional[Pose] = None) -> Optional[Pose]:
        step = self._require_step(step_idx)
        if start_pose is None:
            start_pos, start_rotmat = self.current_pose
        else:
            start_pos = np.asarray(start_pose[0], dtype=np.float32)
            start_rotmat = np.asarray(start_pose[1], dtype=np.float32)

        if step.action_type == 'place':
            return self._home_pose()
        if step.action_type == 'fold':
            before_tf = oum.tf_from_rotmat_pos(start_rotmat, start_pos)
            delta_tf = oum.tf_from_rotmat_pos(step.rel_rotmat, step.rel_pos)
            after_tf = before_tf @ delta_tf
            return after_tf[:3, 3].astype(np.float32), after_tf[:3, :3].astype(np.float32)
        if step.action_type == 'screw':
            screw_pos = start_pos + start_rotmat @ step.rel_pos
            screw_rotmat = start_rotmat @ step.rel_rotmat
            return screw_pos.astype(np.float32), screw_rotmat.astype(np.float32)
        return None

    def pose_after_actions(self,
                           step_indices: Sequence[int],
                           start_pose: Optional[Pose] = None) -> Pose:
        pose = self.current_pose if start_pose is None else (
            np.asarray(start_pose[0], dtype=np.float32),
            np.asarray(start_pose[1], dtype=np.float32))
        for step_idx in step_indices:
            next_pose = self.pose_after_action(step_idx, start_pose=pose)
            if next_pose is None:
                raise ValueError(f'Failed to evaluate action {step_idx} for {self.name}')
            if self.steps[step_idx].action_type != 'screw':
                pose = next_pose
        return pose[0].copy(), pose[1].copy()

    def apply_action(self, step_idx: int) -> Optional[Pose]:
        next_pose = self.pose_after_action(step_idx)
        if next_pose is None:
            return None
        if self.steps[step_idx].action_type in {'place', 'fold'}:
            self.set_pose(*next_pose)
        return next_pose

    def apply_actions(self, step_indices: Sequence[int]) -> Pose:
        pose = self.current_pose
        for step_idx in step_indices:
            result = self.apply_action(step_idx)
            if result is None:
                raise ValueError(f'Failed to apply action {step_idx} for {self.name}')
            if self.steps[step_idx].action_type in {'place', 'fold'}:
                pose = result
        return pose[0].copy(), pose[1].copy()

    def attach_to(self, scene):
        self.model.attach_to(scene)

    def detach(self, scene):
        self.model.detach_from(scene)

    def action(self, num=0):
        return self.apply_action(int(num))

    def place(self, num=0):
        step = self._require_step(int(num))
        if step.action_type != 'place':
            return None
        return self.apply_action(int(num))

    def fold(self, num=0):
        step = self._require_step(int(num))
        if step.action_type != 'fold':
            return None
        return self.apply_action(int(num))

    def screw(self, num=0, toggle_dbg=False):
        del toggle_dbg
        step = self._require_step(int(num))
        if step.action_type != 'screw':
            return None
        return self.pose_after_action(int(num))

    def obstacle(self, num=0, n_steps=2):
        step = self._require_step(int(num))
        if step.action_type != 'fold':
            return []
        start_pose = self.current_pose
        goal_pose = self.pose_after_action(int(num), start_pose=start_pose)
        if goal_pose is None:
            return []
        obstacle_list = []
        for pos, rotmat in interpolate_fold(start_pose=start_pose, goal_pose=goal_pose, n_steps=n_steps):
            obj_cpy = self.copy_cmodel()
            obj_cpy.pos = pos
            obj_cpy.rotmat = rotmat
            obstacle_list.append(obj_cpy)
        return obstacle_list

    def copy_cmodel(self):
        return self.model.clone()
