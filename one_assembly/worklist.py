import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

import one.scene.scene_object as osso
import one.utils.constant as ouc
import one.utils.math as oum

from one_assembly.work import Work


Pose = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class LayoutEntry:
    work_idx: int
    preplace: bool
    pos: np.ndarray
    rotmat: np.ndarray


@dataclass(frozen=True)
class LayoutSpec:
    base_mesh_file: Optional[str]
    base_pos_offset: np.ndarray
    base_rotmat: np.ndarray
    screw_origin: np.ndarray
    screw_rotmat: np.ndarray
    screw_pitch: np.ndarray
    part_entries: Tuple[LayoutEntry, ...]


def _default_root_dir() -> str:
    asset_root = os.path.join(os.path.dirname(__file__), 'worklists', 'electric_assembly')
    if not os.path.isdir(asset_root):
        raise FileNotFoundError(f'Assembly asset root not found: {asset_root}')
    return asset_root


def _as_vec3(values, *, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f'{field_name} must have shape (3,), got {array.shape}')
    return array


def _as_rotmat(values, *, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (3, 3):
        raise ValueError(f'{field_name} must have shape (3, 3), got {array.shape}')
    return array


class WorkList:
    def __init__(self,
                 pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                 rotmat=None,
                 yamlpath=None,
                 meshpath=None,
                 grasp_path=None,
                 alpha=1.0,
                 collision_type=ouc.CollisionType.AABB):
        default_root = _default_root_dir()
        yamlpath = yamlpath or os.path.join(default_root, 'yamls')
        meshpath = meshpath or os.path.join(default_root, 'meshes')
        grasp_path = grasp_path or os.path.join(default_root, 'grasps')
        if rotmat is None:
            rotmat = oum.rotmat_from_euler(np.pi / 2, 0, 0, order='rxyz')

        self.yamlpath = yamlpath
        self.meshpath = meshpath
        self.grasp_path = grasp_path
        self.pos = np.asarray(pos, dtype=np.float32)
        self.rotmat = np.asarray(rotmat, dtype=np.float32)
        self.alpha = float(alpha)
        self.collision_type = collision_type
        self.screw_counter = 0
        self.layout_name: Optional[str] = None
        self.work_base = None

        self.work = self._load_work_items(alpha=self.alpha)
        self._work_by_name = {work.name: work for work in self.work}
        self._object_key_to_index = {f'object{work.obj_num}': idx for idx, work in enumerate(self.work)}
        self.layout_specs = self._load_layout_specs()

    def _load_work_items(self, alpha: float) -> List[Work]:
        object_indices = self._discover_object_indices()
        colors = [
            ouc.ExtendedColor.ORANGE_RED,
            ouc.BasicColor.GRAY,
            ouc.BasicColor.BLUE,
            ouc.ExtendedColor.STEEL_GRAY,
            ouc.ExtendedColor.DEEP_SKY_BLUE,
            ouc.BasicColor.YELLOW,
        ]
        work_items = []
        for idx in object_indices:
            color = colors[idx] if idx < len(colors) else ouc.ExtendedColor.ORANGE_RED
            work_items.append(Work(
                pos=self.pos,
                rotmat=self.rotmat,
                yamlpath=self.yamlpath,
                meshpath=self.meshpath,
                grasp_path=self.grasp_path,
                obj_num=idx,
                rgb=color,
                alpha=alpha,
                collision_type=self.collision_type))
        return work_items

    def _discover_object_indices(self) -> List[int]:
        if not os.path.isdir(self.yamlpath):
            return list(range(6))
        object_indices = []
        for file_name in os.listdir(self.yamlpath):
            if not file_name.startswith('object') or not file_name.endswith('.yaml'):
                continue
            suffix = file_name[len('object'):-len('.yaml')]
            if not suffix.isdigit():
                continue
            object_indices.append(int(suffix))
        return sorted(object_indices)

    def _layout_file_path(self) -> str:
        return os.path.join(self.yamlpath, 'layouts.yaml')

    def _load_layout_specs(self) -> Dict[str, LayoutSpec]:
        layout_file = self._layout_file_path()
        if not os.path.isfile(layout_file):
            raise FileNotFoundError(f'Layout config not found: {layout_file}')
        with open(layout_file, 'r', encoding='utf-8') as f:
            raw_layouts = yaml.safe_load(f) or {}
        if not isinstance(raw_layouts, dict):
            raise ValueError(f'Layout config must be a mapping: {layout_file}')

        layout_specs = {}
        for layout_name, layout_data in raw_layouts.items():
            if not isinstance(layout_data, dict):
                raise ValueError(f'Layout "{layout_name}" must be a mapping')
            work_base_data = layout_data.get('work_base') or {}
            screw_data = layout_data.get('screw') or {}
            parts_data = layout_data.get('parts') or {}
            if not isinstance(parts_data, dict):
                raise ValueError(f'Layout "{layout_name}" parts must be a mapping')

            part_entries = []
            for part_key, part_data in parts_data.items():
                if not isinstance(part_data, dict):
                    raise ValueError(f'Layout "{layout_name}" part "{part_key}" must be a mapping')
                work_idx = self._resolve_layout_part_index(part_key)
                part_entries.append(LayoutEntry(
                    work_idx=work_idx,
                    preplace=bool(part_data.get('preplace', False)),
                    pos=_as_vec3(part_data.get('pos', [0.0, 0.0, 0.0]),
                                 field_name=f'{layout_name}.{part_key}.pos'),
                    rotmat=_as_rotmat(part_data.get('rotmat', np.eye(3, dtype=np.float32)),
                                      field_name=f'{layout_name}.{part_key}.rotmat')))

            base_mesh_value = work_base_data.get('mesh', 'work_base.stl')
            base_mesh = None if base_mesh_value in (None, '') else str(base_mesh_value)
            layout_specs[layout_name] = LayoutSpec(
                base_mesh_file=None if base_mesh is None else os.path.join(self.meshpath, base_mesh),
                base_pos_offset=_as_vec3(work_base_data.get('pos', [0.0, 0.0, 0.0]),
                                         field_name=f'{layout_name}.work_base.pos'),
                base_rotmat=_as_rotmat(work_base_data.get('rotmat', np.eye(3, dtype=np.float32)),
                                       field_name=f'{layout_name}.work_base.rotmat'),
                screw_origin=_as_vec3(screw_data.get('origin', [0.0, 0.0, 0.0]),
                                      field_name=f'{layout_name}.screw.origin'),
                screw_rotmat=_as_rotmat(screw_data.get('rotmat', np.eye(3, dtype=np.float32)),
                                        field_name=f'{layout_name}.screw.rotmat'),
                screw_pitch=_as_vec3(screw_data.get('pitch', [0.0, 0.0, 0.0]),
                                     field_name=f'{layout_name}.screw.pitch'),
                part_entries=tuple(part_entries))
        return layout_specs

    def _resolve_layout_part_index(self, part_key: str) -> int:
        if part_key in self._object_key_to_index:
            return self._object_key_to_index[part_key]
        idx = self.index_of(part_key)
        if idx is not None:
            return idx
        raise KeyError(f'Unknown layout part key: {part_key}')

    def __len__(self) -> int:
        return len(self.work)

    def __iter__(self):
        return iter(self.work)

    def __getitem__(self, index: int) -> Work:
        return self.work[index]

    def names(self) -> List[str]:
        return [work.name for work in self.work]

    def get_work(self, name: str) -> Optional[Work]:
        return self._work_by_name.get(name)

    def index_of(self, name: str) -> Optional[int]:
        for idx, work in enumerate(self.work):
            if work.name == name:
                return idx
        return None

    def attach_to(self, scene):
        if self.work_base is not None:
            self.work_base.attach_to(scene)
        for work in self.work:
            work.attach_to(scene)

    def detach(self, scene):
        if self.work_base is not None:
            self.work_base.detach_from(scene)
        for work in self.work:
            work.detach(scene)

    def reset_home_poses(self):
        for work in self.work:
            work.reset_pose()

    def _base_pos(self, layout: LayoutSpec) -> np.ndarray:
        return self.rotmat @ layout.base_pos_offset + self.pos

    def _ensure_work_base(self):
        if self.work_base is not None:
            return self.work_base
        if self.layout_name is None:
            raise ValueError('Cannot create work base before a layout is selected')
        work_base_file = self.layout_specs[self.layout_name].base_mesh_file
        if not work_base_file or not os.path.isfile(work_base_file):
            return None
        self.work_base = osso.SceneObject.from_file(
            work_base_file,
            collision_type=self.collision_type,
            rgb=ouc.ExtendedColor.BEIGE,
            alpha=1.0)
        return self.work_base

    def init_pose(self, seed='home'):
        layout = self.layout_specs.get(seed)
        if layout is None:
            return

        self.layout_name = seed
        self.screw_counter = 0

        base_pos = self._base_pos(layout)
        work_base = self._ensure_work_base()
        if work_base is not None:
            work_base.pos = base_pos
            work_base.rotmat = self.rotmat @ layout.base_rotmat

        for entry in layout.part_entries:
            if entry.work_idx >= len(self.work):
                continue
            if entry.preplace:
                part_pos = self.rotmat @ entry.pos + self.pos
                part_rotmat = self.rotmat @ entry.rotmat
            else:
                if work_base is None:
                    part_pos = self.rotmat @ entry.pos + self.pos
                    part_rotmat = self.rotmat @ entry.rotmat
                else:
                    part_pos = work_base.rotmat @ entry.pos + work_base.pos
                    part_rotmat = work_base.rotmat @ entry.rotmat
            self.work[entry.work_idx].set_pose(part_pos, part_rotmat)

    def init_pos(self, seed=0):
        rng = np.random.default_rng(seed)
        for work in self.work:
            rand_pos = rng.random(3, dtype=np.float32) / 2.0 + np.array([-0.5, -0.25, 0.1], dtype=np.float32)
            work.model.pos = rand_pos

    def init_rotmat(self, seed=0):
        rng = np.random.default_rng(seed)
        for work in self.work:
            axis = rng.random(3, dtype=np.float32)
            angle = float(rng.random())
            work.model.rotmat = oum.rotmat_from_axangle(axis, angle)

    def get_screw_pose(self):
        if self.work_base is None or self.layout_name not in self.layout_specs:
            return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)

        layout = self.layout_specs[self.layout_name]
        rel_pos = layout.screw_origin + layout.screw_pitch * self.screw_counter
        before_tf = oum.tf_from_rotmat_pos(self.work_base.rotmat, self.work_base.pos)
        after_tf = before_tf @ oum.tf_from_rotmat_pos(layout.screw_rotmat, rel_pos)
        self.screw_counter += 1
        return after_tf[:3, 3].astype(np.float32), after_tf[:3, :3].astype(np.float32)

    def current_part_poses(self) -> Dict[str, Pose]:
        return {work.name: work.current_pose for work in self.work}

    def set_part_poses(self, part_poses: Dict[str, Pose]):
        for name, pose in part_poses.items():
            work = self.get_work(name)
            if work is None:
                continue
            work.set_pose(*pose)

    def apply_actions(self, actions: Sequence[Tuple[int, int]]):
        results = []
        for work_idx, action_idx in actions:
            results.append(self.work[work_idx].apply_action(action_idx))
        return results

    def actions(self, work_num: int, act_num):
        return self.work[work_num].action(act_num)

    def copy(self, alpha=1.0):
        copied = WorkList(
            pos=self.pos.copy(),
            rotmat=self.rotmat.copy(),
            yamlpath=self.yamlpath,
            meshpath=self.meshpath,
            grasp_path=self.grasp_path,
            alpha=alpha,
            collision_type=self.collision_type)
        part_poses = self.current_part_poses()
        copied.set_part_poses(part_poses)
        copied.screw_counter = self.screw_counter
        copied.layout_name = self.layout_name
        if self.work_base is not None:
            copied_base = copied._ensure_work_base()
            copied_base.pos = self.work_base.pos.copy()
            copied_base.rotmat = self.work_base.rotmat.copy()
        return copied
