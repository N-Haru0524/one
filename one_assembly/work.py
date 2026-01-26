import builtins
import os
import pickle
from typing import List

import numpy as np
import yaml

from one import oum
import one.utils.constant as ouc
import one.scene.scene_object as osso
from motion_planner.foldplanner import interpolate_fold


class Work:
    def __init__(self,
                 pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                 rotmat=np.eye(3, dtype=np.float32),
                 collision_type=ouc.CollisionType.AABB,
                 rgb=ouc.ExtendedColor.STEEL_GRAY,
                 alpha=1.0,
                 yamlpath=None,
                 meshpath=None,
                 grasp_path=None,
                 obj_num=0):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        default_root = os.path.join(base_dir, 'from_wrs')
        yamlpath = yamlpath or os.path.join(default_root, 'yamls')
        meshpath = meshpath or os.path.join(default_root, 'meshes')
        grasp_path = grasp_path or os.path.join(default_root, 'pickles')

        obj_path = os.path.join(yamlpath, f'object{obj_num}.yaml')
        with open(obj_path, 'r', encoding='utf-8') as f:
            obj = yaml.safe_load(f)

        self.yamlpath = yamlpath
        self.meshpath = meshpath
        self.grasp_path = grasp_path
        self.home_pos = np.asarray(pos, dtype=np.float32)
        self.home_rotmat = np.asarray(rotmat, dtype=np.float32)
        self.name = obj['name']
        self.obj_num = obj_num
        self.immediate = []
        self.type = []
        self.pos: List[np.ndarray] = []
        self.rotmat: List[np.ndarray] = []
        self._load_steps(obj)

        self.grasp_collection = self._load_grasps()

        mesh_file = os.path.join(meshpath, f'{self.name}.stl')
        self.model = osso.SceneObject.from_file(
            mesh_file,
            collision_type=collision_type,
            rgb=rgb,
            alpha=alpha)
        self._reset_pose()

    def _load_steps(self, obj):
        for time_idx in range(10):
            key = f'time{time_idx}'
            if key not in obj:
                break
            self.type.append(obj[key]['type'])
            self.immediate.append(obj[key]['immediate'])
            self.pos.append(np.asarray(obj[key]['pos'], dtype=np.float32))
            self.rotmat.append(np.asarray(obj[key]['rotmat'], dtype=np.float32))

    def _load_grasps(self):
        grasp_file = os.path.join(self.grasp_path, f'{self.name}_specialized.pickle')
        if not os.path.exists(grasp_file):
            return None
        try:
            with open(grasp_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def _reset_pose(self):
        if not self.pos:
            return
        if self.obj_num == 0:
            self.model.pos = self.pos[0] + self.home_pos
            self.model.rotmat = self.rotmat[0] @ self.home_rotmat
        else:
            self.model.pos = self.home_rotmat @ self.pos[0] + self.home_pos
            self.model.rotmat = self.home_rotmat @ self.rotmat[0]

    def attach_to(self, scene):
        self.model.attach_to(scene)

    def detach(self, scene):
        self.model.detach_from(scene)

    def action(self, num=0):
        if num >= len(self.type):
            return None
        if self.type[num] == 'place':
            return self.place(num)
        if self.type[num] == 'fold':
            return self.fold(num)
        if self.type[num] == 'screw':
            return self.screw(num)
        return None

    def place(self, num=0):
        if num >= len(self.type):
            return None
        if self.type[num] != 'place':
            return None
        self._reset_pose()
        return self.model.pos.copy(), self.model.rotmat.copy()

    def fold(self, num=0):
        if num >= len(self.type):
            return None
        if self.type[num] != 'fold':
            return None
        before_tf = oum.tf_from_rotmat_pos(self.model.rotmat, self.model.pos)
        delta_tf = oum.tf_from_rotmat_pos(self.rotmat[num], self.pos[num])
        after_tf = before_tf @ delta_tf
        self.model.pos = after_tf[:3, 3]
        self.model.rotmat = after_tf[:3, :3]
        return self.model.pos.copy(), self.model.rotmat.copy()

    def screw(self, num=0, toggle_dbg=False):
        if num >= len(self.type):
            return None
        if self.type[num] != 'screw':
            return None
        screw_pos = self.model.pos + self.model.rotmat @ self.pos[num]
        screw_rotmat = self.model.rotmat @ self.rotmat[num]
        if toggle_dbg:
            try:
                from one import orsd
                ee_sd = orsd.ORSD()
                screw_tf = oum.tf_from_rotmat_pos(screw_rotmat, screw_pos)
                base_tf = screw_tf @ oum.tf_inverse(ee_sd.tcp_tf)
                ee_sd.set_rotmat_pos(base_tf[:3, :3], base_tf[:3, 3])
                ee_sd.attach_to(builtins.base.scene)
            except Exception:
                pass
        return screw_pos, screw_rotmat

    def obstacle(self, num=0):
        obstacle_list = []
        if num >= len(self.type):
            return obstacle_list
        if self.type[num] == 'fold':
            obj_cpy = self.copy_cmodel()
            start_pose = (obj_cpy.pos.copy(), obj_cpy.rotmat.copy())
            end_pose = self.action(num)
            if end_pose is None:
                return obstacle_list
            poses = interpolate_fold(start_pose=start_pose, goal_pose=end_pose, n_steps=2)
            for pose in poses:
                obj_cpy.pos = pose[0]
                obj_cpy.rotmat = pose[1]
                obstacle_list.append(obj_cpy.clone())
        return obstacle_list

    def copy_cmodel(self):
        return self.model.clone()
