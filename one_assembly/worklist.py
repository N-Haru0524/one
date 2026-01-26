import os
from typing import List

import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.scene.scene_object as osso
from one_assembly.work import Work


class WorkList:
    def __init__(self,
                 pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                 rotmat=None,
                 yamlpath=None,
                 meshpath=None,
                 alpha=1.0,
                 collision_type=ouc.CollisionType.AABB):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        default_root = os.path.join(base_dir, 'from_wrs')
        yamlpath = yamlpath or os.path.join(default_root, 'yamls')
        meshpath = meshpath or os.path.join(default_root, 'meshes')
        rotmat = oum.rotmat_from_euler(np.pi / 2, 0, 0, order='rxyz') if rotmat is None else rotmat

        self.yamlpath = yamlpath
        self.meshpath = meshpath
        self.rotmat = np.asarray(rotmat, dtype=np.float32)
        self.pos = np.asarray(pos, dtype=np.float32)
        self.screw_counter = 0

        self.work: List[Work] = []
        if yamlpath == os.path.join(default_root, 'yamls'):
            colors = [
                ouc.ExtendedColor.ORANGE_RED,
                ouc.BasicColor.GRAY,
                ouc.BasicColor.BLUE,
                ouc.ExtendedColor.STEEL_GRAY,
                ouc.ExtendedColor.DEEP_SKY_BLUE,
                ouc.BasicColor.YELLOW,
            ]
            for idx, color in enumerate(colors):
                self.work.append(Work(pos=pos, rotmat=self.rotmat,
                                      yamlpath=yamlpath, meshpath=meshpath,
                                      obj_num=idx, rgb=color, alpha=alpha,
                                      collision_type=collision_type))
        else:
            obj_num = 0
            for file in os.listdir(yamlpath):
                if file.endswith('.yaml'):
                    obj_num += 1
            for i in range(obj_num):
                self.work.append(Work(pos=pos, rotmat=self.rotmat,
                                      yamlpath=yamlpath, meshpath=meshpath,
                                      obj_num=i, rgb=ouc.ExtendedColor.ORANGE_RED,
                                      alpha=alpha, collision_type=collision_type))

    def attach_to(self, scene):
        if hasattr(self, 'work_base'):
            self.work_base.attach_to(scene)
        for work in self.work:
            work.attach_to(scene)

    def detach(self, scene):
        if hasattr(self, 'work_base'):
            self.work_base.detach_from(scene)
        for work in self.work:
            work.detach(scene)

    def init_pose(self, seed='home', pos=np.array([0.015, 0.0, -0.025], dtype=np.float32)):
        if seed not in ('home', 'factory'):
            return
        x = pos[0]
        y = pos[1]
        z = pos[2]
        work_base_file = os.path.join(self.meshpath, 'work_base.stl')
        self.work_base = osso.SceneObject.from_file(
            work_base_file,
            collision_type=ouc.CollisionType.AABB,
            rgb=ouc.ExtendedColor.BEIGE,
            alpha=1.0)
        base_rot = oum.rotmat_from_euler(np.pi / 2, 0, 0, order='rxyz')
        base_pos = base_rot @ np.array([0.24 + x, -0.115 + z, -y], dtype=np.float32) + self.pos
        base_rot = oum.rotmat_from_euler(0, -np.pi / 2, -np.pi / 2, order='rxyz')
        self.work_base.pos = base_pos
        self.work_base.rotmat = base_rot

        self.work[0].model.pos = self.work[0].model.pos
        self.work[0].model.rotmat = self.work[0].model.rotmat
        self.work[1].model.pos = self.work[1].model.pos
        self.work[1].model.rotmat = self.work[1].model.rotmat
        if seed == 'home':
            self.work[2].model.pos = base_rot @ np.array([0.025, 0.03, -0.0215], dtype=np.float32) + base_pos
            self.work[2].model.rotmat = base_rot @ oum.rotmat_from_euler(
                0, 0, np.pi / 2 + np.pi / 24, order='rxyz')
            self.work[3].model.pos = base_rot @ np.array([-0.029, 0.007, -0.04], dtype=np.float32) + base_pos
            self.work[3].model.rotmat = base_rot @ oum.rotmat_from_euler(
                -np.pi / 2, 0, -np.pi / 2, order='rxyz')
            self.work[4].model.pos = base_rot @ np.array([-0.001, 0.0325, -0.0688], dtype=np.float32) + base_pos
            self.work[4].model.rotmat = base_rot @ oum.rotmat_from_euler(
                np.pi / 2, 0, -np.pi / 2, order='rxyz')
            self.work[5].model.pos = base_rot @ np.array([-0.053, 0.007, 0.04], dtype=np.float32) + base_pos
            self.work[5].model.rotmat = base_rot @ oum.rotmat_from_euler(
                0, np.pi, 0, order='rxyz')
        else:
            self.work[2].model.pos = base_rot @ np.array([0.025, 0.03, -0.022], dtype=np.float32) + base_pos
            self.work[2].model.rotmat = base_rot @ oum.rotmat_from_euler(
                0, 0, np.pi / 2 + np.pi / 24, order='rxyz')
            self.work[3].model.pos = base_rot @ np.array([-0.029, 0.006, -0.04], dtype=np.float32) + base_pos
            self.work[3].model.rotmat = base_rot @ oum.rotmat_from_euler(
                -np.pi / 2, 0, -np.pi / 2, order='rxyz')
            self.work[4].model.pos = base_rot @ np.array([-0.001, 0.0325, -0.0688], dtype=np.float32) + base_pos
            self.work[4].model.rotmat = base_rot @ oum.rotmat_from_euler(
                np.pi / 2, 0, -np.pi / 2, order='rxyz')
            self.work[5].model.pos = base_rot @ np.array([-0.053, 0.006, 0.04], dtype=np.float32) + base_pos
            self.work[5].model.rotmat = base_rot @ oum.rotmat_from_euler(
                0, np.pi, 0, order='rxyz')

    def init_pos(self, seed=0):
        np.random.seed(seed)
        for work in self.work:
            work.model.pos = np.random.rand(3) / 2 + np.array([-0.5, -0.25, 0.1], dtype=np.float32)

    def init_rotmat(self, seed=0):
        np.random.seed(seed)
        for work in self.work:
            axis = np.random.rand(3)
            angle = float(np.random.rand())
            work.model.rotmat = oum.rotmat_from_axangle(axis, angle)

    def get_screw_pose(self):
        distance = -0.02
        if hasattr(self, 'work_base'):
            before_tf = oum.tf_from_rotmat_pos(self.work_base.rotmat, self.work_base.pos)
            rel_pos = np.array([0.075, 0.006, 0.06 + distance * self.screw_counter], dtype=np.float32)
            rel_rot = oum.rotmat_from_euler(np.pi / 2, 0, 0, order='rxyz')
            after_tf = before_tf @ oum.tf_from_rotmat_pos(rel_rot, rel_pos)
            screw_pos, screw_rotmat = after_tf[:3, 3], after_tf[:3, :3]
            self.screw_counter += 1
        else:
            screw_pos = np.zeros(3, dtype=np.float32)
            screw_rotmat = np.eye(3, dtype=np.float32)
        return screw_pos, screw_rotmat

    def copy(self, alpha=1.0):
        worklist = WorkList(pos=self.pos, rotmat=self.rotmat, alpha=alpha)
        return worklist

    def actions(self, work_num: int, act_num):
        return self.work[work_num].action(act_num)

if __name__ == '__main__':
    import builtins
    from one import ovw, ossop

    base = ovw.World(cam_pos=(0.8, 0.8, 0.8), cam_lookat_pos=(0.0, 0.0, 0.2))
    builtins.base = base
    ossop.gen_frame().attach_to(base.scene)

    worklist = WorkList()
    worklist.init_pose(seed='home')
    worklist.attach_to(base.scene)

    for _ in range(5):
        screw_pos, screw_rotmat = worklist.get_screw_pose()
        ossop.gen_frame(pos=screw_pos, rotmat=screw_rotmat, ax_length=0.05).attach_to(base.scene)

    base.run()
