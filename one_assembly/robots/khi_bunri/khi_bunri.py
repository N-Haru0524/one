import os
import sys

import numpy as np
import one.utils.constant as ouc
import one.utils.math as oum
import one.robots.base.mech_base as orbmb
import one.robots.base.mech_structure as orbms
import one.robots.manipulators.kawasaki.rs007l.rs007l as orkrs
import one.scene.scene_object as osso
import one.scene.render_model_primitive as osrmp

from one_assembly.robots.or_2fg7.or_2fg7 import OR2FG7
from one_assembly.robots.or_sd.or_sd import ORSD

_LFT_ARM_HOME = np.radians(oum.vec(0.0, 0.0, -90.0, 0.0, -90.0, -90.0))
_RGT_ARM_HOME = np.radians(
    oum.vec(20.966, 9.271, -104.327, 81.122, -109.085, 244.963)
)
_ARM_MOUNT_ROT = oum.rotmat_from_euler(0.0, 0.0, -np.pi / 2.0)
_LFT_ARM_MOUNT_TF = oum.tf_from_rotmat_pos(rotmat=_ARM_MOUNT_ROT, pos=(0.0, 0.25, 0.0))
_RGT_ARM_MOUNT_TF = oum.tf_from_rotmat_pos(rotmat=_ARM_MOUNT_ROT, pos=(0.0, -0.25, 0.0))
_LFT_EE_ENGAGE_TF = oum.tf_from_rotmat_pos(pos=(0.0, 0.0, 0.105))
_RGT_EE_ENGAGE_TF = oum.tf_from_rotmat_pos(
    rotmat=oum.rotmat_from_euler(0.0, 0.0, np.pi / 2.0), pos=(0.0, 0.0, 0.105)
)


def prepare_body_ms():
    structure = orbms.MechStruct()
    mesh_path = os.path.join(structure.res_dir, 'meshes', 'base_bunri.stl')
    body_lnk = orbms.Link.from_file(
        mesh_path,
        collision_type=ouc.CollisionType.MESH,
        rgb=(0.42, 0.42, 0.42),
        scale=(1.0, 1.0, 1.0),
    )
    structure.add_lnk(body_lnk)
    structure.compile()
    return structure


class KHIBunriBody(orbmb.MechBase):

    @classmethod
    def _build_structure(cls):
        return prepare_body_ms()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos, is_free=True)


def _make_flange_adapter(length=0.105, radius=0.035, rgb=(0.35, 0.35, 0.35)):
    adapter = osso.SceneObject(collision_type=ouc.CollisionType.OBB, is_free=True)
    adapter.add_visual(
        osrmp.gen_cylinder_rmodel(
            length=length,
            radius=radius,
            n_segs=6,
            rotmat=oum.rotmat_from_euler(np.pi, 0.0, 0.0),
            rgb=rgb,
        ),
        auto_make_collision=True,
    )
    return adapter


class KHIBunri:
    """Dual-arm KHI bunri cell for one."""

    def __init__(self, rotmat=None, pos=None):
        self._rotmat = oum.ensure_rotmat(rotmat)
        self._pos = oum.ensure_pos(pos)

        self.body = KHIBunriBody(rotmat=self._rotmat, pos=self._pos)
        self.lft_arm = orkrs.RS007L()
        self.rgt_arm = orkrs.RS007L()
        self.lft_gripper = OR2FG7()
        self.rgt_screwdriver = ORSD()
        self.lft_adapter = _make_flange_adapter()
        self.rgt_adapter = _make_flange_adapter()
        self.lft_arm.is_free = True
        self.rgt_arm.is_free = True

        self.body.mount(self.lft_arm, self.body.runtime_root_lnk, _LFT_ARM_MOUNT_TF)
        self.body.mount(self.rgt_arm, self.body.runtime_root_lnk, _RGT_ARM_MOUNT_TF)
        self.body._update_mounting(self.body._mountings[self.lft_arm])
        self.body._update_mounting(self.body._mountings[self.rgt_arm])

        orbmb.MechBase.mount(self.lft_arm, self.lft_adapter, self.lft_arm.runtime_lnks[-1], _LFT_EE_ENGAGE_TF)
        orbmb.MechBase.mount(self.rgt_arm, self.rgt_adapter, self.rgt_arm.runtime_lnks[-1], _RGT_EE_ENGAGE_TF)
        self.lft_arm._update_mounting(self.lft_arm._mountings[self.lft_adapter])
        self.rgt_arm._update_mounting(self.rgt_arm._mountings[self.rgt_adapter])

        self.lft_arm.engage(self.lft_gripper, engage_tf=_LFT_EE_ENGAGE_TF)
        self.rgt_arm.engage(self.rgt_screwdriver, engage_tf=_RGT_EE_ENGAGE_TF)

        self.lft_arm.home_qs = _LFT_ARM_HOME
        self.rgt_arm.home_qs = _RGT_ARM_HOME
        self.goto_home_conf()

    def goto_home_conf(self):
        self.lft_arm.fk(self.lft_arm.home_qs)
        self.rgt_arm.fk(self.rgt_arm.home_qs)

    def attach_to(self, scene):
        self.body.attach_to(scene)

    def detach_from(self, scene):
        self.body.detach_from(scene)

    @property
    def rotmat(self):
        return self.body.rotmat

    @property
    def pos(self):
        return self.body.pos

    @property
    def lft_tcp_tf(self):
        return self.lft_arm.gl_tcp_tf

    @property
    def rgt_tcp_tf(self):
        return self.rgt_arm.gl_tcp_tf

    def set_rotmat_pos(self, rotmat=None, pos=None):
        self.body.set_rotmat_pos(rotmat=rotmat, pos=pos)
        self.body._update_mounting(self.body._mountings[self.lft_arm])
        self.body._update_mounting(self.body._mountings[self.rgt_arm])
