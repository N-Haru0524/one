import os
import sys

import numpy as np
import one.geom.loader as ogl
import one.utils.constant as ouc
import one.utils.math as oum
import one.robots.base.mech_base as orbmb
import one.robots.base.mech_structure as orbms
import one.robots.base.kine.kinematic_chain as orbkkc
import one.robots.manipulators.kawasaki.rs007l.rs007l as orkrs
import one.scene.collision_shape as osc
import one.scene.render_model as osrm
import one.scene.scene_object as osso
import one.scene.render_model_primitive as osrmp

from one_assembly.robots.or_2fg7.or_2fg7 import OR2FG7
# from one_assembly.robots.or_sd.or_sd import ORSD
from one_assembly.robots.or_sd_cam.or_sd import ORSD

_LFT_ARM_HOME = np.radians(oum.vec(0.0, 0.0, -90.0, 0.0, -90.0, -90.0))
_RGT_ARM_HOME = np.radians(
    oum.vec(20.966, 9.271, -104.327, 81.122, -109.085, 244.963)
)
_LFT_ARM_LMT_LO = np.radians(oum.vec(-180.0, -40.0, -157.0, -100.0, -125.0, -175.0))
_LFT_ARM_LMT_UP = np.radians(oum.vec(110.0, 90.0, -20.0, 100.0, 0.0, 130.0))
_RGT_ARM_LMT_LO = np.radians(oum.vec(-110.0, -40.0, -157.0, -40.0, -125.0, 0.0))
_RGT_ARM_LMT_UP = np.radians(oum.vec(180.0, 90.0, -20.0, 150.0, 0.0, 360.0))

_ARM_MOUNT_ROT = oum.rotmat_from_euler(0.0, 0.0, -np.pi / 2.0)
_LFT_ARM_MOUNT_TF = oum.tf_from_rotmat_pos(rotmat=_ARM_MOUNT_ROT, pos=(0.0, 0.25, 0.0))
_RGT_ARM_MOUNT_TF = oum.tf_from_rotmat_pos(rotmat=_ARM_MOUNT_ROT, pos=(0.0, -0.25, 0.0))
_LFT_EE_ENGAGE_TF = oum.tf_from_rotmat_pos(pos=(0.0, 0.0, 0.105))
_RGT_EE_ENGAGE_TF = oum.tf_from_rotmat_pos(
    rotmat=oum.rotmat_from_euler(0.0, 0.0, np.pi / 2.0), pos=(0.0, 0.0, 0.105)
)


def prepare_body_ms():
    structure = orbms.MechStruct()
    mesh_dir = os.path.join(structure.res_dir, 'meshes')
    mesh_names = sorted(
        name for name in os.listdir(mesh_dir)
        if name.endswith('.stl') and name != 'base_bunri.stl'
    )
    if not mesh_names:
        mesh_names = ['base_bunri.stl']
    body_lnk = orbms.Link(collision_type=ouc.CollisionType.MESH)
    body_lnk.file_path = os.path.join(mesh_dir, mesh_names[0])
    for mesh_name in mesh_names:
        mesh_path = os.path.join(mesh_dir, mesh_name)
        render_model = osrm.RenderModel(
            geom=ogl.load_geometry(mesh_path, scale=(1.0, 1.0, 1.0)),
            rgb=(0.42, 0.42, 0.42),
        )
        body_lnk.add_visual(render_model, auto_make_collision=False)
        body_lnk.add_collision(
            osc.MeshCollisionShape(
                file_path=mesh_path,
                geom=render_model.geom,
                rotmat=render_model.rotmat,
                pos=render_model.pos,
            )
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


def _set_arm_joint_limits(arm, lmt_lo, lmt_up):
    compiled = arm.structure.compiled
    arm._chain = orbkkc.KinematicChain(arm.structure, compiled.root_lnk, compiled.tip_lnks[0])
    arm._solver = arm.get_solver(arm._chain)
    lmt_lo = np.asarray(lmt_lo, dtype=np.float32)
    lmt_up = np.asarray(lmt_up, dtype=np.float32)
    arm._chain.lmt_lo = lmt_lo.copy()
    arm._chain.lmt_up = lmt_up.copy()
    arm._solver.joint_limits = (arm._chain.lmt_lo, arm._chain.lmt_up)


class KHIBunriRS007L(orkrs.RS007L):
    def _normalize_active_qs_to_limits(self, qs_active, ref_qs=None, return_debug=False):
        lmt_lo = np.asarray(self._chain.lmt_lo, dtype=np.float32)
        lmt_up = np.asarray(self._chain.lmt_up, dtype=np.float32)
        qs_active = np.asarray(qs_active, dtype=np.float32)
        if ref_qs is not None:
            ref_qs = np.asarray(ref_qs, dtype=np.float32)
        normalized = np.zeros_like(qs_active, dtype=np.float32)
        period = np.float32(2.0 * np.pi)
        debug_rows = []
        for idx, q in enumerate(qs_active):
            candidates = []
            for k in range(-4, 5):
                q_shifted = np.float32(q + k * period)
                if lmt_lo[idx] <= q_shifted <= lmt_up[idx]:
                    candidates.append(q_shifted)
            if return_debug:
                debug_rows.append({
                    'joint_idx': idx,
                    'raw_q': float(q),
                    'limit_lo': float(lmt_lo[idx]),
                    'limit_up': float(lmt_up[idx]),
                    'candidates': [float(candidate) for candidate in candidates],
                })
            if not candidates:
                if return_debug:
                    return None, {
                        'failed_joint_idx': idx,
                        'joint_rows': debug_rows,
                    }
                return None
            if ref_qs is not None:
                normalized[idx] = min(candidates, key=lambda cand: abs(float(cand - ref_qs[idx])))
            else:
                mid = 0.5 * float(lmt_lo[idx] + lmt_up[idx])
                normalized[idx] = min(candidates, key=lambda cand: abs(float(cand - mid)))
        if return_debug:
            return normalized, {
                'failed_joint_idx': None,
                'joint_rows': debug_rows,
            }
        return normalized

    def _filter_active_joint_limits(self, qs_active_list, ref_qs=None):
        filtered = []
        for qs_active in qs_active_list:
            normalized = self._normalize_active_qs_to_limits(qs_active, ref_qs=ref_qs)
            if normalized is not None:
                filtered.append(normalized)
        return filtered

    def _ik_active(self, tgt_rotmat, tgt_pos, max_solutions=None, ref_qs=None, toggle_dbg=False):
        del toggle_dbg
        tgt_tcp_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        tgt_lastlnk_tf = tgt_tcp_tf @ np.linalg.inv(
            self._loc_flange_tf @ self._loc_tcp_tf
        )
        ref_qs_active = None
        if ref_qs is not None:
            ref_qs = np.asarray(ref_qs, dtype=np.float32)
            ref_qs_active = self._chain.extract_active_qs(ref_qs)
        ik_results = self._solver.ik(
            root_rotmat=self.rotmat,
            root_pos=self.pos,
            tgt_rotmat=tgt_lastlnk_tf[:3, :3],
            tgt_pos=tgt_lastlnk_tf[:3, 3],
            max_solutions=max_solutions,
            ref_qs=ref_qs_active,
        )
        if len(ik_results) == 0:
            return []
        return self._filter_active_joint_limits(ik_results, ref_qs=ref_qs_active)

    def ik_tcp(self, tgt_rotmat, tgt_pos, max_solutions=8, toggle_dbg=False):
        ik_results = self._ik_active(
            tgt_rotmat=tgt_rotmat,
            tgt_pos=tgt_pos,
            max_solutions=None,
            toggle_dbg=toggle_dbg,
        )
        if not ik_results:
            return None
        if max_solutions is not None:
            ik_results = ik_results[:max_solutions]
        return [
            self._chain.embed_active_qs(qs_active, self.qs)
            for qs_active in ik_results
        ]

    def ik_tcp_nearest(self, tgt_rotmat, tgt_pos, ref_qs=None, toggle_dbg=False):
        if ref_qs is None:
            ref_qs = self.qs
        ik_results = self._ik_active(
            tgt_rotmat=tgt_rotmat,
            tgt_pos=tgt_pos,
            max_solutions=None,
            ref_qs=ref_qs,
            toggle_dbg=toggle_dbg,
        )
        if not ik_results:
            return None
        return self._chain.embed_active_qs(ik_results[0], self.qs)


class KHIBunri:
    """Dual-arm KHI bunri cell for one."""

    def __init__(self, rotmat=None, pos=None):
        self._rotmat = oum.ensure_rotmat(rotmat)
        self._pos = oum.ensure_pos(pos)

        self.body = KHIBunriBody(rotmat=self._rotmat, pos=self._pos)
        self.lft_arm = KHIBunriRS007L()
        self.rgt_arm = KHIBunriRS007L()
        self.lft_gripper = OR2FG7()
        self.rgt_screwdriver = ORSD()
        self.lft_adapter = _make_flange_adapter()
        self.rgt_adapter = _make_flange_adapter()
        self.lft_arm.is_free = True
        self.rgt_arm.is_free = True
        _set_arm_joint_limits(self.lft_arm, _LFT_ARM_LMT_LO, _LFT_ARM_LMT_UP)
        _set_arm_joint_limits(self.rgt_arm, _RGT_ARM_LMT_LO, _RGT_ARM_LMT_UP)

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
