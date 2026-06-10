import gc
import os

import numpy as np

import one.robots.base.mech_structure as orbms
import one.robots.end_effectors.ee_base as oreb
import one.utils.constant as ouc
import one.utils.math as oum


def prepare_ms():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir

    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, 'or_screwdriver_camera.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SILVER)
    shank_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, 'shank.stl'),
        loc_rotmat=oum.rotmat_from_euler(0, oum.pi / 2, 0),
        loc_pos=oum.vec(-0.041,0,0),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SILVER)

    jnt_shank = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk,
        child_lnk=shank_lnk,
        axis=ouc.StandardAxis.X,
        pos=oum.vec(0.14435446, -0.00044894, 0.09494043),
        lmt_lo=0.0,
        lmt_up=0.041)

    structure.add_lnk(base_lnk)
    structure.add_lnk(shank_lnk)
    structure.add_jnt(jnt_shank)
    structure.compile()
    return structure


class ORSD(oreb.EndEffectorBase, oreb.PointMixin):
    """OnRobot ScrewDriver with a single prismatic shank joint."""

    @classmethod
    def _build_structure(cls):
        return prepare_ms()

    def __init__(self):
        self.tcp_pos = oum.vec(0.199714, 0, 0.09509044)
        self.tcp_rotmat = oum.rotmat_from_axangle(ouc.StandardAxis.Y, oum.pi / 2)
        super().__init__(loc_tcp_tf=oum.tf_from_rotmat_pos(self.tcp_rotmat, self.tcp_pos))
        self.shank_range = oum.vec(0.0, 0.041).astype('float32')
        self.fk(qs=[self.shank_range[0]])
        self._is_activated = False
        # dual cameras: poses are offsets in the flange (base link) frame.
        # flange-based (not TCP-based) on purpose: the TCP slides with the
        # shank, so a TCP-relative camera would drift. the camera looks along
        # its local +z axis. values are CAD-measured in the flange frame.
        self.flange_to_cam_pos = [oum.vec(0.15006732, 0.03773201, 0.09494043),
                                  oum.vec(0.15006732, -0.03862989, 0.09494043)]
        # aim each camera's optical axis (local +z) at the screw acting point
        # (the TCP, expressed in the flange frame). this gives a converging
        # stereo pair pointed at the screw. override afterwards to re-aim.
        self.flange_to_cam_rotmat = [
            oum.rotmat_from_normal(self.tcp_pos - pos)
            for pos in self.flange_to_cam_pos]

    def fk(self, qs=None):
        result = super().fk(qs)
        if not hasattr(self, '_loc_tcp_tf'):
            return result
        shank_len = self.qs[0] if len(self.qs) > 0 else 0.0
        tcp_pos = self.tcp_pos + oum.vec(shank_len, 0, 0)
        self._loc_tcp_tf[:] = oum.tf_from_rotmat_pos(self.tcp_rotmat, tcp_pos)
        return result

    def _sync_parent_tcp(self):
        for ref in gc.get_referrers(self):
            if not isinstance(ref, dict) or self not in ref:
                continue
            for owner in gc.get_referrers(ref):
                if getattr(owner, '_mountings', None) is not ref:
                    continue
                if not hasattr(owner, '_loc_tcp_tf'):
                    continue
                mounting = ref.get(self)
                if mounting is None or not hasattr(mounting, 'engage_tf'):
                    continue
                owner._loc_tcp_tf[:] = mounting.engage_tf @ self.loc_tcp_tf

    def clone(self):
        new = super().clone()
        new.tcp_pos = self.tcp_pos.copy()
        new.tcp_rotmat = self.tcp_rotmat.copy()
        new.shank_range = self.shank_range.copy()
        return new
        
    def set_shank_len(self, length):
        if length < self.shank_range[0] or length > self.shank_range[1]:
            raise ValueError(f'shank length {length} out of range {self.shank_range}')
        self.fk(qs=[length])
        self._sync_parent_tcp()

    # ------------------------------------------------------------------ #
    # dual-camera view-collision (ported from from_wrs orsd_dualcamera)  #
    # ------------------------------------------------------------------ #
    def camera_tfs(self):
        """World (4,4) tfs of the two cameras; +z column is the view axis.

        Cameras are rigidly attached to the flange (base link), so they do not
        drift when the shank extends/retracts.
        """
        flange_tf = self.runtime_root_lnk.tf
        return [flange_tf @ oum.tf_from_rotmat_pos(rotmat, pos)
                for pos, rotmat in zip(self.flange_to_cam_pos, self.flange_to_cam_rotmat)]

    def get_view_rays(self, tgt_pos, radius=0.001):
        """SceneObject sticks from each camera to tgt_pos (for visualization).

        Caller sets `.rgb` (e.g. red when occluded, green when clear).
        """
        import one.scene.scene_object_primitive as ossop
        tgt_pos = np.asarray(tgt_pos, dtype=np.float32)
        return [ossop.cylinder(spos=cam_tf[:3, 3], epos=tgt_pos, radius=radius)
                for cam_tf in self.camera_tfs()]

    def is_view_collided(self, tgt_pos, obstacle_list=(), toggle_dbg=False):
        """True if any camera's line of sight to tgt_pos is blocked.

        A ray is cast from each camera toward tgt_pos; the view is occluded if
        any obstacle is hit strictly between the camera and the target.
        :param obstacle_list: SceneObjects with MESH collision (raycast support)
        """
        import one.collider.raycast as ocr
        tgt_pos = np.asarray(tgt_pos, dtype=np.float32)
        collided = False
        for cam_tf in self.camera_tfs():
            cam_pos = cam_tf[:3, 3]
            dist, dir_u = oum.unit_vec(tgt_pos - cam_pos, return_length=True)
            dist = float(dist)
            blocked = False
            for obstacle in obstacle_list:
                res = ocr.ray_shoot_scene_object(obstacle, cam_pos, dir_u)
                if res is None:
                    continue
                hit_t = res[2]
                if np.any((hit_t > 1e-6) & (hit_t < dist - 1e-6)):
                    blocked = True
                    break
            if toggle_dbg:
                print(f"camera view {'collided' if blocked else 'clear'}")
            collided = collided or blocked
        return collided
