import os

import one.robots.base.mech_structure as orbms
import one.robots.end_effectors.ee_base as oreb
import one.utils.constant as ouc
import one.utils.math as oum


def prepare_ms():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir

    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, 'or_screwdriver.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SILVER)
    shank_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, 'shank.stl'),
        loc_rotmat=oum.rotmat_from_euler(0, 0, 0),
        loc_pos=oum.vec(-0.199714, 0, -0.09511483),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SILVER)

    jnt_shank = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk,
        child_lnk=shank_lnk,
        axis=ouc.StandardAxis.X,
        pos=oum.vec(0.199714, 0, 0.09511483),
        lmt_lo=-0.033164,
        lmt_up=0.0)

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
        self.shank_range = oum.vec(-0.033164, 0.0)
        self.fk(qs=[self.shank_range[0]])
        self._is_activated = False

    def fk(self, qs=None):
        result = super().fk(qs)
        if not hasattr(self, '_loc_tcp_tf'):
            return result
        shank_len = self.qs[0] if len(self.qs) > 0 else 0.0
        tcp_pos = self.tcp_pos + oum.vec(shank_len, 0, 0)
        self._loc_tcp_tf[:] = oum.tf_from_rotmat_pos(self.tcp_rotmat, tcp_pos)
        return result
        
    def set_shank_len(self, length):
        if length < self.shank_range[0] or length > self.shank_range[1]:
            raise ValueError(f'shank length {length} out of range {self.shank_range}')
        self.fk(qs=[length])
