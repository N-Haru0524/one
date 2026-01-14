import os
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.end_effectors.ee_base as oreb

def get_structure():
    structure = orbms.MechStruct()
    mesh_dir=structure.default_mesh_dir

    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "or_screwdriver.stl"),
        local_rotmat=oum.rotmat_from_euler(0, 0, np.pi / 2),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SILVER)
    shank_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "shank.stl"),
        local_rotmat=oum.rotmat_from_euler(0, 0, np.pi / 2),
        collision_type=ouc.CollisionType.AABB,
        rgb=ouc.ExtendedColor.SILVER)
    
    jnt_shank = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk, child_lnk=shank_lnk,
        axis=ouc.StandardAxis.Y,
        pos=np.array([0, -0.033164, 0], dtype=np.float32),
        lmt_low=0.0, lmt_up=0.033164)
    
    structure.add_lnk(base_lnk)
    structure.add_lnk(shank_lnk)
    structure.add_jnt(jnt_shank)
    structure.compile()
    return structure

class ORSD(oreb.EndEffectorBase):
    @classmethod
    def _build_structure(cls):
        return get_structure()

    def __init__(self):
        super().__init__(
            tcp_tf=oum.tf_from_rotmat_pos(pos=(0, 0, 0.15)))
        self.shank_range = np.array([0.0, 0.033164], dtype=np.float32)
        self.set_shank_len(self.shank_range[0])
        
    def set_shank_len(self, length):
        if length < self.shank_range[0] or length > self.shank_range[1]:
            raise ValueError(f"Shank length {length} out of range {self.shank_range}")
        self.fk(qs=[length])
