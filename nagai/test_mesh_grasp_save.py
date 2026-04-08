import argparse
import time

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from one import osso, ossop, ouc, ovw
import one.collider.cpu_simd as occs
from one.grasp.antipodal import antipodal_iter
import one.utils.math as oum
from one_assembly.robots.or_2fg7 import or_2fg7


def compute_mesh_center_of_mass(target):
    verts, faces = occs.cols_to_vfs(target.collisions)
    if verts is None or faces is None or len(faces) == 0:
        raise ValueError('target mesh must have triangle collisions to compute center of mass')

    tri = verts[faces]
    v0 = tri[:, 0]
    v1 = tri[:, 1]
    v2 = tri[:, 2]

    tetra_volume6 = np.einsum('ij,ij->i', v0, np.cross(v1, v2))
    volume6_sum = float(np.sum(tetra_volume6))
    if abs(volume6_sum) > float(oum.eps):
        tetra_centroids = (v0 + v1 + v2) * 0.25
        center_of_mass = np.einsum('i,ij->j', tetra_volume6, tetra_centroids) / volume6_sum
        return center_of_mass.astype(np.float32)

    mean, _ = oum.area_weighted_pca(verts, faces)
    return np.asarray(mean, dtype=np.float32)


def point_segment_distance(point, seg_start, seg_end):
    seg = seg_end - seg_start
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq <= float(oum.eps):
        return float(np.linalg.norm(point - seg_start))
    t = float(np.dot(point - seg_start, seg) / seg_len_sq)
    t = float(np.clip(t, 0.0, 1.0))
    closest = seg_start + t * seg
    return float(np.linalg.norm(point - closest))


def filter_grasps_by_center_of_mass(grasps, gripper, center_of_mass, threshold):
    if threshold is None:
        return list(grasps)

    open_dir = np.asarray(gripper.open_dir, dtype=np.float32)
    open_dir_norm = np.linalg.norm(open_dir)
    if open_dir_norm <= float(oum.eps):
        raise ValueError('gripper.open_dir must be non-zero')
    open_dir = open_dir / open_dir_norm

    filtered = []
    for pose, pre_pose, jaw_width, score in grasps:
        world_open_dir = pose[:3, :3] @ open_dir
        half_span = 0.5 * float(jaw_width) * world_open_dir
        seg_start = pose[:3, 3] - half_span
        seg_end = pose[:3, 3] + half_span
        com_distance = point_segment_distance(center_of_mass, seg_start, seg_end)
        if com_distance <= threshold:
            filtered.append((pose, pre_pose, jaw_width, score))
    return filtered


def rotation_angle_between(rotmat_a, rotmat_b):
    rel_rotmat = rotmat_a.T @ rotmat_b
    cos_theta = np.clip((np.trace(rel_rotmat) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def filter_grasps_by_orientation_diversity(grasps, min_angle_deg):
    if min_angle_deg is None or min_angle_deg <= 0.0:
        return list(grasps)

    min_angle_rad = float(np.deg2rad(min_angle_deg))
    filtered = []
    for grasp in grasps:
        rotmat = grasp[0][:3, :3]
        if all(rotation_angle_between(rotmat, selected[0][:3, :3]) >= min_angle_rad
               for selected in filtered):
            filtered.append(grasp)
    return filtered


def augment_grasps_with_tcp_z_flip(grasps):
    rotz_pi = np.array([[-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, 1.0]], dtype=np.float32)
    augmented = []
    for pose, pre_pose, jaw_width, score in grasps:
        augmented.append((pose, pre_pose, jaw_width, score))

        flipped_pose = pose.copy()
        flipped_pose[:3, :3] = pose[:3, :3] @ rotz_pi

        flipped_pre_pose = pre_pose.copy()
        flipped_pre_pose[:3, :3] = pre_pose[:3, :3] @ rotz_pi
        augmented.append((flipped_pose, flipped_pre_pose, jaw_width, score))
    return augmented


def save_grasps_npz(grasps, output_path):
    pose_list = []
    pre_pose_list = []
    jaw_width_list = []
    score_list = []
    for pose, pre_pose, jaw_width, score in grasps:
        pose_list.append(pose.astype(np.float32))
        pre_pose_list.append(pre_pose.astype(np.float32))
        jaw_width_list.append(np.float32(jaw_width))
        score_list.append(np.float32(score))

    if pose_list:
        poses = np.asarray(pose_list, dtype=np.float32)
        pre_poses = np.asarray(pre_pose_list, dtype=np.float32)
        jaw_widths = np.asarray(jaw_width_list, dtype=np.float32)
        scores = np.asarray(score_list, dtype=np.float32)
    else:
        poses = np.zeros((0, 4, 4), dtype=np.float32)
        pre_poses = np.zeros((0, 4, 4), dtype=np.float32)
        jaw_widths = np.zeros((0,), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)

    np.savez(
        output_path,
        pose=poses,
        pre_pose=pre_poses,
        jaw_width=jaw_widths,
        score=scores,
    )


def visualize_grasps(mesh_path, grasps, max_visualized):
    base = ovw.World(
        cam_pos=(0.5, 0.5, 0.5),
        cam_lookat_pos=(0.0, 0.0, 0.0),
        toggle_auto_cam_orbit=True,
    )
    ossop.frame(radius_scale=0.3).attach_to(base.scene)

    target = osso.SceneObject.from_file(
        mesh_path,
        collision_type=ouc.CollisionType.MESH,
    )
    target.rgb = ouc.ExtendedColor.BEIGE
    target.alpha = 1.0
    target.attach_to(base.scene)

    gripper = or_2fg7.OR2FG7()
    n_draw = min(len(grasps), max_visualized)
    for i, (pose, pre_pose, jaw_width, score) in enumerate(grasps[:n_draw]):
        ratio = 1.0 if n_draw <= 1 else i / (n_draw - 1)

        grasp_ghost = gripper.clone()
        grasp_ghost.grip_at(pose[:3, 3], pose[:3, :3], jaw_width)
        grasp_ghost.rgb = np.array([1.0 - ratio, 0.8, ratio], dtype=np.float32)
        grasp_ghost.alpha = 0.28
        grasp_ghost.attach_to(base.scene)

        pre_grasp_ghost = gripper.clone()
        pre_grasp_ghost.grip_at(pre_pose[:3, 3], pre_pose[:3, :3], jaw_width)
        pre_grasp_ghost.rgb = ouc.BasicColor.YELLOW
        pre_grasp_ghost.alpha = 0.18
        pre_grasp_ghost.attach_to(base.scene)

    print(f'visualized_grasps: {n_draw}')
    print('viewer colors: grasp=gradient, pre_grasp=yellow')
    base.run()


def main():
    parser = argparse.ArgumentParser(
        description='Load a mesh, generate antipodal grasps, and save them to an NPZ file.',
    )
    parser.add_argument('mesh_path', help='Input mesh path such as bunny.stl')
    parser.add_argument(
        '-o',
        '--output',
        default='grasps.npz',
        help='Output NPZ path',
    )
    parser.add_argument(
        '--density',
        type=float,
        default=0.001,
        help='Surface sampling density in meters',
    )
    parser.add_argument(
        '--normal_tol_deg',
        type=float,
        default=20.0,
        help='Antipodal normal tolerance in degrees',
    )
    parser.add_argument(
        '--roll_step_deg',
        type=float,
        default=30.0,
        help='Roll search step in degrees',
    )
    parser.add_argument(
        '--clearance',
        type=float,
        default=0.0001,
        help='Extra jaw clearance in meters',
    )
    parser.add_argument(
        '--max_grasps',
        type=int,
        default=50,
        help='Maximum number of grasps to save',
    )
    parser.add_argument(
        '--max_visualized',
        type=int,
        default=30,
        help='Maximum number of grasps to draw in the viewer',
    )
    parser.add_argument(
        '--com_distance_threshold',
        type=float,
        default=0.05,
        help='Keep only grasps whose finger-line distance to the mesh center of mass is within this threshold in meters',
    )
    parser.add_argument(
        '--min_orientation_angle_deg',
        type=float,
        default=15.0,
        help='Greedily discard grasps whose orientation difference from an already selected grasp is smaller than this angle in degrees',
    )
    parser.add_argument(
        '--no_view',
        action='store_true',
        help='Skip viewer and only save grasps',
    )
    args = parser.parse_args()

    gripper = or_2fg7.OR2FG7()
    target = osso.SceneObject.from_file(
        args.mesh_path,
        collision_type=ouc.CollisionType.MESH,
    )

    tic = time.perf_counter()
    grasps = []
    for pose, pre_pose, jaw_width, score, collided in antipodal_iter(
            gripper=gripper,
            tgt_sobj=target,
            density=args.density,
            normal_tol_deg=args.normal_tol_deg,
            roll_step_deg=args.roll_step_deg,
            clearance=args.clearance):
        if not collided:
            grasps.append((pose, pre_pose, jaw_width, score))
    collision_free_grasp_count = len(grasps)
    center_of_mass = compute_mesh_center_of_mass(target)
    grasps = filter_grasps_by_center_of_mass(
        grasps=grasps,
        gripper=gripper,
        center_of_mass=center_of_mass,
        threshold=args.com_distance_threshold,
    )
    com_filtered_grasp_count = len(grasps)
    grasps = filter_grasps_by_orientation_diversity(
        grasps=grasps,
        min_angle_deg=args.min_orientation_angle_deg,
    )
    orientation_filtered_grasp_count = len(grasps)
    grasps = augment_grasps_with_tcp_z_flip(grasps)
    if args.max_grasps is not None:
        grasps = grasps[:args.max_grasps]
    toc = time.perf_counter()

    save_grasps_npz(grasps, args.output)

    print(f'mesh: {args.mesh_path}')
    print(f'center_of_mass: {center_of_mass}')
    print(f'com_distance_threshold: {args.com_distance_threshold:.4f}')
    print(f'collision_free_grasps: {collision_free_grasp_count}')
    print(f'com_filtered_grasps: {com_filtered_grasp_count}')
    print(f'min_orientation_angle_deg: {args.min_orientation_angle_deg:.1f}')
    print(f'orientation_filtered_grasps: {orientation_filtered_grasp_count}')
    print(f'grasps: {len(grasps)}')
    print(f'output: {args.output}')
    print(f'elapsed_sec: {toc - tic:.3f}')
    if grasps:
        best_pose, best_pre_pose, best_jaw_width, best_score = grasps[0]
        print(f'best_score: {best_score:.4f}')
        print(f'best_jaw_width: {best_jaw_width:.4f}')
        print(f'best_pose_pos: {best_pose[:3, 3]}')
        print(f'best_pre_pose_pos: {best_pre_pose[:3, 3]}')

    if not args.no_view:
        visualize_grasps(
            mesh_path=args.mesh_path,
            grasps=grasps,
            max_visualized=args.max_visualized,
        )


if __name__ == '__main__':
    main()
