import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from one import ossop, ouc, ovw
import one.collider.cpu_simd as occs
from one.grasp.antipodal import (
    _antipodal_candidates,
    _rotmat_about_axis_batch_vec,
    build_grasp_rotmat_batch,
)
import one.utils.math as oum
from one_assembly.motion_planner import utils as omp_utils
from one_assembly.robots.or_2fg7.or_2fg7 import OR2FG7
from one_assembly.robots.or_sd.or_sd import ORSD
from one_assembly.worklist import WorkList


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate grasps for a worklist object and keep only screw-clear candidates.',
    )
    parser.add_argument(
        'worklist_dir',
        help='Worklist root directory containing grasps/meshes/yamls',
    )
    parser.add_argument(
        'object_name',
        help='Target object name in the worklist to generate grasps for',
    )
    parser.add_argument(
        '--layout',
        default='home',
        help='Layout name defined in layouts.yaml',
    )
    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help='Output NPZ path, default: <worklist_dir>/grasps/<object_name>.npz',
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
        help='Maximum number of grasps to save after all filters',
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
        '--no_screw_filter',
        action='store_true',
        help='Skip screw-obstacle filtering',
    )
    parser.add_argument(
        '--include_preplaced_obstacles',
        action='store_true',
        help='Also treat other preplaced work objects as collision obstacles',
    )
    parser.add_argument(
        '--no_view',
        action='store_true',
        help='Skip viewer and only save grasps',
    )
    return parser.parse_args()


def resolve_worklist_paths(worklist_dir):
    root_dir = Path(worklist_dir).expanduser().resolve()
    yaml_dir = root_dir / 'yamls'
    mesh_dir = root_dir / 'meshes'
    grasp_dir = root_dir / 'grasps'
    for path in (yaml_dir, mesh_dir, grasp_dir):
        if not path.is_dir():
            raise FileNotFoundError(f'Required directory not found: {path}')
    return root_dir, yaml_dir, mesh_dir, grasp_dir


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


def antipodal_iter_with_fallback(gripper, tgt_sobj,
                                 density=0.02, normal_tol_deg=20,
                                 roll_step_deg=30, clearance=0.002,
                                 score_weights=(0.7, 0.3)):
    gripper = gripper.clone()
    tgt_vs, tgt_fs, tgt_fns = occs.cols_to_vffns(tgt_sobj.collisions)
    cand = _antipodal_candidates(
        tgt_vs, tgt_fs, tgt_fns, density,
        normal_tol_deg, roll_step_deg, clearance)
    if cand is None:
        return

    (center, jaw_width, n_all, nq_all,
     cos_vals, normal_cos_th, roll_step) = cand
    jaw_min, jaw_max = gripper.jaw_range
    jaw_mid = 0.5 * (jaw_min + jaw_max)
    jaw_span = jaw_max - jaw_min + oum.eps
    mask = ((cos_vals >= normal_cos_th) &
            (jaw_width >= jaw_min) &
            (jaw_width <= jaw_max))
    if not np.any(mask):
        return

    center_sel = center[mask]
    jaw_sel = jaw_width[mask]
    n_sel = n_all[mask]
    nq_sel = nq_all[mask]
    ray_dirs = -n_sel
    open_dir = gripper.open_dir / (np.linalg.norm(gripper.open_dir))
    rot_base = build_grasp_rotmat_batch(ray_dirs, open_dir)
    angles = np.arange(0.0, 2 * np.pi, roll_step)
    roll_axes = np.einsum('nij,j->ni', rot_base, open_dir)
    roll_rots = _rotmat_about_axis_batch_vec(roll_axes, angles)
    rot_all = roll_rots @ rot_base[:, None, :, :]
    pose_tf = np.tile(np.eye(4, dtype=np.float32), (rot_all.shape[0], rot_all.shape[1], 1, 1))
    pose_tf[:, :, :3, :3] = rot_all
    pose_tf[:, :, :3, 3] = center_sel[:, None, :]
    pose_all = pose_tf.reshape(-1, 4, 4)
    jaw_all = np.repeat(jaw_sel, len(angles))
    normal_align = (1.0 + np.einsum('ij,ij->i', n_sel, -nq_sel) /
                    (np.linalg.norm(n_sel, axis=1) *
                     np.linalg.norm(nq_sel, axis=1) + oum.eps)) * 0.5
    jaw_close = 1.0 - np.abs(jaw_sel - jaw_mid) / jaw_span
    score = score_weights[0] * normal_align + score_weights[1] * jaw_close
    score_all = np.repeat(score, len(angles))
    order = np.argsort(score_all)[::-1]
    pose_all = pose_all[order]
    jaw_all = jaw_all[order]
    score_all = score_all[order]

    items = gripper.runtime_lnks + [tgt_sobj]
    tgt_idx = len(items) - 1
    pairs = [(i, tgt_idx) for i in range(len(gripper.runtime_lnks))]
    detector = occs.create_detector()
    batch = occs.build_batch(items, pairs)
    tcp_len = np.linalg.norm(gripper.loc_tcp_tf[:3, 3])
    retreat_dist = 0.5 * tcp_len

    for pose, jw, sc in zip(pose_all, jaw_all, score_all):
        collided = False
        gripper.grip_at(pose[:3, 3], pose[:3, :3], jw)
        results = detector.detect_collision_batch(batch)
        if results is not None:
            collided = True

        pre_pos = pose[:3, 3] - retreat_dist * pose[:3, 2]
        pre_pose = pose.copy()
        pre_pose[:3, 3] = pre_pos
        gripper.grip_at(pre_pose[:3, 3], pre_pose[:3, :3], jw)
        results = detector.detect_collision_batch(batch)
        if results is not None:
            collided = True
        yield pose, pre_pose, jw, float(sc), collided


def transform_grasp_to_world(grasp, obj_pose):
    obj_tf = oum.tf_from_rotmat_pos(obj_pose[1], obj_pose[0])
    pose, pre_pose, jaw_width, score = grasp
    pose_world = obj_tf @ pose
    pre_pose_world = obj_tf @ pre_pose
    return pose_world, pre_pose_world, jaw_width, score


def attach_orsd_ghost(scene, tcp_pose, alpha=0.2):
    ghost = ORSD()
    ghost.set_shank_len(float(ghost.shank_range[1]))
    ghost_tf = oum.tf_from_rotmat_pos(tcp_pose[1], tcp_pose[0]) @ oum.np.linalg.inv(ghost.loc_tcp_tf)
    ghost.set_rotmat_pos(rotmat=ghost_tf[:3, :3], pos=ghost_tf[:3, 3])
    ghost.alpha = alpha
    ghost.attach_to(scene)
    return ghost


def compute_target_screening_poses(worklist, target_work):
    screening_poses = []
    worklist.screw_counter = 0
    for work in worklist.work:
        start_pose = work.current_pose
        for step_idx, step in enumerate(work.steps):
            pose_start = start_pose
            if step.action_type in {'fold', 'screw'} and step_idx > 0:
                pose_start = work.pose_after_actions(range(step_idx), start_pose=work.current_pose)
            pose = work.pose_after_action(step_idx, start_pose=pose_start)
            if pose is None:
                continue
            if step.action_type == 'screw':
                _pick_pose = worklist.get_screw_pose()
                if work.name == target_work.name:
                    screening_poses.append((f'pre_screw_step_{step_idx}', pose_start))
            elif step.action_type != 'screw':
                start_pose = pose
    if not screening_poses:
        screening_poses.append(('current', target_work.current_pose))
    return screening_poses


def build_target_screw_obstacles(worklist, target_work):
    obstacles = []
    worklist.screw_counter = 0
    for work in worklist.work:
        start_pose = work.current_pose
        for step_idx, step in enumerate(work.steps):
            pose_start = start_pose
            if step.action_type in {'fold', 'screw'} and step_idx > 0:
                pose_start = work.pose_after_actions(range(step_idx), start_pose=work.current_pose)
            pose = work.pose_after_action(step_idx, start_pose=pose_start)
            if pose is None:
                continue
            if step.action_type == 'screw':
                worklist.get_screw_pose()
                if work.name == target_work.name:
                    obstacles.append((f'{work.name}:step{step_idx}:goal', pose))
            elif step.action_type != 'screw':
                start_pose = pose
    return obstacles


def build_filter_collider(screw_poses, extra_obstacles=None):
    gripper = OR2FG7()
    obstacles = []
    for _label, tcp_pose in screw_poses:
        screw = ORSD()
        screw.set_shank_len(float(screw.shank_range[1]))
        screw_tf = oum.tf_from_rotmat_pos(tcp_pose[1], tcp_pose[0]) @ oum.np.linalg.inv(screw.loc_tcp_tf)
        screw.set_rotmat_pos(rotmat=screw_tf[:3, :3], pos=screw_tf[:3, 3])
        obstacles.append(screw)
    if extra_obstacles:
        obstacles.extend(extra_obstacles)
    collider = omp_utils.build_collider([gripper], obstacles=obstacles)
    return gripper, collider


def build_preplaced_obstacles(worklist, target_work):
    layout_name = getattr(worklist, 'layout_name', None)
    if layout_name is None:
        return []
    layout = worklist.layout_specs.get(layout_name)
    if layout is None:
        return []

    obstacles = []
    for entry in layout.part_entries:
        if not entry.preplace:
            continue
        work = worklist[entry.work_idx]
        if work.name == target_work.name:
            continue
        obstacle = work.model.clone()
        obstacle.set_rotmat_pos(rotmat=work.current_pose[1], pos=work.current_pose[0])
        obstacles.append(obstacle)
    return obstacles


def filter_grasps_by_screw_clearance(grasps, screw_poses, object_poses, extra_obstacles=None):
    if (not screw_poses and not extra_obstacles) or not object_poses:
        return list(grasps), {}

    gripper, collider = build_filter_collider(screw_poses, extra_obstacles=extra_obstacles)
    kept = []
    rejected_by_pose = {label: 0 for label, _ in object_poses}
    for grasp in grasps:
        grasp_ok = True
        for pose_label, obj_pose in object_poses:
            pose_world, pre_pose_world, jaw_width, _score = transform_grasp_to_world(grasp, obj_pose)
            gripper.grip_at(pose_world[:3, 3], pose_world[:3, :3], jaw_width)
            if collider.is_collided(gripper.qs.copy()):
                rejected_by_pose[pose_label] += 1
                grasp_ok = False
                break
            gripper.grip_at(pre_pose_world[:3, 3], pre_pose_world[:3, :3], jaw_width)
            if collider.is_collided(gripper.qs.copy()):
                rejected_by_pose[pose_label] += 1
                grasp_ok = False
                break
        if grasp_ok:
            kept.append(grasp)
    return kept, rejected_by_pose


def visualize_grasps(worklist, target_work, screw_poses, screening_pose, grasps, max_visualized):
    base = ovw.World(
        cam_pos=(0.55, 0.45, 0.35),
        cam_lookat_pos=(0.18, 0.0, 0.02),
        toggle_auto_cam_orbit=False,
    )
    ossop.frame(radius_scale=0.3).attach_to(base.scene)

    target_marker_base = target_work.model.clone()
    target_marker_base.set_rotmat_pos(
        rotmat=screening_pose[1],
        pos=screening_pose[0],
    )
    target_marker_base.rgb = ouc.ExtendedColor.STEEL_GRAY
    target_marker_base.alpha = 0.18
    target_marker_base.attach_to(base.scene)

    for _label, tcp_pose in screw_poses:
        attach_orsd_ghost(base.scene, tcp_pose, alpha=0.16)

    marker = target_work.model.clone()
    marker.set_rotmat_pos(rotmat=screening_pose[1], pos=screening_pose[0])
    marker.rgb = target_work.model.rgb
    marker.alpha = 1.00
    marker.attach_to(base.scene)
    
    worklist.attach_to(base.scene)

    gripper = OR2FG7()
    n_draw = min(len(grasps), max_visualized)
    for i, grasp in enumerate(grasps[:n_draw]):
        ratio = 1.0 if n_draw <= 1 else i / (n_draw - 1)
        pose, pre_pose, jaw_width, _score = transform_grasp_to_world(grasp, screening_pose)

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
    print(f'visualized_pose: pos={screening_pose[0]}, rotmat=\n{screening_pose[1]}')
    print(f'screw_obstacles: {len(screw_poses)}')
    base.run()


def main():
    args = parse_args()
    root_dir, yaml_dir, mesh_dir, grasp_dir = resolve_worklist_paths(args.worklist_dir)
    output_path = Path(args.output) if args.output else grasp_dir / f'{args.object_name}.npz'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    worklist = WorkList(
        yaml_path=str(yaml_dir),
        mesh_path=str(mesh_dir),
        grasp_path=str(grasp_dir),
        collision_type=ouc.CollisionType.MESH,
    )
    worklist.init_pose(seed=args.layout)

    target_work = worklist.get_work(args.object_name)
    if target_work is None:
        raise KeyError(f'Object "{args.object_name}" not found in worklist: {root_dir}')

    target_local = target_work.model.clone()
    target_local.set_rotmat_pos(rotmat=np.eye(3, dtype=np.float32), pos=np.zeros(3, dtype=np.float32))

    gripper = OR2FG7()
    tic = time.perf_counter()
    grasps = []
    for pose, pre_pose, jaw_width, score, collided in antipodal_iter_with_fallback(
            gripper=gripper,
            tgt_sobj=target_local,
            density=args.density,
            normal_tol_deg=args.normal_tol_deg,
            roll_step_deg=args.roll_step_deg,
            clearance=args.clearance):
        if not collided:
            grasps.append((pose, pre_pose, jaw_width, score))
    elapsed = time.perf_counter() - tic
    collision_free_grasp_count = len(grasps)

    center_of_mass = compute_mesh_center_of_mass(target_local)
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
    augmented_grasp_count = len(grasps)

    screw_poses = build_target_screw_obstacles(worklist, target_work)
    screening_poses = compute_target_screening_poses(worklist, target_work)
    extra_obstacles = []
    if args.include_preplaced_obstacles:
        extra_obstacles = build_preplaced_obstacles(worklist, target_work)
    rejected_by_pose = {}
    if not args.no_screw_filter:
        grasps, rejected_by_pose = filter_grasps_by_screw_clearance(
            grasps=grasps,
            screw_poses=screw_poses,
            object_poses=screening_poses,
            extra_obstacles=extra_obstacles,
        )
    screw_filtered_grasp_count = len(grasps)

    if args.max_grasps is not None:
        grasps = grasps[:args.max_grasps]

    save_grasps_npz(grasps, output_path)

    print(f'worklist_dir: {root_dir}')
    print(f'object_name: {target_work.name}')
    print(f'layout: {args.layout}')
    print(f'output: {output_path}')
    print(f'center_of_mass: {center_of_mass}')
    print(f'grasp_generation_sec: {elapsed:.3f}')
    print(f'collision_free_grasps: {collision_free_grasp_count}')
    print(f'com_filtered_grasps: {com_filtered_grasp_count}')
    print(f'orientation_filtered_grasps: {orientation_filtered_grasp_count}')
    print(f'augmented_grasps: {augmented_grasp_count}')
    print(f'screw_obstacles: {len(screw_poses)}')
    print(f'preplaced_obstacles: {len(extra_obstacles)}')
    print(f'screening_poses: {[label for label, _ in screening_poses]}')
    if rejected_by_pose:
        print(f'screw_rejected_by_pose: {rejected_by_pose}')
    print(f'screw_filtered_grasps: {screw_filtered_grasp_count}')
    print(f'saved_grasps: {len(grasps)}')

    if not args.no_view:
        view_pose = screening_poses[0][1] if screening_poses else target_work.current_pose
        visualize_grasps(
            worklist=worklist,
            target_work=target_work,
            screw_poses=screw_poses,
            screening_pose=view_pose,
            grasps=grasps,
            max_visualized=args.max_visualized,
        )


if __name__ == '__main__':
    main()
