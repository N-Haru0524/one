import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from one import osso, ossop, ouc, ovw
import one.collider.cpu_simd as occs
from one.grasp.antipodal import _antipodal_candidates, _sample_count_from_area, antipodal_iter
from one_assembly.robots.or_2fg7 import or_2fg7


def signed_axis_label(vec):
    axis_names = ('+X', '+Y', '+Z')
    neg_axis_names = ('-X', '-Y', '-Z')
    idx = int(np.argmax(np.abs(vec)))
    return axis_names[idx] if vec[idx] >= 0.0 else neg_axis_names[idx]


def count_labels(vectors):
    counter = Counter()
    for vec in vectors:
        counter[signed_axis_label(vec)] += 1
    return counter


def print_counter(title, counter):
    print(title)
    if not counter:
        print('  none')
        return
    for key in ('+X', '-X', '+Y', '-Y', '+Z', '-Z'):
        print(f'  {key}: {counter.get(key, 0)}')


def visualize_candidates(mesh_path, results, max_visualized):
    base = ovw.World(
        cam_pos=(0.5, 0.5, 0.5),
        cam_lookat_pos=(0.0, 0.0, 0.0),
        toggle_auto_cam_orbit=True,
    )
    ossop.frame().attach_to(base.scene)

    target = osso.SceneObject.from_file(
        mesh_path,
        collision_type=ouc.CollisionType.MESH,
    )
    target.rgb = ouc.ExtendedColor.BEIGE
    target.attach_to(base.scene)

    gripper = or_2fg7.OR2FG7()
    n_draw = min(len(results), max_visualized)
    for pose, pre_pose, jaw_width, score, collided in results[:n_draw]:
        ghost = gripper.clone()
        ghost.grip_at(pose[:3, 3], pose[:3, :3], jaw_width)
        ghost.rgb = ouc.BasicColor.RED if collided else ouc.BasicColor.GREEN
        ghost.alpha = 0.25
        ghost.attach_to(base.scene)

        if collided:
            continue

        pre_ghost = gripper.clone()
        pre_ghost.grip_at(pre_pose[:3, 3], pre_pose[:3, :3], jaw_width)
        pre_ghost.rgb = ouc.BasicColor.YELLOW
        pre_ghost.alpha = 0.18
        pre_ghost.attach_to(base.scene)

    print(f'visualized_candidates: {n_draw}')
    print('viewer colors: free=green, collided=red, pre_grasp=yellow')
    base.run()


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose antipodal grasp generation stage by stage.',
    )
    parser.add_argument('mesh_path', help='Input mesh path such as relay.stl')
    parser.add_argument('--density', type=float, default=0.005)
    parser.add_argument('--normal_tol_deg', type=float, default=20.0)
    parser.add_argument('--roll_step_deg', type=float, default=60.0)
    parser.add_argument('--clearance', type=float, default=0.0001)
    parser.add_argument('--max_visualized', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--view', action='store_true', help='Visualize candidates including collided ones')
    args = parser.parse_args()

    np.random.seed(args.seed)

    gripper = or_2fg7.OR2FG7()
    target = osso.SceneObject.from_file(
        args.mesh_path,
        collision_type=ouc.CollisionType.MESH,
    )

    tgt_vs, tgt_fs, tgt_fns = occs.cols_to_vffns(target.collisions)
    bbox_min = tgt_vs.min(axis=0)
    bbox_max = tgt_vs.max(axis=0)
    bbox_size = bbox_max - bbox_min
    area_sample_count = _sample_count_from_area(tgt_vs, tgt_fs, args.density)

    print(f'mesh: {args.mesh_path}')
    print(f'bbox_min: {bbox_min}')
    print(f'bbox_max: {bbox_max}')
    print(f'bbox_size: {bbox_size}')
    print(f'gripper_jaw_range: {gripper.jaw_range}')
    print(f'density: {args.density}')
    print(f'estimated_surface_samples: {area_sample_count}')

    cand = _antipodal_candidates(
        tgt_vs,
        tgt_fs,
        tgt_fns,
        args.density,
        args.normal_tol_deg,
        args.roll_step_deg,
        args.clearance,
    )
    if cand is None:
        print('candidate_stage: no valid opposite-side ray hits')
        return

    center, jaw_width, n_all, nq_all, cos_vals, normal_cos_th, _ = cand
    jaw_min, jaw_max = gripper.jaw_range
    normal_ok = cos_vals >= normal_cos_th
    jaw_low_fail = jaw_width < jaw_min
    jaw_high_fail = jaw_width > jaw_max
    jaw_ok = (~jaw_low_fail) & (~jaw_high_fail)
    pre_collision_mask = normal_ok & jaw_ok
    ray_dirs = -n_all

    print(f'candidate_ray_hits: {len(center)}')
    print(f'normal_ok: {int(np.sum(normal_ok))}')
    print(f'jaw_low_fail: {int(np.sum(jaw_low_fail))}')
    print(f'jaw_high_fail: {int(np.sum(jaw_high_fail))}')
    print(f'pre_collision_candidates: {int(np.sum(pre_collision_mask))}')
    if np.any(pre_collision_mask):
        print(f'pre_collision_jaw_width_min: {float(np.min(jaw_width[pre_collision_mask])):.6f}')
        print(f'pre_collision_jaw_width_max: {float(np.max(jaw_width[pre_collision_mask])):.6f}')
    else:
        print('pre_collision_jaw_width_min: none')
        print('pre_collision_jaw_width_max: none')

    print_counter(
        'pre_collision_approach_axes:',
        count_labels(ray_dirs[pre_collision_mask]),
    )

    tic = time.perf_counter()
    results = list(
        antipodal_iter(
            gripper=gripper,
            tgt_sobj=target,
            density=args.density,
            normal_tol_deg=args.normal_tol_deg,
            roll_step_deg=args.roll_step_deg,
            clearance=args.clearance,
        )
    )
    toc = time.perf_counter()

    free_results = [r for r in results if not r[4]]
    collided_results = [r for r in results if r[4]]
    pose_dirs_all = np.asarray([pose[:3, 1] for pose, _, _, _, _ in results], dtype=np.float32) if results else np.zeros((0, 3), dtype=np.float32)
    pose_dirs_free = np.asarray([pose[:3, 1] for pose, _, _, _, c in results if not c], dtype=np.float32) if free_results else np.zeros((0, 3), dtype=np.float32)
    pose_dirs_collided = np.asarray([pose[:3, 1] for pose, _, _, _, c in results if c], dtype=np.float32) if collided_results else np.zeros((0, 3), dtype=np.float32)

    print(f'iter_candidates_total: {len(results)}')
    print(f'iter_candidates_free: {len(free_results)}')
    print(f'iter_candidates_collided: {len(collided_results)}')
    print(f'iter_elapsed_sec: {toc - tic:.3f}')
    print_counter('iter_axes_all:', count_labels(pose_dirs_all))
    print_counter('iter_axes_free:', count_labels(pose_dirs_free))
    print_counter('iter_axes_collided:', count_labels(pose_dirs_collided))

    if args.view:
        visualize_candidates(args.mesh_path, results, args.max_visualized)


if __name__ == '__main__':
    main()
