import argparse
import time

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from one import osso, ossop, ouc, ovw
from one.grasp.antipodal import antipodal
from one_assembly.robots.or_2fg7 import or_2fg7

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
        default=0.01,
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
    grasps = antipodal(
        gripper=gripper,
        target_sobj=target,
        density=args.density,
        normal_tol_deg=args.normal_tol_deg,
        roll_step_deg=args.roll_step_deg,
        clearance=args.clearance,
        max_grasps=args.max_grasps,
    )
    toc = time.perf_counter()

    save_grasps_npz(grasps, args.output)

    print(f'mesh: {args.mesh_path}')
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
