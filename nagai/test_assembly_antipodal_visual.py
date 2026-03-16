import argparse
import builtins
import pickle
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from one import ossop, ouc, ovw
from one.grasp.antipodal import antipodal
from one_assembly.worklist import WorkList
from one_assembly.robots.or_2fg7 import or_2fg7

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY_ROOT = ROOT / 'one_assembly' / 'worklists' / 'electric_assembly'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize electric assembly placement and antipodal grasps.')
    parser.add_argument(
        '--target',
        default='belt',
        help='Target part name or object key like object4.')
    parser.add_argument(
        '--layout',
        default='home',
        help='Layout name defined in layouts.yaml.')
    parser.add_argument(
        '--layout-offset',
        nargs=3,
        type=float,
        default=(0.015, 0.0, -0.025),
        metavar=('X', 'Y', 'Z'),
        help='Layout offset passed to WorkList.init_pose.')
    parser.add_argument(
        '--density',
        type=float,
        default=0.01,
        help='Surface sampling density for antipodal grasp generation.')
    parser.add_argument(
        '--normal-tol-deg',
        type=float,
        default=20.0,
        help='Normal tolerance in degrees.')
    parser.add_argument(
        '--roll-step-deg',
        type=float,
        default=30.0,
        help='Roll angle step in degrees.')
    parser.add_argument(
        '--max-grasps',
        type=int,
        default=80,
        help='Maximum number of collision-free grasps to keep.')
    parser.add_argument(
        '--clearance',
        type=float,
        default=0.002,
        help='Jaw clearance used by antipodal grasp generation.')
    parser.add_argument(
        '--show-count',
        type=int,
        default=30,
        help='Number of top grasps to visualize.')
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save generated grasp collection to pickles/<name>_specialized.pickle.')
    return parser.parse_args()


def resolve_target(worklist: WorkList, target_key: str):
    work = worklist.get_work(target_key)
    if work is not None:
        return work
    if target_key.startswith('object') and target_key[6:].isdigit():
        object_num = int(target_key[6:])
        for item in worklist.work:
            if item.obj_num == object_num:
                return item
    raise KeyError(f'Unknown target "{target_key}". Available: {worklist.names()}')


def score_color(fraction: float):
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return (1.0 - fraction, fraction, 0.2)


def print_summary(target_work, grasps, planning_time, layout_name):
    print(f'target: {target_work.name}')
    print(f'layout: {layout_name}')
    print(f'planning_time: {planning_time:.3f}s')
    print(f'grasp_count: {len(grasps)}')
    if not grasps:
        return
    scores = [grasp[3] for grasp in grasps]
    print(f'score_range: [{min(scores):.4f}, {max(scores):.4f}]')
    print('top grasps:')
    for idx, (pose, _pre_pose, jaw_width, score) in enumerate(grasps[:10]):
        pos = pose[:3, 3]
        print(
            f'  {idx + 1:2d}. score={score:.4f} jaw={jaw_width:.4f} '
            f'pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]')


def visualize_grasps(base, gripper, grasps, show_count):
    if not grasps:
        return
    count = min(show_count, len(grasps))
    denom = max(count - 1, 1)
    for idx, (pose, pre_pose, jaw_width, _score) in enumerate(grasps[:count]):
        rgb = score_color(1.0 - idx / denom)

        grasp_ghost = gripper.clone()
        grasp_ghost.grip_at(pose[:3, 3], pose[:3, :3], jaw_width)
        grasp_ghost.rgb = rgb
        grasp_ghost.alpha = 0.28
        grasp_ghost.attach_to(base.scene)

        pre_ghost = gripper.clone()
        pre_ghost.grip_at(pre_pose[:3, 3], pre_pose[:3, :3], jaw_width)
        pre_ghost.rgb = rgb
        pre_ghost.alpha = 0.12
        pre_ghost.attach_to(base.scene)


def maybe_save_grasps(target_work, grasps):
    pickle_dir = ASSEMBLY_ROOT / 'pickles'
    pickle_dir.mkdir(parents=True, exist_ok=True)
    file_path = pickle_dir / f'{target_work.name}_specialized.pickle'
    with open(file_path, 'wb') as f:
        pickle.dump(grasps, f)
    print(f'saved: {file_path}')


def main():
    args = parse_args()

    base = ovw.World(
        cam_pos=(0.55, 0.45, 0.35),
        cam_lookat_pos=(0.18, 0.0, 0.02),
        toggle_auto_cam_orbit=False,
    )
    builtins.base = base
    ossop.frame(ax_length=0.05).attach_to(base.scene)

    worklist = WorkList()
    worklist.init_pose(
        seed=args.layout,
        pos=np.asarray(args.layout_offset, dtype=np.float32))
    worklist.attach_to(base.scene)

    target_work = resolve_target(worklist, args.target)
    target_work.model.alpha = 1.0
    target_work.model.rgb = ouc.ExtendedColor.ORANGE_RED

    target_marker = target_work.model.clone()
    target_marker.alpha = 0.15
    target_marker.rgb = ouc.ExtendedColor.ORANGE_RED
    target_marker.attach_to(base.scene)

    gripper = or_2fg7.OR2FG7()

    print('building antipodal grasp set...')
    tic = time.perf_counter()
    grasps = antipodal(
        gripper=gripper,
        target_sobj=target_work.model,
        density=args.density,
        normal_tol_deg=args.normal_tol_deg,
        roll_step_deg=args.roll_step_deg,
        clearance=args.clearance,
        max_grasps=args.max_grasps,
    )
    toc = time.perf_counter()

    print_summary(target_work, grasps, toc - tic, args.layout)
    visualize_grasps(base, gripper, grasps, args.show_count)

    if args.save:
        maybe_save_grasps(target_work, grasps)

    base.run()


if __name__ == '__main__':
    main()
