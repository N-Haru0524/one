import argparse
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ossop, ouc, ovw
from one_assembly.worklist import WorkList


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize worklist action results for manual inspection.')
    parser.add_argument(
        '--layout',
        default='home',
        help='Layout name defined in layouts.yaml.')
    parser.add_argument(
        '--action-view',
        choices=('state', 'ghost'),
        default='state',
        help='Update place/fold start pose sequentially or keep all actions from the initial state.')
    return parser.parse_args()


def print_action_summary(worklist: WorkList):
    for work_idx, work in enumerate(worklist.work):
        action_names = [step.action_type for step in work.steps]
        print(f'{work_idx}: {work.name} -> {action_names}')


def visualize_action_results(base, worklist: WorkList, action_view='state'):
    worklist.screw_counter = 0
    for work in worklist.work:
        start_pose = work.current_pose
        for step_idx, step in enumerate(work.steps):
            pose = work.pose_after_action(step_idx, start_pose=start_pose)
            if pose is None:
                continue

            if step.action_type == 'place':
                marker = work.model.clone()
                marker.set_rotmat_pos(rotmat=pose[1], pos=pose[0])
                marker.rgb = ouc.BasicColor.GREEN
                marker.alpha = 0.2
                marker.attach_to(base.scene)
            elif step.action_type == 'fold':
                marker = work.model.clone()
                marker.set_rotmat_pos(rotmat=pose[1], pos=pose[0])
                marker.rgb = ouc.BasicColor.YELLOW
                marker.alpha = 0.2
                marker.attach_to(base.scene)
            elif step.action_type == 'screw':
                pick_pose = worklist.get_screw_pose()
                pick_marker = ossop.frame(
                    pos=pick_pose[0],
                    rotmat=pick_pose[1],
                    length_scale=0.14,
                    radius_scale=0.7,
                    color_mat=ouc.CoordColor.DYO,
                    alpha=0.9)
                pick_marker.attach_to(base.scene)

                marker = ossop.frame(
                    pos=pose[0],
                    rotmat=pose[1],
                    length_scale=0.18,
                    radius_scale=0.8,
                    color_mat=ouc.CoordColor.MYC,
                    alpha=0.9)
                marker.attach_to(base.scene)
            else:
                continue

            if action_view == 'state' and step.action_type != 'screw':
                start_pose = pose


def main():
    args = parse_args()

    base = ovw.World(
        cam_pos=(0.55, 0.45, 0.35),
        cam_lookat_pos=(0.18, 0.0, 0.02),
        toggle_auto_cam_orbit=False,
    )
    builtins.base = base
    ossop.frame(ax_length=0.05).attach_to(base.scene)

    worklist = WorkList(collision_type=ouc.CollisionType.MESH)
    worklist.init_pose(seed=args.layout)
    worklist.attach_to(base.scene)

    print_action_summary(worklist)
    visualize_action_results(base, worklist, action_view=args.action_view)
    base.run()


if __name__ == '__main__':
    main()
