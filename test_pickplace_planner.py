import builtins
import numpy as np

from one import oum, ovw, ouc, ossop, osso, khi_rs007l, or_2fg7
from one.grasp.antipodal import antipodal
from planner import utils
from planner.ppplanner import PickPlacePlanner


def main():
    base = ovw.World(cam_pos=(-1.2, 1.2, 1.0), cam_lookat_pos=(0.3, 0.0, 0.3))
    builtins.base = base
    ossop.gen_frame().attach_to(base.scene)

    robot = khi_rs007l.RS007L()
    robot.set_rotmat_pos(pos=(0.0, 0.0, 0.01))
    gripper = or_2fg7.OR2FG7()
    robot.engage(gripper)
    robot.attach_to(base.scene)

    table = ossop.gen_box(half_extents=(0.4, 0.4, 0.02), pos=(0.4, 0.0, 0.01),
                          collision_type=ouc.CollisionType.AABB)
    table.rgb = ouc.ExtendedColor.SILVER_GRAY
    table.attach_to(base.scene)

    belt = osso.SceneObject.from_file('meshes/belt.stl',
                                      collision_type=ouc.CollisionType.MESH)
    belt.set_rotmat_pos(pos=(0.35, 0.0, 0.06))
    belt.attach_to(base.scene)

    target_pose = (np.array([0.55, 0.0, 0.06], dtype=np.float32), belt.rotmat)
    target_marker = belt.clone()
    target_marker.set_rotmat_pos(rotmat=target_pose[1], pos=target_pose[0])
    target_marker.alpha = 0.2
    target_marker.attach_to(base.scene)

    print('Computing grasps for belt...')
    grasps = antipodal(scene_obj=belt, gripper=gripper, obstacles=[belt],
                       density=0.01, normal_tol_deg=20, roll_step_deg=30,
                       max_grasps=10)
    if not grasps:
        raise RuntimeError('No valid grasps found for meshes/belt.stl')

    collider = utils.build_collider([robot], obstacles=[table])
    print('Start state collided?', collider.is_collided(robot.qs))

    planner = PickPlacePlanner(robot, collider=collider)
    pick_plan = None
    place_plan = None
    pick_pose_tf = None
    jaw_width = None
    for idx, (pose_tf, jw, score) in enumerate(grasps):
        ik_solutions = robot.ik_tcp(pose_tf[:3, :3], pose_tf[:3, 3])
        print(f'Grasp {idx + 1}: score={score:.4f}, ik={len(ik_solutions)}')
        if not ik_solutions:
            continue
        pick_pose_tf = pose_tf
        jaw_width = jw
        place_pose_tf = oum.tf_from_rotmat_pos(pose_tf[:3, :3], target_pose[0])
        pick_plan = planner.plan_pick(pose_tf, jw, obstacles=[table],
                                      step_size=np.pi / 72,
                                      max_iters=5000)
        if pick_plan is None:
            print('  pick plan failed with table, retrying without obstacles')
            pick_plan = planner.plan_pick(pose_tf, jw, obstacles=None,
                                          step_size=np.pi / 72,
                                          max_iters=5000)
            if pick_plan is None:
                print('  pick plan failed')
                continue
        place_plan = planner.plan_place(place_pose_tf, start_qs=pick_plan.qs_list[-1],
                                        obstacles=[table],
                                        step_size=np.pi / 72,
                                        max_iters=5000)
        if place_plan is None:
            print('  place plan failed with table, retrying without obstacles')
            place_plan = planner.plan_place(place_pose_tf, start_qs=pick_plan.qs_list[-1],
                                            obstacles=None,
                                            step_size=np.pi / 72,
                                            max_iters=5000)
            if place_plan is None:
                print('  place plan failed')
                pick_plan = None
                continue
        break

    if pick_plan is None or place_plan is None:
        raise RuntimeError('Pick/place planning failed for all grasps')

    plan_qs = pick_plan.qs_list + place_plan.qs_list[1:]
    pick_end_idx = len(pick_plan.qs_list) - 1
    place_end_idx = len(plan_qs) - 1
    start_pose = (belt.pos.copy(), belt.rotmat.copy())

    def reset_scene():
        if belt in gripper._mountings:
            gripper.release(belt)
        belt.set_rotmat_pos(rotmat=start_pose[1], pos=start_pose[0])
        gripper.open()

    def update(dt, state):
        if state['idx'] >= len(plan_qs):
            reset_scene()
            state['idx'] = 0
            return
        robot.fk(qs=plan_qs[state['idx']])
        if state['idx'] == pick_end_idx:
            gripper.grasp(belt, jaw_width=jaw_width)
        if state['idx'] == place_end_idx:
            gripper.release(belt)
            gripper.open()
        state['idx'] += 1

    reset_scene()
    base.schedule_interval(update, interval=0.1, state={'idx': 0})
    base.run()


if __name__ == '__main__':
    main()
