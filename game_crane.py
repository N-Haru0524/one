from one import oum, ovw, khi_rs007l, ossop, or_2fg7

base = ovw.World(cam_pos=(1.5, 1, 1.5), cam_lookat_pos=(0, 0, .5),
                toggle_auto_cam_orbit=True)
robot = khi_rs007l.RS007L()
robot.attach_to(base.scene)
robot.toggle_render_collision=True

gripper = or_2fg7.OR2FG7(base_pos=(0.3, 0, 0.5))
gripper.attach_to(base.scene)

target_box = ossop.gen_box(spos=(.3, 0, .5), size=(.05, .05, .05))
target_box.attach_to(base.scene)

robot.engage(gripper)

def catch_target(dt):
    tgt_homo = robot.fk(robot.state.qs, update=False)[-1]
    tgt_pos = tgt_homo[:3, 3]
    tgt_rotmat = tgt_homo[:3, :3]
    print("TCP homo:", tgt_homo)
    print("TCP pos:", tgt_pos)
    print("TCP rotmat:", tgt_rotmat)
    ossop.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base.scene)

base.schedule_interval(catch_target, interval=.01)
base.run()