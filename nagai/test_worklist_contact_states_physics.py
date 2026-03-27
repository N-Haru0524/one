import argparse
import builtins
import sys
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ouc, ossop, ovw
import one.scene.scene as oss
import one.physics.mj_env as opme
from one_assembly.assembly_planning import execute_sequence_string, reset_work_state
from one_assembly.worklist import WorkList


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run MuJoCo physics on one WorkList state and inspect contacts.',
    )
    parser.add_argument('--layout', default='home')
    parser.add_argument(
        '--sequence',
        default=None,
        help='Optional symbolic sequence string. If omitted, simulate the root state.',
    )
    parser.add_argument(
        '--collision-type',
        default='mesh',
        choices=('mesh', 'aabb', 'obb'),
        help='Collision shape used when building the WorkList.',
    )
    parser.add_argument(
        '--use-visual-offset',
        action='store_true',
        default=True,
        help='Use the same WorkList base offset as test_assembly_sequence_visual.py.',
    )
    parser.add_argument(
        '--no-use-visual-offset',
        dest='use_visual_offset',
        action='store_false',
        help='Use the default WorkList origin instead of the visual test offset.',
    )
    parser.add_argument(
        '--free-part',
        action='append',
        default=[],
        help='Temporarily set the named part model to is_free=True before simulating. Repeatable.',
    )
    parser.add_argument(
        '--treat-parts-as-object',
        action='store_true',
        help='Override all WorkList scene objects to OBJECT collision group before simulating.',
    )
    parser.add_argument('--margin', type=float, default=0.0)
    parser.add_argument('--duration', type=float, default=0.5, help='Simulation duration in seconds.')
    parser.add_argument('--dt', type=float, default=0.5, help='Viewer update interval in seconds.')
    parser.add_argument('--show-collision', action='store_true')
    parser.add_argument('--show-contact-points', action='store_true')
    parser.add_argument(
        '--print-contact-updates',
        action='store_true',
        help='Print contact pair changes while the simulation runs.',
    )
    return parser.parse_args()


def resolve_collision_type(label):
    mapping = {
        'mesh': ouc.CollisionType.MESH,
        'aabb': ouc.CollisionType.AABB,
        'obb': ouc.CollisionType.OBB,
    }
    return mapping[label]


def build_worklist(collision_type, use_visual_offset=True):
    root_path = os.path.join('/home/wrs/nagai/one/one_assembly/worklists/move_object')
    yaml_path = os.path.join(root_path, 'yamls')
    mesh_path = os.path.join(root_path, 'meshes')
    grasp_path = os.path.join(root_path, 'grasps')
    if use_visual_offset:
        pos = np.array([0.2 + 0.09 + 0.035, 0.0, 0.11 + 0.018 + 0.0902], dtype=np.float32)
        return WorkList(pos=pos, collision_type=collision_type, yaml_path=yaml_path, mesh_path=mesh_path, grasp_path=grasp_path)
    return WorkList(collision_type=collision_type, yaml_path=yaml_path, mesh_path=mesh_path, grasp_path=grasp_path)


def scene_items(worklist):
    items = []
    if worklist.work_base is not None:
        items.append(('work_base', worklist.work_base))
    for work in worklist.work:
        items.append((work.name, work.model))
    return items


def set_parts_collision_group(worklist, group):
    items = []
    if worklist.work_base is not None:
        items.append(worklist.work_base)
    items.extend(work.model for work in worklist.work)
    previous = []
    for item in items:
        previous.append((item, getattr(item, '_collision_group_override', None)))
        item.collision_group = group
    return previous


def restore_collision_group(previous):
    for item, override in previous:
        item._collision_group_override = override


def set_parts_free(worklist, part_names):
    requested = {name.strip() for name in part_names if name and name.strip()}
    previous = []
    for work in worklist.work:
        if work.name not in requested:
            continue
        previous.append((work.model, bool(work.model.is_free)))
        work.model.is_free = True
    return previous


def restore_parts_free(previous):
    for item, is_free in previous:
        item.is_free = is_free


def set_collision_rendering(worklist, flag=True):
    if worklist.work_base is not None:
        worklist.work_base.toggle_render_collision = flag
    for work in worklist.work:
        work.model.toggle_render_collision = flag


def contact_report(mjenv, named_items):
    body_name_to_label = {}
    for label, sobj in named_items:
        body = mjenv.sync.sobj2bdy.get(sobj)
        if body is not None:
            body_name_to_label[body.name] = label
    contacts = {}
    points = []
    for idx in range(int(mjenv.data.ncon)):
        contact = mjenv.data.contact[idx]
        body_id_a = int(mjenv.model.geom_bodyid[contact.geom1])
        body_id_b = int(mjenv.model.geom_bodyid[contact.geom2])
        body_name_a = mjenv.model.body(body_id_a).name
        body_name_b = mjenv.model.body(body_id_b).name
        label_a = body_name_to_label.get(body_name_a)
        label_b = body_name_to_label.get(body_name_b)
        points.append(np.asarray(contact.pos, dtype=np.float32).copy())
        if label_a is None or label_b is None or label_a == label_b:
            continue
        pair = tuple(sorted((label_a, label_b)))
        contacts[pair] = contacts.get(pair, 0) + 1
    return contacts, points


def attach_contact_markers(base, max_contacts=64, radius=0.003):
    markers = []
    for _ in range(max_contacts):
        marker = ossop.sphere(
            pos=np.zeros(3, dtype=np.float32),
            radius=radius,
            rgb=ouc.BasicColor.RED,
            alpha=0.0,
            collision_type=None,
        )
        marker.attach_to(base.scene)
        markers.append(marker)
    return markers


def update_contact_markers(markers, points):
    for idx, marker in enumerate(markers):
        if idx < len(points):
            marker.pos = points[idx]
            marker.alpha = 0.5
        else:
            marker.alpha = 0.0


def main():
    args = parse_args()
    collision_type = resolve_collision_type(args.collision_type)
    worklist = build_worklist(collision_type, use_visual_offset=args.use_visual_offset)
    worklist.init_pose(seed=args.layout)
    if args.sequence:
        result = execute_sequence_string(worklist, args.sequence, layout_name=args.layout)
        state = result.work_states[-1] if result.work_states else reset_work_state(worklist, layout_name=args.layout)
    else:
        state = reset_work_state(worklist, layout_name=args.layout)
    reset_work_state(worklist, state=state)

    previous_free = set_parts_free(worklist, args.free_part) if args.free_part else None
    previous_groups = set_parts_collision_group(worklist, ouc.CollisionGroup.OBJECT) if args.treat_parts_as_object else None

    scene = oss.Scene()
    named_items = scene_items(worklist)
    for _name, item in named_items:
        scene.add(item)
    mjenv = opme.MJEnv(scene, margin=args.margin)

    if args.show_collision:
        base = ovw.World(
            cam_pos=(1,-1,1),
            cam_lookat_pos=(0,0,0.5),
            toggle_auto_cam_orbit=False,
        )
        builtins.base = base
        # ossop.frame(ax_length=0.05).attach_to(base.scene)
        set_collision_rendering(worklist, flag=True)
        worklist.attach_to(base.scene)
        markers = attach_contact_markers(base) if args.show_contact_points else []
        elapsed = {'t': 0.0}
        last_contacts = {'pairs': None}

        print(f'sequence: {args.sequence or "root"}')

        def update(_dt):
            if elapsed['t'] >= args.duration:
                return
            mjenv.step(args.dt)
            elapsed['t'] += args.dt
            contacts, points = contact_report(mjenv, named_items)
            if args.show_contact_points:
                update_contact_markers(markers, points)
            pair_items = tuple(sorted(contacts.items()))
            if args.print_contact_updates and pair_items != last_contacts['pairs']:
                print(f't={elapsed["t"]:.3f}s ncon={int(mjenv.data.ncon)}')
                if not contacts:
                    print('contacts: none')
                else:
                    for pair, count in pair_items:
                        print(f'contact: {pair[0]} <-> {pair[1]}, points={count}')
                last_contacts['pairs'] = pair_items

        update(0.0)
        base.schedule_interval(update, args.dt)
        base.run()
    else:
        n_steps = max(1, int(round(args.duration / mjenv.get_timestep())))
        mjenv.runtime.exit_cd()
        mjenv.runtime.step(n_steps)
        mjenv.sync.pull_all_sobj_pose()
        contacts, _points = contact_report(mjenv, named_items)
        print(f'sequence: {args.sequence or "root"}')
        print(f'ncon: {int(mjenv.data.ncon)}')
        if not contacts:
            print('contacts: none')
        else:
            for pair, count in sorted(contacts.items()):
                print(f'contact: {pair[0]} <-> {pair[1]}, points={count}')

    if previous_groups is not None:
        restore_collision_group(previous_groups)
    if previous_free is not None:
        restore_parts_free(previous_free)


if __name__ == '__main__':
    main()
