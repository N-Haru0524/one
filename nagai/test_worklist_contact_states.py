import argparse
import builtins
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one import ouc, ossop, ovw
import one.scene.scene as oss
import one.physics.mj_env as opme
from one_assembly.assembly_planning import (
    assembly_sequence_planning,
    execute_sequence_string,
    reset_work_state,
    sequence_labels,
)
from one_assembly.precise_collision import precise_mesh_is_collided
from one_assembly.worklist import WorkList


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check pairwise WorkList contacts for every reachable assembly action state.',
    )
    parser.add_argument('--layout', default='home')
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
        '--collision-type',
        default='mesh',
        choices=('mesh', 'aabb', 'obb'),
        help='Collision shape used when building the WorkList.',
    )
    parser.add_argument(
        '--only-collided',
        action='store_true',
        help='Print only states that contain at least one collision.',
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=None,
        help='Optional assembly planning depth limit. Default checks all reachable action states.',
    )
    parser.add_argument(
        '--treat-parts-as-object',
        action='store_true',
        help='Override all WorkList scene objects to OBJECT collision group during contact checks.',
    )
    parser.add_argument(
        '--free-part',
        action='append',
        default=[],
        help='Temporarily set the named part model to is_free=True during contact checks. Repeatable.',
    )
    parser.add_argument(
        '--show-collision',
        action='store_true',
        help='Open a viewer and render collision shapes for a selected state.',
    )
    parser.add_argument(
        '--show-sequence',
        default=None,
        help='Explicit symbolic sequence string to visualize, such as cpctr or wrkbnch-brckt-cpctr.',
    )
    parser.add_argument(
        '--show-first-collided',
        action='store_true',
        help='When visualizing, show the first state that contains contacts under the current overrides.',
    )
    parser.add_argument(
        '--compare-direct-mesh',
        action='store_true',
        help='Also run pairwise direct mesh-mesh checks with one.collider.tbd_collider.',
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
    if use_visual_offset:
        pos = np.array([0.2 + 0.09 + 0.035, 0.0, 0.11 + 0.018 + 0.0902], dtype=np.float32)
        return WorkList(pos=pos, collision_type=collision_type)
    return WorkList(collision_type=collision_type)


def scene_items(worklist):
    items = []
    if worklist.work_base is not None:
        items.append(('work_base', worklist.work_base))
    for work in worklist.work:
        items.append((work.name, work.model))
    return items


def contact_pairs(worklist):
    scene = oss.Scene()
    named_items = scene_items(worklist)
    for _name, item in named_items:
        scene.add(item)
    mjenv = opme.MJEnv(scene, margin=0.0)
    mjenv.runtime.enter_cd()
    collided = mjenv.runtime.is_collided()
    if not collided:
        return []

    body_name_to_label = {}
    for label, sobj in named_items:
        body = mjenv.sync.sobj2bdy.get(sobj)
        if body is not None:
            body_name_to_label[body.name] = label

    contacts = {}
    for idx in range(int(mjenv.data.ncon)):
        contact = mjenv.data.contact[idx]
        body_id_a = int(mjenv.model.geom_bodyid[contact.geom1])
        body_id_b = int(mjenv.model.geom_bodyid[contact.geom2])
        body_name_a = mjenv.model.body(body_id_a).name
        body_name_b = mjenv.model.body(body_id_b).name
        label_a = body_name_to_label.get(body_name_a)
        label_b = body_name_to_label.get(body_name_b)
        if label_a is None or label_b is None or label_a == label_b:
            continue
        pair = tuple(sorted((label_a, label_b)))
        contacts[pair] = contacts.get(pair, 0) + 1

    return [
        {
            'pair': pair,
            'n_points': n_points,
        }
        for pair, n_points in sorted(contacts.items())
    ]


def contact_points(worklist):
    scene = oss.Scene()
    named_items = scene_items(worklist)
    for _name, item in named_items:
        scene.add(item)
    mjenv = opme.MJEnv(scene, margin=0.0)
    mjenv.runtime.enter_cd()
    collided = mjenv.runtime.is_collided()
    if not collided:
        return []
    points = []
    for idx in range(int(mjenv.data.ncon)):
        contact = mjenv.data.contact[idx]
        points.append(np.asarray(contact.pos, dtype=np.float32).copy())
    return points


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
    if not requested:
        return previous
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


def apply_contact_overrides(worklist, args):
    previous_free = None
    previous_groups = None
    if args.free_part:
        previous_free = set_parts_free(worklist, args.free_part)
    if args.treat_parts_as_object:
        previous_groups = set_parts_collision_group(worklist, ouc.CollisionGroup.OBJECT)
    return previous_free, previous_groups


def restore_contact_overrides(previous_free, previous_groups):
    if previous_groups is not None:
        restore_collision_group(previous_groups)
    if previous_free is not None:
        restore_parts_free(previous_free)


def print_state_report(node, contacts, direct_mesh_contacts=None):
    seq = sequence_labels(node)
    seq_text = 'root' if not seq else ' -> '.join(seq)
    print(f'state: {seq_text}')
    if not contacts:
        print('  contacts: none')
    else:
        for item in contacts:
            name_a, name_b = item['pair']
            print(f'  contact: {name_a} <-> {name_b}, points={item["n_points"]}')
    if direct_mesh_contacts is None:
        return
    if not direct_mesh_contacts:
        print('  direct_mesh: none')
        return
    for item in direct_mesh_contacts:
        name_a, name_b = item['pair']
        print(f'  direct_mesh: {name_a} <-> {name_b}, points={item["n_points"]}')


def direct_mesh_pairs(worklist):
    named_items = scene_items(worklist)
    results = []
    for idx, (name_a, sobj_a) in enumerate(named_items):
        if len(getattr(sobj_a, 'collisions', ())) != 1 or getattr(sobj_a, '_collision_type', None) != ouc.CollisionType.MESH:
            continue
        for name_b, sobj_b in named_items[idx + 1:]:
            if len(getattr(sobj_b, 'collisions', ())) != 1 or getattr(sobj_b, '_collision_type', None) != ouc.CollisionType.MESH:
                continue
            points = precise_mesh_is_collided(sobj_a, sobj_b)
            if points is None or len(points) == 0:
                continue
            results.append({
                'pair': (name_a, name_b),
                'n_points': len(points),
            })
    return results


def find_visual_state(args, nodes, worklist):
    if args.show_sequence:
        result = execute_sequence_string(worklist, args.show_sequence, layout_name=args.layout)
        return result.work_states[-1] if result.work_states else reset_work_state(worklist, layout_name=args.layout)
    if args.show_first_collided or args.show_collision:
        ordered_nodes = sorted(nodes.values(), key=lambda node: (node.depth, node.node_id))
        for node in ordered_nodes:
            if node.work_state is None:
                continue
            reset_work_state(worklist, state=node.work_state)
            previous_free, previous_groups = apply_contact_overrides(worklist, args)
            try:
                contacts = contact_pairs(worklist)
            finally:
                restore_contact_overrides(previous_free, previous_groups)
            if contacts:
                return node.work_state
    return reset_work_state(worklist, layout_name=args.layout)


def visualize_state(worklist, state):
    base = ovw.World(
        cam_pos=(0.55, 0.45, 0.35),
        cam_lookat_pos=(0.18, 0.0, 0.02),
        toggle_auto_cam_orbit=False,
    )
    builtins.base = base
    ossop.frame(ax_length=0.05).attach_to(base.scene)
    reset_work_state(worklist, state=state)
    set_collision_rendering(worklist, flag=True)
    worklist.attach_to(base.scene)
    for pos in contact_points(worklist):
        marker = ossop.sphere(
            pos=pos,
            radius=0.003,
            rgb=ouc.BasicColor.RED,
            alpha=0.5,
            collision_type=None,
        )
        marker.attach_to(base.scene)
    base.run()


def main():
    args = parse_args()
    collision_type = resolve_collision_type(args.collision_type)
    worklist = build_worklist(
        collision_type=collision_type,
        use_visual_offset=args.use_visual_offset,
    )
    nodes = assembly_sequence_planning(
        worklist,
        initial_layout=args.layout,
        max_depth=args.max_depth,
    )
    ordered_nodes = sorted(nodes.values(), key=lambda node: (node.depth, node.node_id))

    total_states = 0
    collided_states = 0
    total_contacts = 0
    for node in ordered_nodes:
        if node.work_state is None:
            continue
        total_states += 1
        reset_work_state(worklist, state=node.work_state)
        previous_free, previous_groups = apply_contact_overrides(worklist, args)
        try:
            contacts = contact_pairs(worklist)
            direct_contacts = direct_mesh_pairs(worklist) if args.compare_direct_mesh else None
        finally:
            restore_contact_overrides(previous_free, previous_groups)
        if contacts:
            collided_states += 1
            total_contacts += len(contacts)
        if args.only_collided and not contacts:
            continue
        print_state_report(node, contacts, direct_mesh_contacts=direct_contacts)

    print('summary:')
    print(f'  states_checked: {total_states}')
    print(f'  collided_states: {collided_states}')
    print(f'  total_contact_pairs: {total_contacts}')
    if args.show_collision:
        visual_worklist = build_worklist(
            collision_type=collision_type,
            use_visual_offset=args.use_visual_offset,
        )
        visual_state = find_visual_state(args, nodes, visual_worklist)
        previous_free, previous_groups = apply_contact_overrides(visual_worklist, args)
        try:
            visualize_state(visual_worklist, visual_state)
        finally:
            restore_contact_overrides(previous_free, previous_groups)


if __name__ == '__main__':
    main()
