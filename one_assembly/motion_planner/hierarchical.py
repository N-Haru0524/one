from dataclasses import dataclass
from pathlib import Path
import builtins
import time

import one.utils.constant as ouc
import one.utils.math as oum
from one import ossop
import one.robots.base.mech_base as orbmb

from .approach_depart_planner import ADPlanner
from . import utils


@dataclass
class ScreenResult:
    goal_qs: oum.np.ndarray
    ee_qs: oum.np.ndarray | None
    depart_plan: utils.MotionData | None = None


@dataclass
class CandidateRecord:
    key: int
    pose_tf: oum.np.ndarray
    screen_result: ScreenResult


class HierarchicalPlannerBase(ADPlanner):
    def __init__(self,
                 robot,
                 pln_ctx,
                 ee_actor=None,
                 reasoning_backend='simd',
                 transit_backend=None):
        super().__init__(robot=robot, pln_ctx=pln_ctx, ee_actor=ee_actor)
        if reasoning_backend not in ('mujoco', 'simd'):
            raise ValueError(f'Unsupported reasoning backend: {reasoning_backend}')
        if transit_backend is None:
            transit_backend = reasoning_backend
        if transit_backend not in ('mujoco', 'simd'):
            raise ValueError(f'Unsupported transit backend: {transit_backend}')
        self._filtered_pln_ctx_cache = {}
        self._hold_pln_ctx_cache = {}
        self._specialized_pln_ctx_cache = {}
        self.reasoning_backend = reasoning_backend
        self.transit_backend = transit_backend

    def _tag_mecba_debug_links(self, mecba, owner_label):
        mecba._debug_owner_label = owner_label
        for lnk in getattr(mecba, 'runtime_lnks', ()):
            file_path = getattr(lnk, 'file_path', None)
            stem = Path(file_path).stem if isinstance(file_path, str) and file_path else type(lnk).__name__
            lnk._debug_label = f'{owner_label}/{stem}'

    def _grasp_fields(self, grasp):
        if not isinstance(grasp, (tuple, list)) or len(grasp) < 4:
            raise ValueError('grasp must be a tuple like (pose_tf, pre_pose_tf, jaw_width, score)')
        pose_tf = oum.np.asarray(grasp[0], dtype=oum.np.float32)
        pre_pose_tf = oum.np.asarray(grasp[1], dtype=oum.np.float32)
        jaw_width = float(oum.np.asarray(grasp[2], dtype=oum.np.float32).reshape(-1)[0])
        score = float(grasp[3])
        if pose_tf.shape != (4, 4):
            raise ValueError('grasp pose_tf must be a (4, 4) transform')
        if pre_pose_tf.shape != (4, 4):
            raise ValueError('grasp pre_pose_tf must be a (4, 4) transform')
        return pose_tf, pre_pose_tf, jaw_width, score

    def _grasp_has_collision(self, grasp):
        if not isinstance(grasp, (tuple, list)) or len(grasp) < 5:
            return False
        return bool(grasp[4])

    def _pose_to_tf(self, pose):
        if isinstance(pose, (tuple, list)) and len(pose) == 2:
            return oum.tf_from_rotmat_pos(pose[1], pose[0])
        pose = oum.np.asarray(pose, dtype=oum.np.float32)
        if pose.shape == (4, 4):
            return pose
        raise ValueError('pose must be a (pos, rotmat) pair or a (4, 4) transform')

    def _scene_entities(self, exclude_entities=None):
        excluded_ids = set() if exclude_entities is None else {id(entity) for entity in exclude_entities}
        entities = []
        for sobj in self.pln_ctx.collider.scene.sobjs:
            if id(sobj) not in excluded_ids:
                entities.append(sobj)
        for mecba in self.pln_ctx.collider.scene.mecbas:
            if id(mecba) not in excluded_ids:
                entities.append(mecba)
        return entities

    def _pruned_scene_entities(self, exclude_entities=None):
        excluded_ids = set() if exclude_entities is None else {id(entity) for entity in exclude_entities}
        entities = []
        for sobj in self.pln_ctx.collider.scene.sobjs:
            if id(sobj) not in excluded_ids:
                entities.append(sobj)
        for mecba in self.pln_ctx.collider.scene.mecbas:
            if id(mecba) in excluded_ids:
                continue
            if any(id(child) in excluded_ids for child in getattr(mecba, '_mountings', {}).keys()):
                entities.append(self._clone_mechbase_without_excluded_children(mecba, excluded_ids))
            else:
                entities.append(mecba)
        return entities

    def _clone_mechbase_without_excluded_children(self, mecba, excluded_ids):
        clone = mecba.clone()
        self._prune_mechbase_clone_mountings(mecba, clone, excluded_ids)
        clone.fk(clone.qs)
        return clone

    def _prune_mechbase_clone_mountings(self, original, clone, excluded_ids):
        kept_mountings = {}
        original_items = list(getattr(original, '_mountings', {}).items())
        clone_items = list(getattr(clone, '_mountings', {}).items())
        for (orig_child, _orig_mount), (clone_child, clone_mount) in zip(original_items, clone_items):
            if id(orig_child) in excluded_ids:
                continue
            if isinstance(orig_child, orbmb.MechBase):
                self._prune_mechbase_clone_mountings(orig_child, clone_child, excluded_ids)
            kept_mountings[clone_child] = clone_mount
        clone._mountings = kept_mountings

    def _ctx_cache_key(self, exclude_entities=None, actors=None):
        exclude_entities = () if exclude_entities is None else tuple(sorted(id(entity) for entity in exclude_entities))
        actors = self._desired_actors() if actors is None else actors
        actor_ids = tuple(id(actor) for actor in actors)
        return actor_ids, exclude_entities

    def _hold_ctx_cache_key(self,
                            obj_model,
                            exclude_entities=None,
                            backend=None,
                            actor_link_masks=None,
                            actor_include_mounted=None,
                            collision_mode=None):
        exclude_entities = () if exclude_entities is None else tuple(sorted(id(entity) for entity in exclude_entities))
        if actor_link_masks is None:
            link_mask_key = ()
        else:
            link_mask_key = tuple(
                sorted((id(actor), None if mask is None else tuple(sorted(mask))) for actor, mask in actor_link_masks.items())
            )
        if actor_include_mounted is None:
            mounted_key = ()
        else:
            mounted_key = tuple(sorted((id(actor), bool(flag)) for actor, flag in actor_include_mounted.items()))
        return (
            id(obj_model),
            exclude_entities,
            self.reasoning_backend if backend is None else backend,
            link_mask_key,
            mounted_key,
            collision_mode,
        )

    def _specialized_ctx_cache_key(self,
                                   label,
                                   exclude_entities=None,
                                   actors=None,
                                   backend=None,
                                   actor_link_masks=None,
                                   actor_include_mounted=None):
        exclude_ids = () if exclude_entities is None else tuple(sorted(id(entity) for entity in exclude_entities))
        actor_ids = () if actors is None else tuple(id(actor) for actor in actors)
        if actor_link_masks is None:
            link_mask_key = ()
        else:
            link_mask_key = tuple(
                sorted((id(actor), None if mask is None else tuple(sorted(mask))) for actor, mask in actor_link_masks.items())
            )
        if actor_include_mounted is None:
            mounted_key = ()
        else:
            mounted_key = tuple(sorted((id(actor), bool(flag)) for actor, flag in actor_include_mounted.items()))
        return label, exclude_ids, actor_ids, self.reasoning_backend if backend is None else backend, link_mask_key, mounted_key

    def _specialized_pln_ctx(self,
                             label,
                             exclude_entities=None,
                             actors=None,
                             backend=None,
                             actor_link_masks=None,
                             actor_include_mounted=None):
        desired_actors = self._desired_actors() if actors is None else list(actors)
        cache_key = self._specialized_ctx_cache_key(
            label,
            exclude_entities=exclude_entities,
            actors=desired_actors,
            backend=backend,
            actor_link_masks=actor_link_masks,
            actor_include_mounted=actor_include_mounted,
        )
        if cache_key in self._specialized_pln_ctx_cache:
            return self._specialized_pln_ctx_cache[cache_key]
        excluded = [] if exclude_entities is None else list(exclude_entities)
        excluded.extend(entity for entity in desired_actors if entity not in excluded)
        collider = utils.build_collider(
            desired_actors,
            obstacles=self._scene_entities(excluded),
            backend=self.reasoning_backend if backend is None else backend,
            actor_link_masks=actor_link_masks,
            actor_include_mounted=actor_include_mounted,
        )
        pln_ctx = utils.build_planning_context(collider, max_edge_step=self.pln_ctx.cd_step_size)
        self._specialized_pln_ctx_cache[cache_key] = pln_ctx
        return pln_ctx

    def _all_link_indices(self, actor):
        return tuple(range(len(getattr(actor, 'runtime_lnks', ()))))

    def _empty_link_indices(self):
        return tuple()

    def _arm_only_goal_pln_ctx(self, exclude_entities=None):
        if self.ee_actor is None:
            return self._specialized_pln_ctx(
                label='arm_only_goal',
                exclude_entities=exclude_entities,
                actors=[self.robot],
                backend='mujoco',
                actor_link_masks={self.robot: self._all_link_indices(self.robot)},
                actor_include_mounted={self.robot: False},
            )
        return self._specialized_pln_ctx(
            label='arm_only_goal',
            exclude_entities=exclude_entities,
            actors=[self.robot, self.ee_actor],
            backend='mujoco',
            actor_link_masks={
                self.robot: self._all_link_indices(self.robot),
                self.ee_actor: self._empty_link_indices(),
            },
            actor_include_mounted={
                self.robot: False,
                self.ee_actor: False,
            },
        )

    def _ee_local_goal_pln_ctx(self, exclude_entities=None, backend='simd'):
        if self.ee_actor is None:
            return self._specialized_pln_ctx(
                label='ee_local_goal',
                exclude_entities=exclude_entities,
                actors=[self.robot],
                backend=backend,
                actor_link_masks={self.robot: self._empty_link_indices()},
                actor_include_mounted={self.robot: False},
            )
        return self._specialized_pln_ctx(
            label='ee_local_goal',
            exclude_entities=exclude_entities,
            actors=[self.robot, self.ee_actor],
            backend=backend,
            actor_link_masks={
                self.robot: self._empty_link_indices(),
                self.ee_actor: self._all_link_indices(self.ee_actor),
            },
            actor_include_mounted={
                self.robot: False,
                self.ee_actor: True,
            },
        )

    def _filtered_pln_ctx(self, exclude_entities=None, actors=None):
        if exclude_entities is None:
            return self.pln_ctx
        desired_actors = self._desired_actors() if actors is None else list(actors)
        cache_key = self._ctx_cache_key(exclude_entities=exclude_entities, actors=desired_actors)
        if cache_key in self._filtered_pln_ctx_cache:
            return self._filtered_pln_ctx_cache[cache_key]
        excluded = list(exclude_entities) + list(desired_actors)
        collider = utils.build_collider(desired_actors, obstacles=self._scene_entities(excluded))
        pln_ctx = utils.build_planning_context(collider, max_edge_step=self.pln_ctx.cd_step_size)
        self._filtered_pln_ctx_cache[cache_key] = pln_ctx
        return pln_ctx

    def _backend_pln_ctx(self, exclude_entities=None, actors=None, backend=None):
        desired_actors = self._desired_actors() if actors is None else list(actors)
        excluded = [] if exclude_entities is None else list(exclude_entities)
        excluded.extend(desired_actors)
        collider = utils.build_collider(
            desired_actors,
            obstacles=self._scene_entities(excluded),
            backend=self.reasoning_backend if backend is None else backend,
        )
        return utils.build_planning_context(collider, max_edge_step=self.pln_ctx.cd_step_size)

    def _precise_pln_ctx(self, exclude_entities=None, actors=None, backend=None):
        return self._backend_pln_ctx(exclude_entities=exclude_entities, actors=actors, backend=backend)

    def _resolve_goal_ee_qs(self, ee_value=None):
        if self.ee_actor is None:
            return None
        if ee_value is None:
            return self._default_ee_qs()
        ee_value_arr = oum.np.asarray(ee_value, dtype=oum.np.float32)
        if ee_value_arr.ndim == 0 or ee_value_arr.size == 1:
            if hasattr(self.ee_actor, 'set_jaw_width'):
                ee_actor = self.ee_actor.clone()
                ee_actor.set_jaw_width(float(ee_value_arr.reshape(-1)[0]))
                return oum.np.asarray(ee_actor.qs[:ee_actor.ndof], dtype=oum.np.float32)
        return self._resolve_ee_qs(ee_value_arr)

    def _grasp_world_data(self, grasp, obj_pose):
        obj_tf = self._pose_to_tf(obj_pose)
        pose_tf, pre_pose_tf, jaw_width, score = self._grasp_fields(grasp)
        pose_tf = obj_tf @ pose_tf
        pre_pose_tf = obj_tf @ pre_pose_tf
        ee_qs = self._resolve_goal_ee_qs(jaw_width)
        return pose_tf, pre_pose_tf, ee_qs, jaw_width, score

    def _grasp_pose(self, grasp, obj_pose):
        pose_tf, _pre_pose_tf, ee_qs, _jaw_width, _score = self._grasp_world_data(grasp, obj_pose)
        return pose_tf, ee_qs

    def _grasp_pre_pose(self, grasp, obj_pose):
        _pose_tf, pre_pose_tf, ee_qs, _jaw_width, _score = self._grasp_world_data(grasp, obj_pose)
        return pre_pose_tf, ee_qs

    def _max_open_ee_qs(self):
        if self.ee_actor is None:
            return None
        if hasattr(self.ee_actor, 'set_jaw_width') and hasattr(self.ee_actor, 'jaw_range'):
            ee_actor = self.ee_actor.clone()
            ee_actor.set_jaw_width(float(oum.np.asarray(ee_actor.jaw_range, dtype=oum.np.float32)[1]))
            return oum.np.asarray(ee_actor.qs[:ee_actor.ndof], dtype=oum.np.float32)
        return self._default_ee_qs()

    def _compose_with_ee_qs(self, qs, ee_qs):
        robot_qs, _ = self._split_state(qs, ee_values=ee_qs)
        return self._compose_state(robot_qs, ee_qs)

    def _mounted_ee_actor(self, robot_clone):
        if self.ee_actor is None:
            return None
        for mounting in robot_clone._mountings.values():
            if isinstance(mounting.child, type(self.ee_actor)):
                return mounting.child
        return None

    def _release_mounted_objects(self):
        if self.ee_actor is None or not hasattr(self.ee_actor, '_mountings'):
            return []
        mounted_children = list(self.ee_actor._mountings.keys())
        for child in mounted_children:
            if hasattr(self.ee_actor, 'release'):
                self.ee_actor.release(child)
            elif hasattr(self.ee_actor, 'detach'):
                self.ee_actor.detach(child)
            else:
                self.ee_actor.unmount(child)
        return mounted_children

    def _hold_pln_ctx(self,
                      obj_model,
                      grasp,
                      pick_pose,
                      exclude_entities=None,
                      backend=None,
                      actor_link_masks=None,
                      actor_include_mounted=None,
                      collision_mode=None):
        if self.ee_actor is None:
            return self._filtered_pln_ctx(exclude_entities=exclude_entities)
        if obj_model is None or not hasattr(obj_model, 'clone'):
            return self._filtered_pln_ctx(exclude_entities=exclude_entities)
        cache_key = self._hold_ctx_cache_key(
            obj_model,
            exclude_entities=exclude_entities,
            backend=backend,
            actor_link_masks=actor_link_masks,
            actor_include_mounted=actor_include_mounted,
        )
        cache_entry = self._hold_pln_ctx_cache.get(cache_key)
        pick_tf = self._pose_to_tf(pick_pose)
        pose_tf, _pre_pose_tf, _ee_qs, jaw_width, _score = self._grasp_world_data(grasp, obj_pose=pick_pose)
        if cache_entry is None:
            robot_clone = self.robot.clone()
            ee_clone = self._mounted_ee_actor(robot_clone)
            if ee_clone is None:
                raise RuntimeError('Failed to locate mounted end effector clone for hold planning context')
            obj_clone = obj_model.clone()
            obj_clone.is_free = True
            ee_root_tf = pose_tf @ oum.tf_inverse(ee_clone.loc_tcp_tf)
            engage_tf = oum.tf_inverse(ee_root_tf) @ pick_tf
            obj_clone.set_rotmat_pos(rotmat=pick_tf[:3, :3], pos=pick_tf[:3, 3])
            ee_clone.set_jaw_width(jaw_width)
            ee_clone.mount(obj_clone, ee_clone.runtime_root_lnk, engage_tf)
            obj_mounting = ee_clone._mountings[obj_clone]
            ee_clone._update_mounting(obj_mounting)
            robot_clone.fk(robot_clone.qs)
            desired_actors = [robot_clone, ee_clone]
            link_masks = actor_link_masks
            include_mounted = actor_include_mounted
            if collision_mode == 'arm_only':
                link_masks = {
                    robot_clone: self._all_link_indices(robot_clone),
                    ee_clone: self._empty_link_indices(),
                }
                include_mounted = {
                    robot_clone: False,
                    ee_clone: False,
                }
            elif collision_mode == 'ee_only':
                link_masks = {
                    robot_clone: self._empty_link_indices(),
                    ee_clone: self._all_link_indices(ee_clone),
                }
                include_mounted = {
                    robot_clone: False,
                    ee_clone: True,
                }
            excluded = [] if exclude_entities is None else list(exclude_entities)
            excluded.extend(self._desired_actors())
            excluded.append(obj_model)
            collider = utils.build_collider(
                desired_actors,
                obstacles=self._pruned_scene_entities(excluded),
                backend=self.reasoning_backend if backend is None else backend,
                actor_link_masks=link_masks,
                actor_include_mounted=include_mounted,
            )
            pln_ctx = utils.build_planning_context(collider, max_edge_step=self.pln_ctx.cd_step_size)
            pln_ctx._held_obj_debug = obj_clone
            cache_entry = {
                'pln_ctx': pln_ctx,
                'robot_clone': robot_clone,
                'ee_clone': ee_clone,
                'obj_clone': obj_clone,
                'obj_mounting': obj_mounting,
            }
            self._hold_pln_ctx_cache[cache_key] = cache_entry
        ee_clone = cache_entry['ee_clone']
        obj_clone = cache_entry['obj_clone']
        robot_clone = cache_entry['robot_clone']
        ee_root_tf = pose_tf @ oum.tf_inverse(ee_clone.loc_tcp_tf)
        engage_tf = oum.tf_inverse(ee_root_tf) @ pick_tf

        obj_clone.set_rotmat_pos(rotmat=pick_tf[:3, :3], pos=pick_tf[:3, 3])
        ee_clone.set_jaw_width(jaw_width)
        obj_mounting = cache_entry['obj_mounting']
        obj_mounting.engage_tf[:] = engage_tf
        ee_clone._update_mounting(obj_mounting)
        robot_clone.fk(robot_clone.qs)
        cache_entry['pln_ctx'].clear_cache()
        return cache_entry['pln_ctx']

    def _pose_error(self, src_tf, dst_tf):
        pos_err = float(oum.np.linalg.norm(src_tf[:3, 3] - dst_tf[:3, 3]))
        rot_delta = src_tf[:3, :3].T @ dst_tf[:3, :3]
        cos_theta = oum.np.clip((oum.np.trace(rot_delta) - 1.0) * 0.5, -1.0, 1.0)
        rot_err = float(oum.np.arccos(cos_theta))
        return 10.0 * pos_err + rot_err

    def _sort_pose_candidates(self, pose_tf_list):
        current_tf = self.robot.gl_tcp_tf
        return sorted(
            pose_tf_list,
            key=lambda item: self._pose_error(current_tf, item[1]),
        )

    def _sorted_ik_solutions(self, tgt_pos, tgt_rotmat, ref_qs):
        ik_solutions = self.robot.ik_tcp(tgt_rotmat=tgt_rotmat, tgt_pos=tgt_pos)
        if not ik_solutions:
            return []
        ref_qs = oum.np.asarray(ref_qs, dtype=oum.np.float32)
        return sorted(
            (oum.np.asarray(qs, dtype=oum.np.float32) for qs in ik_solutions),
            key=lambda qs: oum.np.linalg.norm(qs - ref_qs),
        )

    def _contact_snapshot_for_state(self, pln_ctx, state):
        collider = getattr(pln_ctx, 'collider', None)
        if collider is None or not hasattr(collider, '_mjenv') or collider._mjenv is None:
            return [], []
        if not collider.is_collided(state):
            return [], []
        sync = collider._mjenv.sync
        if self.robot is not None:
            self._tag_mecba_debug_links(self.robot, 'active_arm')
        if self.ee_actor is not None:
            self._tag_mecba_debug_links(self.ee_actor, 'active_ee')
        body_name_to_label = {}
        for sobj, body in sync.sobj2bdy.items():
            body_name_to_label[body.name] = self._scene_entity_label(sobj, collider.scene)
        for rutl, body in sync.rutl2bdy.items():
            body_name_to_label.setdefault(body.name, self._scene_entity_label(rutl, collider.scene))

        pair_counts = {}
        contact_points = []
        model = collider._mjenv.model
        data = collider._mjenv.data
        contact_indices = getattr(collider, 'iter_collision_contact_indices', None)
        if callable(contact_indices):
            contact_indices = contact_indices()
        else:
            contact_indices = range(int(data.ncon))
        for cidx in contact_indices:
            contact = data.contact[cidx]
            body_id_a = int(model.geom_bodyid[contact.geom1])
            body_id_b = int(model.geom_bodyid[contact.geom2])
            label_a = body_name_to_label.get(model.body(body_id_a).name)
            label_b = body_name_to_label.get(model.body(body_id_b).name)
            contact_pos = oum.np.asarray(contact.pos, dtype=oum.np.float32).copy()
            contact_points.append(contact_pos)
            if label_a is None or label_b is None or label_a == label_b:
                continue
            if self._is_redundant_active_arm_ghost_pair(label_a, label_b):
                continue
            pair = tuple(sorted((label_a, label_b)))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        ranked_pairs = sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))
        return ranked_pairs, contact_points

    def _debug_contact_color(self, idx):
        palette = (
            ouc.BasicColor.RED,
            ouc.BasicColor.GREEN,
            ouc.BasicColor.BLUE,
            ouc.BasicColor.YELLOW,
            ouc.BasicColor.MAGENTA,
            ouc.BasicColor.CYAN,
            ouc.ExtendedColor.ORANGE_RED,
            ouc.ExtendedColor.DEEP_PINK,
        )
        return palette[idx % len(palette)]

    def _debug_attach_contact_points(self, contact_points, max_points=12, radius=0.008, rgb=None):
        if not contact_points:
            return
        base = getattr(builtins, 'base', None)
        scene = None if base is None else getattr(base, 'scene', None)
        if scene is None:
            return
        color = ouc.BasicColor.RED if rgb is None else rgb
        for pos in contact_points[:max_points]:
            marker = ossop.sphere(
                pos=pos,
                radius=radius,
                rgb=color,
                alpha=0.5,
                collision_type=None,
            )
            marker.attach_to(scene)

    def _scene_entity_label(self, entity, scene=None):
        debug_label = getattr(entity, '_debug_label', None)
        if isinstance(debug_label, str) and debug_label:
            return debug_label
        if scene is not None:
            for mecba in scene.mecbas:
                if entity in getattr(mecba, 'runtime_lnks', ()):
                    owner_label = self._scene_entity_owner_label(mecba)
                    file_path = getattr(entity, 'file_path', None)
                    if isinstance(file_path, str) and file_path:
                        return f'{owner_label}/{Path(file_path).stem}'
                    return f'{owner_label}/{type(entity).__name__}'
        name = getattr(entity, 'name', None)
        if isinstance(name, str) and name:
            return name
        file_path = getattr(entity, 'file_path', None)
        if isinstance(file_path, str) and file_path:
            return Path(file_path).stem
        return type(entity).__name__

    def _scene_entity_owner_label(self, mecba):
        debug_owner_label = getattr(mecba, '_debug_owner_label', None)
        if isinstance(debug_owner_label, str) and debug_owner_label:
            return debug_owner_label
        if mecba is self.robot:
            return 'active_arm'
        if self.ee_actor is not None and mecba is self.ee_actor:
            return 'active_ee'
        if type(mecba) is type(self.robot):
            return f'other_arm@{float(mecba.pos[1]):+.3f}'
        if self.ee_actor is not None and type(mecba) is type(self.ee_actor):
            return f'other_ee@{float(mecba.pos[1]):+.3f}'
        return type(mecba).__name__

    def _is_redundant_active_arm_ghost_pair(self, label_a, label_b):
        if not isinstance(label_a, str) or not isinstance(label_b, str):
            return False
        owner_a = label_a.split('/', 1)[0]
        owner_b = label_b.split('/', 1)[0]
        ghost_owners = {'active_arm_clone', 'active_arm_ghost'}
        if owner_a == owner_b:
            return False
        if owner_a == 'active_arm' and owner_b not in ghost_owners:
            return False
        if owner_b == 'active_arm' and owner_a not in ghost_owners:
            return False
        return owner_a == 'active_arm' or owner_b == 'active_arm'

    def _collision_pairs_for_state(self, pln_ctx, state, max_pairs=3):
        ranked_pairs, _contact_points = self._contact_snapshot_for_state(pln_ctx, state)
        return ranked_pairs[:max_pairs]

    def _classify_screen_stats(self, stats):
        if stats['rejected_no_ik']:
            return 'no_ik'
        if stats.get('rejected_coarse_collision', 0):
            return 'coarse_goal_in_collision'
        if stats['rejected_goal_collision']:
            return 'goal_in_collision'
        if stats['rejected_depart_seed_ik']:
            return 'depart_seed_ik_failed'
        if stats['rejected_depart_start_collision']:
            return 'depart_start_in_collision'
        if stats['rejected_depart_path_collision']:
            return 'depart_path_in_collision'
        if stats['rejected_depart_failure'] or stats['rejected_depart_other']:
            return 'depart_failed'
        if stats['survived']:
            return 'survived'
        return 'unknown'

    def _debug_visualize_failed_pose_entries(self, label, debug_entries, linear_granularity=0.03, max_entries=8):
        if not debug_entries:
            return
        base = getattr(builtins, 'base', None)
        scene = None if base is None else getattr(base, 'scene', None)
        if scene is None:
            print(f'[{label}] debug visualization skipped: builtins.base.scene is not available')
            return
        print(f'[{label}] survived=0, launching debug visualization for {min(len(debug_entries), max_entries)} poses')
        sorted_entries = self._sort_pose_candidates(
            [(entry['key'], entry['pose_tf'], entry['ee_qs']) for entry in debug_entries]
        )
        entry_map = {entry['key']: entry for entry in debug_entries}
        for idx, (key, pose_tf, ee_qs) in enumerate(sorted_entries[:max_entries]):
            entry = entry_map[key]
            color = self._debug_contact_color(idx)
            ik_solutions = self._sorted_ik_solutions(
                tgt_pos=pose_tf[:3, 3],
                tgt_rotmat=pose_tf[:3, :3],
                ref_qs=entry['ref_qs'],
            )
            if ik_solutions:
                ghost_robot = self.robot.clone()
                ghost_robot.fk(ik_solutions[0])
                ghost_robot.rgb = color
                ghost_robot.alpha = 0.18
                ghost_robot.attach_to(scene)
            if self.ee_actor is not None:
                ghost = self.ee_actor.clone()
                if ee_qs is not None:
                    ghost.fk(ee_qs)
                ghost_tf = pose_tf @ oum.np.linalg.inv(ghost.loc_tcp_tf)
                ghost.set_rotmat_pos(rotmat=ghost_tf[:3, :3], pos=ghost_tf[:3, 3])
                ghost.rgb = color
                ghost.alpha = 0.2
                ghost.attach_to(scene)
            _result, stats = self._screen_pose_with_stats(
                tcp_pos=pose_tf[:3, 3],
                tcp_rotmat=pose_tf[:3, :3],
                ee_qs=ee_qs,
                pln_ctx=entry['pln_ctx'],
                linear_granularity=linear_granularity,
                ref_qs=entry['ref_qs'],
                diagnose_collision_pairs=True,
                debug_visualize_contacts=True,
                debug_contact_rgb=color,
            )
            first_qs = None if not ik_solutions else oum.np.asarray(ik_solutions[0], dtype=oum.np.float32)
            print(
                f'[{label}] key={key}, reason={self._classify_screen_stats(stats)}, '
                f'ik_count={len(ik_solutions)}, '
                f'first_qs={None if first_qs is None else first_qs.tolist()}, '
                f'collision_pairs={stats.get("collision_pairs", [])}'
            )
        base.run()

    def _held_object_pose_for_state(self, pln_ctx, state):
        held_obj = getattr(pln_ctx, '_held_obj_debug', None)
        collider = getattr(pln_ctx, 'collider', None)
        if held_obj is None or collider is None:
            return None
        if not hasattr(collider, '_mjenv') or collider._mjenv is None:
            return None
        collider.is_collided(state)
        sync = collider._mjenv.sync
        body = sync.sobj2bdy.get(held_obj)
        if body is None:
            return None
        model = collider._mjenv.model
        data = collider._mjenv.data
        bid = model.body(body.name).id
        pos = oum.np.asarray(data.xpos[bid], dtype=oum.np.float32).copy()
        rotmat = oum.np.asarray(data.xmat[bid], dtype=oum.np.float32).reshape(3, 3).copy()
        return pos, rotmat

    def _pose_delta(self, src_pose, dst_pose):
        if src_pose is None or dst_pose is None:
            return None
        src_pos = oum.np.asarray(src_pose[0], dtype=oum.np.float32)
        src_rotmat = oum.np.asarray(src_pose[1], dtype=oum.np.float32)
        dst_pos = oum.np.asarray(dst_pose[0], dtype=oum.np.float32)
        dst_rotmat = oum.np.asarray(dst_pose[1], dtype=oum.np.float32)
        pos_err = float(oum.np.linalg.norm(src_pos - dst_pos))
        rot_delta = src_rotmat.T @ dst_rotmat
        cos_theta = oum.np.clip((oum.np.trace(rot_delta) - 1.0) * 0.5, -1.0, 1.0)
        rot_err = float(oum.np.arccos(cos_theta))
        return pos_err, rot_err

    def _screen_pose_with_stats(self,
                                tcp_pos,
                                tcp_rotmat,
                                ee_qs=None,
                                coarse_pln_ctx=None,
                                pln_ctx=None,
                                depart_pln_ctx=None,
                                exclude_entities=None,
                                depart_direction=None,
                                depart_distance=0.0,
                                linear_granularity=0.03,
                                ref_qs=None,
                                timing_prefix=None,
                                diagnose_collision_pairs=False,
                                debug_visualize_contacts=False,
                                debug_contact_rgb=None):
        plan_pln_ctx = self._filtered_pln_ctx(exclude_entities=exclude_entities) if pln_ctx is None else pln_ctx
        depart_pln_ctx = plan_pln_ctx if depart_pln_ctx is None else depart_pln_ctx
        ref_qs = self.robot.qs if ref_qs is None else ref_qs
        total_start_time = time.perf_counter()
        ik_start_time = time.perf_counter()
        ik_solutions = self._sorted_ik_solutions(tcp_pos, tcp_rotmat, ref_qs)
        ik_elapsed = time.perf_counter() - ik_start_time
        stats = {
            'rejected_no_ik': 0,
            'rejected_coarse_collision': 0,
            'rejected_goal_collision': 0,
            'rejected_depart_failure': 0,
            'rejected_depart_seed_ik': 0,
            'rejected_depart_start_collision': 0,
            'rejected_depart_path_collision': 0,
            'rejected_depart_other': 0,
            'survived': 0,
            'collision_pairs': [],
        }
        coarse_collision_elapsed = 0.0
        goal_collision_elapsed = 0.0
        depart_elapsed = 0.0
        if not ik_solutions:
            stats['rejected_no_ik'] = 1
            if timing_prefix is not None:
                self._record_timing(f'{timing_prefix}.ik_solve', ik_elapsed)
                self._record_timing(f'{timing_prefix}.total', time.perf_counter() - total_start_time)
            return None, stats
        saw_state_valid = False
        for goal_qs in ik_solutions:
            combined_qs = self._compose_state(goal_qs, ee_qs)
            if coarse_pln_ctx is not None:
                coarse_collision_start_time = time.perf_counter()
                is_coarse_valid = coarse_pln_ctx.is_state_valid(combined_qs)
                coarse_collision_elapsed += time.perf_counter() - coarse_collision_start_time
                if not is_coarse_valid:
                    stats['rejected_coarse_collision'] = 1
                    continue
            goal_collision_start_time = time.perf_counter()
            is_state_valid = plan_pln_ctx.is_state_valid(combined_qs)
            goal_collision_elapsed += time.perf_counter() - goal_collision_start_time
            if not is_state_valid:
                if diagnose_collision_pairs and not stats['collision_pairs']:
                    ranked_pairs, contact_points = self._contact_snapshot_for_state(
                        plan_pln_ctx,
                        combined_qs,
                    )
                    stats['collision_pairs'] = ranked_pairs
                    if debug_visualize_contacts:
                        self._debug_attach_contact_points(contact_points, rgb=debug_contact_rgb)
                continue
            saw_state_valid = True
            depart_plan = None
            if depart_distance > 0.0:
                depart_start_time = time.perf_counter()
                depart_plan = self.gen_depart(
                    goal_qs=combined_qs,
                    pln_ctx=depart_pln_ctx,
                    depart_direction=depart_direction,
                    depart_distance=depart_distance,
                    linear=True,
                    linear_granularity=linear_granularity,
                    use_rrt=False,
                )
                depart_elapsed += time.perf_counter() - depart_start_time
                if depart_plan is None:
                    failure = self._last_plan_failure
                    if failure is None:
                        stats['rejected_depart_other'] = 1
                    elif failure['reason'] == 'seed_ik_failed':
                        stats['rejected_depart_seed_ik'] = 1
                    elif failure['reason'] == 'start_state_in_collision':
                        stats['rejected_depart_start_collision'] = 1
                    elif failure['reason'] == 'path_in_collision':
                        stats['rejected_depart_path_collision'] = 1
                    else:
                        stats['rejected_depart_other'] = 1
                    continue
            stats['survived'] = 1
            if timing_prefix is not None:
                self._record_timing(f'{timing_prefix}.ik_solve', ik_elapsed)
                if coarse_pln_ctx is not None:
                    self._record_timing(f'{timing_prefix}.coarse_goal_check', coarse_collision_elapsed)
                self._record_timing(f'{timing_prefix}.goal_state_check', goal_collision_elapsed)
                if depart_distance > 0.0:
                    self._record_timing(f'{timing_prefix}.depart_screen', depart_elapsed)
                self._record_timing(f'{timing_prefix}.total', time.perf_counter() - total_start_time)
            return ScreenResult(goal_qs=combined_qs, ee_qs=ee_qs, depart_plan=depart_plan), stats
        if saw_state_valid:
            stats['rejected_depart_failure'] = 1
        else:
            stats['rejected_goal_collision'] = 1
        if timing_prefix is not None:
            self._record_timing(f'{timing_prefix}.ik_solve', ik_elapsed)
            if coarse_pln_ctx is not None:
                self._record_timing(f'{timing_prefix}.coarse_goal_check', coarse_collision_elapsed)
            self._record_timing(f'{timing_prefix}.goal_state_check', goal_collision_elapsed)
            if depart_distance > 0.0:
                self._record_timing(f'{timing_prefix}.depart_screen', depart_elapsed)
            self._record_timing(f'{timing_prefix}.total', time.perf_counter() - total_start_time)
        return None, stats

    def _screen_pose(self,
                     tcp_pos,
                     tcp_rotmat,
                     ee_qs=None,
                     coarse_pln_ctx=None,
                     pln_ctx=None,
                     depart_pln_ctx=None,
                     exclude_entities=None,
                     depart_direction=None,
                     depart_distance=0.0,
                     linear_granularity=0.03,
                     ref_qs=None,
                     timing_prefix=None):
        result, _stats = self._screen_pose_with_stats(
            tcp_pos=tcp_pos,
            tcp_rotmat=tcp_rotmat,
            ee_qs=ee_qs,
            coarse_pln_ctx=coarse_pln_ctx,
            pln_ctx=pln_ctx,
            depart_pln_ctx=depart_pln_ctx,
            exclude_entities=exclude_entities,
            depart_direction=depart_direction,
            depart_distance=depart_distance,
            linear_granularity=linear_granularity,
            ref_qs=ref_qs,
            timing_prefix=timing_prefix,
        )
        return result

    def _screen_pose_list(self,
                          keyed_pose_list,
                          coarse_pln_ctx=None,
                          pln_ctx=None,
                          depart_pln_ctx=None,
                          exclude_entities=None,
                          depart_direction=None,
                          depart_distance=0.0,
                          linear_granularity=0.03,
                          ref_qs=None,
                          timing_prefix=None,
                          toggle_dbg=False,
                          debug_label='reason'):
        plan_pln_ctx = self._filtered_pln_ctx(exclude_entities=exclude_entities) if pln_ctx is None else pln_ctx
        records = []
        cur_ref_qs = self.robot.qs if ref_qs is None else oum.np.asarray(ref_qs, dtype=oum.np.float32)
        total_poses = len(keyed_pose_list)
        rejected_no_ik = 0
        rejected_coarse_collision = 0
        rejected_goal_collision = 0
        rejected_depart_failure = 0
        rejected_depart_seed_ik = 0
        rejected_depart_start_collision = 0
        rejected_depart_path_collision = 0
        rejected_depart_other = 0
        for key, pose_tf, ee_qs in self._sort_pose_candidates(keyed_pose_list):
            result, stats = self._screen_pose_with_stats(
                tcp_pos=pose_tf[:3, 3],
                tcp_rotmat=pose_tf[:3, :3],
                ee_qs=ee_qs,
                coarse_pln_ctx=coarse_pln_ctx,
                pln_ctx=plan_pln_ctx,
                depart_pln_ctx=depart_pln_ctx,
                exclude_entities=exclude_entities,
                depart_direction=depart_direction,
                depart_distance=depart_distance,
                linear_granularity=linear_granularity,
                ref_qs=cur_ref_qs,
                timing_prefix=timing_prefix,
            )
            rejected_no_ik += stats['rejected_no_ik']
            rejected_coarse_collision += stats['rejected_coarse_collision']
            rejected_goal_collision += stats['rejected_goal_collision']
            rejected_depart_failure += stats['rejected_depart_failure']
            rejected_depart_seed_ik += stats['rejected_depart_seed_ik']
            rejected_depart_start_collision += stats['rejected_depart_start_collision']
            rejected_depart_path_collision += stats['rejected_depart_path_collision']
            rejected_depart_other += stats['rejected_depart_other']
            if result is None:
                continue
            records.append(CandidateRecord(key=key, pose_tf=pose_tf, screen_result=result))
            cur_ref_qs = result.goal_qs[:self.robot.ndof]
        if toggle_dbg:
            print(
                f'[{debug_label}] '
                f'candidates_tested={total_poses}, '
                f'removed_no_ik={rejected_no_ik}, '
                f'removed_coarse_collision={rejected_coarse_collision}, '
                f'removed_goal_state_in_collision={rejected_goal_collision}, '
                f'removed_depart_motion_failed={rejected_depart_failure}, '
                f'removed_depart_seed_ik={rejected_depart_seed_ik}, '
                f'removed_depart_start_state_in_collision={rejected_depart_start_collision}, '
                f'removed_depart_path_in_collision={rejected_depart_path_collision}, '
                f'removed_depart_other={rejected_depart_other}, '
                f'survived={len(records)}'
            )
        return records
