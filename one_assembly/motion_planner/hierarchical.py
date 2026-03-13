from dataclasses import dataclass

import one.utils.math as oum

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
    def __init__(self, robot, pln_ctx, ee_actor=None):
        super().__init__(robot=robot, pln_ctx=pln_ctx, ee_actor=ee_actor)
        self._filtered_pln_ctx_cache = {}

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

    def _ctx_cache_key(self, exclude_entities=None, actors=None):
        exclude_entities = () if exclude_entities is None else tuple(sorted(id(entity) for entity in exclude_entities))
        actors = self._desired_actors() if actors is None else actors
        actor_ids = tuple(id(actor) for actor in actors)
        return actor_ids, exclude_entities

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

    def _hold_pln_ctx(self, obj_model, grasp, pick_pose, exclude_entities=None):
        if self.ee_actor is None:
            return self._filtered_pln_ctx(exclude_entities=exclude_entities)
        if obj_model is None or not hasattr(obj_model, 'clone'):
            return self._filtered_pln_ctx(exclude_entities=exclude_entities)
        robot_clone = self.robot.clone()
        ee_clone = self._mounted_ee_actor(robot_clone)
        if ee_clone is None:
            raise RuntimeError('Failed to locate mounted end effector clone for hold planning context')
        pick_tf = self._pose_to_tf(pick_pose)
        obj_clone = obj_model.clone()
        obj_clone.set_rotmat_pos(rotmat=pick_tf[:3, :3], pos=pick_tf[:3, 3])
        pose_tf, _pre_pose_tf, _ee_qs, jaw_width, _score = self._grasp_world_data(grasp, obj_pose=pick_pose)
        ee_clone.grip_at(pose_tf[:3, 3], pose_tf[:3, :3], jaw_width)
        ee_clone.grasp(obj_clone, jaw_width=jaw_width)
        desired_actors = [robot_clone, ee_clone]
        excluded = [] if exclude_entities is None else list(exclude_entities)
        excluded.extend(self._desired_actors())
        excluded.append(obj_model)
        collider = utils.build_collider(desired_actors, obstacles=self._scene_entities(excluded))
        return utils.build_planning_context(collider, max_edge_step=self.pln_ctx.cd_step_size)

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

    def _screen_pose_with_stats(self,
                                tcp_pos,
                                tcp_rotmat,
                                ee_qs=None,
                                pln_ctx=None,
                                exclude_entities=None,
                                depart_direction=None,
                                depart_distance=0.0,
                                linear_granularity=0.03,
                                ref_qs=None):
        plan_pln_ctx = self._filtered_pln_ctx(exclude_entities=exclude_entities) if pln_ctx is None else pln_ctx
        ref_qs = self.robot.qs if ref_qs is None else ref_qs
        ik_solutions = self._sorted_ik_solutions(tcp_pos, tcp_rotmat, ref_qs)
        stats = {
            'rejected_no_ik': 0,
            'rejected_goal_collision': 0,
            'rejected_depart_failure': 0,
            'rejected_depart_seed_ik': 0,
            'rejected_depart_start_collision': 0,
            'rejected_depart_path_collision': 0,
            'rejected_depart_other': 0,
            'survived': 0,
        }
        if not ik_solutions:
            stats['rejected_no_ik'] = 1
            return None, stats
        saw_state_valid = False
        for goal_qs in ik_solutions:
            combined_qs = self._compose_state(goal_qs, ee_qs)
            if not plan_pln_ctx.is_state_valid(combined_qs):
                continue
            saw_state_valid = True
            depart_plan = None
            if depart_distance > 0.0:
                depart_plan = self.gen_depart(
                    goal_qs=combined_qs,
                    pln_ctx=plan_pln_ctx,
                    depart_direction=depart_direction,
                    depart_distance=depart_distance,
                    linear=True,
                    linear_granularity=linear_granularity,
                    use_rrt=False,
                )
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
            return ScreenResult(goal_qs=combined_qs, ee_qs=ee_qs, depart_plan=depart_plan), stats
        if saw_state_valid:
            stats['rejected_depart_failure'] = 1
        else:
            stats['rejected_goal_collision'] = 1
        return None, stats

    def _screen_pose(self,
                     tcp_pos,
                     tcp_rotmat,
                     ee_qs=None,
                     pln_ctx=None,
                     exclude_entities=None,
                     depart_direction=None,
                     depart_distance=0.0,
                     linear_granularity=0.03,
                     ref_qs=None):
        result, _stats = self._screen_pose_with_stats(
            tcp_pos=tcp_pos,
            tcp_rotmat=tcp_rotmat,
            ee_qs=ee_qs,
            pln_ctx=pln_ctx,
            exclude_entities=exclude_entities,
            depart_direction=depart_direction,
            depart_distance=depart_distance,
            linear_granularity=linear_granularity,
            ref_qs=ref_qs,
        )
        return result

    def _screen_pose_list(self,
                          keyed_pose_list,
                          pln_ctx=None,
                          exclude_entities=None,
                          depart_direction=None,
                          depart_distance=0.0,
                          linear_granularity=0.03,
                          ref_qs=None,
                          toggle_dbg=False,
                          debug_label='reason'):
        plan_pln_ctx = self._filtered_pln_ctx(exclude_entities=exclude_entities) if pln_ctx is None else pln_ctx
        records = []
        cur_ref_qs = self.robot.qs if ref_qs is None else oum.np.asarray(ref_qs, dtype=oum.np.float32)
        total_poses = len(keyed_pose_list)
        rejected_no_ik = 0
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
                pln_ctx=plan_pln_ctx,
                exclude_entities=exclude_entities,
                depart_direction=depart_direction,
                depart_distance=depart_distance,
                linear_granularity=linear_granularity,
                ref_qs=cur_ref_qs,
            )
            rejected_no_ik += stats['rejected_no_ik']
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
                f'removed_goal_state_in_collision={rejected_goal_collision}, '
                f'removed_depart_motion_failed={rejected_depart_failure}, '
                f'removed_depart_seed_ik={rejected_depart_seed_ik}, '
                f'removed_depart_start_state_in_collision={rejected_depart_start_collision}, '
                f'removed_depart_path_in_collision={rejected_depart_path_collision}, '
                f'removed_depart_other={rejected_depart_other}, '
                f'survived={len(records)}'
            )
        return records
