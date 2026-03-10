import numpy as np

from one import oum

from .hierarchical import HierarchicalPlannerBase
from . import utils


class PickPlacePlanner(HierarchicalPlannerBase):
    def _grasp_fields(self, grasp):
        if not isinstance(grasp, (tuple, list)) or len(grasp) < 4:
            raise ValueError('grasp must be a tuple like (pose_tf, pre_pose_tf, jaw_width, score)')
        pose_tf = np.asarray(grasp[0], dtype=np.float32)
        pre_pose_tf = np.asarray(grasp[1], dtype=np.float32)
        jaw_width = float(np.asarray(grasp[2], dtype=np.float32).reshape(-1)[0])
        score = float(grasp[3])
        if pose_tf.shape != (4, 4):
            raise ValueError('grasp pose_tf must be a (4, 4) transform')
        if pre_pose_tf.shape != (4, 4):
            raise ValueError('grasp pre_pose_tf must be a (4, 4) transform')
        return pose_tf, pre_pose_tf, jaw_width, score

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
            ee_actor.set_jaw_width(float(np.asarray(ee_actor.jaw_range, dtype=np.float32)[1]))
            return np.asarray(ee_actor.qs[:ee_actor.ndof], dtype=np.float32)
        return self._default_ee_qs()

    def _compose_with_ee_qs(self, qs, ee_qs):
        robot_qs, _ = self._split_state(qs, ee_values=ee_qs)
        return self._compose_state(robot_qs, ee_qs)

    def _offset_tcp_pose(self, tcp_tf, direction, distance, motion_type):
        tcp_tf = np.asarray(tcp_tf, dtype=np.float32)
        offset_direction = self._normalize_direction(tcp_tf[:3, :3], direction)
        offset = offset_direction * float(distance)
        endpoint_tf = tcp_tf.copy()
        if motion_type == 'approach':
            endpoint_tf[:3, 3] = tcp_tf[:3, 3] - offset
        elif motion_type == 'depart':
            endpoint_tf[:3, 3] = tcp_tf[:3, 3] + offset
        else:
            raise ValueError(f'Unsupported motion_type: {motion_type}')
        return endpoint_tf

    def _approach_from_pre_pose(self, pose_tf, pre_pose_tf, fallback_direction, fallback_distance):
        pose_tf = np.asarray(pose_tf, dtype=np.float32)
        pre_pose_tf = np.asarray(pre_pose_tf, dtype=np.float32)
        delta = pose_tf[:3, 3] - pre_pose_tf[:3, 3]
        distance = float(np.linalg.norm(delta))
        if distance > 1e-6:
            return (delta / distance).astype(np.float32), distance
        return fallback_direction, fallback_distance

    def _gen_approach_via_pre_pose(self,
                                   goal_tf,
                                   pre_pose_tf,
                                   goal_qs,
                                   approach_ee_qs,
                                   final_ee_qs,
                                   start_qs,
                                   fallback_direction,
                                   fallback_distance,
                                   linear_granularity,
                                   pln_ctx,
                                   use_rrt):
        goal_tf = np.asarray(goal_tf, dtype=np.float32)
        pre_pose_tf = np.asarray(pre_pose_tf, dtype=np.float32)
        goal_robot_qs, _ = self._split_state(goal_qs, ee_values=final_ee_qs)
        approach_goal_qs = self._compose_state(goal_robot_qs, approach_ee_qs)
        pre_pos_err = float(np.linalg.norm(goal_tf[:3, 3] - pre_pose_tf[:3, 3]))
        pre_rot_err = float(np.linalg.norm(goal_tf[:3, :3] - pre_pose_tf[:3, :3]))
        if pre_pos_err <= 1e-6 and pre_rot_err <= 1e-6:
            approach_direction, approach_distance = self._approach_from_pre_pose(
                goal_tf,
                pre_pose_tf,
                fallback_direction,
                fallback_distance,
            )
            return self.gen_approach(
                goal_tcp_pos=goal_tf[:3, 3],
                goal_tcp_rotmat=goal_tf[:3, :3],
                goal_qs=approach_goal_qs,
                start_qs=start_qs,
                approach_direction=approach_direction,
                approach_distance=approach_distance,
                linear=approach_distance > 0.0,
                linear_granularity=linear_granularity,
                pln_ctx=pln_ctx,
                use_rrt=use_rrt,
            )

        linear_app = self._linear_motion_between_poses(
            start_pos=pre_pose_tf[:3, 3],
            start_rotmat=pre_pose_tf[:3, :3],
            goal_pos=goal_tf[:3, 3],
            goal_rotmat=goal_tf[:3, :3],
            seed_qs=approach_goal_qs,
            ee_values=approach_ee_qs,
            pln_ctx=pln_ctx,
            pos_step=linear_granularity,
        )
        if linear_app is None:
            return None

        full_plan = linear_app
        if start_qs is not None:
            start2pre = self._connect_motion(
                start_qs=start_qs,
                goal_qs=linear_app.qs_list[0],
                ee_values=approach_ee_qs,
                pln_ctx=pln_ctx,
                use_rrt=use_rrt,
            )
            if start2pre is None:
                return None
            full_plan = self._merge_plans(start2pre, linear_app)

        goal_state = self._compose_state(goal_robot_qs, final_ee_qs)
        if not np.allclose(full_plan.qs_list[-1], goal_state):
            end_connect = self._connect_motion(
                start_qs=full_plan.qs_list[-1],
                goal_qs=goal_state,
                ee_values=final_ee_qs,
                pln_ctx=pln_ctx,
                use_rrt=use_rrt,
            )
            if end_connect is None:
                return None
            full_plan = self._merge_plans(full_plan, end_connect)
        return full_plan

    def reason_common_grasp_ids(self,
                                grasp_collection,
                                goal_pose_list,
                                pick_pose=None,
                                pick_approach_direction=None,
                                pick_approach_distance=0.0,
                                pick_depart_direction=None,
                                pick_depart_distance=0.0,
                                place_approach_direction=None,
                                place_approach_distance=0.0,
                                place_depart_distance=0.1,
                                place_depart_direction=None,
                                linear_granularity=0.03,
                                exclude_entities=None,
                                toggle_dbg=False):
        available_ids = list(range(len(grasp_collection)))
        plan_pln_ctx = self._filtered_pln_ctx(exclude_entities=exclude_entities)
        self._last_reason_common_grasp_report = {
            'survived_gids': [],
            'failures': {},
        }

        def classify_reason(stats):
            if stats['rejected_no_ik']:
                return 'no_ik'
            if stats['rejected_goal_collision']:
                return 'goal_in_collision'
            if stats['rejected_depart_failure']:
                return 'depart_failed'
            if stats['rejected_depart_seed_ik']:
                return 'depart_seed_ik_failed'
            if stats['rejected_depart_start_collision']:
                return 'depart_start_in_collision'
            if stats['rejected_depart_path_collision']:
                return 'depart_path_in_collision'
            if stats['rejected_depart_other']:
                return 'depart_other'
            return 'unknown'

        def build_pose_entries(label, obj_pose, direction=None, distance=0.0, motion_type=None, use_pick_pose=False):
            pose_entries = []
            open_ee_qs = self._max_open_ee_qs()
            for gid in available_ids:
                if label.endswith('approach_start'):
                    tcp_tf, _ee_qs = self._grasp_pre_pose(
                        grasp_collection[gid],
                        obj_pose=pick_pose if use_pick_pose else obj_pose,
                    )
                    ee_qs = open_ee_qs
                elif use_pick_pose:
                    tcp_tf, ee_qs = self._grasp_pose(grasp_collection[gid], obj_pose=pick_pose)
                else:
                    if pick_pose is None:
                        raise ValueError('pick_pose is required for place grasp reasoning')
                    tcp_tf, ee_qs = self._grasp_pose(grasp_collection[gid], obj_pose=obj_pose)
                if motion_type == 'depart' and distance > 0.0:
                    tcp_tf = self._offset_tcp_pose(
                        tcp_tf,
                        direction=direction,
                        distance=distance,
                        motion_type=motion_type,
                    )
                pose_entries.append((gid, tcp_tf, ee_qs))
            return label, pose_entries

        endpoint_checks = []
        if pick_pose is not None:
            if pick_approach_distance > 0.0:
                endpoint_checks.append(build_pose_entries(
                    'pick_approach_start',
                    pick_pose,
                    direction=pick_approach_direction,
                    distance=pick_approach_distance,
                    motion_type='approach',
                    use_pick_pose=True,
                ))
            endpoint_checks.append(build_pose_entries('pick_goal', pick_pose, use_pick_pose=True))
            if pick_depart_distance > 0.0:
                endpoint_checks.append(build_pose_entries(
                    'pick_depart_end',
                    pick_pose,
                    direction=pick_depart_direction,
                    distance=pick_depart_distance,
                    motion_type='depart',
                    use_pick_pose=True,
                ))
        for goal_idx, goal_pose in enumerate(goal_pose_list):
            if place_approach_distance > 0.0:
                endpoint_checks.append(build_pose_entries(
                    f'place_{goal_idx}_approach_start',
                    goal_pose,
                    direction=place_approach_direction,
                    distance=place_approach_distance,
                    motion_type='approach',
                ))
            endpoint_checks.append(build_pose_entries(f'place_{goal_idx}_goal', goal_pose))
            if place_depart_distance > 0.0:
                endpoint_checks.append(build_pose_entries(
                    f'place_{goal_idx}_depart_end',
                    goal_pose,
                    direction=place_depart_direction,
                    distance=place_depart_distance,
                    motion_type='depart',
                ))

        for label, pose_entries in endpoint_checks:
            next_available_ids = []
            for gid, tcp_tf, ee_qs in pose_entries:
                result, stats = self._screen_pose_with_stats(
                    tcp_pos=tcp_tf[:3, 3],
                    tcp_rotmat=tcp_tf[:3, :3],
                    ee_qs=ee_qs,
                    pln_ctx=plan_pln_ctx,
                    linear_granularity=linear_granularity,
                    ref_qs=self.robot.qs.copy(),
                )
                if result is None:
                    reason = classify_reason(stats)
                    self._last_reason_common_grasp_report['failures'].setdefault(
                        gid,
                        {
                            'label': label,
                            'reason': reason,
                            'tcp_pos': tcp_tf[:3, 3].copy(),
                            'tcp_rotmat': tcp_tf[:3, :3].copy(),
                        },
                    )
                    if toggle_dbg:
                        pos = np.array2string(tcp_tf[:3, 3], precision=4)
                        print(f'[pickplace_reason gid={gid}] label={label}, reason={reason}, tcp_pos={pos}')
                    continue
                next_available_ids.append(gid)
            available_ids = next_available_ids
            if not available_ids:
                break
        self._last_reason_common_grasp_report['survived_gids'] = available_ids.copy()
        return available_ids

    def reason_common_gids(self, *args, **kwargs):
        return self.reason_common_grasp_ids(*args, **kwargs)

    def _repeat_param(self, values, count, default_value):
        if values is None:
            return [default_value] * count
        if isinstance(values, (list, tuple)):
            if len(values) != count:
                raise ValueError(f'Expected {count} values, got {len(values)}')
            return list(values)
        return [values] * count

    def gen_pick_and_moveto(self,
                            obj_model,
                            grasp,
                            goal_pose_list,
                            gid=None,
                            start_qs=None,
                            pick_approach_direction=None,
                            pick_approach_distance=0.07,
                            pick_depart_direction=None,
                            pick_depart_distance=0.07,
                            approach_direction_list=None,
                            approach_distance_list=None,
                            linear_granularity=0.03,
                            use_rrt=True,
                            toggle_dbg=False):
        if start_qs is None:
            start_qs = self.robot.qs.copy()
        if pick_approach_direction is None:
            pick_approach_direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        if pick_depart_direction is None:
            pick_depart_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        default_moveto_direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        open_ee_qs = self._max_open_ee_qs()
        approach_direction_list = self._repeat_param(
            approach_direction_list, len(goal_pose_list), default_moveto_direction)
        approach_distance_list = self._repeat_param(
            approach_distance_list, len(goal_pose_list), 0.07)

        obj_pose_tf = oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos)
        pick_tf, pick_pre_tf, ee_qs, _jaw_width, _score = self._grasp_world_data(grasp, obj_pose=obj_pose_tf)
        pick_goal, pick_stats = self._screen_pose_with_stats(
            tcp_pos=pick_tf[:3, 3],
            tcp_rotmat=pick_tf[:3, :3],
            ee_qs=ee_qs,
            pln_ctx=self.pln_ctx,
            depart_direction=pick_depart_direction,
            depart_distance=pick_depart_distance,
            linear_granularity=linear_granularity,
            ref_qs=start_qs,
        )
        if toggle_dbg:
            if pick_stats['survived']:
                print(f'[pickmove_pick gid={gid}] survived=1')
            elif pick_stats['rejected_goal_collision']:
                print(f'[pickmove_pick gid={gid}] removed_goal_state_in_collision=1')
            elif pick_stats['rejected_depart_failure']:
                print(f'[pickmove_pick gid={gid}] removed_pick_depart_motion_failed=1')
            elif pick_stats['rejected_no_ik']:
                print(f'[pickmove_pick gid={gid}] removed_no_ik=1')
        if pick_goal is None:
            return None
        current_state = self._compose_state(*self._split_state(start_qs, ee_values=ee_qs))
        pick_plan = self._gen_approach_via_pre_pose(
            goal_tf=pick_tf,
            pre_pose_tf=pick_pre_tf,
            goal_qs=pick_goal.goal_qs,
            approach_ee_qs=open_ee_qs,
            final_ee_qs=ee_qs,
            start_qs=current_state,
            fallback_direction=pick_approach_direction,
            fallback_distance=pick_approach_distance,
            linear_granularity=linear_granularity,
            pln_ctx=self.pln_ctx,
            use_rrt=use_rrt,
        )
        if pick_plan is None:
            if toggle_dbg:
                failure = self._last_plan_failure
                if failure is None:
                    print(f'[pickmove_pick gid={gid}] motion planning failed')
                else:
                    print(
                        f'[pickmove_pick gid={gid}] '
                        f'{failure["stage"]} failed: {failure["reason"]}'
                    )
            return None
        moveto_plan = utils.MotionData([pick_plan.qs_list[-1]])
        current_state = pick_plan.qs_list[-1]
        for goal_idx, goal_pose in enumerate(goal_pose_list):
            goal_tf, goal_pre_tf, _goal_ee_qs, _goal_jaw_width, _goal_score = self._grasp_world_data(
                grasp,
                obj_pose=goal_pose,
            )
            goal_result = self._screen_pose(
                tcp_pos=goal_tf[:3, 3],
                tcp_rotmat=goal_tf[:3, :3],
                ee_qs=ee_qs,
                pln_ctx=self.pln_ctx,
                ref_qs=current_state[:self.robot.ndof],
            )
            if goal_result is None:
                if toggle_dbg:
                    print(f'[pickmove_hold gid={gid} goal={goal_idx}] goal screening failed')
                return None
            goal_plan = self._gen_approach_via_pre_pose(
                goal_tf=goal_tf,
                pre_pose_tf=goal_pre_tf,
                goal_qs=goal_result.goal_qs,
                approach_ee_qs=ee_qs,
                final_ee_qs=ee_qs,
                start_qs=current_state,
                fallback_direction=approach_direction_list[goal_idx],
                fallback_distance=approach_distance_list[goal_idx],
                linear_granularity=linear_granularity,
                pln_ctx=self.pln_ctx,
                use_rrt=use_rrt,
            )
            if goal_plan is None:
                if toggle_dbg:
                    failure = self._last_plan_failure
                    if failure is None:
                        print(f'[pickmove_hold gid={gid} goal={goal_idx}] motion planning failed')
                    else:
                        print(
                            f'[pickmove_hold gid={gid} goal={goal_idx}] '
                            f'{failure["stage"]} failed: {failure["reason"]}'
                        )
                return None
            moveto_plan = self._merge_plans(moveto_plan, goal_plan)
            current_state = moveto_plan.qs_list[-1]
        full_plan = pick_plan.copy()
        full_plan.extend(moveto_plan.qs_list[1:])
        release_state = self._compose_with_ee_qs(full_plan.qs_list[-1], open_ee_qs)
        if not np.allclose(full_plan.qs_list[-1], release_state):
            full_plan.extend([release_state])
        full_plan.events['attach'] = len(pick_plan.qs_list) - 1
        full_plan.events['release'] = len(full_plan.qs_list) - 1
        full_plan.events['gid'] = gid
        return full_plan

    def gen_pick_and_place(self,
                           obj_model,
                           grasp_collection,
                           goal_pose_list,
                           start_qs=None,
                           approach_direction_list=None,
                           approach_distance_list=None,
                           pick_approach_direction=None,
                           pick_approach_distance=0.07,
                           pick_depart_direction=None,
                           pick_depart_distance=0.07,
                           place_approach_direction=None,
                           place_approach_distance=0.07,
                           place_depart_direction=None,
                           place_depart_distance=0.07,
                           linear_granularity=0.03,
                           reason_grasps=True,
                           use_rrt=True,
                           toggle_dbg=False):
        if start_qs is None:
            start_qs = self.robot.qs.copy()
        if pick_approach_direction is None:
            pick_approach_direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        if pick_depart_direction is None:
            pick_depart_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if approach_direction_list is None:
            if place_approach_direction is None:
                place_approach_direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            approach_direction_list = [place_approach_direction] * len(goal_pose_list)
        if approach_distance_list is None:
            if place_approach_distance is None:
                place_approach_distance = 0.07
            approach_distance_list = [place_approach_distance] * len(goal_pose_list)
        obj_pose_tf = oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos)
        if reason_grasps:
            common_gids = self.reason_common_grasp_ids(
                grasp_collection=grasp_collection,
                goal_pose_list=goal_pose_list,
                pick_pose=obj_pose_tf,
                pick_approach_direction=pick_approach_direction,
                pick_approach_distance=pick_approach_distance,
                pick_depart_direction=pick_depart_direction,
                pick_depart_distance=pick_depart_distance,
                place_approach_direction=place_approach_direction,
                place_approach_distance=place_approach_distance,
                place_depart_distance=place_depart_distance,
                place_depart_direction=place_depart_direction,
                linear_granularity=linear_granularity,
                toggle_dbg=toggle_dbg,
            )
        else:
            common_gids = list(range(len(grasp_collection)))
        if not common_gids:
            if toggle_dbg:
                print('[pickplace] no common grasp ids after reasoning')
            return None
        pick_pose_tf_list = []
        for gid in common_gids:
            tcp_tf, _ = self._grasp_pose(grasp_collection[gid], obj_pose=obj_pose_tf)
            pick_pose_tf_list.append((gid, tcp_tf))
        for gid, _ in self._sort_pose_candidates(pick_pose_tf_list):
            grasp = grasp_collection[gid]
            full_plan = self.gen_pick_and_moveto(
                obj_model=obj_model,
                grasp=grasp,
                goal_pose_list=goal_pose_list,
                gid=gid,
                start_qs=start_qs,
                pick_approach_direction=pick_approach_direction,
                pick_approach_distance=pick_approach_distance,
                pick_depart_direction=pick_depart_direction,
                pick_depart_distance=pick_depart_distance,
                approach_direction_list=approach_direction_list,
                approach_distance_list=approach_distance_list,
                linear_granularity=linear_granularity,
                use_rrt=use_rrt,
                toggle_dbg=toggle_dbg,
            )
            if full_plan is None:
                continue
            if toggle_dbg:
                print(f'[pickplace] selected gid={gid}, waypoints={len(full_plan.qs_list)}')
            return full_plan
        if toggle_dbg:
            print('[pickplace] no feasible pick-and-place plan found')
        return None
