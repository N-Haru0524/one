import one.utils.math as oum

from .hierarchical import HierarchicalPlannerBase
from . import utils


class PickPlacePlanner(HierarchicalPlannerBase):
    def reason_common_grasp_ids(self,
                                obj_model,
                                grasp_collection,
                                goal_pose_list,
                                pick_pose=None,
                                linear_granularity=0.03,
                                exclude_entities=None,
                                toggle_dbg=False):
        available_ids = list(range(len(grasp_collection)))
        pick_exclude_entities = [] if exclude_entities is None else list(exclude_entities)
        pick_exclude_entities.append(obj_model)
        pick_pln_ctx = self._filtered_pln_ctx(exclude_entities=pick_exclude_entities)
        hold_ctx_cache = {}
        self._last_reason_common_grasp_report = {
            'survived_gids': [],
            'failures': {},
        }
        collision_free_ids = []
        for gid in available_ids:
            if self._grasp_has_collision(grasp_collection[gid]):
                self._last_reason_common_grasp_report['failures'][gid] = {
                    'label': 'grasp',
                    'reason': 'grasp_collision',
                }
                continue
            collision_free_ids.append(gid)
        available_ids = collision_free_ids

        def classify_reason(stats):
            if stats['rejected_no_ik']:
                return 'no_ik'
            if stats['rejected_goal_collision']:
                return 'goal_in_collision'
            return 'unknown'

        def hold_pln_ctx_for_gid(gid):
            return hold_ctx_cache.setdefault(
                gid,
                self._hold_pln_ctx(
                    obj_model=obj_model,
                    grasp=grasp_collection[gid],
                    pick_pose=pick_pose,
                    exclude_entities=exclude_entities,
                ),
            )

        def record_failure(gid, label, reason, tcp_tf):
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
                pos = oum.np.array2string(tcp_tf[:3, 3], precision=4)
                print(f'[pickplace_reason gid={gid}] label={label}, reason={reason}, tcp_pos={pos}')

        ref_qs_map = {gid: self.robot.qs.copy() for gid in available_ids}
        open_ee_qs = self._max_open_ee_qs()
        if pick_pose is not None:
            approach_entries = []
            for gid in available_ids:
                tcp_tf, _ee_qs = self._grasp_pre_pose(grasp_collection[gid], obj_pose=pick_pose)
                approach_entries.append((gid, tcp_tf, open_ee_qs))
            approach_records = self._screen_pose_list(
                approach_entries,
                pln_ctx=pick_pln_ctx,
                linear_granularity=linear_granularity,
                ref_qs=self.robot.qs.copy(),
                toggle_dbg=toggle_dbg,
                debug_label='pickplace_reason pick_approach_start',
            )
            available_ids = [record.key for record in approach_records]
            ref_qs_map = {record.key: record.screen_result.goal_qs[:self.robot.ndof].copy() for record in approach_records}
            survived_set = set(available_ids)
            for gid, tcp_tf, _ee_qs in approach_entries:
                if gid not in survived_set:
                    record_failure(gid, 'pick_approach_start', 'unreachable_approach', tcp_tf)
            if not available_ids:
                self._last_reason_common_grasp_report['survived_gids'] = []
                return []

            next_available_ids = []
            next_ref_qs_map = {}
            for gid in available_ids:
                tcp_tf, ee_qs = self._grasp_pose(grasp_collection[gid], obj_pose=pick_pose)
                result, stats = self._screen_pose_with_stats(
                    tcp_pos=tcp_tf[:3, 3],
                    tcp_rotmat=tcp_tf[:3, :3],
                    ee_qs=ee_qs,
                    pln_ctx=pick_pln_ctx,
                    linear_granularity=linear_granularity,
                    ref_qs=ref_qs_map[gid],
                )
                if result is None:
                    record_failure(gid, 'pick_goal', classify_reason(stats), tcp_tf)
                    continue
                next_available_ids.append(gid)
                next_ref_qs_map[gid] = result.goal_qs[:self.robot.ndof].copy()
            available_ids = next_available_ids
            ref_qs_map = next_ref_qs_map
            if not available_ids:
                self._last_reason_common_grasp_report['survived_gids'] = []
                return []

        for goal_idx, goal_pose in enumerate(goal_pose_list):
            if pick_pose is None:
                raise ValueError('pick_pose is required for place grasp reasoning')
            next_available_ids = []
            next_ref_qs_map = {}
            approach_label = f'place_{goal_idx}_approach_start'
            goal_label = f'place_{goal_idx}_goal'
            for gid in available_ids:
                pre_tf, pre_ee_qs = self._grasp_pre_pose(grasp_collection[gid], obj_pose=goal_pose)
                pre_result, pre_stats = self._screen_pose_with_stats(
                    tcp_pos=pre_tf[:3, 3],
                    tcp_rotmat=pre_tf[:3, :3],
                    ee_qs=pre_ee_qs,
                    pln_ctx=hold_pln_ctx_for_gid(gid),
                    linear_granularity=linear_granularity,
                    ref_qs=ref_qs_map[gid],
                )
                if pre_result is None:
                    record_failure(gid, approach_label, classify_reason(pre_stats), pre_tf)
                    continue
                goal_tf, ee_qs = self._grasp_pose(grasp_collection[gid], obj_pose=goal_pose)
                goal_result, goal_stats = self._screen_pose_with_stats(
                    tcp_pos=goal_tf[:3, 3],
                    tcp_rotmat=goal_tf[:3, :3],
                    ee_qs=ee_qs,
                    pln_ctx=hold_pln_ctx_for_gid(gid),
                    linear_granularity=linear_granularity,
                    ref_qs=pre_result.goal_qs[:self.robot.ndof],
                )
                if goal_result is None:
                    record_failure(gid, goal_label, classify_reason(goal_stats), goal_tf)
                    continue
                next_available_ids.append(gid)
                next_ref_qs_map[gid] = goal_result.goal_qs[:self.robot.ndof].copy()
            available_ids = next_available_ids
            ref_qs_map = next_ref_qs_map
            if not available_ids:
                break
        self._last_reason_common_grasp_report['survived_gids'] = available_ids.copy()
        return available_ids

    def gen_pick_and_moveto(self,
                            obj_model,
                            grasp,
                            goal_pose_list,
                            gid=None,
                            start_qs=None,
                            linear_granularity=0.03,
                            use_rrt=True,
                            toggle_dbg=False):
        if self._grasp_has_collision(grasp):
            if toggle_dbg:
                print(f'[pickmove_pick gid={gid}] skipped grasp_collision=1')
            return None
        if start_qs is None:
            start_qs = self.robot.qs.copy()
        open_ee_qs = self._max_open_ee_qs()
        pick_pln_ctx = self._filtered_pln_ctx(exclude_entities=[obj_model])
        hold_pln_ctx = self._hold_pln_ctx(
            obj_model=obj_model,
            grasp=grasp,
            pick_pose=oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos),
        )

        obj_pose_tf = oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos)
        pick_tf, pick_pre_tf, ee_qs, _jaw_width, _score = self._grasp_world_data(grasp, obj_pose=obj_pose_tf)
        pick_goal, pick_stats = self._screen_pose_with_stats(
            tcp_pos=pick_tf[:3, 3],
            tcp_rotmat=pick_tf[:3, :3],
            ee_qs=ee_qs,
            pln_ctx=pick_pln_ctx,
            linear_granularity=linear_granularity,
            ref_qs=start_qs,
        )
        if toggle_dbg:
            if pick_stats['survived']:
                print(f'[pickmove_pick gid={gid}] survived=1')
            elif pick_stats['rejected_goal_collision']:
                print(f'[pickmove_pick gid={gid}] removed_goal_state_in_collision=1')
            elif pick_stats['rejected_no_ik']:
                print(f'[pickmove_pick gid={gid}] removed_no_ik=1')
        if pick_goal is None:
            return None
        current_state = self._compose_state(*self._split_state(start_qs, ee_values=ee_qs))
        pick_plan = self.gen_approach_via_pose(
            goal_tf=pick_tf,
            via_tf=pick_pre_tf,
            goal_qs=pick_goal.goal_qs,
            via_ee_qs=open_ee_qs,
            final_ee_qs=ee_qs,
            start_qs=current_state,
            linear_granularity=linear_granularity,
            pln_ctx=pick_pln_ctx,
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
                pln_ctx=hold_pln_ctx,
                ref_qs=current_state[:self.robot.ndof],
            )
            if goal_result is None:
                if toggle_dbg:
                    print(f'[pickmove_hold gid={gid} goal={goal_idx}] goal screening failed')
                return None
            goal_plan = self.gen_approach_via_pose(
                goal_tf=goal_tf,
                via_tf=goal_pre_tf,
                goal_qs=goal_result.goal_qs,
                via_ee_qs=ee_qs,
                final_ee_qs=ee_qs,
                start_qs=current_state,
                linear_granularity=linear_granularity,
                pln_ctx=hold_pln_ctx,
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
        if not oum.np.allclose(full_plan.qs_list[-1], release_state):
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
                           linear_granularity=0.03,
                           reason_grasps=True,
                           use_rrt=True,
                           toggle_dbg=False):
        if start_qs is None:
            start_qs = self.robot.qs.copy()
        obj_pose_tf = oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos)
        if reason_grasps:
            common_gids = self.reason_common_grasp_ids(
                obj_model=obj_model,
                grasp_collection=grasp_collection,
                goal_pose_list=goal_pose_list,
                pick_pose=obj_pose_tf,
                linear_granularity=linear_granularity,
                toggle_dbg=toggle_dbg,
            )
        else:
            common_gids = [
                gid for gid, grasp in enumerate(grasp_collection)
                if not self._grasp_has_collision(grasp)
            ]
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
