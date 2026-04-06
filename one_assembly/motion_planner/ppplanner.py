from collections import Counter
import time

from one_assembly.assembly_data import EEEvent, HeldGrasp, PlannerActionDraft, PlannerSegmentDraft
import one.motion.trajectory.cartesian as omtc
import one.utils.math as oum

from . import utils
from .hierarchical import HierarchicalPlannerBase


class PickPlacePlanner(HierarchicalPlannerBase):
    def _grasp_event_payload(self, obj_pose, grasp):
        pose_tf = oum.ensure_tf(obj_pose) @ oum.np.asarray(grasp[0], dtype=oum.np.float32)
        pick_tf = oum.ensure_tf(obj_pose)
        jaw_width = float(oum.np.asarray(grasp[2], dtype=oum.np.float32).reshape(-1)[0])
        ee_base_tf = pose_tf @ oum.np.linalg.inv(self.ee_actor.loc_tcp_tf)
        engage_tf = oum.np.linalg.inv(ee_base_tf) @ pick_tf
        return jaw_width, engage_tf.astype(oum.np.float32)

    def _grasp_motion_params(self, grasp, obj_pose, direction=None, distance=0.0):
        if direction is not None:
            return direction, float(distance)
        pose_tf, pre_pose_tf, _ee_qs, _jaw_width, _score = self._grasp_world_data(grasp, obj_pose=obj_pose)
        delta = pose_tf[:3, 3] - pre_pose_tf[:3, 3]
        delta_norm = float(oum.np.linalg.norm(delta))
        if delta_norm <= 1e-8:
            return None, 0.0
        return delta / delta_norm, delta_norm

    def _grasp_pose_motion(self, grasp, obj_pose, direction=None, distance=0.0, motion_type='sink'):
        pose_tf, _pre_pose_tf, ee_qs, _jaw_width, _score = self._grasp_world_data(grasp, obj_pose=obj_pose)
        resolved_direction, resolved_distance = self._grasp_motion_params(
            grasp,
            obj_pose=obj_pose,
            direction=direction,
            distance=distance,
        )
        motion_tf = self._offset_pose_tf(
            goal_tcp_pos=pose_tf[:3, 3],
            goal_tcp_rotmat=pose_tf[:3, :3],
            direction=resolved_direction,
            distance=resolved_distance,
            motion_type=motion_type,
        )
        return pose_tf, motion_tf, ee_qs

    def _needs_prefix_start_depart(self, start_qs):
        start_robot_qs, _start_ee_qs = self._split_state(start_qs)
        home_qs = getattr(self.robot, 'home_qs', None)
        if home_qs is None:
            return False
        home_qs = oum.np.asarray(home_qs, dtype=oum.np.float32)
        return not oum.np.allclose(start_robot_qs, home_qs)

    def _prefix_start_depart(self, start_qs, ee_qs, linear_granularity, toggle_dbg=False):
        start_robot_qs, start_ee_qs = self._split_state(start_qs, ee_values=ee_qs)
        home_qs = getattr(self.robot, 'home_qs', None)
        if home_qs is None:
            if toggle_dbg:
                print('[pickmove_prefix] skipped: robot has no home_qs')
            return None
        home_qs = oum.np.asarray(home_qs, dtype=oum.np.float32)
        if oum.np.allclose(start_robot_qs, home_qs):
            if toggle_dbg:
                print('[pickmove_prefix] skipped: start_qs already at home_qs')
            return None
        start_state = self._compose_state(start_robot_qs, start_ee_qs)
        start_tcp_pos, start_tcp_rotmat = self._tcp_pose_from_qs(start_state)
        depart_tf = self._offset_pose_tf(
            goal_tcp_pos=start_tcp_pos,
            goal_tcp_rotmat=start_tcp_rotmat,
            distance=0.07,
            motion_type='sink',
        )
        q_seq, _ = omtc.cartesian_to_jtraj(
            robot=self.robot,
            start_rotmat=start_tcp_rotmat,
            start_pos=start_tcp_pos,
            goal_rotmat=depart_tf[:3, :3],
            goal_pos=depart_tf[:3, 3],
            pos_step=linear_granularity,
            rot_step=oum.np.deg2rad(2.0),
            ref_qs=start_robot_qs,
        )
        if q_seq is None:
            self._set_last_plan_failure('prefix_depart_ik', 'cartesian_ik_failed')
            depart_plan = None
        else:
            depart_plan = self._motion_plan([
                self._compose_state(qs, start_ee_qs)
                for qs in q_seq
            ])
        if toggle_dbg:
            if depart_plan is None:
                failure = self._last_plan_failure
                if failure is None:
                    print('[pickmove_prefix] failed: linear depart unavailable')
                else:
                    print(
                        '[pickmove_prefix] failed: '
                        f'{failure["stage"]} {failure["reason"]}'
                    )
            else:
                print(f'[pickmove_prefix] planned: waypoints={len(depart_plan.qs_list)}')
        return depart_plan

    def reason_common_grasp_ids(self,
                                obj_model,
                                grasp_collection,
                                goal_pose_list,
                                pick_pose=None,
                                pick_approach_direction=None,
                                pick_approach_distance=0.1,
                                pick_depart_direction=None,
                                pick_depart_distance=0.1,
                                approach_direction=None,
                                approach_distance=0.07,
                                depart_direction=None,
                                depart_distance=0.07,
                                linear_granularity=0.03,
                                exclude_entities=None,
                                toggle_dbg=False):
        total_start_time = time.perf_counter()
        try:
            available_ids = list(range(len(grasp_collection)))
            pick_exclude_entities = [] if exclude_entities is None else list(exclude_entities)
            pick_exclude_entities.append(obj_model)

            ctx_start_time = time.perf_counter()
            pick_pln_ctx = self._precise_pln_ctx(
                exclude_entities=pick_exclude_entities,
                backend=self.reasoning_backend,
            )
            pick_goal_arm_pln_ctx = self._arm_only_goal_pln_ctx(
                exclude_entities=pick_exclude_entities,
            )
            pick_goal_ee_pln_ctx = self._ee_local_goal_pln_ctx(
                exclude_entities=pick_exclude_entities,
                backend=self.reasoning_backend,
            )
            pick_transit_pln_ctx = self._precise_pln_ctx(
                exclude_entities=pick_exclude_entities,
                backend=self.transit_backend,
            )
            self._record_timing('pickplace.reason.context_setup', time.perf_counter() - ctx_start_time)

            self._last_reason_common_grasp_report = {
                'survived_gids': [],
                'failures': {},
            }

            def print_stage_stats(label, total, survived, reason_counts, pair_counts=None):
                if not toggle_dbg:
                    return
                parts = [f'[{label}]', f'tested={int(total)}']
                removed = int(total) - int(survived)
                parts.append(f'removed={removed}')
                for reason, count in sorted(reason_counts.items()):
                    if count <= 0:
                        continue
                    parts.append(f'{reason}={int(count)}')
                parts.append(f'survived={int(survived)}')
                if pair_counts:
                    top_pairs = sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
                    parts.append(
                        'contacts=' + '; '.join(
                            f'{pair[0]}<->{pair[1]}:{count}' for pair, count in top_pairs
                        )
                    )
                print(', '.join(parts))

            def classify_reason(stats):
                return self._classify_screen_stats(stats)

            def hold_pln_ctx_for_gid(gid):
                start_time = time.perf_counter()
                pln_ctx = self._hold_pln_ctx(
                    obj_model=obj_model,
                    grasp=grasp_collection[gid],
                    pick_pose=pick_pose,
                    exclude_entities=exclude_entities,
                    backend=self.reasoning_backend,
                )
                self._record_timing('pickplace.reason.hold_ctx_setup', time.perf_counter() - start_time)
                return pln_ctx

            def hold_goal_arm_pln_ctx_for_gid(gid):
                start_time = time.perf_counter()
                pln_ctx = self._hold_pln_ctx(
                    obj_model=obj_model,
                    grasp=grasp_collection[gid],
                    pick_pose=pick_pose,
                    exclude_entities=exclude_entities,
                    backend='mujoco',
                    collision_mode='arm_only',
                )
                self._record_timing('pickplace.reason.hold_arm_ctx_setup', time.perf_counter() - start_time)
                return pln_ctx

            def hold_goal_ee_pln_ctx_for_gid(gid):
                start_time = time.perf_counter()
                pln_ctx = self._hold_pln_ctx(
                    obj_model=obj_model,
                    grasp=grasp_collection[gid],
                    pick_pose=pick_pose,
                    exclude_entities=exclude_entities,
                    backend=self.reasoning_backend,
                    collision_mode='ee_only',
                )
                self._record_timing('pickplace.reason.hold_ee_ctx_setup', time.perf_counter() - start_time)
                return pln_ctx

            def transit_hold_pln_ctx_for_gid(gid):
                start_time = time.perf_counter()
                pln_ctx = self._hold_pln_ctx(
                    obj_model=obj_model,
                    grasp=grasp_collection[gid],
                    pick_pose=pick_pose,
                    exclude_entities=exclude_entities,
                    backend=self.transit_backend,
                )
                self._record_timing('pickplace.reason.transit_hold_ctx_setup', time.perf_counter() - start_time)
                return pln_ctx

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

            collision_free_ids = []
            grasp_reason_counts = {
                'grasp_collision': 0,
            }
            grasp_stage_start_time = time.perf_counter()
            for gid in available_ids:
                if self._grasp_has_collision(grasp_collection[gid]):
                    self._last_reason_common_grasp_report['failures'][gid] = {
                        'label': 'grasp',
                        'reason': 'grasp_collision',
                    }
                    grasp_reason_counts['grasp_collision'] += 1
                    continue
                collision_free_ids.append(gid)
            self._record_timing('pickplace.reason.grasp_collision_screen', time.perf_counter() - grasp_stage_start_time)
            available_ids = collision_free_ids
            print_stage_stats('pickplace_reason grasp', len(grasp_collection), len(available_ids), grasp_reason_counts)

            ref_qs_map = {gid: self.robot.qs.copy() for gid in available_ids}
            open_ee_qs = self._max_open_ee_qs()
            if pick_pose is not None:
                tested_count = len(available_ids)
                approach_entries = []
                for gid in available_ids:
                    _pick_tf, tcp_tf, _ee_qs = self._grasp_pose_motion(
                        grasp_collection[gid],
                        obj_pose=pick_pose,
                        direction=pick_approach_direction,
                        distance=pick_approach_distance,
                        motion_type='sink',
                    )
                    approach_entries.append((gid, tcp_tf, open_ee_qs))

                pick_approach_start_time = time.perf_counter()
                approach_records = self._screen_pose_list(
                    approach_entries,
                    pln_ctx=pick_transit_pln_ctx,
                    depart_pln_ctx=pick_transit_pln_ctx,
                    depart_direction=pick_depart_direction,
                    depart_distance=pick_depart_distance,
                    linear_granularity=linear_granularity,
                    ref_qs=self.robot.qs.copy(),
                    timing_prefix='pickplace.reason.pick_approach_start.detail',
                    toggle_dbg=False,
                    debug_label='pickplace_reason pick_approach_start',
                )
                self._record_timing('pickplace.reason.pick_approach_start', time.perf_counter() - pick_approach_start_time)

                available_ids = [record.key for record in approach_records]
                ref_qs_map = {
                    record.key: record.screen_result.goal_qs[:self.robot.ndof].copy()
                    for record in approach_records
                }
                survived_set = set(available_ids)
                stage_reason_counts = {
                    'unreachable_approach': 0,
                }
                for gid, tcp_tf, _ee_qs in approach_entries:
                    if gid not in survived_set:
                        record_failure(gid, 'pick_approach_start', 'unreachable_approach', tcp_tf)
                        stage_reason_counts['unreachable_approach'] += 1
                print_stage_stats(
                    'pickplace_reason pick_approach_start',
                    tested_count,
                    len(available_ids),
                    stage_reason_counts,
                )
                if not available_ids:
                    if toggle_dbg and approach_entries:
                        self._debug_visualize_failed_pose_entries(
                            label='pickplace_reason pick_approach_start',
                            debug_entries=[
                                {
                                    'key': gid,
                                    'pose_tf': tcp_tf,
                                    'ee_qs': ee_qs,
                                    'pln_ctx': pick_transit_pln_ctx,
                                    'ref_qs': self.robot.qs.copy(),
                                }
                                for gid, tcp_tf, ee_qs in approach_entries
                            ],
                            linear_granularity=linear_granularity,
                        )
                    self._last_reason_common_grasp_report['survived_gids'] = []
                    return []

                next_available_ids = []
                next_ref_qs_map = {}
                stage_reason_counts = {}
                stage_pair_counts = Counter()
                pick_goal_start_time = time.perf_counter()
                for gid in available_ids:
                    tcp_tf, ee_qs = self._grasp_pose(grasp_collection[gid], obj_pose=pick_pose)
                    pick_goal_depart_direction, pick_goal_depart_distance = self._grasp_motion_params(
                        grasp_collection[gid],
                        obj_pose=pick_pose,
                        direction=pick_depart_direction,
                        distance=pick_depart_distance,
                    )
                    result, stats = self._screen_pose_with_stats(
                        tcp_pos=tcp_tf[:3, 3],
                        tcp_rotmat=tcp_tf[:3, :3],
                        ee_qs=ee_qs,
                        coarse_pln_ctx=pick_goal_arm_pln_ctx,
                        pln_ctx=pick_goal_ee_pln_ctx,
                        depart_pln_ctx=pick_transit_pln_ctx,
                        depart_direction=pick_goal_depart_direction,
                        depart_distance=pick_goal_depart_distance,
                        linear_granularity=linear_granularity,
                        ref_qs=ref_qs_map[gid],
                        timing_prefix='pickplace.reason.pick_goal.detail',
                        diagnose_collision_pairs=toggle_dbg,
                        debug_visualize_contacts=toggle_dbg,
                    )
                    if result is None:
                        reason = classify_reason(stats)
                        record_failure(gid, 'pick_goal', reason, tcp_tf)
                        stage_reason_counts[reason] = stage_reason_counts.get(reason, 0) + 1
                        for pair, count in stats.get('collision_pairs', []):
                            stage_pair_counts[pair] += count
                        continue
                    next_available_ids.append(gid)
                    next_ref_qs_map[gid] = result.goal_qs[:self.robot.ndof].copy()
                self._record_timing('pickplace.reason.pick_goal', time.perf_counter() - pick_goal_start_time)
                print_stage_stats(
                    'pickplace_reason pick_goal',
                    len(available_ids),
                    len(next_available_ids),
                    stage_reason_counts,
                    pair_counts=stage_pair_counts,
                )
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
                approach_reason_counts = {}
                goal_reason_counts = {}
                approach_pair_counts = Counter()
                goal_pair_counts = Counter()
                tested_count = len(available_ids)
                debug_entries = []
                place_approach_start_time = time.perf_counter()
                place_goal_elapsed = 0.0
                for gid in available_ids:
                    _goal_tf, pre_tf, pre_ee_qs = self._grasp_pose_motion(
                        grasp_collection[gid],
                        obj_pose=goal_pose,
                        direction=approach_direction,
                        distance=approach_distance,
                        motion_type='sink',
                    )
                    hold_transit_pln_ctx = transit_hold_pln_ctx_for_gid(gid)
                    debug_entries.append(
                        {
                            'key': gid,
                            'pose_tf': pre_tf,
                            'ee_qs': pre_ee_qs,
                            'pln_ctx': hold_transit_pln_ctx,
                            'ref_qs': ref_qs_map[gid],
                        }
                    )
                    pre_result, pre_stats = self._screen_pose_with_stats(
                        tcp_pos=pre_tf[:3, 3],
                        tcp_rotmat=pre_tf[:3, :3],
                        ee_qs=pre_ee_qs,
                        pln_ctx=hold_transit_pln_ctx,
                        linear_granularity=linear_granularity,
                        ref_qs=ref_qs_map[gid],
                        timing_prefix=f'pickplace.reason.{approach_label}.detail',
                        diagnose_collision_pairs=toggle_dbg,
                        debug_visualize_contacts=toggle_dbg,
                    )
                    if pre_result is None:
                        reason = classify_reason(pre_stats)
                        record_failure(gid, approach_label, reason, pre_tf)
                        approach_reason_counts[reason] = approach_reason_counts.get(reason, 0) + 1
                        for pair, count in pre_stats.get('collision_pairs', []):
                            approach_pair_counts[pair] += count
                        continue
                    goal_tf, ee_qs = self._grasp_pose(grasp_collection[gid], obj_pose=goal_pose)
                    goal_depart_direction, goal_depart_distance = self._grasp_motion_params(
                        grasp_collection[gid],
                        obj_pose=goal_pose,
                        direction=depart_direction,
                        distance=depart_distance,
                    )
                    goal_stage_start_time = time.perf_counter()
                    goal_result, goal_stats = self._screen_pose_with_stats(
                        tcp_pos=goal_tf[:3, 3],
                        tcp_rotmat=goal_tf[:3, :3],
                        ee_qs=ee_qs,
                        coarse_pln_ctx=hold_goal_arm_pln_ctx_for_gid(gid),
                        pln_ctx=hold_goal_ee_pln_ctx_for_gid(gid),
                        depart_pln_ctx=transit_hold_pln_ctx_for_gid(gid),
                        depart_direction=goal_depart_direction,
                        depart_distance=goal_depart_distance,
                        linear_granularity=linear_granularity,
                        ref_qs=pre_result.goal_qs[:self.robot.ndof],
                        timing_prefix=f'pickplace.reason.{goal_label}.detail',
                        diagnose_collision_pairs=toggle_dbg,
                        debug_visualize_contacts=toggle_dbg,
                    )
                    place_goal_elapsed += time.perf_counter() - goal_stage_start_time
                    if goal_result is None:
                        reason = classify_reason(goal_stats)
                        record_failure(gid, goal_label, reason, goal_tf)
                        goal_reason_counts[reason] = goal_reason_counts.get(reason, 0) + 1
                        for pair, count in goal_stats.get('collision_pairs', []):
                            goal_pair_counts[pair] += count
                        if toggle_dbg and reason == 'goal_in_collision':
                            goal_state = self._compose_state(
                                self._sorted_ik_solutions(
                                    goal_tf[:3, 3],
                                    goal_tf[:3, :3],
                                    pre_result.goal_qs[:self.robot.ndof],
                                )[0],
                                ee_qs,
                            )
                            held_pose = self._held_object_pose_for_state(hold_pln_ctx_for_gid(gid), goal_state)
                            pose_delta = self._pose_delta(goal_pose, held_pose)
                            if held_pose is None or pose_delta is None:
                                print(f'[pickplace_debug {goal_label} gid={gid}] held_pose_unavailable=1')
                            else:
                                pos_err, rot_err = pose_delta
                                print(
                                    f'[pickplace_debug {goal_label} gid={gid}] '
                                    f'goal_pos={goal_pose[0].tolist()}, held_pos={held_pose[0].tolist()}, '
                                    f'pos_err={pos_err:.6f}, rot_err={rot_err:.6f}'
                                )
                        continue
                    next_available_ids.append(gid)
                    next_ref_qs_map[gid] = goal_result.goal_qs[:self.robot.ndof].copy()

                self._record_timing(
                    f'pickplace.reason.{approach_label}',
                    time.perf_counter() - place_approach_start_time - place_goal_elapsed,
                )
                self._record_timing(f'pickplace.reason.{goal_label}', place_goal_elapsed)

                after_approach_count = tested_count - sum(approach_reason_counts.values())
                print_stage_stats(
                    f'pickplace_reason {approach_label}',
                    tested_count,
                    after_approach_count,
                    approach_reason_counts,
                    pair_counts=approach_pair_counts,
                )
                if toggle_dbg and tested_count > 0 and after_approach_count == 0:
                    self._debug_visualize_failed_pose_entries(
                        label=f'pickplace_reason {approach_label}',
                        debug_entries=debug_entries,
                        linear_granularity=linear_granularity,
                    )
                print_stage_stats(
                    f'pickplace_reason {goal_label}',
                    after_approach_count,
                    len(next_available_ids),
                    goal_reason_counts,
                    pair_counts=goal_pair_counts,
                )
                available_ids = next_available_ids
                ref_qs_map = next_ref_qs_map
                if not available_ids:
                    break

            self._last_reason_common_grasp_report['survived_gids'] = available_ids.copy()
            if toggle_dbg:
                print(f'[pickplace_reason final], survived={len(available_ids)}')
            return available_ids
        finally:
            self._record_timing('pickplace.reason.total', time.perf_counter() - total_start_time)

    def gen_pick_and_moveto(self,
                            obj_model,
                            grasp,
                            goal_pose_list,
                            gid=None,
                            start_qs=None,
                            pick_approach_direction=None,
                            pick_approach_distance=0.1,
                            pick_depart_direction=None,
                            pick_depart_distance=0.1,
                            approach_direction=None,
                            approach_distance=0.07,
                            depart_direction=None,
                            depart_distance=0.07,
                            linear_granularity=0.03,
                            use_rrt=True,
                            pln_jnt=False,
                            toggle_dbg=False):
        total_start_time = time.perf_counter()
        try:
            if start_qs is None:
                start_qs = self.robot.qs.copy()
            self._release_mounted_objects()
            open_ee_qs = self._max_open_ee_qs()
            prepend_release = False
            prefix_depart_plan = None
            if open_ee_qs is not None:
                _start_robot_qs, start_ee_qs = self._split_state(start_qs, ee_values=open_ee_qs)
                if start_ee_qs is not None and not oum.np.allclose(start_ee_qs, open_ee_qs):
                    prepend_release = True
                start_qs = self._compose_with_ee_qs(start_qs, open_ee_qs)
            require_prefix_depart = self._needs_prefix_start_depart(start_qs)
            prefix_depart_plan = self._prefix_start_depart(
                start_qs=start_qs,
                ee_qs=open_ee_qs,
                linear_granularity=linear_granularity,
                toggle_dbg=toggle_dbg,
            )
            if require_prefix_depart and prefix_depart_plan is None:
                self._set_last_plan_failure('prefix_depart', 'required_depart_failed')
                if toggle_dbg:
                    print('[pickmove_prefix] required depart missing, aborting pick plan')
                return None
            if prefix_depart_plan is None:
                current_start_qs = start_qs
            else:
                current_start_qs = prefix_depart_plan.qs_list[-1]
            if self._grasp_has_collision(grasp):
                if toggle_dbg:
                    print(f'[pickmove_pick gid={gid}] skipped grasp_collision=1')
                return None
            start_robot_qs, _start_ee_qs = self._split_state(current_start_qs)
            pick_pln_ctx = self._filtered_pln_ctx(exclude_entities=[obj_model])
            hold_pln_ctx = self._hold_pln_ctx(
                obj_model=obj_model,
                grasp=grasp,
                pick_pose=oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos),
                backend=self.reasoning_backend,
            )

            obj_pose_tf = oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos)
            pick_tf, pick_pre_tf, ee_qs = self._grasp_pose_motion(
                grasp,
                obj_pose=obj_pose_tf,
                direction=pick_approach_direction,
                distance=pick_approach_distance,
                motion_type='sink',
            )
            pick_depart_direction_resolved, pick_depart_distance_resolved = self._grasp_motion_params(
                grasp,
                obj_pose=obj_pose_tf,
                direction=pick_depart_direction,
                distance=pick_depart_distance,
            )
            pick_goal_start_time = time.perf_counter()
            pick_goal, pick_stats = self._screen_pose_with_stats(
                tcp_pos=pick_tf[:3, 3],
                tcp_rotmat=pick_tf[:3, :3],
                ee_qs=ee_qs,
                pln_ctx=pick_pln_ctx,
                depart_pln_ctx=hold_pln_ctx,
                depart_direction=pick_depart_direction_resolved,
                depart_distance=pick_depart_distance_resolved,
                linear_granularity=linear_granularity,
                ref_qs=start_robot_qs,
            )
            self._record_timing('pickplace.motion.pick_goal_screen', time.perf_counter() - pick_goal_start_time)
            if toggle_dbg:
                if pick_stats['survived']:
                    print(f'[pickmove_pick gid={gid}] survived=1')
                elif pick_stats['rejected_goal_collision']:
                    print(f'[pickmove_pick gid={gid}] removed_goal_state_in_collision=1')
                elif pick_stats['rejected_no_ik']:
                    print(f'[pickmove_pick gid={gid}] removed_no_ik=1')
            if pick_goal is None:
                return None
            current_state = self._compose_state(*self._split_state(current_start_qs, ee_values=ee_qs))
            pick_plan_start_time = time.perf_counter()
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
                pln_jnt=pln_jnt,
                connect_timing_prefix='pickmove_pick',
            )
            self._record_timing('pickplace.motion.pick_approach_plan', time.perf_counter() - pick_plan_start_time)
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

            pick_depart_start_time = time.perf_counter()
            pick_depart_plan = self.gen_depart(
                goal_qs=pick_plan.qs_list[-1],
                depart_direction=pick_depart_direction_resolved,
                depart_distance=pick_depart_distance_resolved,
                linear=pick_depart_distance_resolved > 0.0,
                linear_granularity=linear_granularity,
                pln_ctx=hold_pln_ctx,
                use_rrt=False,
            )
            self._record_timing('pickplace.motion.pick_depart_plan', time.perf_counter() - pick_depart_start_time)
            if pick_depart_plan is None:
                if toggle_dbg:
                    failure = self._last_plan_failure
                    if failure is None:
                        print(f'[pickmove_pick gid={gid}] depart planning failed')
                    else:
                        print(
                            f'[pickmove_pick gid={gid}] '
                            f'{failure["stage"]} failed: {failure["reason"]}'
                        )
                return None

            full_pick_plan = self._merge_plans(pick_plan, pick_depart_plan)
            if prefix_depart_plan is not None:
                full_pick_plan = self._merge_plans(prefix_depart_plan, full_pick_plan)
            moveto_plan = utils.MotionData([full_pick_plan.qs_list[-1]])
            current_state = full_pick_plan.qs_list[-1]
            goal_plan_events = []
            for goal_idx, goal_pose in enumerate(goal_pose_list):
                goal_tf, goal_pre_tf, _goal_ee_qs = self._grasp_pose_motion(
                    grasp,
                    obj_pose=goal_pose,
                    direction=approach_direction,
                    distance=approach_distance,
                    motion_type='sink',
                )
                goal_depart_direction_resolved, goal_depart_distance_resolved = self._grasp_motion_params(
                    grasp,
                    obj_pose=goal_pose,
                    direction=depart_direction,
                    distance=depart_distance,
                )
                goal_screen_start_time = time.perf_counter()
                goal_result = self._screen_pose(
                    tcp_pos=goal_tf[:3, 3],
                    tcp_rotmat=goal_tf[:3, :3],
                    ee_qs=ee_qs,
                    pln_ctx=hold_pln_ctx,
                    depart_direction=goal_depart_direction_resolved,
                    depart_distance=goal_depart_distance_resolved,
                    ref_qs=current_state[:self.robot.ndof],
                    timing_prefix=f'pickplace.motion.place_{goal_idx}_goal_screen.detail',
                )
                self._record_timing(
                    f'pickplace.motion.place_{goal_idx}_goal_screen',
                    time.perf_counter() - goal_screen_start_time,
                )
                if goal_result is None:
                    if toggle_dbg:
                        print(f'[pickmove_hold gid={gid} goal={goal_idx}] goal screening failed')
                    return None
                goal_plan_start_time = time.perf_counter()
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
                    pln_jnt=pln_jnt,
                    connect_timing_prefix=f'pickmove_hold.goal_{goal_idx}',
                )
                self._record_timing(
                    f'pickplace.motion.place_{goal_idx}_approach_plan',
                    time.perf_counter() - goal_plan_start_time,
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
                goal_plan_events.append(dict(goal_plan.events))
                moveto_plan = self._merge_plans(moveto_plan, goal_plan)
                current_state = moveto_plan.qs_list[-1]

            full_plan = full_pick_plan.copy()
            full_plan.extend(moveto_plan.qs_list[1:])
            attach_idx = len(pick_plan.qs_list) - 1
            if prefix_depart_plan is not None:
                attach_idx += len(prefix_depart_plan.qs_list) - 1
                full_plan.events['prefix_end'] = len(prefix_depart_plan.qs_list) - 1
            full_plan.events['pick_pre'] = int(pick_plan.events.get('via', 0)) + (
                0 if prefix_depart_plan is None else len(prefix_depart_plan.qs_list) - 1
            )
            if prepend_release and open_ee_qs is not None:
                full_plan.events['pre_release'] = 0
            full_plan.events['attach'] = attach_idx
            full_plan.events['post_attach'] = len(full_pick_plan.qs_list) - 1
            if goal_plan_events:
                goal0_via = int(goal_plan_events[0].get('via', 0))
                full_plan.events['place_pre'] = len(full_pick_plan.qs_list) - 1 + goal0_via
            full_plan.events['gid'] = gid
            return full_plan
        finally:
            self._record_timing('pickplace.motion.total', time.perf_counter() - total_start_time)

    def gen_pick_and_place(self,
                           obj_model,
                           grasp_collection,
                           goal_pose_list,
                           start_qs=None,
                           pick_approach_direction=None,
                           pick_approach_distance=0.1,
                           pick_depart_direction=None,
                           pick_depart_distance=0.1,
                           approach_direction=None,
                           approach_distance=0.07,
                           depart_direction=None,
                           depart_distance=0.07,
                           linear_granularity=0.03,
                           reason_grasps=True,
                           use_rrt=True,
                           pln_jnt=False,
                           preferred_gid=None,
                           release_at_end=True,
                           toggle_dbg=False):
        total_start_time = time.perf_counter()
        self._reset_timing_report()
        try:
            if start_qs is None:
                start_qs = self.robot.qs.copy()
            self._release_mounted_objects()
            obj_pose_tf = oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos)
            if reason_grasps:
                common_gids = self.reason_common_grasp_ids(
                    obj_model=obj_model,
                    grasp_collection=grasp_collection,
                    goal_pose_list=goal_pose_list,
                    pick_pose=obj_pose_tf,
                    pick_approach_direction=pick_approach_direction,
                    pick_approach_distance=pick_approach_distance,
                    pick_depart_direction=pick_depart_direction,
                    pick_depart_distance=pick_depart_distance,
                    approach_direction=approach_direction,
                    approach_distance=approach_distance,
                    depart_direction=depart_direction,
                    depart_distance=depart_distance,
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
            if preferred_gid is not None:
                if preferred_gid not in common_gids:
                    if toggle_dbg:
                        print(f'[pickplace] preferred gid={preferred_gid} rejected by reasoning')
                    return None
                common_gids = [preferred_gid]

            pick_pose_tf_list = []
            sort_start_time = time.perf_counter()
            for gid in common_gids:
                tcp_tf, _ = self._grasp_pose(grasp_collection[gid], obj_pose=obj_pose_tf)
                pick_pose_tf_list.append((gid, tcp_tf))
            sorted_gids = [gid for gid, _ in self._sort_pose_candidates(pick_pose_tf_list)]
            self._record_timing('pickplace.motion.sort_candidates', time.perf_counter() - sort_start_time)

            for idx, gid in enumerate(sorted_gids):
                grasp = grasp_collection[gid]
                try_gid_start_time = time.perf_counter()
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
                    approach_direction=approach_direction,
                    approach_distance=approach_distance,
                    depart_direction=depart_direction,
                    depart_distance=depart_distance,
                    linear_granularity=linear_granularity,
                    use_rrt=use_rrt,
                    pln_jnt=pln_jnt,
                    toggle_dbg=toggle_dbg,
                )
                self._record_timing('pickplace.motion.try_gid', time.perf_counter() - try_gid_start_time)
                if full_plan is None:
                    if toggle_dbg and idx + 1 < len(sorted_gids):
                        next_gid = sorted_gids[idx + 1]
                        failure = self._last_plan_failure
                        if failure is None:
                            print(f'[pickplace] gid={gid} failed, trying next gid={next_gid}')
                        else:
                            print(
                                f'[pickplace] gid={gid} failed at {failure["stage"]}: '
                                f'{failure["reason"]}, trying next gid={next_gid}'
                            )
                    continue
                if release_at_end:
                    full_plan.events['release'] = len(full_plan.qs_list) - 1
                else:
                    full_plan.events.pop('release', None)
                if toggle_dbg:
                    print(f'[pickplace] selected gid={gid}, waypoints={len(full_plan.qs_list)}')
                return full_plan

            if toggle_dbg:
                print('[pickplace] no feasible pick-and-place plan found')
            return None
        finally:
            self._record_timing('pickplace.total', time.perf_counter() - total_start_time)

    def gen_place_draft(self,
                        obj_model,
                        work_name,
                        grasp_collection,
                        goal_pose_list,
                        segment_label=None,
                        end_sync_label=None,
                        keep_holding=False,
                        start_qs=None,
                        pick_approach_direction=None,
                        pick_approach_distance=0.1,
                        pick_depart_direction=None,
                        pick_depart_distance=0.1,
                        approach_direction=None,
                        approach_distance=0.07,
                        depart_direction=None,
                        depart_distance=0.07,
                        linear_granularity=0.03,
                        reason_grasps=True,
                        use_rrt=True,
                        pln_jnt=False,
                        preferred_gid=None,
                        toggle_dbg=False):
        plan = self.gen_pick_and_place(
            obj_model=obj_model,
            grasp_collection=grasp_collection,
            goal_pose_list=goal_pose_list,
            start_qs=start_qs,
            pick_approach_direction=pick_approach_direction,
            pick_approach_distance=pick_approach_distance,
            pick_depart_direction=pick_depart_direction,
            pick_depart_distance=pick_depart_distance,
            approach_direction=approach_direction,
            approach_distance=approach_distance,
            depart_direction=depart_direction,
            depart_distance=depart_distance,
            linear_granularity=linear_granularity,
            reason_grasps=reason_grasps,
            use_rrt=use_rrt,
            pln_jnt=pln_jnt,
            preferred_gid=preferred_gid,
            release_at_end=not keep_holding,
            toggle_dbg=toggle_dbg,
        )
        if plan is None:
            return None

        left_path = [
            oum.np.asarray(qs[:self.robot.ndof], dtype=oum.np.float32).copy()
            for qs in plan.qs_list
        ]
        selected_gid = int(plan.events['gid']) if 'gid' in plan.events else None
        grasp = None
        if selected_gid is not None and 0 <= selected_gid < len(grasp_collection):
            grasp = grasp_collection[selected_gid]

        jaw_width = None
        engage_tf = None
        if grasp is not None:
            pick_pose_tf = oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos)
            jaw_width, engage_tf = self._grasp_event_payload(
                obj_pose=pick_pose_tf,
                grasp=grasp,
            )

        held_after = None
        if keep_holding and selected_gid is not None:
            held_after = HeldGrasp(work_name=work_name, gid=selected_gid)

        pre_open_events = [EEEvent(
            actor='left_gripper',
            action='open',
            timing='start',
            label=f'open before picking {work_name}',
        )] if 'pre_release' in plan.events else []
        segment_specs = []
        if 'prefix_end' in plan.events:
            segment_specs.append((
                0,
                int(plan.events['prefix_end']),
                f'clear for picking {work_name}',
                f'{work_name} pick_clear',
                pre_open_events,
            ))
            pre_open_events = []

        pick_start = 0 if 'prefix_end' not in plan.events else int(plan.events['prefix_end'])
        pick_pre = int(plan.events.get('pick_pre', pick_start))
        attach_idx = int(plan.events['attach'])
        post_attach = int(plan.events.get('post_attach', attach_idx))
        place_pre = int(plan.events.get('place_pre', post_attach))
        release_idx = int(plan.events.get('release', len(left_path) - 1))

        segment_specs.append((
            pick_start,
            pick_pre,
            f'approach {work_name} pregrasp',
            f'{work_name} pregrasp',
            pre_open_events,
        ))
        segment_specs.append((
            pick_pre,
            attach_idx,
            f'grasp {work_name}',
            f'{work_name} grasp',
            [EEEvent(
                actor='left_gripper',
                action='attach',
                timing='end',
                value=jaw_width,
                work_name=work_name,
                grasp_id=selected_gid,
                engage_tf=engage_tf,
                label=f'attach {work_name}',
            )],
        ))
        if post_attach > attach_idx:
            segment_specs.append((
                attach_idx,
                post_attach,
                f'lift {work_name}',
                f'{work_name} postgrasp',
                [],
            ))
        if place_pre > post_attach:
            segment_specs.append((
                post_attach,
                place_pre,
                f'transport {work_name}',
                f'{work_name} preplace',
                [],
            ))
        place_events = []
        if not keep_holding and 'release' in plan.events:
            place_events.append(EEEvent(
                actor='left_gripper',
                action='release',
                timing='end',
                work_name=work_name,
                label=f'release {work_name}',
            ))
        segment_specs.append((
            place_pre,
            release_idx,
            segment_label or f'place {work_name}',
            end_sync_label or f'{work_name} place',
            place_events,
        ))

        segments = []
        for start_idx, end_idx, seg_label, sync_label, ee_events in segment_specs:
            if end_idx < start_idx:
                continue
            seg_path = [qs.copy() for qs in left_path[start_idx:end_idx + 1]]
            if not seg_path:
                continue
            segments.append(PlannerSegmentDraft(
                segment_label=seg_label,
                left_path=seg_path,
                right_path=[],
                ee_events=ee_events,
                end_sync_label=sync_label,
                held_after=None,
            ))

        return PlannerActionDraft(
            segments=segments,
            held_after=held_after,
        )
