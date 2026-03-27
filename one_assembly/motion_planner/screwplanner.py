from collections import Counter
import time

import one.motion.trajectory.cartesian as omtc
import one.utils.math as oum

from .hierarchical import CandidateRecord, HierarchicalPlannerBase
from . import utils


class ScrewPlanner(HierarchicalPlannerBase):
    def __init__(self, robot, pln_ctx, ee_actor=None):
        super().__init__(robot=robot, pln_ctx=pln_ctx, ee_actor=ee_actor)
        self._active_ee_qs_for_ik = None

    def _sync_robot_tcp_from_ee_qs(self, robot_obj, ee_qs=None):
        if (
            ee_qs is None or
            self.ee_actor is None or
            not hasattr(robot_obj, '_loc_tcp_tf') or
            not hasattr(robot_obj, '_mountings')
        ):
            return
        mounted_ee = self.ee_actor if robot_obj is self.robot else self._mounted_ee_actor(robot_obj)
        if mounted_ee is None or mounted_ee not in robot_obj._mountings:
            return
        mounted_ee.fk(qs=ee_qs)
        mounting = robot_obj._mountings[mounted_ee]
        robot_obj._loc_tcp_tf[:] = mounting.engage_tf @ mounted_ee.loc_tcp_tf

    def _ik_tcp_nearest_with_ee(self, tgt_rotmat, tgt_pos, ref_qs, ee_qs=None):
        self._sync_robot_tcp_from_ee_qs(self.robot, ee_qs=ee_qs)
        return self.robot.ik_tcp_nearest(
            tgt_rotmat=tgt_rotmat,
            tgt_pos=tgt_pos,
            ref_qs=oum.np.asarray(ref_qs, dtype=oum.np.float32),
        )

    def _robot_at_qs(self, qs):
        robot_qs, ee_qs = self._split_state(qs)
        self._scratch_robot.fk(qs=robot_qs)
        self._sync_robot_tcp_from_ee_qs(self._scratch_robot, ee_qs=ee_qs)
        return self._scratch_robot

    def _sorted_ik_solutions(self, tgt_pos, tgt_rotmat, ref_qs):
        self._sync_robot_tcp_from_ee_qs(self.robot, ee_qs=self._active_ee_qs_for_ik)
        return super()._sorted_ik_solutions(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, ref_qs=ref_qs)

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
        prev_ee_qs = self._active_ee_qs_for_ik
        self._active_ee_qs_for_ik = None if ee_qs is None else oum.np.asarray(ee_qs, dtype=oum.np.float32)
        try:
            return super()._screen_pose_with_stats(
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
                diagnose_collision_pairs=diagnose_collision_pairs,
                debug_visualize_contacts=debug_visualize_contacts,
                debug_contact_rgb=debug_contact_rgb,
            )
        finally:
            self._active_ee_qs_for_ik = prev_ee_qs

    def _solve_pose_state(self, pose_tf, ref_qs, ee_qs=None, pln_ctx=None, failure_stage='pose_ik'):
        if pln_ctx is None:
            pln_ctx = self.pln_ctx
        pose_tf = oum.np.asarray(pose_tf, dtype=oum.np.float32)
        ref_qs = oum.np.asarray(ref_qs, dtype=oum.np.float32)
        goal_qs = self._ik_tcp_nearest_with_ee(
            tgt_rotmat=pose_tf[:3, :3],
            tgt_pos=pose_tf[:3, 3],
            ref_qs=ref_qs,
            ee_qs=ee_qs,
        )
        if goal_qs is None:
            self._set_last_plan_failure(failure_stage, 'goal_ik_failed')
            return None
        state = self._compose_state(goal_qs, ee_qs)
        if not pln_ctx.is_state_valid(state):
            self._set_last_plan_failure(failure_stage, 'goal_state_in_collision')
            return None
        return state

    def _linear_motion_to_pose(self, goal_tcp_pos, goal_tcp_rotmat,
                               direction=None,
                               distance=0.07,
                               ee_values=None,
                               ref_qs=None,
                               pln_ctx=None,
                               granularity=0.02,
                               motion_type='sink'):
        goal_tcp_pos = oum.np.asarray(goal_tcp_pos, dtype=oum.np.float32)
        goal_tcp_rotmat = oum.np.asarray(goal_tcp_rotmat, dtype=oum.np.float32)
        direction = self._normalize_direction(goal_tcp_rotmat, direction)
        offset = direction * float(distance)
        resolved_ee_qs = None if ee_values is None else self._resolve_ee_qs(ee_values)
        if ref_qs is None:
            ref_qs = oum.np.asarray(self.robot.qs, dtype=oum.np.float32)
        else:
            ref_qs, split_ee_qs = self._split_state(ref_qs, ee_values=ee_values)
            if resolved_ee_qs is None:
                resolved_ee_qs = split_ee_qs
        if motion_type == 'sink':
            start_pos = goal_tcp_pos - offset
            goal_pos = goal_tcp_pos
        elif motion_type == 'source':
            start_pos = goal_tcp_pos
            goal_pos = goal_tcp_pos - offset
        else:
            raise ValueError(f'Unsupported motion_type: {motion_type}')
        seed_qs = self._ik_tcp_nearest_with_ee(
            tgt_rotmat=goal_tcp_rotmat,
            tgt_pos=start_pos,
            ref_qs=ref_qs,
            ee_qs=resolved_ee_qs,
        )
        if seed_qs is None:
            self._set_last_plan_failure(f'{motion_type}_seed_ik', 'seed_ik_failed')
            return None
        return self._linear_motion_between_poses(
            start_pos=start_pos,
            start_rotmat=goal_tcp_rotmat,
            goal_pos=goal_pos,
            goal_rotmat=goal_tcp_rotmat,
            seed_qs=self._compose_state(seed_qs, resolved_ee_qs),
            ee_values=resolved_ee_qs,
            pln_ctx=pln_ctx,
            pos_step=granularity,
        )

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
                print('[screw_prefix] skipped: robot has no home_qs')
            return None
        home_qs = oum.np.asarray(home_qs, dtype=oum.np.float32)
        if oum.np.allclose(start_robot_qs, home_qs):
            if toggle_dbg:
                print('[screw_prefix] skipped: start_qs already at home_qs')
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
                    print('[screw_prefix] failed: linear depart unavailable')
                else:
                    print(
                        '[screw_prefix] failed: '
                        f'{failure["stage"]} {failure["reason"]}'
                    )
            else:
                print(f'[screw_prefix] planned: waypoints={len(depart_plan.qs_list)}')
        return depart_plan

    def _unit_vec(self, vec):
        vec = oum.np.asarray(vec, dtype=oum.np.float32).reshape(-1)
        if vec.size != 3:
            raise ValueError('expected a 3D vector')
        norm = float(oum.np.linalg.norm(vec))
        if norm <= 0.0:
            raise ValueError('vector must be non-zero')
        return (vec / norm).astype(oum.np.float32)

    def _supports_shank_extension(self):
        return (
            self.ee_actor is not None and
            hasattr(self.ee_actor, 'set_shank_len') and
            hasattr(self.ee_actor, 'shank_range')
        )

    def _shank_ee_qs(self, length):
        if not self._supports_shank_extension():
            return None
        ee_actor = self.ee_actor.clone()
        ee_actor.set_shank_len(float(length))
        return oum.np.asarray(ee_actor.qs[:ee_actor.ndof], dtype=oum.np.float32)

    def _retracted_ee_qs(self):
        if not self._supports_shank_extension():
            return None
        shank_range = oum.np.asarray(self.ee_actor.shank_range, dtype=oum.np.float32).reshape(-1)
        return self._shank_ee_qs(float(shank_range[0]))

    def _extended_ee_qs(self):
        if not self._supports_shank_extension():
            return None
        shank_range = oum.np.asarray(self.ee_actor.shank_range, dtype=oum.np.float32).reshape(-1)
        return self._shank_ee_qs(float(shank_range[1]))

    def _pick_state(self, pick_pose=None, pick_qs=None, ref_qs=None):
        if pick_qs is not None:
            return self._compose_state(*self._split_state(pick_qs))
        if pick_pose is None:
            return None
        pick_tf = self._pose_to_tf(pick_pose)
        if ref_qs is None:
            ref_qs = self.robot.qs.copy()
        goal_qs = self._ik_tcp_nearest_with_ee(
            tgt_rotmat=pick_tf[:3, :3],
            tgt_pos=pick_tf[:3, 3],
            ref_qs=ref_qs,
            ee_qs=self._active_ee_qs_for_ik,
        )
        if goal_qs is None:
            return None
        return self._compose_state(goal_qs)

    def gen_goal_pose_list(self,
                           tgt_pos,
                           tgt_vec,
                           resolution=20,
                           angle_offset=0.0):
        tgt_pos = oum.np.asarray(tgt_pos, dtype=oum.np.float32)
        if tgt_pos.shape != (3,):
            raise ValueError('tgt_pos must be a 3D position')
        if resolution <= 0:
            raise ValueError('resolution must be positive')
        z_ax = self._unit_vec(tgt_vec)
        x_ax = oum.orth_vec(z_ax, toggle_unit=True)
        y_ax = oum.np.cross(z_ax, x_ax).astype(oum.np.float32)
        y_ax = self._unit_vec(y_ax)
        initial_rotmat = oum.np.column_stack((x_ax, y_ax, z_ax)).astype(oum.np.float32)
        goal_pose_list = []
        for angle in oum.np.linspace(0.0, 2.0 * oum.pi, int(resolution), endpoint=False, dtype=oum.np.float32):
            rotmat = oum.rotmat_from_axangle(z_ax, float(angle + angle_offset)) @ initial_rotmat
            goal_pose_list.append((tgt_pos.copy(), rotmat.astype(oum.np.float32)))
        return goal_pose_list

    def _goal_pose_angle_offset(self, goal_pose):
        goal_tf = self._pose_to_tf(goal_pose)
        goal_rotmat = goal_tf[:3, :3]
        z_ax = self._unit_vec(goal_rotmat[:, 2])
        x0 = oum.orth_vec(z_ax, toggle_unit=True).astype(oum.np.float32)
        y0 = self._unit_vec(oum.np.cross(z_ax, x0))
        desired_x = oum.np.asarray(goal_rotmat[:, 0], dtype=oum.np.float32)
        desired_x = desired_x - z_ax * float(oum.np.dot(desired_x, z_ax))
        x_norm = float(oum.np.linalg.norm(desired_x))
        if x_norm <= 0.0:
            desired_y = oum.np.asarray(goal_rotmat[:, 1], dtype=oum.np.float32)
            desired_x = oum.np.cross(desired_y, z_ax).astype(oum.np.float32)
            x_norm = float(oum.np.linalg.norm(desired_x))
            if x_norm <= 0.0:
                return 0.0
        desired_x = (desired_x / x_norm).astype(oum.np.float32)
        return float(oum.np.arctan2(oum.np.dot(desired_x, y0), oum.np.dot(desired_x, x0)))

    def _resolve_goal_pose_list(self, goal_pose_list, tgt_pos, tgt_vec, resolution, angle_offset):
        if goal_pose_list is None:
            if tgt_pos is None or tgt_vec is None:
                raise ValueError('goal_pose_list or tgt_pos/tgt_vec is required')
            return self.gen_goal_pose_list(
                tgt_pos=tgt_pos,
                tgt_vec=tgt_vec,
                resolution=resolution,
                angle_offset=angle_offset,
            )
        resolved_goal_pose_list = list(goal_pose_list)
        if len(resolved_goal_pose_list) == 1 and resolution > 1:
            goal_tf = self._pose_to_tf(resolved_goal_pose_list[0])
            return self.gen_goal_pose_list(
                tgt_pos=goal_tf[:3, 3],
                tgt_vec=goal_tf[:3, 2],
                resolution=resolution,
                angle_offset=self._goal_pose_angle_offset(resolved_goal_pose_list[0]),
            )
        return resolved_goal_pose_list

    def gen_pose_roll_candidates(self, pose, resolution=20):
        if resolution <= 1:
            goal_tf = self._pose_to_tf(pose)
            return [(goal_tf[:3, 3].copy(), goal_tf[:3, :3].copy())]
        goal_tf = self._pose_to_tf(pose)
        return self.gen_goal_pose_list(
            tgt_pos=goal_tf[:3, 3],
            tgt_vec=goal_tf[:3, 2],
            resolution=resolution,
            angle_offset=self._goal_pose_angle_offset(pose),
        )

    def reason_common_sids(self,
                           goal_pose_list,
                           pick_pose_list=None,
                           pick_pose=None,
                           pick_approach_direction=None,
                           pick_approach_distance=0.0,
                           pick_depart_direction=None,
                           pick_depart_distance=0.0,
                           approach_direction=None,
                           approach_distance=0.0,
                           place_depart_distance=0.1,
                           place_depart_direction=None,
                           linear_granularity=0.03,
                           ref_qs=None,
                           exclude_entities=None,
                           toggle_dbg=False):
        return list(self._reason_screw_candidates(
            goal_pose_list=goal_pose_list,
            pick_pose_list=pick_pose_list,
            pick_pose=pick_pose,
            pick_approach_direction=pick_approach_direction,
            pick_approach_distance=pick_approach_distance,
            pick_depart_direction=pick_depart_direction,
            pick_depart_distance=pick_depart_distance,
            approach_direction=approach_direction,
            approach_distance=approach_distance,
            place_depart_distance=place_depart_distance,
            place_depart_direction=place_depart_direction,
            linear_granularity=linear_granularity,
            ref_qs=ref_qs,
            exclude_entities=exclude_entities,
            toggle_dbg=toggle_dbg,
        ).keys())

    def _reason_screw_candidates(self,
                                 goal_pose_list,
                                 pick_pose_list=None,
                                 pick_pose=None,
                                 pick_approach_direction=None,
                                 pick_approach_distance=0.0,
                                 pick_depart_direction=None,
                                 pick_depart_distance=0.0,
                                 approach_direction=None,
                                 approach_distance=0.0,
                                 place_depart_distance=0.1,
                                 place_depart_direction=None,
                                 linear_granularity=0.03,
                                 ref_qs=None,
                                 exclude_entities=None,
                                 toggle_dbg=False):
        total_start_time = time.perf_counter()
        try:
            pose_tf_list = [(sid, self._pose_to_tf(goal_pose), None) for sid, goal_pose in enumerate(goal_pose_list)]
            ref_qs = self.robot.qs.copy() if ref_qs is None else oum.np.asarray(ref_qs, dtype=oum.np.float32)
            shank_extend_pick = self._extended_ee_qs() if self._supports_shank_extension() else None
            shank_extend_place = self._extended_ee_qs() if self._supports_shank_extension() else None

            ctx_start_time = time.perf_counter()
            pick_transit_pln_ctx = self._precise_pln_ctx(
                exclude_entities=exclude_entities,
                backend=self.transit_backend,
            )
            pick_goal_arm_pln_ctx = self._arm_only_goal_pln_ctx(
                exclude_entities=exclude_entities,
            )
            pick_goal_ee_pln_ctx = self._ee_local_goal_pln_ctx(
                exclude_entities=exclude_entities,
                backend=self.reasoning_backend,
            )
            screw_goal_arm_pln_ctx = self._arm_only_goal_pln_ctx(
                exclude_entities=exclude_entities,
            )
            screw_goal_ee_pln_ctx = self._ee_local_goal_pln_ctx(
                exclude_entities=exclude_entities,
                backend=self.reasoning_backend,
            )
            screw_transit_pln_ctx = self._precise_pln_ctx(
                exclude_entities=exclude_entities,
                backend=self.transit_backend,
            )
            self._record_timing('screw.reason.context_setup', time.perf_counter() - ctx_start_time)

            self._last_reason_common_screw_report = {
                'survived_sids': [],
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

            def record_failure(sid, label, reason, pose_tf):
                self._last_reason_common_screw_report['failures'].setdefault(
                    sid,
                    {
                        'label': label,
                        'reason': reason,
                        'tcp_pos': pose_tf[:3, 3].copy(),
                        'tcp_rotmat': pose_tf[:3, :3].copy(),
                    },
                )

            self._last_reason_common_screw_pick_record = None
            current_ref_qs = ref_qs.copy()
            pick_pose_entries = []
            if pick_pose_list is not None:
                pick_pose_entries = [(idx, self._pose_to_tf(pose), None) for idx, pose in enumerate(pick_pose_list)]
            elif pick_pose is not None:
                pick_pose_entries = [(0, self._pose_to_tf(pick_pose), None)]
            if pick_pose_entries:
                pick_records = []
                if pick_approach_distance > 0.0:
                    pre_entries = []
                    for pid, pick_tf, _ee_qs in pick_pose_entries:
                        pick_pre_tf = self._offset_pose_tf(
                            goal_tcp_pos=pick_tf[:3, 3],
                            goal_tcp_rotmat=pick_tf[:3, :3],
                            direction=pick_approach_direction,
                            distance=pick_approach_distance,
                            motion_type='sink',
                        )
                        pre_entries.append((pid, pick_pre_tf, shank_extend_pick))
                    pick_approach_start_time = time.perf_counter()
                    pick_records = self._screen_pose_list(
                        pre_entries,
                        pln_ctx=pick_transit_pln_ctx,
                        linear_granularity=linear_granularity,
                        ref_qs=current_ref_qs,
                        timing_prefix='screw.reason.pick_approach_start.detail',
                        toggle_dbg=False,
                        debug_label='screw_reason pick_approach_start',
                    )
                    self._record_timing('screw.reason.pick_approach_start', time.perf_counter() - pick_approach_start_time)
                    survived_ids = {record.key for record in pick_records}
                    reason_counts = {'unreachable_approach': 0}
                    for pid, pick_pre_tf, _ee_qs in pre_entries:
                        if pid not in survived_ids:
                            record_failure(pid, 'pick_approach_start', 'unreachable_approach', pick_pre_tf)
                            reason_counts['unreachable_approach'] += 1
                    print_stage_stats(
                        'screw_reason pick_approach_start',
                        len(pre_entries),
                        len(pick_records),
                        reason_counts,
                    )
                    if not pick_records:
                        if toggle_dbg and pre_entries:
                            self._debug_visualize_failed_pose_entries(
                                label='screw_reason pick_approach_start',
                                debug_entries=[
                                    {
                                        'key': pid,
                                        'pose_tf': pick_pre_tf,
                                        'ee_qs': shank_extend_pick,
                                        'pln_ctx': pick_transit_pln_ctx,
                                        'ref_qs': current_ref_qs.copy(),
                                    }
                                    for pid, pick_pre_tf, _ee_qs in pre_entries
                                ],
                                linear_granularity=linear_granularity,
                            )
                        self._last_reason_common_screw_report['survived_sids'] = []
                        return {}
                    pick_ref_qs_map = {
                        record.key: record.screen_result.goal_qs[:self.robot.ndof].copy()
                        for record in pick_records
                    }
                else:
                    pick_ref_qs_map = {pid: current_ref_qs.copy() for pid, _pick_tf, _ee_qs in pick_pose_entries}
                    pick_records = [CandidateRecord(key=pid, pose_tf=pick_tf, screen_result=None) for pid, pick_tf, _ee_qs in pick_pose_entries]

                next_pick_records = []
                pick_goal_reason_counts = {}
                pick_goal_pair_counts = Counter()
                pick_goal_start_time = time.perf_counter()
                pick_tf_map = {pid: pick_tf for pid, pick_tf, _ee_qs in pick_pose_entries}
                pick_goal_debug_entries = []
                for record in pick_records:
                    pick_tf = pick_tf_map[record.key]
                    pick_goal_debug_entries.append(
                        {
                            'key': record.key,
                            'pose_tf': pick_tf,
                            'ee_qs': shank_extend_pick,
                            'pln_ctx': pick_goal_ee_pln_ctx,
                            'ref_qs': pick_ref_qs_map[record.key],
                        }
                    )
                    pick_goal_result, pick_goal_stats = self._screen_pose_with_stats(
                        tcp_pos=pick_tf[:3, 3],
                        tcp_rotmat=pick_tf[:3, :3],
                        ee_qs=shank_extend_pick,
                        coarse_pln_ctx=pick_goal_arm_pln_ctx,
                        pln_ctx=pick_goal_ee_pln_ctx,
                        depart_pln_ctx=pick_transit_pln_ctx,
                        depart_direction=pick_depart_direction,
                        depart_distance=pick_depart_distance,
                        linear_granularity=linear_granularity,
                        ref_qs=pick_ref_qs_map[record.key],
                        timing_prefix='screw.reason.pick_goal.detail',
                        diagnose_collision_pairs=toggle_dbg,
                        debug_visualize_contacts=toggle_dbg,
                    )
                    if pick_goal_result is None:
                        pick_reason = self._classify_screen_stats(pick_goal_stats)
                        record_failure(record.key, 'pick_goal', pick_reason, pick_tf)
                        pick_goal_reason_counts[pick_reason] = pick_goal_reason_counts.get(pick_reason, 0) + 1
                        for pair, count in pick_goal_stats.get('collision_pairs', []):
                            pick_goal_pair_counts[pair] += count
                        continue
                    next_pick_records.append(CandidateRecord(key=record.key, pose_tf=pick_tf, screen_result=pick_goal_result))
                self._record_timing('screw.reason.pick_goal', time.perf_counter() - pick_goal_start_time)
                print_stage_stats(
                    'screw_reason pick_goal',
                    len(pick_records),
                    len(next_pick_records),
                    pick_goal_reason_counts,
                    pair_counts=pick_goal_pair_counts,
                )
                if not next_pick_records:
                    if toggle_dbg and pick_goal_debug_entries:
                        self._debug_visualize_failed_pose_entries(
                            label='screw_reason pick_goal',
                            debug_entries=pick_goal_debug_entries,
                            linear_granularity=linear_granularity,
                        )
                    self._last_reason_common_screw_report['survived_sids'] = []
                    return {}
                self._last_reason_common_screw_pick_record = next_pick_records[0]
                current_ref_qs = next_pick_records[0].screen_result.goal_qs[:self.robot.ndof].copy()

            approach_entries = []
            for sid, pose_tf, ee_qs in pose_tf_list:
                if self._grasp_has_collision((pose_tf, pose_tf, 0.0, 0.0, False)):
                    continue
                if approach_distance > 0.0:
                    pre_tf = self._offset_pose_tf(
                        goal_tcp_pos=pose_tf[:3, 3],
                        goal_tcp_rotmat=pose_tf[:3, :3],
                        direction=approach_direction,
                        distance=approach_distance,
                        motion_type='sink',
                    )
                else:
                    pre_tf = pose_tf
                entry_ee_qs = shank_extend_place if approach_distance > 0.0 else ee_qs
                approach_entries.append((sid, pre_tf, entry_ee_qs))

            debug_entries = [
                {
                    'key': sid,
                    'pose_tf': pre_tf,
                    'ee_qs': ee_qs,
                    'pln_ctx': screw_transit_pln_ctx,
                    'ref_qs': current_ref_qs.copy(),
                }
                for sid, pre_tf, ee_qs in approach_entries
            ]
            screw_approach_start_time = time.perf_counter()
            approach_records = self._screen_pose_list(
                approach_entries,
                pln_ctx=screw_transit_pln_ctx,
                linear_granularity=linear_granularity,
                ref_qs=current_ref_qs,
                timing_prefix='screw.reason.screw_approach_start.detail',
                toggle_dbg=True,
                debug_label='screw_reason screw_approach_start',
            )
            self._record_timing('screw.reason.screw_approach_start', time.perf_counter() - screw_approach_start_time)
            survived_approach_ids = {record.key for record in approach_records}
            approach_reason_counts = {'unreachable_approach': 0}
            for sid, pre_tf, _ee_qs in approach_entries:
                if sid not in survived_approach_ids:
                    record_failure(sid, 'screw_approach_start', 'unreachable_approach', pre_tf)
                    approach_reason_counts['unreachable_approach'] += 1
            print_stage_stats(
                'screw_reason screw_approach_start',
                len(approach_entries),
                len(approach_records),
                approach_reason_counts,
            )
            if not approach_records:
                if toggle_dbg and approach_entries:
                    self._debug_visualize_failed_pose_entries(
                        label='screw_reason screw_approach_start',
                        debug_entries=debug_entries,
                        linear_granularity=linear_granularity,
                    )
                self._last_reason_common_screw_report['survived_sids'] = []
                return {}

            next_records = []
            self._last_reason_common_screw_approach_record_map = {
                record.key: record
                for record in approach_records
            }
            next_ref_qs_map = {
                record.key: record.screen_result.goal_qs[:self.robot.ndof].copy()
                for record in approach_records
            }
            reason_counts = {}
            pair_counts = Counter()
            stage_start_time = time.perf_counter()
            goal_tf_map = {sid: pose_tf for sid, pose_tf, _ee_qs in pose_tf_list}
            for record in approach_records:
                sid = record.key
                pose_tf = goal_tf_map[sid]
                ee_qs = shank_extend_place if approach_distance > 0.0 else None
                result, stats = self._screen_pose_with_stats(
                    tcp_pos=pose_tf[:3, 3],
                    tcp_rotmat=pose_tf[:3, :3],
                    ee_qs=ee_qs,
                    coarse_pln_ctx=screw_goal_arm_pln_ctx,
                    pln_ctx=screw_goal_ee_pln_ctx,
                    depart_pln_ctx=screw_transit_pln_ctx,
                    exclude_entities=exclude_entities,
                    depart_direction=place_depart_direction,
                    depart_distance=place_depart_distance,
                    linear_granularity=linear_granularity,
                    ref_qs=next_ref_qs_map[sid],
                    timing_prefix='screw.reason.goal.detail',
                    diagnose_collision_pairs=toggle_dbg,
                    debug_visualize_contacts=toggle_dbg,
                )
                if result is None:
                    reason = self._classify_screen_stats(stats)
                    record_failure(sid, 'screw_goal', reason, pose_tf)
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    for pair, count in stats.get('collision_pairs', []):
                        pair_counts[pair] += count
                    continue
                next_records.append(CandidateRecord(key=sid, pose_tf=pose_tf, screen_result=result))
            self._record_timing('screw.reason.goal', time.perf_counter() - stage_start_time)
            print_stage_stats(
                'screw_reason screw_goal',
                len(approach_records),
                len(next_records),
                reason_counts,
                pair_counts=pair_counts,
            )
            if toggle_dbg and approach_records and not next_records:
                self._debug_visualize_failed_pose_entries(
                    label='screw_reason screw_goal',
                    debug_entries=debug_entries,
                    linear_granularity=linear_granularity,
                )
            self._last_reason_common_screw_report['survived_sids'] = [record.key for record in next_records]
            if toggle_dbg:
                print(f'[screw_reason final], survived={len(next_records)}')
            return {record.key: record for record in next_records}
        finally:
            self._record_timing('screw.reason.total', time.perf_counter() - total_start_time)

    def gen_screw(self,
                  goal_pose_list=None,
                  start_qs=None,
                  tgt_pos=None,
                  tgt_vec=None,
                  resolution=20,
                  angle_offset=0.0,
                  pick_pose_list=None,
                  pick_pose=None,
                  pick_qs=None,
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
        if start_qs is None:
            start_qs = self.robot.qs.copy()
        start_state = self._compose_state(*self._split_state(start_qs))
        prefix_plan = None
        require_prefix_depart = self._needs_prefix_start_depart(start_state)
        prefix_depart_plan = self._prefix_start_depart(
            start_qs=start_state,
            ee_qs=None,
            linear_granularity=linear_granularity,
            toggle_dbg=toggle_dbg,
        )
        if require_prefix_depart and prefix_depart_plan is None:
            self._set_last_plan_failure('prefix_depart', 'required_depart_failed')
            if toggle_dbg:
                print('[screw_prefix] required depart missing, aborting screw plan')
            return None
        if prefix_depart_plan is None:
            current_state = start_state
        else:
            prefix_plan = prefix_depart_plan
            current_state = prefix_depart_plan.qs_list[-1]
        goal_pose_list = self._resolve_goal_pose_list(
            goal_pose_list=goal_pose_list,
            tgt_pos=tgt_pos,
            tgt_vec=tgt_vec,
            resolution=resolution,
            angle_offset=angle_offset,
        )
        candidate_map = self._reason_screw_candidates(
            goal_pose_list=goal_pose_list,
            pick_pose_list=pick_pose_list,
            pick_pose=pick_pose,
            pick_approach_direction=pick_approach_direction,
            pick_approach_distance=pick_approach_distance,
            pick_depart_direction=pick_depart_direction,
            pick_depart_distance=pick_depart_distance,
            approach_direction=approach_direction,
            approach_distance=approach_distance,
            place_depart_distance=depart_distance,
            place_depart_direction=depart_direction,
            linear_granularity=linear_granularity,
            ref_qs=current_state[:self.robot.ndof],
            toggle_dbg=toggle_dbg,
        )
        if not candidate_map:
            if toggle_dbg:
                print('[screw] no feasible screw reasoning candidate found')
            return None
        pick_record = getattr(self, '_last_reason_common_screw_pick_record', None)
        retracted_ee_qs = self._retracted_ee_qs()
        extended_ee_qs = self._extended_ee_qs()
        pick_state = None
        if pick_record is not None:
            pick_state = pick_record.screen_result.goal_qs
        else:
            prev_ee_qs = self._active_ee_qs_for_ik
            self._active_ee_qs_for_ik = extended_ee_qs
            pick_state = self._pick_state(
                pick_pose=pick_pose,
                pick_qs=pick_qs,
                ref_qs=current_state[:self.robot.ndof],
            )
            self._active_ee_qs_for_ik = prev_ee_qs
        if pick_state is None and (pick_pose is not None or pick_qs is not None):
            if toggle_dbg:
                print('[screw] failed to resolve pick pose')
            return None
        if pick_state is not None:
            if extended_ee_qs is not None and (pick_approach_distance > 0.0 or pick_depart_distance > 0.0):
                pick_state = self._compose_with_ee_qs(pick_state, extended_ee_qs)
            pick_plan = self.gen_approach_depart(
                goal_qs=pick_state,
                start_qs=current_state,
                end_qs=None,
                approach_direction=pick_approach_direction,
                approach_distance=pick_approach_distance,
                depart_direction=pick_depart_direction,
                depart_distance=pick_depart_distance,
                approach_linear=pick_approach_distance > 0.0,
                depart_linear=pick_depart_distance > 0.0,
                linear_granularity=linear_granularity,
                pln_ctx=self.pln_ctx,
                use_rrt=use_rrt,
                pln_jnt=pln_jnt,
            )
            if pick_plan is None:
                if toggle_dbg:
                    failure = self._last_plan_failure
                    if failure is None:
                        print('[screw] failed to generate pick motion')
                    else:
                        print(
                            '[screw] failed to generate pick motion: '
                            f'{failure["stage"]} {failure["reason"]}'
                        )
                return None
            if extended_ee_qs is not None and retracted_ee_qs is not None:
                retract_pick_state = self._compose_with_ee_qs(pick_plan.qs_list[-1], retracted_ee_qs)
                if not oum.np.allclose(retract_pick_state, pick_plan.qs_list[-1]):
                    retract_pick_plan = self._connect_motion(
                        start_qs=pick_plan.qs_list[-1],
                        goal_qs=retract_pick_state,
                        pln_ctx=self.pln_ctx,
                        use_rrt=False,
                    )
                    if retract_pick_plan is None:
                        return None
                    pick_plan = self._merge_plans(pick_plan, retract_pick_plan)
            if prefix_plan is None:
                prefix_plan = pick_plan
            else:
                prefix_plan = self._merge_plans(prefix_plan, pick_plan)
            current_state = pick_plan.qs_list[-1]
        for sid, record in candidate_map.items():
            result = record.screen_result
            approach_record_map = getattr(self, '_last_reason_common_screw_approach_record_map', {})
            approach_record = approach_record_map.get(sid)
            goal_state = result.goal_qs
            if extended_ee_qs is not None and approach_distance > 0.0:
                goal_state = self._compose_with_ee_qs(goal_state, extended_ee_qs)
            if approach_record is not None and approach_distance > 0.0:
                via_tf = approach_record.pose_tf
                via_state = self._compose_with_ee_qs(approach_record.screen_result.goal_qs, extended_ee_qs)
                connect_plan = self._connect_motion(
                    start_qs=current_state,
                    goal_qs=via_state,
                    pln_ctx=self.pln_ctx,
                    use_rrt=use_rrt,
                )
                if connect_plan is None:
                    if toggle_dbg:
                        failure = self._last_plan_failure
                        if failure is None:
                            print(f'[screw] sid={sid} pre_screw_connect failed')
                        else:
                            print(
                                f'[screw] sid={sid} pre_screw_connect '
                                f'{failure["stage"]} failed: {failure["reason"]}'
                            )
                    plan = None
                else:
                    linear_plan = self._linear_motion_between_poses(
                        start_pos=via_tf[:3, 3],
                        start_rotmat=via_tf[:3, :3],
                        goal_pos=record.pose_tf[:3, 3],
                        goal_rotmat=record.pose_tf[:3, :3],
                        seed_qs=via_state,
                        pln_ctx=self.pln_ctx,
                        pos_step=linear_granularity,
                    )
                    if linear_plan is None:
                        if toggle_dbg:
                            failure = self._last_plan_failure
                            if failure is None:
                                print(f'[screw] sid={sid} pre_screw_linear failed')
                            else:
                                print(
                                    f'[screw] sid={sid} pre_screw_linear '
                                    f'{failure["stage"]} failed: {failure["reason"]}'
                                )
                        plan = None
                    else:
                        plan = self._merge_plans(connect_plan, linear_plan)
            else:
                plan = self.gen_approach(
                    goal_qs=goal_state,
                    start_qs=current_state,
                    approach_direction=approach_direction,
                    approach_distance=approach_distance,
                    linear=approach_distance > 0.0,
                    linear_granularity=linear_granularity,
                    pln_ctx=self.pln_ctx,
                    use_rrt=use_rrt,
                    pln_jnt=pln_jnt,
                )
            if plan is not None:
                if extended_ee_qs is not None and retracted_ee_qs is not None and approach_distance > 0.0:
                    retract_place_state = self._compose_with_ee_qs(plan.qs_list[-1], retracted_ee_qs)
                    if not oum.np.allclose(retract_place_state, plan.qs_list[-1]):
                        retract_place_plan = self._connect_motion(
                            start_qs=plan.qs_list[-1],
                            goal_qs=retract_place_state,
                            pln_ctx=self.pln_ctx,
                            use_rrt=False,
                        )
                        if retract_place_plan is None:
                            plan = None
                        else:
                            plan = self._merge_plans(plan, retract_place_plan)
            if plan is not None:
                if prefix_plan is not None:
                    plan = self._merge_plans(prefix_plan, plan)
                plan.events['sid'] = sid
                if toggle_dbg:
                    print(f'[screw] selected sid={sid}, waypoints={len(plan.qs_list)}')
                return plan
            if toggle_dbg:
                failure = self._last_plan_failure
                if failure is None:
                    print(f'[screw] sid={sid} motion planning failed after reasoning')
                else:
                    print(
                        f'[screw] sid={sid} '
                        f'{failure["stage"]} failed: {failure["reason"]}'
                    )
        if toggle_dbg:
            print('[screw] no feasible screw plan found')
        return None
