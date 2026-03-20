import one.motion.trajectory.cartesian as omtc
import one.robots.manipulators.manipulator_base as orm
import one.utils.math as oum
import time

from . import utils


class ADPlanner:
    """Approach/depart planner built on top of Cartesian IK and joint-space connect planning."""

    def __init__(self, robot: orm.ManipulatorBase, pln_ctx, ee_actor=None):
        self.robot = robot
        self.ee_actor = ee_actor
        self.pln_ctx = pln_ctx
        self._scratch_robot = robot.clone()
        self._last_plan_failure = None
        self._timing_stats = {}
        self._timing_events = []
        self._metric_stats = {}

    def _clear_last_plan_failure(self):
        self._last_plan_failure = None

    def _set_last_plan_failure(self, stage, reason):
        self._last_plan_failure = {
            'stage': stage,
            'reason': reason,
        }

    def _reset_timing_report(self):
        self._timing_stats = {}
        self._timing_events = []
        self._metric_stats = {}

    def _record_timing(self, label, duration):
        stats = self._timing_stats.setdefault(
            label,
            {
                'total_s': 0.0,
                'count': 0,
                'max_s': 0.0,
            },
        )
        duration = float(duration)
        stats['total_s'] += duration
        stats['count'] += 1
        stats['max_s'] = max(stats['max_s'], duration)
        self._timing_events.append((label, duration))

    def _record_metric(self, label, value):
        stats = self._metric_stats.setdefault(
            label,
            {
                'count': 0,
                'sum': 0.0,
                'min': None,
                'max': None,
                'last': None,
            },
        )
        value = float(value)
        stats['count'] += 1
        stats['sum'] += value
        stats['last'] = value
        stats['min'] = value if stats['min'] is None else min(stats['min'], value)
        stats['max'] = value if stats['max'] is None else max(stats['max'], value)

    def timing_report(self):
        return {
            'stats': {
                label: value.copy()
                for label, value in self._timing_stats.items()
            },
            'events': list(self._timing_events),
            'metrics': {
                label: value.copy()
                for label, value in self._metric_stats.items()
            },
        }

    def _normalize_direction(self, rotmat, direction):
        if direction is None:
            vec = oum.np.asarray(rotmat[:, 2], dtype=oum.np.float32)
        else:
            vec = oum.np.asarray(direction, dtype=oum.np.float32)
        norm = float(oum.np.linalg.norm(vec))
        if norm <= 0.0:
            raise ValueError('direction must be non-zero')
        return vec / norm

    def _ee_aux_map(self, ee_values):
        del ee_values
        return None

    def _desired_actors(self):
        actors = [self.robot]
        if self.ee_actor is not None:
            actors.append(self.ee_actor)
        return actors

    def _default_ee_qs(self):
        if self.ee_actor is None:
            return None
        return oum.np.asarray(self.ee_actor.qs[:self.ee_actor.ndof], dtype=oum.np.float32)

    def _resolve_ee_qs(self, ee_values=None):
        if self.ee_actor is None:
            return None
        if ee_values is None:
            return self._default_ee_qs()
        ee_qs = oum.np.asarray(ee_values, dtype=oum.np.float32)
        if ee_qs.size == self.ee_actor.ndof:
            return ee_qs
        full_qs_len = len(self.ee_actor.qs)
        if ee_qs.size == full_qs_len:
            return oum.np.asarray(ee_qs[:self.ee_actor.ndof], dtype=oum.np.float32)
        if self.ee_actor.ndof == 1 and ee_qs.size == 1:
            return ee_qs
        if self.ee_actor.ndof == 1 and ee_qs.size == 2 and oum.np.allclose(ee_qs[0], ee_qs[1]):
            return oum.np.asarray([ee_qs[0]], dtype=oum.np.float32)
        if ee_qs.size != self.ee_actor.ndof:
            raise ValueError(
                f'Expected ee state with {self.ee_actor.ndof} active values '
                f'or {full_qs_len} full qs values, got {ee_qs.size}'
            )
        return ee_qs

    def _split_state(self, qs, ee_values=None):
        qs = oum.np.asarray(qs, dtype=oum.np.float32)
        robot_ndof = self.robot.ndof
        if self.ee_actor is None:
            if qs.size != robot_ndof:
                raise ValueError(f'Expected {robot_ndof} robot qs values, got {qs.size}')
            return qs, None
        ee_ndof = self.ee_actor.ndof
        if qs.size == robot_ndof:
            return qs, self._resolve_ee_qs(ee_values)
        if qs.size == robot_ndof + ee_ndof:
            return qs[:robot_ndof], qs[robot_ndof:]
        raise ValueError(
            f'Expected {robot_ndof} robot qs values or '
            f'{robot_ndof + ee_ndof} combined qs values, got {qs.size}'
        )

    def _compose_state(self, robot_qs, ee_qs=None):
        robot_qs = oum.np.asarray(robot_qs, dtype=oum.np.float32)
        if self.ee_actor is None:
            return robot_qs
        if ee_qs is None:
            ee_qs = self._default_ee_qs()
        ee_qs = oum.np.asarray(ee_qs, dtype=oum.np.float32)
        return oum.np.concatenate([robot_qs, ee_qs]).astype(oum.np.float32)

    def _robot_at_qs(self, qs):
        qs, _ = self._split_state(qs)
        self._scratch_robot.fk(qs=qs)
        return self._scratch_robot

    def _tcp_pose_from_qs(self, qs):
        robot = self._robot_at_qs(qs)
        tcp_tf = robot.gl_tcp_tf
        return tcp_tf[:3, 3].copy(), tcp_tf[:3, :3].copy()

    def _motion_plan(self, q_list):
        q_list = [oum.np.asarray(qs, dtype=oum.np.float32) for qs in q_list]
        return utils.MotionData(q_list)

    def _compose_start_state(self, start_qs, ee_values=None):
        if start_qs is None:
            return None
        return self._compose_state(*self._split_state(start_qs, ee_values=ee_values))

    def _solve_pose_state(self, pose_tf, ref_qs, ee_qs=None, pln_ctx=None, failure_stage='pose_ik'):
        if pln_ctx is None:
            pln_ctx = self.pln_ctx
        pose_tf = oum.np.asarray(pose_tf, dtype=oum.np.float32)
        ref_qs = oum.np.asarray(ref_qs, dtype=oum.np.float32)
        goal_qs = self.robot.ik_tcp_nearest(
            tgt_rotmat=pose_tf[:3, :3],
            tgt_pos=pose_tf[:3, 3],
            ref_qs=ref_qs,
        )
        if goal_qs is None:
            self._set_last_plan_failure(failure_stage, 'goal_ik_failed')
            return None
        state = self._compose_state(goal_qs, ee_qs)
        if not pln_ctx.is_state_valid(state):
            self._set_last_plan_failure(failure_stage, 'goal_state_in_collision')
            return None
        return state

    def _offset_pose_tf(self, goal_tcp_pos, goal_tcp_rotmat, direction=None, distance=0.07, motion_type='sink'):
        goal_tcp_pos = oum.np.asarray(goal_tcp_pos, dtype=oum.np.float32)
        goal_tcp_rotmat = oum.np.asarray(goal_tcp_rotmat, dtype=oum.np.float32)
        direction = self._normalize_direction(goal_tcp_rotmat, direction)
        offset = direction * float(distance)
        pose_tf = oum.np.eye(4, dtype=oum.np.float32)
        pose_tf[:3, :3] = goal_tcp_rotmat
        if motion_type == 'sink':
            pose_tf[:3, 3] = goal_tcp_pos - offset
        elif motion_type == 'source':
            pose_tf[:3, 3] = goal_tcp_pos + offset
        else:
            raise ValueError(f'Unsupported motion_type: {motion_type}')
        return pose_tf

    def _keyframe_motion_plan(self, state_list, pln_ctx=None, validate_edges=True, failure_stage='keyframe_path'):
        start_time = time.perf_counter()
        try:
            if pln_ctx is None:
                pln_ctx = self.pln_ctx
            compact = []
            for state in state_list:
                if state is None:
                    continue
                state = oum.np.asarray(state, dtype=oum.np.float32)
                if len(compact) == 0 or not oum.np.allclose(compact[-1], state):
                    compact.append(state)
            if not compact:
                return None
            for state in compact:
                if not pln_ctx.is_state_valid(state):
                    self._set_last_plan_failure(failure_stage, 'state_in_collision')
                    return None
            if validate_edges and len(compact) > 1:
                for qs0, qs1 in zip(compact[:-1], compact[1:]):
                    if not pln_ctx.is_motion_valid(qs0, qs1):
                        self._set_last_plan_failure(failure_stage, 'path_in_collision')
                        return None
            return self._motion_plan(compact)
        finally:
            self._record_timing('ad.keyframe_motion_plan', time.perf_counter() - start_time)

    def _linear_motion_between_poses(self, start_pos, start_rotmat,
                                     goal_pos, goal_rotmat,
                                     seed_qs,
                                     ee_values=None,
                                     pln_ctx=None,
                                     pos_step=0.01,
                                     rot_step=oum.np.deg2rad(2.0),
                                     max_edge_step=oum.pi / 180):
        start_time = time.perf_counter()
        try:
            del max_edge_step
            if pln_ctx is None:
                pln_ctx = self.pln_ctx
            seed_robot_qs, ee_qs = self._split_state(seed_qs, ee_values=ee_values)
            q_seq, _ = omtc.cartesian_to_jtraj(
                robot=self.robot,
                start_rotmat=start_rotmat,
                start_pos=start_pos,
                goal_rotmat=goal_rotmat,
                goal_pos=goal_pos,
                pos_step=pos_step,
                rot_step=rot_step,
                ref_qs=seed_robot_qs,
            )
            if q_seq is None:
                self._set_last_plan_failure('linear_ik', 'cartesian_ik_failed')
                return None
            q_list = [self._compose_state(qs, ee_qs) for qs in q_seq]
            if not q_list or not pln_ctx.is_state_valid(q_list[0]):
                self._set_last_plan_failure('linear_start', 'start_state_in_collision')
                return None
            for qs0, qs1 in zip(q_list[:-1], q_list[1:]):
                if not pln_ctx.is_motion_valid(qs0, qs1):
                    self._set_last_plan_failure('linear_path', 'path_in_collision')
                    return None
            return self._motion_plan(q_list)
        finally:
            self._record_timing('ad.linear_motion_between_poses', time.perf_counter() - start_time)

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
        if ref_qs is None:
            ref_qs = oum.np.asarray(self.robot.qs, dtype=oum.np.float32)
        else:
            ref_qs, _ = self._split_state(ref_qs, ee_values=ee_values)
        if motion_type == 'sink':
            start_pos = goal_tcp_pos - offset
            goal_pos = goal_tcp_pos
            seed_qs = self.robot.ik_tcp_nearest(
                tgt_rotmat=goal_tcp_rotmat,
                tgt_pos=start_pos,
                ref_qs=ref_qs,
            )
        elif motion_type == 'source':
            start_pos = goal_tcp_pos
            goal_pos = goal_tcp_pos + offset
            seed_qs = self.robot.ik_tcp_nearest(
                tgt_rotmat=goal_tcp_rotmat,
                tgt_pos=start_pos,
                ref_qs=ref_qs,
            )
        else:
            raise ValueError(f'Unsupported motion_type: {motion_type}')
        if seed_qs is None:
            self._set_last_plan_failure(f'{motion_type}_seed_ik', 'seed_ik_failed')
            return None
        return self._linear_motion_between_poses(
            start_pos=start_pos,
            start_rotmat=goal_tcp_rotmat,
            goal_pos=goal_pos,
            goal_rotmat=goal_tcp_rotmat,
            seed_qs=seed_qs,
            ee_values=ee_values,
            pln_ctx=pln_ctx,
            pos_step=granularity,
        )

    def gen_approach_via_pose(self,
                              goal_tf,
                              via_tf,
                              goal_qs,
                              start_qs=None,
                              via_ee_qs=None,
                              final_ee_qs=None,
                              linear_granularity=0.03,
                              pln_ctx=None,
                              use_rrt=True,
                              pln_jnt=False,
                              connect_timing_prefix='gen_approach_via_pose'):
        goal_tf = oum.np.asarray(goal_tf, dtype=oum.np.float32)
        via_tf = oum.np.asarray(via_tf, dtype=oum.np.float32)
        goal_robot_qs, resolved_final_ee_qs = self._split_state(goal_qs, ee_values=final_ee_qs)
        resolved_via_ee_qs = self._resolve_ee_qs(via_ee_qs) if self.ee_actor is not None else None
        if self.ee_actor is not None and resolved_via_ee_qs is None:
            resolved_via_ee_qs = resolved_final_ee_qs
        via_goal_qs = self._compose_state(goal_robot_qs, resolved_via_ee_qs)
        if pln_ctx is None:
            pln_ctx = self.pln_ctx
        if pln_jnt:
            start_state = self._compose_start_state(start_qs, ee_values=resolved_via_ee_qs)
            via_state = self._solve_pose_state(
                pose_tf=via_tf,
                ref_qs=goal_robot_qs,
                ee_qs=resolved_via_ee_qs,
                pln_ctx=pln_ctx,
                failure_stage='via_pose',
            )
            if via_state is None:
                return None
            goal_state = self._compose_state(goal_robot_qs, resolved_final_ee_qs)
            return self._keyframe_motion_plan(
                [start_state, via_state, goal_state],
                pln_ctx=pln_ctx,
                validate_edges=True,
                failure_stage='via_keyframe_path',
            )
        via_pos_err = float(oum.np.linalg.norm(goal_tf[:3, 3] - via_tf[:3, 3]))
        via_rot_err = float(oum.np.linalg.norm(goal_tf[:3, :3] - via_tf[:3, :3]))
        if via_pos_err <= 1e-6 and via_rot_err <= 1e-6:
            full_plan = self._motion_plan([via_goal_qs])
            if start_qs is not None:
                start2via = self._connect_motion(
                    start_qs=start_qs,
                    goal_qs=via_goal_qs,
                    ee_values=resolved_via_ee_qs,
                    pln_ctx=pln_ctx,
                    use_rrt=use_rrt,
                    timing_label=f'{connect_timing_prefix}.start_to_via',
                )
                if start2via is None:
                    return None
                full_plan = start2via
            goal_state = self._compose_state(goal_robot_qs, resolved_final_ee_qs)
            if not oum.np.allclose(full_plan.qs_list[-1], goal_state):
                end_connect = self._connect_motion(
                    start_qs=full_plan.qs_list[-1],
                    goal_qs=goal_state,
                    ee_values=resolved_final_ee_qs,
                    pln_ctx=pln_ctx,
                    use_rrt=use_rrt,
                    timing_label=f'{connect_timing_prefix}.end_to_goal',
                )
                if end_connect is None:
                    return None
                full_plan = self._merge_plans(full_plan, end_connect)
            return full_plan

        linear_app = self._linear_motion_between_poses(
            start_pos=via_tf[:3, 3],
            start_rotmat=via_tf[:3, :3],
            goal_pos=goal_tf[:3, 3],
            goal_rotmat=goal_tf[:3, :3],
            seed_qs=via_goal_qs,
            ee_values=resolved_via_ee_qs,
            pln_ctx=pln_ctx,
            pos_step=linear_granularity,
        )
        if linear_app is None:
            return None

        full_plan = linear_app
        if start_qs is not None:
            start2via = self._connect_motion(
                start_qs=start_qs,
                goal_qs=linear_app.qs_list[0],
                ee_values=resolved_via_ee_qs,
                pln_ctx=pln_ctx,
                use_rrt=use_rrt,
                timing_label=f'{connect_timing_prefix}.start_to_via',
            )
            if start2via is None:
                return None
            full_plan = self._merge_plans(start2via, linear_app)

        goal_state = self._compose_state(goal_robot_qs, resolved_final_ee_qs)
        if not oum.np.allclose(full_plan.qs_list[-1], goal_state):
            end_connect = self._connect_motion(
                start_qs=full_plan.qs_list[-1],
                goal_qs=goal_state,
                ee_values=resolved_final_ee_qs,
                pln_ctx=pln_ctx,
                use_rrt=use_rrt,
                timing_label=f'{connect_timing_prefix}.end_to_goal',
            )
            if end_connect is None:
                return None
            full_plan = self._merge_plans(full_plan, end_connect)
        return full_plan

    def _connect_motion(self, start_qs, goal_qs,
                        ee_values=None,
                        pln_ctx=None,
                        use_rrt=True,
                        step_size=oum.pi / 36,
                        max_iters=2000,
                        time_limit=None,
                        max_edge_step=oum.pi / 180,
                        timing_label=None):
        start_time = time.perf_counter()
        connect_label = 'ad.connect_motion' if timing_label is None else f'ad.connect_motion.{timing_label}'
        rrt_label = 'joint_path.rrt_connect.solve' if timing_label is None else f'joint_path.rrt_connect.solve.{timing_label}'
        try:
            if pln_ctx is None:
                pln_ctx = self.pln_ctx
            rrt_pln_ctx = utils.all_inclusive_mujoco_pln_ctx(pln_ctx) if use_rrt else pln_ctx
            expected_start = self._compose_state(*self._split_state(start_qs, ee_values=ee_values))
            expected_goal = self._compose_state(*self._split_state(goal_qs, ee_values=ee_values))
            connect_distance = float(pln_ctx.distance(expected_start, expected_goal))
            if timing_label is not None:
                self._record_metric(f'ad.connect_metric.{timing_label}.distance', connect_distance)
                direct_check_start = time.perf_counter()
                direct_path = utils.interpolate_qs(expected_start, expected_goal, step_size=step_size)
                direct_valid = utils.path_is_valid(direct_path, rrt_pln_ctx)
                self._record_timing(
                    f'ad.connect_motion.{timing_label}.direct_interp_check',
                    time.perf_counter() - direct_check_start,
                )
                self._record_metric(f'ad.connect_metric.{timing_label}.direct_interp_valid', 1.0 if direct_valid else 0.0)
                self._record_metric(f'ad.connect_metric.{timing_label}.direct_interp_points', len(direct_path))
            path = utils.plan_joint_path(
                start_qs=expected_start,
                goal_qs=expected_goal,
                pln_ctx=rrt_pln_ctx,
                use_rrt=use_rrt,
                step_size=step_size,
                max_iters=max_iters,
                time_limit=time_limit,
                timing_hook=self._record_timing,
                timing_label=rrt_label,
            )
            if path is None:
                return None
            path = [oum.np.asarray(qs, dtype=oum.np.float32) for qs in path]
            forward_error = (
                pln_ctx.distance(path[0], expected_start) +
                pln_ctx.distance(path[-1], expected_goal)
            )
            reverse_error = (
                pln_ctx.distance(path[0], expected_goal) +
                pln_ctx.distance(path[-1], expected_start)
            )
            if reverse_error < forward_error:
                path = list(reversed(path))
            return self._motion_plan(path)
        finally:
            self._record_timing(connect_label, time.perf_counter() - start_time)

    def _merge_plans(self, first, second):
        if first is None or second is None:
            return None
        merged = first.copy()
        qs_list = second.qs_list
        if merged.qs_list and qs_list:
            qs_list = qs_list[1:]
        merged.extend(qs_list)
        return merged

    def gen_linear_approach(self,
                            goal_tcp_pos,
                            goal_tcp_rotmat,
                            direction=None,
                            distance=0.07,
                            ee_values=None,
                            ref_qs=None,
                            pln_ctx=None,
                            granularity=0.02,
                            toggle_dbg=False):
        del toggle_dbg
        return self._linear_motion_to_pose(
            goal_tcp_pos=goal_tcp_pos,
            goal_tcp_rotmat=goal_tcp_rotmat,
            direction=direction,
            distance=distance,
            ee_values=ee_values,
            ref_qs=ref_qs,
            pln_ctx=pln_ctx,
            granularity=granularity,
            motion_type='sink',
        )

    def gen_linear_depart(self,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          direction=None,
                          distance=0.07,
                          ee_values=None,
                          ref_qs=None,
                          pln_ctx=None,
                          granularity=0.03,
                          toggle_dbg=False):
        del toggle_dbg
        return self._linear_motion_to_pose(
            goal_tcp_pos=start_tcp_pos,
            goal_tcp_rotmat=start_tcp_rotmat,
            direction=direction,
            distance=distance,
            ee_values=ee_values,
            ref_qs=ref_qs,
            pln_ctx=pln_ctx,
            granularity=granularity,
            motion_type='source',
        )

    def gen_approach(self,
                     goal_tcp_pos=None,
                     goal_tcp_rotmat=None,
                     goal_qs=None,
                     start_qs=None,
                     end_qs=None,
                     approach_direction=None,
                     approach_distance=0.1,
                     linear=True,
                     linear_granularity=0.03,
                     pln_ctx=None,
                     use_rrt=True,
                     pln_jnt=False,
                     toggle_dbg=False):
        del toggle_dbg
        del end_qs
        if pln_ctx is None:
            pln_ctx = self.pln_ctx
        if goal_qs is not None:
            goal_qs = self._compose_state(*self._split_state(goal_qs))
            goal_tcp_pos, goal_tcp_rotmat = self._tcp_pose_from_qs(goal_qs)
        if goal_tcp_pos is None or goal_tcp_rotmat is None:
            raise ValueError('goal_tcp_pos/goal_tcp_rotmat or goal_qs is required')
        if pln_jnt:
            start_state = self._compose_start_state(start_qs)
            if goal_qs is None:
                ref_qs = self.robot.qs.copy() if start_state is None else start_state[:self.robot.ndof]
                goal_qs = self.robot.ik_tcp_nearest(
                    tgt_rotmat=oum.np.asarray(goal_tcp_rotmat, dtype=oum.np.float32),
                    tgt_pos=oum.np.asarray(goal_tcp_pos, dtype=oum.np.float32),
                    ref_qs=ref_qs,
                )
                if goal_qs is None:
                    self._set_last_plan_failure('approach_goal_ik', 'goal_ik_failed')
                    return None
                goal_qs = self._compose_state(goal_qs)
            key_states = [start_state]
            if linear and approach_distance > 0.0:
                approach_tf = self._offset_pose_tf(
                    goal_tcp_pos=goal_tcp_pos,
                    goal_tcp_rotmat=goal_tcp_rotmat,
                    direction=approach_direction,
                    distance=approach_distance,
                    motion_type='sink',
                )
                ref_qs = goal_qs[:self.robot.ndof]
                approach_state = self._solve_pose_state(
                    pose_tf=approach_tf,
                    ref_qs=ref_qs,
                    ee_qs=goal_qs[self.robot.ndof:] if self.ee_actor is not None else None,
                    pln_ctx=pln_ctx,
                    failure_stage='approach_pose',
                )
                if approach_state is None:
                    return None
                key_states.append(approach_state)
            key_states.append(goal_qs)
            return self._keyframe_motion_plan(
                key_states,
                pln_ctx=pln_ctx,
                validate_edges=True,
                failure_stage='approach_keyframe_path',
            )
        if not linear or approach_distance <= 0.0:
            if start_qs is None:
                return self._motion_plan([goal_qs]) if goal_qs is not None else None
            if goal_qs is None:
                goal_qs = self.robot.ik_tcp_nearest(
                    tgt_rotmat=oum.np.asarray(goal_tcp_rotmat, dtype=oum.np.float32),
                    tgt_pos=oum.np.asarray(goal_tcp_pos, dtype=oum.np.float32),
                    ref_qs=self._split_state(start_qs)[0],
                )
                if goal_qs is None:
                    self._set_last_plan_failure('approach_goal_ik', 'goal_ik_failed')
                    return None
            return self._connect_motion(
                start_qs=start_qs,
                goal_qs=goal_qs,
                pln_ctx=pln_ctx,
                use_rrt=use_rrt,
            )
        linear_app = self.gen_linear_approach(
            goal_tcp_pos=goal_tcp_pos,
            goal_tcp_rotmat=goal_tcp_rotmat,
            direction=approach_direction,
            distance=approach_distance,
            ref_qs=goal_qs,
            pln_ctx=pln_ctx,
            granularity=linear_granularity,
        )
        if linear_app is None:
            return None
        full_plan = linear_app
        if start_qs is not None:
            start2app = self._connect_motion(
                start_qs=start_qs,
                goal_qs=linear_app.qs_list[0],
                pln_ctx=pln_ctx,
                use_rrt=use_rrt,
            )
            if start2app is None:
                return None
            full_plan = self._merge_plans(start2app, linear_app)
        if goal_qs is not None and not oum.np.allclose(full_plan.qs_list[-1], goal_qs):
            end_connect = self._connect_motion(
                start_qs=full_plan.qs_list[-1],
                goal_qs=goal_qs,
                pln_ctx=pln_ctx,
                use_rrt=use_rrt,
            )
            if end_connect is None:
                return None
            full_plan = self._merge_plans(full_plan, end_connect)
        return full_plan

    def gen_depart(self,
                   goal_tcp_pos=None,
                   goal_tcp_rotmat=None,
                   goal_qs=None,
                   start_qs=None,
                   end_qs=None,
                   approach_direction=None,
                   approach_distance=0.1,
                   depart_direction=None,
                   depart_distance=0.1,
                   linear=True,
                   linear_granularity=0.03,
                   pln_ctx=None,
                   use_rrt=True,
                   pln_jnt=False,
                   toggle_dbg=False):
        del approach_direction, approach_distance
        del toggle_dbg
        if pln_ctx is None:
            pln_ctx = self.pln_ctx
        if goal_qs is not None:
            goal_qs = self._compose_state(*self._split_state(goal_qs))
            goal_tcp_pos, goal_tcp_rotmat = self._tcp_pose_from_qs(goal_qs)
        if goal_tcp_pos is None or goal_tcp_rotmat is None:
            raise ValueError('goal_tcp_pos/goal_tcp_rotmat or goal_qs is required')
        if pln_jnt:
            start_state = self._compose_start_state(start_qs)
            end_state = self._compose_start_state(end_qs)
            if goal_qs is None:
                ref_qs = self.robot.qs.copy() if start_state is None else start_state[:self.robot.ndof]
                goal_qs = self.robot.ik_tcp_nearest(
                    tgt_rotmat=oum.np.asarray(goal_tcp_rotmat, dtype=oum.np.float32),
                    tgt_pos=oum.np.asarray(goal_tcp_pos, dtype=oum.np.float32),
                    ref_qs=ref_qs,
                )
                if goal_qs is None:
                    self._set_last_plan_failure('depart_goal_ik', 'goal_ik_failed')
                    return None
                goal_qs = self._compose_state(goal_qs)
            key_states = [start_state, goal_qs]
            if linear and depart_distance > 0.0:
                depart_tf = self._offset_pose_tf(
                    goal_tcp_pos=goal_tcp_pos,
                    goal_tcp_rotmat=goal_tcp_rotmat,
                    direction=depart_direction,
                    distance=depart_distance,
                    motion_type='source',
                )
                ref_qs = goal_qs[:self.robot.ndof]
                depart_state = self._solve_pose_state(
                    pose_tf=depart_tf,
                    ref_qs=ref_qs,
                    ee_qs=goal_qs[self.robot.ndof:] if self.ee_actor is not None else None,
                    pln_ctx=pln_ctx,
                    failure_stage='depart_pose',
                )
                if depart_state is None:
                    return None
                key_states.append(depart_state)
            key_states.append(end_state)
            return self._keyframe_motion_plan(
                key_states,
                pln_ctx=pln_ctx,
                validate_edges=True,
                failure_stage='depart_keyframe_path',
            )
        plan = None
        if start_qs is not None:
            goal_target = goal_qs
            if goal_target is None:
                goal_target = self.robot.ik_tcp_nearest(
                    tgt_rotmat=oum.np.asarray(goal_tcp_rotmat, dtype=oum.np.float32),
                    tgt_pos=oum.np.asarray(goal_tcp_pos, dtype=oum.np.float32),
                    ref_qs=self._split_state(start_qs)[0],
                )
                if goal_target is None:
                    self._set_last_plan_failure('depart_goal_ik', 'goal_ik_failed')
                    return None
            plan = self._connect_motion(
                start_qs=start_qs,
                goal_qs=goal_target,
                pln_ctx=pln_ctx,
                use_rrt=use_rrt,
            )
            if plan is None:
                return None
        elif goal_qs is not None:
            plan = self._motion_plan([goal_qs])
        if not linear or depart_distance <= 0.0:
            if plan is None:
                return None if end_qs is not None else self._motion_plan([goal_qs]) if goal_qs is not None else None
            if end_qs is None:
                return plan
            end_plan = self._connect_motion(
                start_qs=plan.qs_list[-1],
                goal_qs=end_qs,
                pln_ctx=pln_ctx,
                use_rrt=use_rrt,
            )
            if end_plan is None:
                return None
            return self._merge_plans(plan, end_plan)
        linear_dep = self.gen_linear_depart(
            start_tcp_pos=goal_tcp_pos,
            start_tcp_rotmat=goal_tcp_rotmat,
            direction=depart_direction,
            distance=depart_distance,
            ref_qs=goal_qs,
            pln_ctx=pln_ctx,
            granularity=linear_granularity,
        )
        if linear_dep is None:
            return None
        if plan is not None:
            linear_dep = self._merge_plans(plan, linear_dep)
        if end_qs is None:
            return linear_dep
        dep2end = self._connect_motion(
            start_qs=linear_dep.qs_list[-1],
            goal_qs=end_qs,
            pln_ctx=pln_ctx,
            use_rrt=use_rrt,
        )
        if dep2end is None:
            return None
        return self._merge_plans(linear_dep, dep2end)

    def gen_approach_depart(self,
                            goal_tcp_pos=None,
                            goal_tcp_rotmat=None,
                            goal_qs=None,
                            start_qs=None,
                            end_qs=None,
                            approach_direction=None,
                            approach_distance=0.1,
                            depart_direction=None,
                            depart_distance=0.1,
                            approach_linear=True,
                            depart_linear=True,
                            linear_granularity=0.03,
                            pln_ctx=None,
                            use_rrt=True,
                            pln_jnt=False,
                            toggle_dbg=False):
        del toggle_dbg
        app = self.gen_approach(
            goal_tcp_pos=goal_tcp_pos,
            goal_tcp_rotmat=goal_tcp_rotmat,
            goal_qs=goal_qs,
            start_qs=start_qs,
            approach_direction=approach_direction,
            approach_distance=approach_distance,
            linear=approach_linear,
            linear_granularity=linear_granularity,
            pln_ctx=pln_ctx,
            use_rrt=use_rrt,
            pln_jnt=pln_jnt,
        )
        if app is None:
            return None
        dep = self.gen_depart(
            goal_tcp_pos=goal_tcp_pos,
            goal_tcp_rotmat=goal_tcp_rotmat,
            goal_qs=goal_qs,
            end_qs=end_qs,
            depart_direction=depart_direction,
            depart_distance=depart_distance,
            linear=depart_linear,
            linear_granularity=linear_granularity,
            pln_ctx=pln_ctx,
            use_rrt=use_rrt,
            pln_jnt=pln_jnt,
        )
        if dep is None:
            return None
        if len(app.qs_list) <= 1:
            return dep
        app_prefix = utils.MotionData(app.qs_list[:-1])
        return self._merge_plans(app_prefix, dep)
