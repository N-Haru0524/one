from collections import Counter

from one_assembly.assembly_data import EEEvent, PlannerActionDraft, PlannerSegmentDraft
import one.utils.math as oum
import one.motion.trajectory.cartesian as omtc
import builtins

from .hierarchical import HierarchicalPlannerBase
from . import utils


def interpolate_fold(start_pose, goal_pose, n_steps=20):
    if n_steps <= 0:
        raise ValueError('n_steps must be positive')
    start_pos, start_rotmat = start_pose
    goal_pos, goal_rotmat = goal_pose
    start_pos = oum.np.asarray(start_pos, dtype=oum.np.float32)
    start_rotmat = oum.np.asarray(start_rotmat, dtype=oum.np.float32)
    goal_pos = oum.np.asarray(goal_pos, dtype=oum.np.float32)
    goal_rotmat = oum.np.asarray(goal_rotmat, dtype=oum.np.float32)
    fold_axis, fold_angle = oum.axangle_between_rotmat(start_rotmat, goal_rotmat)
    fold_axis = oum.np.asarray(fold_axis, dtype=oum.np.float32)
    chord = goal_pos - start_pos
    chord_len = float(oum.np.linalg.norm(chord))
    poses = []
    rotmat_list = oum.rotmat_slerp(start_rotmat, goal_rotmat, int(n_steps))

    linear_only = fold_angle <= 1e-6 or chord_len <= 1e-6
    center = None
    if not linear_only:
        perp = oum.np.cross(fold_axis, chord).astype(oum.np.float32)
        perp_len = float(oum.np.linalg.norm(perp))
        tan_half = float(oum.np.tan(fold_angle * 0.5))
        if perp_len <= 1e-6 or abs(tan_half) <= 1e-6:
            linear_only = True
        else:
            radius = chord_len / (2.0 * tan_half)
            center = ((start_pos + goal_pos) * 0.5 + perp / perp_len * radius).astype(oum.np.float32)

    for idx, fraction in enumerate(oum.np.linspace(0.0, 1.0, int(n_steps), dtype=oum.np.float32)):
        if linear_only:
            pos = ((1.0 - fraction) * start_pos + fraction * goal_pos).astype(oum.np.float32)
        else:
            rel_pos = start_pos - center
            pos = (oum.rotmat_from_axangle(fold_axis, float(fold_angle * fraction)) @ rel_pos + center)
            pos = oum.np.asarray(pos, dtype=oum.np.float32)
        rotmat = oum.np.asarray(rotmat_list[idx], dtype=oum.np.float32)
        poses.append((pos, rotmat))
    return poses


class FoldPlanner(HierarchicalPlannerBase):
    def _grasp_event_payload(self, obj_pose, grasp):
        pose_tf = oum.ensure_tf(obj_pose) @ oum.np.asarray(grasp[0], dtype=oum.np.float32)
        pick_tf = oum.ensure_tf(obj_pose)
        jaw_width = float(oum.np.asarray(grasp[2], dtype=oum.np.float32).reshape(-1)[0])
        ee_base_tf = pose_tf @ oum.np.linalg.inv(self.ee_actor.loc_tcp_tf)
        engage_tf = oum.np.linalg.inv(ee_base_tf) @ pick_tf
        return jaw_width, engage_tf.astype(oum.np.float32)

    def _debug_attach_ee_clone(self, tcp_tf, ee_qs=None, rgba=(1.0, 0.0, 0.0, 0.35)):
        if self.ee_actor is None:
            return
        base = getattr(builtins, 'base', None)
        scene = None if base is None else getattr(base, 'scene', None)
        if scene is None:
            return
        try:
            ee_clone = self.ee_actor.clone()
            if ee_qs is not None:
                ee_qs = oum.np.asarray(ee_qs, dtype=oum.np.float32).reshape(-1)
                if hasattr(ee_clone, 'fk'):
                    if ee_qs.size == len(ee_clone.qs):
                        ee_clone.fk(qs=ee_qs)
                    elif ee_qs.size == getattr(ee_clone, 'ndof', 0):
                        if ee_qs.size == 1 and len(ee_clone.qs) > 1:
                            ee_clone.fk(qs=[float(ee_qs[0])] * len(ee_clone.qs))
                        else:
                            ee_clone.fk(qs=ee_qs)
                elif hasattr(ee_clone, 'qs'):
                    ee_clone.qs[:ee_qs.size] = ee_qs
            base_tf = oum.np.asarray(tcp_tf, dtype=oum.np.float32)
            if hasattr(ee_clone, 'loc_tcp_tf'):
                base_tf = base_tf @ oum.tf_inverse(ee_clone.loc_tcp_tf)
            ee_clone.set_rotmat_pos(rotmat=base_tf[:3, :3], pos=base_tf[:3, 3])
            if hasattr(ee_clone, 'rgba'):
                ee_clone.rgba = rgba
            elif hasattr(ee_clone, 'rgb'):
                ee_clone.rgb = rgba[:3]
                if hasattr(ee_clone, 'alpha'):
                    ee_clone.alpha = rgba[3]
            ee_clone.attach_to(scene)
        except Exception:
            return

    def _resolve_motion_pln_ctx(self, pln_ctx=None, obstacle_list=None):
        if pln_ctx is not None:
            return pln_ctx
        if obstacle_list:
            collider = utils.build_collider(self._desired_actors(), obstacles=obstacle_list)
            return utils.build_planning_context(collider, max_edge_step=self.pln_ctx.cd_step_size)
        return self.pln_ctx

    def _motion_plan_from_joint_list(self, q_list, ee_values=None, pln_ctx=None):
        if q_list is None:
            return None
        resolved_ee_qs = self._resolve_ee_qs(ee_values) if self.ee_actor is not None else None
        q_list = [
            self._compose_state(oum.np.asarray(qs, dtype=oum.np.float32), resolved_ee_qs)
            for qs in q_list
        ]
        plan_pln_ctx = self.pln_ctx if pln_ctx is None else pln_ctx
        if not utils.path_is_valid(q_list, plan_pln_ctx):
            self._set_last_plan_failure('interpolated_path', 'path_in_collision')
            return None
        return self._motion_plan(q_list)

    def plan_fold(self, start_pose, goal_pose, n_steps=20):
        return interpolate_fold(start_pose, goal_pose, n_steps=n_steps)

    def reason_common_grasp_ids(self,
                                obj_model,
                                grasp_collection,
                                goal_pose_list,
                                pick_pose=None,
                                linear_granularity=0.03,
                                exclude_entities=None,
                                toggle_dbg=False):
        available_ids = list(range(len(grasp_collection)))
        pick_pln_ctx = self._precise_pln_ctx(
            exclude_entities=exclude_entities,
            backend=self.reasoning_backend,
        )
        pick_transit_pln_ctx = self._filtered_pln_ctx(exclude_entities=exclude_entities)
        hold_ctx_cache = {}
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

        def hold_pln_ctx_for_gid(gid, backend):
            return hold_ctx_cache.setdefault(
                (gid, backend),
                self._hold_pln_ctx(
                    obj_model=obj_model,
                    grasp=grasp_collection[gid],
                    pick_pose=pick_pose,
                    exclude_entities=exclude_entities,
                    backend=backend,
                ),
            )

        def classify_reason(stats):
            return self._classify_screen_stats(stats)

        collision_free_ids = []
        grasp_reason_counts = {
            'grasp_collision': 0,
        }
        for gid in available_ids:
            if self._grasp_has_collision(grasp_collection[gid]):
                self._last_reason_common_grasp_report['failures'][gid] = {
                    'label': 'grasp',
                    'reason': 'grasp_collision',
                }
                grasp_reason_counts['grasp_collision'] += 1
                continue
            collision_free_ids.append(gid)
        available_ids = collision_free_ids
        print_stage_stats('fold_reason grasp', len(grasp_collection), len(available_ids), grasp_reason_counts)
        if not available_ids:
            return []

        endpoint_checks = []
        open_ee_qs = self._max_open_ee_qs()
        if pick_pose is not None:
            pose_entries = []
            for gid in available_ids:
                tcp_tf, _ee_qs = self._grasp_pre_pose(grasp_collection[gid], obj_pose=pick_pose)
                pose_entries.append((gid, tcp_tf, open_ee_qs))
            endpoint_checks.append(('pick_approach_start', pose_entries))

            pose_entries = []
            for gid in available_ids:
                tcp_tf, ee_qs = self._grasp_pose(grasp_collection[gid], obj_pose=pick_pose)
                pose_entries.append((gid, tcp_tf, ee_qs))
            endpoint_checks.append(('pick_goal', pose_entries))

        for pose_idx, obj_pose in enumerate(goal_pose_list):
            pose_entries = []
            for gid in available_ids:
                tcp_tf, ee_qs = self._grasp_pose(grasp_collection[gid], obj_pose=obj_pose)
                pose_entries.append((gid, tcp_tf, ee_qs))
            endpoint_checks.append((f'fold_{pose_idx}', pose_entries))

        for label, pose_entries in endpoint_checks:
            next_available_ids = []
            reason_counts = {}
            pair_counts = Counter()
            debug_entries = []
            for gid, tcp_tf, ee_qs in pose_entries:
                stage_pln_ctx = (
                    pick_transit_pln_ctx if label == 'pick_approach_start'
                    else pick_pln_ctx if label == 'pick_goal'
                    else hold_pln_ctx_for_gid(gid, self.transit_backend)
                )
                debug_entries.append(
                    {
                        'key': gid,
                        'pose_tf': tcp_tf,
                        'ee_qs': ee_qs,
                        'pln_ctx': stage_pln_ctx,
                        'ref_qs': self.robot.qs.copy(),
                    }
                )
                result, stats = self._screen_pose_with_stats(
                    tcp_pos=tcp_tf[:3, 3],
                    tcp_rotmat=tcp_tf[:3, :3],
                    ee_qs=ee_qs,
                    pln_ctx=stage_pln_ctx,
                    linear_granularity=linear_granularity,
                    ref_qs=self.robot.qs.copy(),
                    diagnose_collision_pairs=toggle_dbg,
                    debug_visualize_contacts=toggle_dbg,
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
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    for pair, count in stats.get('collision_pairs', []):
                        pair_counts[pair] += count
                    continue
                next_available_ids.append(gid)
            print_stage_stats(
                f'fold_reason {label}',
                len(pose_entries),
                len(next_available_ids),
                reason_counts,
                pair_counts=pair_counts,
            )
            if toggle_dbg and pose_entries and not next_available_ids:
                self._debug_visualize_failed_pose_entries(
                    label=f'fold_reason {label}',
                    debug_entries=debug_entries,
                    linear_granularity=linear_granularity,
                )
            available_ids = next_available_ids
            if not available_ids:
                break

        self._last_reason_common_grasp_report['survived_gids'] = available_ids.copy()
        if toggle_dbg:
            print(f'[fold_reason final], survived={len(available_ids)}')
        return available_ids

    def reason_common_gids(self, *args, **kwargs):
        return self.reason_common_grasp_ids(*args, **kwargs)

    def gen_linear_motion(self,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          goal_tcp_pos,
                          goal_tcp_rotmat,
                          obstacle_list=None,
                          granularity=0.03,
                          ee_values=None,
                          ref_qs=None,
                          pln_ctx=None,
                          toggle_dbg=False):
        plan_pln_ctx = self._resolve_motion_pln_ctx(pln_ctx=pln_ctx, obstacle_list=obstacle_list)
        if ref_qs is None:
            ref_qs = self.robot.qs.copy()
        q_seq, _pose_seq = omtc.cartesian_to_jtraj(
            robot=self.robot,
            start_rotmat=start_tcp_rotmat,
            start_pos=start_tcp_pos,
            goal_rotmat=goal_tcp_rotmat,
            goal_pos=goal_tcp_pos,
            pos_step=granularity,
            ref_qs=ref_qs,
        )
        if q_seq is None:
            self._set_last_plan_failure('interpolated_ik', 'cartesian_ik_failed')
            if toggle_dbg:
                print('[fold_linear] interpolated_ik failed: cartesian_ik_failed')
            return None
        plan = self._motion_plan_from_joint_list(q_seq, ee_values=ee_values, pln_ctx=plan_pln_ctx)
        if toggle_dbg:
            if plan is None:
                failure = self._last_plan_failure
                if failure is None:
                    print('[fold_linear] path validation failed')
                else:
                    print(f'[fold_linear] {failure["stage"]} failed: {failure["reason"]}')
            else:
                print(f'[fold_linear] planned: waypoints={len(plan.qs_list)}')
        return plan

    def gen_piecewise_motion(self,
                             start_tcp_pos,
                             start_tcp_rotmat,
                             goal_tcp_pos_list,
                             goal_tcp_rotmat_list,
                             obstacle_list=None,
                             granularity=0.03,
                             ee_values=None,
                             ref_qs=None,
                             pln_ctx=None,
                             toggle_dbg=False):
        if len(goal_tcp_pos_list) != len(goal_tcp_rotmat_list):
            raise ValueError('goal_tcp_pos_list and goal_tcp_rotmat_list must have the same length')
        plan_pln_ctx = self._resolve_motion_pln_ctx(pln_ctx=pln_ctx, obstacle_list=obstacle_list)
        mot_data = utils.MotionData()
        if ref_qs is None:
            seed_qs = self.robot.qs.copy()
        else:
            seed_qs = oum.np.asarray(ref_qs, dtype=oum.np.float32).copy()
        cur_pos = oum.np.asarray(start_tcp_pos, dtype=oum.np.float32)
        cur_rotmat = oum.np.asarray(start_tcp_rotmat, dtype=oum.np.float32)
        for goal_pos, goal_rotmat in zip(goal_tcp_pos_list, goal_tcp_rotmat_list):
            q_seq, _pose_seq = omtc.cartesian_to_jtraj(
                robot=self.robot,
                start_rotmat=cur_rotmat,
                start_pos=cur_pos,
                goal_rotmat=goal_rotmat,
                goal_pos=goal_pos,
                pos_step=granularity,
                ref_qs=seed_qs,
            )
            if q_seq is None:
                self._set_last_plan_failure('interpolated_ik', 'cartesian_ik_failed')
                if toggle_dbg:
                    print(f'[fold_piecewise seg={len(mot_data)}] interpolated_ik failed: cartesian_ik_failed')
                return None
            if len(q_seq) > 0:
                q_seq = oum.np.asarray(q_seq, dtype=oum.np.float32).copy()
                q_seq[0] = seed_qs
            segment_plan = self._motion_plan_from_joint_list(q_seq, ee_values=ee_values, pln_ctx=plan_pln_ctx)
            if segment_plan is None:
                if toggle_dbg:
                    failure = self._last_plan_failure
                    if failure is None:
                        print(f'[fold_piecewise seg={len(mot_data)}] segment planning failed')
                    else:
                        print(
                            f'[fold_piecewise seg={len(mot_data)}] '
                            f'{failure["stage"]} failed: {failure["reason"]}'
                        )
                return None
            mot_data = segment_plan if len(mot_data) == 0 else self._merge_plans(mot_data, segment_plan)
            seed_qs = q_seq[-1]
            cur_pos = oum.np.asarray(goal_pos, dtype=oum.np.float32)
            cur_rotmat = oum.np.asarray(goal_rotmat, dtype=oum.np.float32)
        if toggle_dbg:
            print(f'[fold_piecewise] planned: segments={len(goal_tcp_pos_list)}, waypoints={len(mot_data.qs_list)}')
        return mot_data

    def _gen_fold_motion(self,
                         grasp,
                         goal_pose_list,
                         start_state,
                         ee_qs,
                         pln_ctx,
                         gid=None,
                         linear_granularity=0.03,
                         pln_jnt=False,
                         toggle_dbg=False):
        current_tf = self._pose_to_tf(self._tcp_pose_from_qs(start_state))
        goal_tf_list = []
        for goal_pose in goal_pose_list:
            goal_tf, _goal_ee_qs = self._grasp_pose(grasp, obj_pose=goal_pose)
            goal_tf_list.append(goal_tf)
        if pln_jnt:
            key_states = [start_state]
            ref_qs = start_state[:self.robot.ndof]
            for goal_tf in goal_tf_list:
                goal_state = self._solve_pose_state(
                    pose_tf=goal_tf,
                    ref_qs=ref_qs,
                    ee_qs=ee_qs,
                    pln_ctx=pln_ctx,
                    failure_stage='fold_pose',
                )
                if goal_state is None:
                    return None
                key_states.append(goal_state)
                ref_qs = goal_state[:self.robot.ndof]
            fold_plan = self._keyframe_motion_plan(
                key_states,
                pln_ctx=pln_ctx,
                validate_edges=True,
                failure_stage='fold_keyframe_path',
            )
        else:
            fold_plan = self.gen_piecewise_motion(
                start_tcp_pos=current_tf[:3, 3],
                start_tcp_rotmat=current_tf[:3, :3],
                goal_tcp_pos_list=[goal_tf[:3, 3] for goal_tf in goal_tf_list],
                goal_tcp_rotmat_list=[goal_tf[:3, :3] for goal_tf in goal_tf_list],
                granularity=linear_granularity,
                ee_values=ee_qs,
                ref_qs=start_state[:self.robot.ndof],
                pln_ctx=pln_ctx,
                toggle_dbg=toggle_dbg,
            )
        if fold_plan is None and toggle_dbg:
            failure = self._last_plan_failure
            if failure is None:
                print(f'[fold gid={gid}] motion planning failed')
            else:
                print(f'[fold gid={gid}] {failure["stage"]} failed: {failure["reason"]}')
        if fold_plan is None:
            return None
        if toggle_dbg:
            print(f'[fold gid={gid}] planned: waypoints={len(fold_plan.qs_list)}')
        if fold_plan.qs_list and oum.np.allclose(fold_plan.qs_list[0], start_state):
            return fold_plan
        return self._merge_plans(utils.MotionData([start_state]), fold_plan)

    def gen_pick_and_fold_motion(self,
                                 obj_model,
                                 grasp,
                                 goal_pose_list,
                                 gid=None,
                                 start_qs=None,
                                 linear_granularity=0.03,
                                 use_rrt=True,
                                 pln_jnt=False,
                                 exclude_entities=None,
                                 toggle_dbg=False):
        if start_qs is None:
            start_qs = self.robot.qs.copy()
        start_robot_qs, _start_ee_qs = self._split_state(start_qs)
        open_ee_qs = self._max_open_ee_qs()
        pick_pln_ctx = self._filtered_pln_ctx(exclude_entities=exclude_entities)
        obj_pose_tf = oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos)
        if self._grasp_has_collision(grasp):
            if toggle_dbg:
                print(f'[pickfold_pick gid={gid}] skipped grasp_collision=1')
            return None
        pick_tf, pick_pre_tf, ee_qs, _jaw_width, _score = self._grasp_world_data(grasp, obj_pose=obj_pose_tf)
        pick_goal, pick_stats = self._screen_pose_with_stats(
            tcp_pos=pick_tf[:3, 3],
            tcp_rotmat=pick_tf[:3, :3],
            ee_qs=ee_qs,
            pln_ctx=pick_pln_ctx,
            linear_granularity=linear_granularity,
            ref_qs=start_robot_qs,
            diagnose_collision_pairs=toggle_dbg,
            debug_visualize_contacts=toggle_dbg,
        )
        if toggle_dbg:
            if pick_stats['survived']:
                print(f'[pickfold_pick gid={gid}] survived=1')
            elif pick_stats['rejected_goal_collision']:
                print(f'[pickfold_pick gid={gid}] removed_goal_state_in_collision=1')
            elif pick_stats['rejected_no_ik']:
                print(f'[pickfold_pick gid={gid}] removed_no_ik=1')
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
            pln_jnt=pln_jnt,
        )
        if pick_plan is None:
            if toggle_dbg:
                failure = self._last_plan_failure
                if failure is None:
                    print(f'[pickfold_pick gid={gid}] motion planning failed')
                else:
                    print(
                        f'[pickfold_pick gid={gid}] '
                        f'{failure["stage"]} failed: {failure["reason"]}'
                    )
            return None

        hold_pln_ctx = self._hold_pln_ctx(
            obj_model=obj_model,
            grasp=grasp,
            pick_pose=obj_pose_tf,
            exclude_entities=exclude_entities,
            backend=self.transit_backend,
        )
        fold_plan = self._gen_fold_motion(
            grasp=grasp,
            goal_pose_list=goal_pose_list,
            start_state=pick_plan.qs_list[-1],
            ee_qs=ee_qs,
            pln_ctx=hold_pln_ctx,
            gid=gid,
            linear_granularity=linear_granularity,
            pln_jnt=pln_jnt,
            toggle_dbg=toggle_dbg,
        )
        if fold_plan is None:
            return None

        full_plan = self._merge_plans(pick_plan, fold_plan)
        full_plan.events['pick_pre'] = int(pick_plan.events.get('via', 0))
        full_plan.events['attach'] = len(pick_plan.qs_list) - 1
        full_plan.events['fold_start'] = len(pick_plan.qs_list) - 1
        full_plan.events['gid'] = gid
        if toggle_dbg:
            print(f'[pickfold gid={gid}] planned: waypoints={len(full_plan.qs_list)}')
        return full_plan

    def gen_pick_and_fold(self,
                          obj_model,
                          grasp_collection,
                          goal_pose_list,
                          start_qs=None,
                          linear_granularity=0.03,
                          reason_grasps=True,
                          use_rrt=True,
                          pln_jnt=False,
                          exclude_entities=None,
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
                exclude_entities=exclude_entities,
                toggle_dbg=toggle_dbg,
            )
        else:
            common_gids = [
                gid for gid, grasp in enumerate(grasp_collection)
                if not self._grasp_has_collision(grasp)
            ]
        if not common_gids:
            if toggle_dbg:
                print('[pickfold] no common grasp ids after reasoning')
            return None

        pick_pose_tf_list = []
        for gid in common_gids:
            tcp_tf, _ee_qs = self._grasp_pose(grasp_collection[gid], obj_pose=obj_pose_tf)
            pick_pose_tf_list.append((gid, tcp_tf))

        sorted_candidates = self._sort_pose_candidates(pick_pose_tf_list)
        for idx, (gid, _) in enumerate(sorted_candidates):
            grasp = grasp_collection[gid]
            full_plan = self.gen_pick_and_fold_motion(
                obj_model=obj_model,
                grasp=grasp,
                goal_pose_list=goal_pose_list,
                gid=gid,
                start_qs=start_qs,
                linear_granularity=linear_granularity,
                use_rrt=use_rrt,
                pln_jnt=pln_jnt,
                exclude_entities=exclude_entities,
                toggle_dbg=toggle_dbg,
            )
            if full_plan is None:
                if toggle_dbg and idx + 1 < len(sorted_candidates):
                    next_gid = sorted_candidates[idx + 1][0]
                    failure = self._last_plan_failure
                    if failure is None:
                        print(f'[pickfold] gid={gid} failed, trying next gid={next_gid}')
                    else:
                        print(
                            f'[pickfold] gid={gid} failed at {failure["stage"]}: '
                            f'{failure["reason"]}, trying next gid={next_gid}'
                        )
                continue
            if toggle_dbg:
                print(f'[pickfold] selected gid={gid}, waypoints={len(full_plan.qs_list)}')
            return full_plan

        if toggle_dbg:
            print('[pickfold] no feasible pick-and-fold plan found')
        return None

    def gen_hold_and_fold(self,
                          obj_model,
                          grasp,
                          goal_pose_list,
                          start_qs=None,
                          linear_granularity=0.03,
                          pln_jnt=False,
                          exclude_entities=None,
                          gid=None,
                          toggle_dbg=False):
        if start_qs is None:
            start_qs = self.robot.qs.copy()
        obj_pose_tf = oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos)
        _pick_tf, _pick_pre_tf, ee_qs, _jaw_width, _score = self._grasp_world_data(
            grasp,
            obj_pose=obj_pose_tf,
        )
        start_state = self._compose_state(*self._split_state(start_qs, ee_values=ee_qs))
        hold_pln_ctx = self._hold_pln_ctx(
            obj_model=obj_model,
            grasp=grasp,
            pick_pose=obj_pose_tf,
            exclude_entities=exclude_entities,
            backend=self.transit_backend,
        )
        if not hold_pln_ctx.is_state_valid(start_state):
            if toggle_dbg:
                print(f'[holdfold gid={gid}] start state invalid while holding')
            return None
        fold_plan = self._gen_fold_motion(
            grasp=grasp,
            goal_pose_list=goal_pose_list,
            start_state=start_state,
            ee_qs=ee_qs,
            pln_ctx=hold_pln_ctx,
            gid=gid,
            linear_granularity=linear_granularity,
            pln_jnt=pln_jnt,
            toggle_dbg=toggle_dbg,
        )
        if fold_plan is None:
            return None
        fold_plan.events['fold_start'] = 0
        fold_plan.events['gid'] = gid
        if toggle_dbg:
            print(f'[holdfold gid={gid}] planned: waypoints={len(fold_plan.qs_list)}')
        return fold_plan

    def gen_fold_draft(self,
                       obj_model,
                       work_name,
                       grasp_collection,
                       goal_pose_list,
                       held_grasp=None,
                       segment_label=None,
                       end_sync_label=None,
                       start_qs=None,
                       linear_granularity=0.03,
                       reason_grasps=True,
                       use_rrt=True,
                       pln_jnt=False,
                       exclude_entities=None,
                       toggle_dbg=False):
        selected_gid = None
        plan = None
        held_mode = held_grasp is not None and held_grasp.work_name == work_name
        if held_mode:
            selected_gid = int(held_grasp.gid)
            if selected_gid < 0 or selected_gid >= len(grasp_collection):
                return None
            plan = self.gen_hold_and_fold(
                obj_model=obj_model,
                grasp=grasp_collection[selected_gid],
                goal_pose_list=goal_pose_list,
                start_qs=start_qs,
                linear_granularity=linear_granularity,
                pln_jnt=pln_jnt,
                exclude_entities=exclude_entities,
                gid=selected_gid,
                toggle_dbg=toggle_dbg,
            )
        else:
            plan = self.gen_pick_and_fold(
                obj_model=obj_model,
                grasp_collection=grasp_collection,
                goal_pose_list=goal_pose_list,
                start_qs=start_qs,
                linear_granularity=linear_granularity,
                reason_grasps=reason_grasps,
                use_rrt=use_rrt,
                pln_jnt=pln_jnt,
                exclude_entities=exclude_entities,
                toggle_dbg=toggle_dbg,
            )
            if plan is not None and 'gid' in plan.events:
                selected_gid = int(plan.events['gid'])
        if plan is None:
            return None

        if selected_gid is None and 'gid' in plan.events:
            selected_gid = int(plan.events['gid'])
        grasp = None
        if selected_gid is not None and 0 <= selected_gid < len(grasp_collection):
            grasp = grasp_collection[selected_gid]

        left_path = [
            oum.np.asarray(qs[:self.robot.ndof], dtype=oum.np.float32).copy()
            for qs in plan.qs_list
        ]
        jaw_width = None
        engage_tf = None
        if grasp is not None:
            pick_pose_tf = oum.tf_from_rotmat_pos(obj_model.rotmat, obj_model.pos)
            jaw_width, engage_tf = self._grasp_event_payload(
                obj_pose=pick_pose_tf,
                grasp=grasp,
            )

        segments = []
        if held_mode:
            segments.append(PlannerSegmentDraft(
                segment_label=segment_label or f'fold {work_name}',
                left_path=[qs.copy() for qs in left_path],
                right_path=[],
                ee_events=[
                    EEEvent(
                        actor='left_gripper',
                        action='release',
                        timing='end',
                        work_name=work_name,
                        label=f'release {work_name}',
                    ),
                ],
                end_sync_label=end_sync_label or f'{work_name} fold',
                held_after=None,
            ))
            return PlannerActionDraft(segments=segments, held_after=None)

        pick_pre = int(plan.events.get('pick_pre', 0))
        attach_idx = int(plan.events['attach'])
        fold_start = int(plan.events.get('fold_start', attach_idx))

        segments.append(PlannerSegmentDraft(
            segment_label=f'approach {work_name} pregrasp',
            left_path=[qs.copy() for qs in left_path[:pick_pre + 1]],
            right_path=[],
            ee_events=[],
            end_sync_label=f'{work_name} pregrasp',
            held_after=None,
        ))
        segments.append(PlannerSegmentDraft(
            segment_label=f'grasp {work_name}',
            left_path=[qs.copy() for qs in left_path[pick_pre:attach_idx + 1]],
            right_path=[],
            ee_events=[
                EEEvent(
                    actor='left_gripper',
                    action='attach',
                    timing='end',
                    value=jaw_width,
                    work_name=work_name,
                    grasp_id=selected_gid,
                    engage_tf=engage_tf,
                    label=f'attach {work_name}',
                ),
            ],
            end_sync_label=f'{work_name} grasp',
            held_after=None,
        ))
        if fold_start > attach_idx:
            segments.append(PlannerSegmentDraft(
                segment_label=f'lift {work_name}',
                left_path=[qs.copy() for qs in left_path[attach_idx:fold_start + 1]],
                right_path=[],
                ee_events=[],
                end_sync_label=f'{work_name} postgrasp',
                held_after=None,
            ))
        segments.append(PlannerSegmentDraft(
            segment_label=segment_label or f'fold {work_name}',
            left_path=[qs.copy() for qs in left_path[fold_start:]],
            right_path=[],
            ee_events=[
                EEEvent(
                    actor='left_gripper',
                    action='release',
                    timing='end',
                    work_name=work_name,
                    label=f'release {work_name}',
                ),
            ],
            end_sync_label=end_sync_label or f'{work_name} fold',
            held_after=None,
        ))
        segments = [segment for segment in segments if segment.left_path]
        return PlannerActionDraft(segments=segments, held_after=None)
