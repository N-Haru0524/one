import one.utils.math as oum

from .hierarchical import HierarchicalPlannerBase
from . import utils


class ScrewPlanner(HierarchicalPlannerBase):
    def __init__(self, robot, pln_ctx, ee_actor=None):
        super().__init__(robot=robot, pln_ctx=pln_ctx, ee_actor=ee_actor)

    def _unit_vec(self, vec):
        vec = oum.np.asarray(vec, dtype=oum.np.float32).reshape(-1)
        if vec.size != 3:
            raise ValueError('expected a 3D vector')
        norm = float(oum.np.linalg.norm(vec))
        if norm <= 0.0:
            raise ValueError('vector must be non-zero')
        return (vec / norm).astype(oum.np.float32)

    def _home_state(self, home_pose=None, home_qs=None, ref_qs=None):
        if home_qs is not None:
            return self._compose_state(*self._split_state(home_qs))
        if home_pose is None:
            return None
        home_tf = self._pose_to_tf(home_pose)
        if ref_qs is None:
            ref_qs = self.robot.qs.copy()
        goal_qs = self.robot.ik_tcp_nearest(
            tgt_rotmat=home_tf[:3, :3],
            tgt_pos=home_tf[:3, 3],
            ref_qs=oum.np.asarray(ref_qs, dtype=oum.np.float32),
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
                           place_depart_distance=0.1,
                           place_depart_direction=None,
                           linear_granularity=0.03,
                           ref_qs=None,
                           exclude_entities=None,
                           toggle_dbg=False):
        return list(self._reason_screw_candidates(
            goal_pose_list=goal_pose_list,
            place_depart_distance=place_depart_distance,
            place_depart_direction=place_depart_direction,
            linear_granularity=linear_granularity,
            ref_qs=ref_qs,
            exclude_entities=exclude_entities,
            toggle_dbg=toggle_dbg,
        ).keys())

    def _reason_screw_candidates(self,
                                 goal_pose_list,
                                 place_depart_distance=0.1,
                                 place_depart_direction=None,
                                 linear_granularity=0.03,
                                 ref_qs=None,
                                 exclude_entities=None,
                                 toggle_dbg=False):
        pose_tf_list = [(sid, self._pose_to_tf(goal_pose), None) for sid, goal_pose in enumerate(goal_pose_list)]
        ref_qs = self.robot.qs.copy() if ref_qs is None else oum.np.asarray(ref_qs, dtype=oum.np.float32)
        records = self._screen_pose_list(
            keyed_pose_list=pose_tf_list,
            exclude_entities=exclude_entities,
            depart_direction=place_depart_direction,
            depart_distance=place_depart_distance,
            linear_granularity=linear_granularity,
            ref_qs=ref_qs,
            toggle_dbg=toggle_dbg,
            debug_label='screw_reason',
        )
        return {record.key: record for record in records}

    def gen_screw(self,
                  goal_pose_list=None,
                  start_qs=None,
                  tgt_pos=None,
                  tgt_vec=None,
                  resolution=20,
                  angle_offset=0.0,
                  home_pose=None,
                  home_qs=None,
                  home_approach_direction=None,
                  home_approach_distance=0.0,
                  home_depart_direction=None,
                  home_depart_distance=0.0,
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
        goal_pose_list = self._resolve_goal_pose_list(
            goal_pose_list=goal_pose_list,
            tgt_pos=tgt_pos,
            tgt_vec=tgt_vec,
            resolution=resolution,
            angle_offset=angle_offset,
        )
        prefix_plan = None
        current_state = start_state
        home_state = self._home_state(
            home_pose=home_pose,
            home_qs=home_qs,
            ref_qs=start_state[:self.robot.ndof],
        )
        if home_state is None and (home_pose is not None or home_qs is not None):
            if toggle_dbg:
                print('[screw] failed to resolve home pose')
            return None
        if home_state is not None:
            home_plan = self.gen_approach_depart(
                goal_qs=home_state,
                start_qs=current_state,
                end_qs=None,
                approach_direction=home_approach_direction,
                approach_distance=home_approach_distance,
                depart_direction=home_depart_direction,
                depart_distance=home_depart_distance,
                approach_linear=home_approach_distance > 0.0,
                depart_linear=home_depart_distance > 0.0,
                linear_granularity=linear_granularity,
                pln_ctx=self.pln_ctx,
                use_rrt=use_rrt,
                pln_jnt=pln_jnt,
            )
            if home_plan is None:
                if toggle_dbg:
                    print('[screw] failed to generate home motion')
                return None
            prefix_plan = home_plan
            current_state = home_plan.qs_list[-1]
        candidate_map = self._reason_screw_candidates(
            goal_pose_list=goal_pose_list,
            place_depart_distance=depart_distance,
            place_depart_direction=depart_direction,
            linear_granularity=linear_granularity,
            ref_qs=current_state[:self.robot.ndof],
            toggle_dbg=toggle_dbg,
        )
        for sid, record in candidate_map.items():
            result = record.screen_result
            plan = self.gen_approach(
                goal_qs=result.goal_qs,
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
                if prefix_plan is not None:
                    plan = self._merge_plans(prefix_plan, plan)
                plan.events['sid'] = sid
                if toggle_dbg:
                    print(f'[screw] selected sid={sid}, waypoints={len(plan.qs_list)}')
                return plan
            if toggle_dbg:
                print(f'[screw] sid={sid} motion planning failed after reasoning')
        if toggle_dbg:
            print('[screw] no feasible screw plan found')
        return None
