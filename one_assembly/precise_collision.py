from types import SimpleNamespace

import numpy as np

import one.scene.scene as oss
import one.collider.cpu_simd as cpu_simd
import one.collider.gpu_simd_batch as gpu_simd_batch


def precise_mesh_is_collided(sobj_a, sobj_b, eps=1e-9, max_points=1000, use_gpu=True):
    if (len(getattr(sobj_a, 'collisions', ())) != 1 or
            len(getattr(sobj_b, 'collisions', ())) != 1):
        return None
    item_a = SimpleNamespace(collisions=list(sobj_a.collisions), tf=sobj_a.tf)
    item_b = SimpleNamespace(collisions=list(sobj_b.collisions), tf=sobj_b.tf)
    if use_gpu:
        try:
            detector = gpu_simd_batch.create_detector()
            batch = gpu_simd_batch.build_batch([item_a, item_b], pairs=[(0, 1)])
            result = detector.detect_collision_batch(batch, eps=eps)
            if result is None:
                return None
            points, _pair_ids = result
            return points
        except Exception:
            pass
    detector = cpu_simd.create_detector(eps=eps, max_points=max_points)
    batch = cpu_simd.build_batch([item_a, item_b], pairs=[(0, 1)])
    result = detector.detect_collision_batch(batch)
    if result is None:
        return None
    points, _pair_ids = result
    return points


class PreciseSIMDCollider:
    def __init__(self, use_gpu=True, eps=1e-9, max_points=1):
        self.scene = oss.Scene()
        self._actors = ()
        self._actor_qs_slice = {}
        self._use_gpu = use_gpu
        self._eps = eps
        self._max_points = max_points
        self._compiled = False
        self._actor_link_masks = {}
        self._actor_include_mounted = {}
        self._self_collision_pairs = []
        self._actor_sobj_pairs = []
        self._actor_mecba_pairs = []
        self._actor_actor_pairs = []
        self._mounted_sobj_pairs = []
        self._cpu_detector = cpu_simd.create_detector(eps=eps, max_points=max_points)
        self._gpu_detector = None if not use_gpu else gpu_simd_batch.create_detector()

    def append(self, entity):
        self.scene.add(entity)

    @property
    def actors(self):
        return self._actors

    @actors.setter
    def actors(self, actors):
        if not actors:
            raise ValueError('actors cannot be empty')
        self._actors = tuple(actors)
        self._rebuild_mapping()

    def get_slice(self, actor):
        return self._actor_qs_slice.get(actor, None)

    def compile(self):
        if not self.actors:
            raise RuntimeError('PreciseSIMDCollider.actors must be set before compile')
        self._self_collision_pairs = []
        for actor in self._actors:
            lnks = actor.runtime_lnks
            ignore_pairs = actor.structure.compiled.collision_ignores_idx
            valid_indices = [(i, lnk.collisions[0]) for i, lnk in self._iter_valid_links(actor)]
            for idx_i, (i, col_i) in enumerate(valid_indices):
                for idx_j in range(idx_i + 1, len(valid_indices)):
                    j, col_j = valid_indices[idx_j]
                    if (min(i, j), max(i, j)) in ignore_pairs:
                        continue
                    if not self._should_collide(lnks[i], lnks[j]):
                        continue
                    self._self_collision_pairs.append((actor, i, j, col_i, col_j))
        self._actor_sobj_pairs = []
        mounted_sobjs_by_actor = {}
        for actor in self._actors:
            mounted_sobjs = []
            if self._actor_include_mounted.get(actor, True):
                self._collect_mounted_scene_objects(actor, mounted_sobjs)
            mounted_sobjs_by_actor[actor] = {id(sobj) for sobj in mounted_sobjs}
        for actor in self._actors:
            valid_links = [(i, lnk, lnk.collisions[0]) for i, lnk in self._iter_valid_links(actor)]
            for sobj in self.scene.sobjs:
                if id(sobj) in mounted_sobjs_by_actor.get(actor, set()):
                    continue
                if not sobj.collisions:
                    continue
                col_sobj = sobj.collisions[0]
                for lidx, lnk, col_lnk in valid_links:
                    if not self._should_collide(lnk, sobj):
                        continue
                    self._actor_sobj_pairs.append((actor, lidx, sobj, col_lnk, col_sobj))
        self._mounted_sobj_pairs = []
        for actor in self._actors:
            if not self._actor_include_mounted.get(actor, True):
                continue
            mounted_sobjs = []
            self._collect_mounted_scene_objects(actor, mounted_sobjs)
            mounted_ids = {id(sobj) for sobj in mounted_sobjs}
            for mounted_sobj in mounted_sobjs:
                if not mounted_sobj.collisions:
                    continue
                col_mounted = mounted_sobj.collisions[0]
                for sobj in self.scene.sobjs:
                    if id(sobj) in mounted_ids or not sobj.collisions:
                        continue
                    col_sobj = sobj.collisions[0]
                    if not self._should_collide(mounted_sobj, sobj):
                        continue
                    self._mounted_sobj_pairs.append((mounted_sobj, sobj, col_mounted, col_sobj))
        self._actor_mecba_pairs = []
        non_actor_robots = [mb for mb in self.scene.mecbas if mb not in self._actors]
        for actor in self._actors:
            actor_valid = [(i, lnk, lnk.collisions[0]) for i, lnk in self._iter_valid_links(actor)]
            for robot_obs in non_actor_robots:
                obs_valid = [(i, lnk, lnk.collisions[0]) for i, lnk in enumerate(robot_obs.runtime_lnks) if lnk.collisions]
                for actor_lidx, actor_lnk, col_a in actor_valid:
                    for obs_lidx, obs_lnk, col_o in obs_valid:
                        if not self._should_collide(actor_lnk, obs_lnk):
                            continue
                        self._actor_mecba_pairs.append(
                            (actor, actor_lidx, robot_obs, obs_lidx, col_a, col_o)
                        )
        self._actor_actor_pairs = []
        n_actors = len(self._actors)
        for i in range(n_actors):
            actor_a = self._actors[i]
            valid_a = [(idx, lnk, lnk.collisions[0]) for idx, lnk in self._iter_valid_links(actor_a)]
            for j in range(i + 1, n_actors):
                actor_b = self._actors[j]
                valid_b = [(idx, lnk, lnk.collisions[0]) for idx, lnk in self._iter_valid_links(actor_b)]
                for lidx_a, lnk_a, col_a in valid_a:
                    for lidx_b, lnk_b, col_b in valid_b:
                        if not self._should_collide(lnk_a, lnk_b):
                            continue
                        self._actor_actor_pairs.append((actor_a, lidx_a, actor_b, lidx_b, col_a, col_b))
        self._compiled = True

    def is_collided(self, qs):
        if not self._compiled:
            raise RuntimeError('PreciseSIMDCollider must be compiled')
        for actor, sl in self._actor_qs_slice.items():
            actor.fk(qs[sl])
        for actor, lidx_i, lidx_j, col_i, col_j in self._self_collision_pairs:
            lnk_i = actor.runtime_lnks[lidx_i]
            lnk_j = actor.runtime_lnks[lidx_j]
            if self._check_pair_direct(col_i, lnk_i.tf, col_j, lnk_j.tf) is not None:
                return True
        for actor, lidx, sobj, col_lnk, col_sobj in self._actor_sobj_pairs:
            lnk = actor.runtime_lnks[lidx]
            if self._check_pair_direct(col_lnk, lnk.tf, col_sobj, sobj.tf) is not None:
                return True
        for actor, actor_lidx, robot_obs, obs_lidx, col_a, col_o in self._actor_mecba_pairs:
            actor_lnk = actor.runtime_lnks[actor_lidx]
            obs_lnk = robot_obs.runtime_lnks[obs_lidx]
            if self._check_pair_direct(col_a, actor_lnk.tf, col_o, obs_lnk.tf) is not None:
                return True
        for mounted_sobj, sobj, col_mounted, col_sobj in self._mounted_sobj_pairs:
            if self._check_pair_direct(col_mounted, mounted_sobj.tf, col_sobj, sobj.tf) is not None:
                return True
        for actor_a, lidx_a, actor_b, lidx_b, col_a, col_b in self._actor_actor_pairs:
            lnk_a = actor_a.runtime_lnks[lidx_a]
            lnk_b = actor_b.runtime_lnks[lidx_b]
            if self._check_pair_direct(col_a, lnk_a.tf, col_b, lnk_b.tf) is not None:
                return True
        return False

    def _iter_valid_links(self, actor):
        allowed = self._actor_link_masks.get(actor, None)
        for idx, lnk in enumerate(actor.runtime_lnks):
            if allowed is not None and idx not in allowed:
                continue
            if not lnk.collisions:
                continue
            yield idx, lnk

    def _collect_mounted_scene_objects(self, entity, out):
        for child in getattr(entity, '_mountings', {}).keys():
            if hasattr(child, 'runtime_lnks'):
                self._collect_mounted_scene_objects(child, out)
            elif hasattr(child, 'collisions'):
                out.append(child)

    def _check_pair_direct(self, col_a, tf_a, col_b, tf_b):
        min_a, max_a = col_a.aabb
        min_b, max_b = col_b.aabb
        if min_a is not None and min_b is not None:
            min_a, max_a = self._transform_aabb(min_a, max_a, tf_a)
            min_b, max_b = self._transform_aabb(min_b, max_b, tf_b)
            if not cpu_simd.aabb_intersect(min_a, max_a, min_b, max_b):
                return None
        if self._use_gpu:
            try:
                item_a = SimpleNamespace(collisions=[col_a], tf=tf_a)
                item_b = SimpleNamespace(collisions=[col_b], tf=tf_b)
                batch = gpu_simd_batch.build_batch([item_a, item_b], pairs=[(0, 1)])
                result = self._gpu_detector.detect_collision_batch(batch, eps=self._eps)
                if result is None:
                    return None
                points, _pair_ids = result
                return points
            except Exception:
                pass
        return self._cpu_detector.detect_collision(
            col_a.geom.vs, col_a.geom.fs, tf_a,
            col_b.geom.vs, col_b.geom.fs, tf_b,
        )

    def _rebuild_mapping(self):
        self._actor_qs_slice.clear()
        offset = 0
        for actor in self._actors:
            if actor not in self.scene.mecbas:
                raise RuntimeError('All actors must be added to the scene')
            ndof = actor.ndof
            self._actor_qs_slice[actor] = slice(offset, offset + ndof)
            offset += ndof

    def _should_collide(self, obj_a, obj_b):
        ga = obj_a.collision_group
        gb = obj_b.collision_group
        aa = obj_a.collision_affinity
        ab = obj_b.collision_affinity
        return bool((aa & gb) and (ab & ga))

    def _transform_aabb(self, min_local, max_local, tf):
        corners = np.array([
            [min_local[0], min_local[1], min_local[2]],
            [min_local[0], min_local[1], max_local[2]],
            [min_local[0], max_local[1], min_local[2]],
            [min_local[0], max_local[1], max_local[2]],
            [max_local[0], min_local[1], min_local[2]],
            [max_local[0], min_local[1], max_local[2]],
            [max_local[0], max_local[1], min_local[2]],
            [max_local[0], max_local[1], max_local[2]],
        ], dtype=np.float32)
        rotmat = tf[:3, :3]
        pos = tf[:3, 3]
        transformed = (rotmat @ corners.T).T + pos
        return transformed.min(axis=0), transformed.max(axis=0)
