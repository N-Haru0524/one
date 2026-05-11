import numpy as np


def retime_global(q_seq, v_max, a_max, dt=0.01):
    """
    Time-parameterize a joint waypoint path via a global per-joint forward-backward pass.

    Unlike retime_trapezoidal (in one.motion.trajectory.time_param), velocity is carried
    across waypoints whenever the path continues in the same direction, giving shorter
    total motion time for dense or nearly-straight paths (e.g. RRT output, Cartesian
    trajectories). A zero-velocity stop is enforced only where a joint reverses direction.

    Parameters
    ----------
    q_seq : array-like, shape (N, D)
        Joint waypoints in radians (or meters).
    v_max : array-like, shape (D,)
        Per-joint velocity limits (absolute, rad/s or m/s).
    a_max : array-like, shape (D,)
        Per-joint acceleration limits (absolute).
    dt : float
        Output sampling period in seconds.

    Returns
    -------
    t_seq   : np.ndarray, shape (M,),    dtype float32
    q_out   : np.ndarray, shape (M, D),  dtype float32
    qd_out  : np.ndarray, shape (M, D),  dtype float32
    qdd_out : np.ndarray, shape (M, D),  dtype float32
    """
    q_seq = np.asarray(q_seq, dtype=np.float32)
    v_max = np.asarray(v_max, dtype=np.float32)
    a_max = np.asarray(a_max, dtype=np.float32)
    if q_seq.ndim != 2:
        raise ValueError(f'q_seq must be 2D, got shape {q_seq.shape}')
    if q_seq.shape[0] < 1:
        raise ValueError('q_seq must contain at least one waypoint')
    n_wp, n_jnts = q_seq.shape
    if v_max.shape != (n_jnts,):
        raise ValueError(f'v_max shape must be ({n_jnts},), got {v_max.shape}')
    if a_max.shape != (n_jnts,):
        raise ValueError(f'a_max shape must be ({n_jnts},), got {a_max.shape}')
    if np.any(v_max <= 0):
        raise ValueError('all v_max must be > 0')
    if np.any(a_max <= 0):
        raise ValueError('all a_max must be > 0')
    if dt <= 0:
        raise ValueError(f'dt must be > 0, got {dt}')

    if n_wp == 1:
        z = np.zeros((1, n_jnts), dtype=np.float32)
        return np.array([0.0], dtype=np.float32), q_seq.copy(), z, z

    # --- per-joint signed velocity profile ---
    vel_seq = np.zeros((n_wp, n_jnts), dtype=np.float64)
    for j in range(n_jnts):
        path_1d = q_seq[:, j].astype(np.float64)
        split_pts = _reversal_splits(path_1d)
        segs = np.split(path_1d, split_pts)
        vel_seq[:, j] = _process_segments(segs, float(v_max[j]), float(a_max[j]))

    # --- waypoint timestamps: bottleneck joint ---
    dists = np.abs(np.diff(q_seq.astype(np.float64), axis=0))           # (N-1, D)
    avg_vels = np.abs((vel_seq[:-1] + vel_seq[1:]) / 2.0)               # (N-1, D)
    avg_vels = np.where(avg_vels < 1e-12, 1e12, avg_vels)               # guard div/0
    dt_intervals = np.max(dists / avg_vels, axis=1)
    t_wp = np.zeros(n_wp, dtype=np.float64)
    t_wp[1:] = np.cumsum(dt_intervals)

    t_total = float(t_wp[-1])
    if t_total < 1e-9:
        z = np.zeros_like(q_seq, dtype=np.float32)
        return np.array([0.0], dtype=np.float32), q_seq.copy(), z, z

    # --- resample at control frequency ---
    n_out = int(t_total / dt) + 1
    t_seq = np.linspace(0.0, t_total, n_out, dtype=np.float32)

    q_out = np.zeros((n_out, n_jnts), dtype=np.float32)
    qd_out = np.zeros((n_out, n_jnts), dtype=np.float32)
    for j in range(n_jnts):
        q_out[:, j] = np.interp(t_seq, t_wp, q_seq[:, j]).astype(np.float32)
        qd_out[:, j] = np.interp(t_seq, t_wp, vel_seq[:, j]).astype(np.float32)

    qdd_out = np.zeros((n_out, n_jnts), dtype=np.float32)
    if n_out > 1:
        qdd_out[:-1] = (np.diff(qd_out, axis=0) / dt).astype(np.float32)

    return t_seq, q_out, qd_out, qdd_out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reversal_splits(path_1d):
    """Return split indices (for np.split) at direction reversals in a 1D path."""
    diffs = np.diff(path_1d)
    eps = 1e-12
    split_pts = []
    last_idx = -1
    last_sign = 0
    for i, d in enumerate(diffs):
        if abs(d) < eps:
            continue
        s = int(np.sign(d))
        if last_sign != 0 and s != last_sign:
            # turning point is waypoint last_idx+1; split so it ends the first segment
            split_pts.append(last_idx + 2)
        last_idx = i
        last_sign = s
    return split_pts


def _forward_pass_1d(path_1d, v_max, a_max):
    """Forward acceleration pass: saturate at v_max, respect a_max."""
    n = len(path_1d)
    v = np.zeros(n)
    dists = np.abs(np.diff(path_1d))
    for i in range(1, n):
        v[i] = min(v_max, np.sqrt(v[i - 1] ** 2 + 2.0 * a_max * dists[i - 1]))
    return v


def _backward_pass_1d(path_1d, fwd_v, a_max):
    """Backward deceleration pass: enforce v=0 at end, respect a_max."""
    n = len(path_1d)
    v = fwd_v.copy()
    v[-1] = 0.0
    dists = np.abs(np.diff(path_1d))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_max * dists[i]))
    return v


def _process_segments(segs, v_max, a_max):
    """
    Forward-backward pass per monotone segment, then stitch into one velocity array.

    Consecutive segments share a zero-velocity junction (direction reversal).
    The junction point is prepended to each non-first segment so the pass sees the
    correct start distance; the prepended slot is then dropped when stitching.
    """
    seg_vels = []
    for idx, seg in enumerate(segs):
        if idx > 0:
            seg = np.concatenate([[segs[idx - 1][-1]], seg])
        fwd = _forward_pass_1d(seg, v_max, a_max)
        bwd = _backward_pass_1d(seg, fwd, a_max)
        if len(seg) > 1 and (seg[1] - seg[0]) < 0:
            bwd = -bwd
        seg_vels.append(bwd)
    return np.concatenate([v[:-1] for v in seg_vels[:-1]] + [seg_vels[-1]])
