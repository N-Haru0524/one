import numpy as np


def split_path_1d(path_1d):
    zero_crossings = []
    distances = np.diff(path_1d)
    for i in range(len(distances) - 1):
        if np.abs(distances[i]) < 1e-12:
            last_j = None
            for j in range(i + 1, len(distances)):
                last_j = j
                if (np.abs(distances[j]) > 1e-12 and
                        np.sign(distances[j]) != np.sign(distances[i - 1])):
                    zero_crossings.append(j + 1)
                    break
                if (np.abs(distances[j]) > 1e-12 and
                        np.sign(distances[j]) == np.sign(distances[i - 1])):
                    break
            if last_j is None or last_j == len(distances) - 1:
                break
        elif (np.sign(distances[i]) != np.sign(distances[i + 1]) and
              np.abs(distances[i + 1]) > 1e-12):
            zero_crossings.append(i + 2)
    if len(zero_crossings) == 0:
        return [path_1d]
    segments = np.split(path_1d, zero_crossings)
    return [subarr for subarr in segments if len(subarr) > 0]


def forward_1d(path_1d, max_vel, max_acc):
    n_waypoints = len(path_1d)
    velocities = np.zeros(n_waypoints)
    distances = np.abs(np.diff(path_1d))
    for i in range(1, n_waypoints):
        velocities[i] = np.minimum(
            max_vel,
            np.sqrt(velocities[i - 1] ** 2 + 2 * max_acc * distances[i - 1]))
    return velocities


def backward_1d(path_1d, velocities, max_acc):
    n_waypoints = len(path_1d)
    distances = np.abs(np.diff(path_1d))
    velocities[-1] = 0
    for i in range(n_waypoints - 2, -1, -1):
        velocities[i] = np.minimum(
            velocities[i],
            np.sqrt(velocities[i + 1] ** 2 + 2 * max_acc * distances[i]))
    return velocities


def proc_segs(path_1d_segs, max_vel, max_acc):
    velocities = []
    for idx, path_1d in enumerate(path_1d_segs):
        if idx > 0:
            path_1d = np.insert(path_1d, 0, path_1d_segs[idx - 1][-1])
        v_fwd = forward_1d(path_1d, max_vel, max_acc)
        v_bwd = backward_1d(path_1d, v_fwd, max_acc)
        if np.diff(path_1d)[0] < 0:
            v_bwd = v_bwd * -1
        velocities.append(v_bwd)
    return velocities


def generate_time_optimal_trajectory(path, max_vels=None, max_accs=None,
                                     ctrl_freq=.005):
    path = np.asarray(path)
    n_waypoints, n_jnts = path.shape
    if max_vels is None:
        max_vels = np.asarray([np.pi * 2 / 3] * n_jnts)
    if max_accs is None:
        max_accs = np.asarray([np.pi] * n_jnts)
    velocities = np.zeros((n_waypoints, n_jnts))
    for id_jnt in range(n_jnts):
        path_1d = path[:, id_jnt]
        path_1d_segs = split_path_1d(path_1d)
        vel_1d_segs = proc_segs(path_1d_segs, max_vels[id_jnt], max_accs[id_jnt])
        vel_1d_merged = np.concatenate(
            [v[:-1] for v in vel_1d_segs[:-1]] + [vel_1d_segs[-1]])
        velocities[:, id_jnt] = vel_1d_merged
    distances = np.abs(np.diff(path, axis=0))
    avg_velocities = np.abs((velocities[:-1] + velocities[1:]) / 2)
    avg_velocities = np.where(avg_velocities == 0, 10e6, avg_velocities)
    time_intervals = np.max(distances / avg_velocities, axis=1)
    time = np.zeros(len(path))
    time[1:] = np.cumsum(time_intervals)
    n_interp_conf = int(time[-1] / ctrl_freq) + 1
    interp_time = np.linspace(0, time[-1], n_interp_conf)
    interp_confs = np.zeros((len(interp_time), path.shape[1]))
    interp_spds = np.zeros((len(interp_time), path.shape[1]))
    for j in range(path.shape[1]):
        interp_confs[:, j] = np.interp(interp_time, time, path[:, j])
        interp_spds[:, j] = np.interp(interp_time, time, velocities[:, j])
    tmp_spds = np.append(interp_spds, np.zeros((1, n_jnts)), axis=0)
    interp_accs = np.diff(tmp_spds, axis=0) / ctrl_freq
    return interp_time, interp_confs, interp_spds, interp_accs
