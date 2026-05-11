"""
Headless test for retime_global (time-parameterization via global forward-backward pass).

Run:
    ./codex_python.sh nagai/test_retime_global.py

Output:
    Console summary + nagai/retime_global_result.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from one_assembly.motion_planner.time_param import retime_global
from one.motion.trajectory.time_param import retime_trapezoidal

# ---------------------------------------------------------------------------
# joint limits (RS007L-like: 6 joints, rad/s and rad/s^2)
# ---------------------------------------------------------------------------
V_MAX = np.array([3.49, 3.49, 3.49, 5.24, 5.24, 5.24], dtype=np.float32)   # rad/s
A_MAX = np.array([6.98, 6.98, 6.98, 10.47, 10.47, 10.47], dtype=np.float32)  # rad/s^2
DT = 0.005  # 200 Hz

# ---------------------------------------------------------------------------
# test cases
# ---------------------------------------------------------------------------

# case 1: monotone 6-DOF path (e.g. a short RRT result)
Q_MONO = np.array([
    [0.0,  0.0,   0.0,   0.0,  0.0,  0.0],
    [0.2,  0.05, -0.1,   0.1,  0.05, 0.0],
    [0.5,  0.1,  -0.3,   0.3,  0.1,  0.1],
    [0.8,  0.2,  -0.5,   0.5,  0.2,  0.2],
    [1.0,  0.3,  -0.7,   0.7,  0.3,  0.3],
], dtype=np.float32)

# case 2: path with a direction reversal on joint 0
Q_REVERSAL = np.array([
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [0.3,  0.1,  0.1,  0.1,  0.0,  0.0],
    [0.6,  0.2,  0.2,  0.2,  0.0,  0.0],
    [0.8,  0.3,  0.3,  0.3,  0.0,  0.0],
    [0.6,  0.4,  0.4,  0.4,  0.0,  0.0],  # joint 0 reverses here
    [0.3,  0.5,  0.5,  0.5,  0.0,  0.0],
    [0.0,  0.6,  0.6,  0.6,  0.0,  0.0],
], dtype=np.float32)

# case 3: dense path (many near-collinear waypoints — where global shines vs trapezoidal)
t_dense = np.linspace(0, 1, 30)
Q_DENSE = np.stack([
    np.sin(t_dense * np.pi),
    np.cos(t_dense * np.pi) * 0.5,
    t_dense * 1.2,
    np.sin(t_dense * 2 * np.pi) * 0.4,
    t_dense * 0.8,
    np.zeros(30),
], axis=1).astype(np.float32)


def run_case(name, q_seq, print_detail=True):
    t_g, q_g, qd_g, qdd_g = retime_global(q_seq, V_MAX, A_MAX, dt=DT)
    t_t, q_t, qd_t, qdd_t = retime_trapezoidal(q_seq, V_MAX, A_MAX, dt=DT)

    # --- assertions ---
    assert np.allclose(q_g[0], q_seq[0], atol=1e-3), 'start mismatch'
    assert np.allclose(q_g[-1], q_seq[-1], atol=1e-3), 'end mismatch'
    assert np.allclose(qd_g[0], 0, atol=1e-3), 'start velocity not zero'
    assert np.allclose(qd_g[-1], 0, atol=1e-3), 'end velocity not zero'
    assert np.all(np.abs(qd_g) <= V_MAX[None, :] + 1e-3), 'v_max violated'
    # total time: global should be <= trapezoidal (allows equal)
    assert t_g[-1] <= t_t[-1] + 1e-3, f'global ({t_g[-1]:.3f}s) slower than trap ({t_t[-1]:.3f}s)'

    if print_detail:
        speedup = (t_t[-1] - t_g[-1]) / t_t[-1] * 100 if t_t[-1] > 0 else 0.0
        print(f'[{name}]')
        print(f'  global:      {t_g[-1]:.4f}s  ({len(t_g)} samples)')
        print(f'  trapezoidal: {t_t[-1]:.4f}s  ({len(t_t)} samples)')
        print(f'  speedup:     {speedup:.1f}%')
        print(f'  qd max: global={np.abs(qd_g).max():.3f}  trap={np.abs(qd_t).max():.3f}')
        print(f'  qdd rms: global={np.sqrt((qdd_g**2).mean()):.3f}  trap={np.sqrt((qdd_t**2).mean()):.3f}')

    return (t_g, q_g, qd_g, qdd_g), (t_t, q_t, qd_t, qdd_t)


print('=' * 60)
print('retime_global test')
print('=' * 60)
(t_g1, q_g1, qd_g1, qdd_g1), (t_t1, q_t1, qd_t1, qdd_t1) = run_case('monotone',  Q_MONO)
(t_g2, q_g2, qd_g2, qdd_g2), (t_t2, q_t2, qd_t2, qdd_t2) = run_case('reversal',  Q_REVERSAL)
(t_g3, q_g3, qd_g3, qdd_g3), (t_t3, q_t3, qd_t3, qdd_t3) = run_case('dense',     Q_DENSE)
print('all assertions passed')
print()

# ---------------------------------------------------------------------------
# plot: q / qd / qdd for all three cases
# ---------------------------------------------------------------------------
N_CASES = 3
fig, axes = plt.subplots(N_CASES, 3, figsize=(15, 4 * N_CASES))
fig.suptitle('retime_global (solid) vs retime_trapezoidal (dashed)', fontsize=13)

CASE_LABELS = ['monotone', 'reversal', 'dense']
RESULTS_G = [(t_g1, q_g1, qd_g1, qdd_g1), (t_g2, q_g2, qd_g2, qdd_g2), (t_g3, q_g3, qd_g3, qdd_g3)]
RESULTS_T = [(t_t1, q_t1, qd_t1, qdd_t1), (t_t2, q_t2, qd_t2, qdd_t2), (t_t3, q_t3, qd_t3, qdd_t3)]

for row, (label, (tg, qg, qdg, qddg), (tt, qt, qdt, qddt)) in enumerate(
        zip(CASE_LABELS, RESULTS_G, RESULTS_T)):
    ax_q, ax_qd, ax_qdd = axes[row]

    for j in range(qg.shape[1]):
        c = f'C{j}'
        ax_q.plot(tg, qg[:, j],   color=c, lw=1.2)
        ax_q.plot(tt, qt[:, j],   color=c, lw=1.2, ls='--', alpha=0.5)
        ax_qd.plot(tg, qdg[:, j], color=c, lw=1.2)
        ax_qd.plot(tt, qdt[:, j], color=c, lw=1.2, ls='--', alpha=0.5)
        ax_qdd.plot(tg, qddg[:, j], color=c, lw=0.9)
        ax_qdd.plot(tt, qddt[:, j], color=c, lw=0.9, ls='--', alpha=0.5)

    ax_q.set_title(f'{label} — position')
    ax_qd.set_title(f'{label} — velocity')
    ax_qdd.set_title(f'{label} — acceleration')
    ax_q.set_ylabel('q [rad]')
    ax_qd.set_ylabel('qd [rad/s]')
    ax_qdd.set_ylabel('qdd [rad/s²]')
    for ax in (ax_q, ax_qd, ax_qdd):
        ax.set_xlabel('t [s]')
        ax.grid(True, lw=0.4, alpha=0.5)

plt.tight_layout()
out_path = Path(__file__).parent / 'retime_global_result.png'
plt.savefig(out_path, dpi=120)
print(f'plot saved → {out_path}')
