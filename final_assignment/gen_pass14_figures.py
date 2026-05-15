"""Pass 14: Adam bias correction, convergence theory summary, RMSprop preconditioning."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
np.random.seed(42)

# ============================================================
# Figure 1: Adam bias correction illustration
# ============================================================
T = 100
beta1 = 0.82
beta2 = 0.999

# Simulated gradient sequence: constant gradient with noise
true_g = np.array([1.0, -0.5])
noise_scale = 2.0

m = np.zeros(2); v = np.zeros(2)
m_raw, m_corrected = [], []
v_raw, v_corrected = [], []

for t in range(1, T+1):
    g = true_g + noise_scale * np.random.randn(2)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    m_raw.append(m.copy())
    v_raw.append(v.copy())
    m_corrected.append(m / (1 - beta1**t))
    v_corrected.append(v / (1 - beta2**t))

m_raw = np.array(m_raw); m_corrected = np.array(m_corrected)
v_raw = np.array(v_raw); v_corrected = np.array(v_corrected)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ts = np.arange(1, T+1)
ax = axes[0]
ax.plot(ts, m_raw[:,0], 'b-', lw=2, label='Raw $m_t[0]$ (biased)')
ax.plot(ts, m_corrected[:,0], 'r-', lw=2, label='Corrected $\\hat{m}_t[0]$')
ax.axhline(true_g[0], color='k', linestyle='--', lw=1.5, label=f'True gradient $g_1={true_g[0]}$')
ax.fill_between(ts[:10], m_raw[:10,0], m_corrected[:10,0], alpha=0.2, color='orange',
                label='Bias region (t small)')
ax.set_xlabel('Time step $t$'); ax.set_ylabel('First moment estimate')
ax.set_title('Adam Bias Correction: First Moment $m_t$\n'
             f'($\\beta_1={beta1}$, true gradient = {true_g[0]})', fontsize=10)
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3); ax.set_xlim(1, T)

ax2 = axes[1]
ax2.semilogy(ts, v_raw[:,0]+1e-10, 'b-', lw=2, label='Raw $v_t[0]$ (biased towards 0)')
ax2.semilogy(ts, v_corrected[:,0]+1e-10, 'r-', lw=2, label='Corrected $\\hat{v}_t[0]$')
ax2.axhline(true_g[0]**2 + noise_scale**2, color='k', linestyle='--', lw=1.5,
            label=f'True $\\mathbb{{E}}[g^2] = {true_g[0]**2 + noise_scale**2:.2f}$')
ax2.set_xlabel('Time step $t$'); ax2.set_ylabel('Second moment estimate (log scale)')
ax2.set_title('Adam Bias Correction: Second Moment $v_t$\n'
              f'($\\beta_2={beta2}$)', fontsize=10)
ax2.legend(fontsize=8.5); ax2.grid(True, which='both', alpha=0.3); ax2.set_xlim(1, T)
ax2.text(0.6, 0.08, 'Without correction: $v_t \\approx 0$ for small $t$\n'
         'causing inflated steps $(\\to \\text{instability})$',
         transform=ax2.transAxes, fontsize=8.5,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Adam Bias Correction: Why Zero Initialisation Causes Underestimation', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q2_adam_bias_correction.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q2_adam_bias_correction.pdf")

# ============================================================
# Figure 2: Nesterov momentum schedule and convergence
# ============================================================
def bench_b(x): return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def bench_b_grad(x): return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

N = 200
x0_B = np.array([-1., 4.])
fstar_B = 0.7244
alpha_nes = 0.035

# Run Nesterov with different beta_max to show effect of momentum schedule
beta_maxs = [0.0, 0.5, 0.7, 0.92, 0.999]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: momentum schedule beta_k vs k for different beta_max
ax = axes[0]
ks = np.arange(1, N+1)
for bmax, color in zip([0.5, 0.7, 0.92, 0.999], ['blue', 'green', 'orange', 'red']):
    beta_k = np.minimum((ks-1)/(ks+2), bmax)
    ax.plot(ks, beta_k, lw=2, color=color, label=f'$\\beta_{{\\max}}={bmax}$')
ax.plot(ks, (ks-1)/(ks+2), 'k--', lw=1, alpha=0.7, label='$(k-1)/(k+2)$ (uncapped)')
ax.set_xlabel('Iteration $k$'); ax.set_ylabel('$\\beta_k = \\min((k-1)/(k+2), \\beta_{\\max})$')
ax.set_title('Nesterov Momentum Schedule $\\beta_k$', fontsize=10)
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
ax.set_xlim(0, N)

# Middle: convergence with different beta_max
ax2 = axes[1]
for bmax, color in zip(beta_maxs, ['gray', 'blue', 'green', 'orange', 'red']):
    x = x0_B.copy().astype(float); z = np.zeros(2); h = [bench_b(x)]
    for k in range(1, N+1):
        bk = min((k-1)/(k+2), bmax)
        la = x + bk * z
        g = bench_b_grad(la)
        z = bk*z - alpha_nes*g
        x += z
        h.append(bench_b(x))
    sub = np.maximum(np.array(h) - fstar_B, 1e-12)
    ax2.semilogy(np.arange(N+1), sub, lw=2, color=color, label=f'$\\beta_{{\\max}}={bmax}$')
ax2.set_xlabel('Iteration $k$'); ax2.set_ylabel('$f(x_k) - f^\\star$')
ax2.set_title('Nesterov Convergence vs $\\beta_{\\max}$\n(Benchmark B, $\\alpha=0.035$)', fontsize=10)
ax2.legend(fontsize=8.5); ax2.grid(True, which='both', alpha=0.3); ax2.set_xlim(0, N)

# Right: O(1/k^2) verification for different beta_max
ax3 = axes[2]
ks_arr = np.arange(1, N+1)
for bmax, color in zip([0.7, 0.92], ['green', 'orange']):
    x = x0_B.copy().astype(float); z = np.zeros(2); h = [bench_b(x)]
    for k in range(1, N+1):
        bk = min((k-1)/(k+2), bmax)
        la = x + bk * z
        g = bench_b_grad(la)
        z = bk*z - alpha_nes*g
        x += z
        h.append(bench_b(x))
    sub = np.maximum(np.array(h[1:]) - fstar_B, 1e-12)
    ax3.loglog(ks_arr, sub, lw=2, color=color, label=f'Nesterov $\\beta_{{\\max}}={bmax}$')

c_ref = (bench_b(x0_B) - fstar_B) * 4
ax3.loglog(ks_arr, c_ref / ks_arr**2, 'k--', lw=1, alpha=0.7, label='$O(1/k^2)$ reference')
ax3.loglog(ks_arr, c_ref / ks_arr, 'k:', lw=1, alpha=0.7, label='$O(1/k)$ reference')
ax3.set_xlabel('Iteration $k$'); ax3.set_ylabel('$f(x_k) - f^\\star$ (log--log)')
ax3.set_title('Rate Verification: $O(1/k^2)$ Nesterov\n(Benchmark B)', fontsize=10)
ax3.legend(fontsize=8.5); ax3.grid(True, which='both', alpha=0.3); ax3.set_xlim(1, N)

plt.suptitle('Nesterov Momentum Schedule: Effect on Convergence Rate', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q2_nesterov_schedule_analysis.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q2_nesterov_schedule_analysis.pdf")

# ============================================================
# Figure 3: Comparison of all constraint methods trajectories
# ============================================================
def bench_b_loss(x): return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def bench_b_grad_fn(x): return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

x0_con = np.array([0.2, 4.0])
n_iter = 80
lb = 0.5

def pgd_traj(x0, alpha, n):
    x = x0.copy().astype(float); xs = [x.copy()]
    for _ in range(n):
        z = x - alpha * bench_b_grad_fn(x)
        x = np.array([max(lb, z[0]), z[1]])
        xs.append(x.copy())
    return np.array(xs)

def pen_traj(x0, alpha, lam, n):
    x = x0.copy().astype(float); xs = [x.copy()]
    for _ in range(n):
        g = bench_b_grad_fn(x)
        if x[0] < lb:
            g = g + np.array([-lam, 0.0])
        x -= alpha * g
        xs.append(x.copy())
    return np.array(xs)

def al_traj(x0, alpha, rho, n):
    x = x0.copy().astype(float); xs = [x.copy()]; mu = 0.0
    for k in range(n):
        c = lb - x[0]; c_pos = max(0.0, c)
        g = bench_b_grad_fn(x)
        g_aug = g + (-1) * (mu + rho * c_pos) * np.array([1.0, 0.0])
        x -= alpha * g_aug
        if (k+1) % 10 == 0:
            mu = max(0.0, mu + rho * (lb - x[0]))
        xs.append(x.copy())
    return np.array(xs)

tr_pgd  = pgd_traj(x0_con, 0.06, n_iter)
tr_pen1 = pen_traj(x0_con, 0.04, 1.0, n_iter)
tr_pen5 = pen_traj(x0_con, 0.03, 4.5, n_iter)
tr_al   = al_traj(x0_con, 0.05, 2.0, n_iter)

# Contour grid
xx = np.linspace(0.0, 2.5, 200); yy = np.linspace(0.0, 5.0, 200)
XX, YY = np.meshgrid(xx, yy)
ZZ = (XX-1)**2 + 5*(YY-2)**2 + np.sin(XX)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(XX, YY, ZZ, levels=20, cmap='YlOrRd', alpha=0.35)
ax.contour(XX, YY, ZZ, levels=20, colors='gray', linewidths=0.4, alpha=0.5)
ax.axvline(lb, color='red', linestyle='--', lw=2, label='Constraint $x_1 = 0.5$')
ax.fill_betweenx([0, 5], 0, lb, alpha=0.15, color='red', label='Infeasible region')

for traj, label, color, ls in [
    (tr_pgd,  'PGD',                    'blue',   '-'),
    (tr_pen1, 'Penalty $\\lambda=1$',   'green',  '--'),
    (tr_pen5, 'Penalty $\\lambda=4.5$', 'purple', '-.'),
    (tr_al,   'Aug. Lagrangian',        'orange', ':'),
]:
    ax.plot(traj[:,0], traj[:,1], color=color, linestyle=ls, lw=1.8, label=label)
    ax.plot(traj[0,0], traj[0,1], 'ks', ms=8, zorder=5)
    ax.plot(traj[-1,0], traj[-1,1], 'o', color=color, ms=8, zorder=5)

ax.plot(0.582, 2.0, 'g*', ms=12, zorder=6, label='$x^\\star = (0.582, 2)$')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Constrained Optimisation: PGD, Penalty, Augmented Lagrangian Trajectories\n'
             '(starting from infeasible $x_0 = (0.2, 4)$, 80 iterations)', fontsize=10)
ax.legend(fontsize=8.5, loc='upper right'); ax.set_xlim(0, 2.5); ax.set_ylim(0, 5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q5_all_methods_trajectories.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q5_all_methods_trajectories.pdf")

print("All Pass 14 figures generated.")
