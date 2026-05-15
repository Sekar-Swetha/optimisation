"""Pass 21: Polyak step size evolution and finite difference optimal step analysis."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(42)

# ============================================================
# Benchmark definitions
# ============================================================

# Benchmark A: proxy MSE loss (quadratic, min at (0,0), f_min = 0.4826)
def loss_A(x):
    return x[0]**2 + x[1]**2 + 0.4826

def grad_A(x):
    return np.array([2*x[0], 2*x[1]])

f_star_true_A = 0.4826
x0_A = np.array([2.0, 3.0])

# Benchmark B: (x1-1)^2 + 5*(x2-2)^2 + sin(x1)
def loss_B(x):
    return (x[0] - 1)**2 + 5*(x[1] - 2)**2 + np.sin(x[0])

def grad_B(x):
    return np.array([2*(x[0] - 1) + np.cos(x[0]), 10*(x[1] - 2)])

f_star_true_B = 0.7244
x0_B = np.array([-1.0, 4.0])

# Benchmark C: Rosenbrock, f* = 0 (correct!)
def loss_C(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_C(x):
    g1 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    g2 = 200*(x[1] - x[0]**2)
    return np.array([g1, g2])

f_star_spec_C = 0.0
x0_C = np.array([-1.0, 1.0])


# ============================================================
# Polyak step gradient descent
# ============================================================

def polyak_gd(loss_fn, grad_fn, x0, f_star_spec, n_iters, eps=1e-4,
              alpha_clip=1e6):
    """Run Polyak step GD, return (alphas, fvals, xs)."""
    x = x0.copy().astype(float)
    alphas = []
    fvals = []
    for k in range(n_iters):
        fk = loss_fn(x)
        gk = grad_fn(x)
        fvals.append(fk)
        gnorm_sq = np.dot(gk, gk)
        if gnorm_sq < 1e-30:
            alphas.append(0.0)
            break
        alpha_k = (fk - f_star_spec) / (gnorm_sq + eps)
        # clip to avoid blow-up destroying the array
        alpha_k = min(alpha_k, alpha_clip)
        alphas.append(alpha_k)
        x = x - alpha_k * gk
    return np.array(alphas), np.array(fvals)


# ---- Run all three benchmarks ----
N_ITERS_AB = 120
N_ITERS_C  = 120

alphas_A, fvals_A = polyak_gd(loss_A, grad_A, x0_A, f_star_spec=0.0,
                               n_iters=N_ITERS_AB, eps=1e-4)
alphas_B, fvals_B = polyak_gd(loss_B, grad_B, x0_B, f_star_spec=0.0,
                               n_iters=N_ITERS_AB, eps=1e-4)
alphas_C, fvals_C = polyak_gd(loss_C, grad_C, x0_C, f_star_spec=0.0,
                               n_iters=N_ITERS_C,  eps=1e-3)

iters_A = np.arange(len(alphas_A))
iters_B = np.arange(len(alphas_B))
iters_C = np.arange(len(alphas_C))

# Divergence thresholds
thresh_A = f_star_true_A / 1e-4   # 0.4826 / 1e-4 = 4826
thresh_B = f_star_true_B / 1e-4   # 0.7244 / 1e-4 = 7244

# ============================================================
# Figure 1
# ============================================================

fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle('Figure 1: Polyak Step Size Evolution — Effect of Wrong $f^*_{\\mathrm{spec}}$',
              fontsize=13, fontweight='bold', y=1.01)

colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
labels = {
    'A': 'Bench A (MSE proxy, $f^*_{\\mathrm{spec}}=0$)',
    'B': 'Bench B ($f^*_{\\mathrm{spec}}=0$)',
    'C': 'Bench C Rosenbrock ($f^*_{\\mathrm{spec}}=0$ correct)',
}

# --- Panel 1: alpha_k vs iteration (semilogy) ---
ax = axes[0]
ax.semilogy(iters_A, alphas_A, color=colors['A'], label=labels['A'], lw=1.5)
ax.semilogy(iters_B, alphas_B, color=colors['B'], label=labels['B'], lw=1.5)
ax.semilogy(iters_C, alphas_C, color=colors['C'], label=labels['C'], lw=1.5)
ax.axhline(thresh_A, color=colors['A'], ls='--', lw=1.0,
           label=f'$f^*_{{\\mathrm{{true,A}}}}/\\varepsilon = {thresh_A:.0f}$')
ax.axhline(thresh_B, color=colors['B'], ls='--', lw=1.0,
           label=f'$f^*_{{\\mathrm{{true,B}}}}/\\varepsilon = {thresh_B:.0f}$')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Step size $\\alpha_k$')
ax.set_title('Polyak Step Size $\\alpha_k$ (log scale)')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(0, N_ITERS_AB)

# --- Panel 2: f(x_k) vs iteration ---
ax = axes[1]
ax.plot(iters_A, fvals_A, color=colors['A'], label=labels['A'], lw=1.5)
ax.plot(iters_B, fvals_B, color=colors['B'], label=labels['B'], lw=1.5)
ax.plot(iters_C, fvals_C, color=colors['C'], label=labels['C'], lw=1.5)
# horizontal lines at f*_true
ax.axhline(f_star_true_A, color=colors['A'], ls=':', lw=1.0,
           label=f'$f^*_{{\\mathrm{{true,A}}}}={f_star_true_A}$')
ax.axhline(f_star_true_B, color=colors['B'], ls=':', lw=1.0,
           label=f'$f^*_{{\\mathrm{{true,B}}}}={f_star_true_B}$')
ax.axhline(0.0, color=colors['C'], ls=':', lw=1.0,
           label='$f^*_C = 0$')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('$f(x_k)$')
ax.set_title('Function Value $f(x_k)$ vs Iteration')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, N_ITERS_AB)

# --- Panel 3: log step sizes with divergence annotations ---
ax = axes[2]
ax.semilogy(iters_A, alphas_A, color=colors['A'], label='Bench A', lw=1.5)
ax.semilogy(iters_B, alphas_B, color=colors['B'], label='Bench B', lw=1.5)
ax.semilogy(iters_C, alphas_C, color=colors['C'], label='Bench C', lw=1.5)

# Mark divergence thresholds with horizontal bands
ax.axhline(thresh_A, color=colors['A'], ls='--', lw=1.5)
ax.axhline(thresh_B, color=colors['B'], ls='--', lw=1.5)

# Annotations
ax.annotate(
    f'Divergence threshold A\n$f^*_{{\\mathrm{{true}}}}/\\varepsilon={thresh_A:.0f}$',
    xy=(N_ITERS_AB * 0.55, thresh_A),
    xytext=(N_ITERS_AB * 0.55, thresh_A * 3.5),
    fontsize=7, color=colors['A'],
    arrowprops=dict(arrowstyle='->', color=colors['A'], lw=1.0),
)
ax.annotate(
    f'Divergence threshold B\n$f^*_{{\\mathrm{{true}}}}/\\varepsilon={thresh_B:.0f}$',
    xy=(N_ITERS_AB * 0.55, thresh_B),
    xytext=(N_ITERS_AB * 0.05, thresh_B * 0.2),
    fontsize=7, color=colors['B'],
    arrowprops=dict(arrowstyle='->', color=colors['B'], lw=1.0),
)
ax.annotate(
    'Bench C converges\n($\\alpha_k \\to 0$)',
    xy=(iters_C[-1], alphas_C[-1] if len(alphas_C) > 0 else 1e-3),
    xytext=(N_ITERS_AB * 0.5, 1e-3),
    fontsize=7, color=colors['C'],
    arrowprops=dict(arrowstyle='->', color=colors['C'], lw=1.0),
)

ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Step size $\\alpha_k$ (log scale)')
ax.set_title('Step Sizes with Divergence Thresholds')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(0, N_ITERS_AB)

fig1.tight_layout()
out1 = os.path.join(OUTDIR, 'q1_polyak_step_evolution.pdf')
fig1.savefig(out1, bbox_inches='tight')
plt.close(fig1)
print(f"Saved: {out1}")


# ============================================================
# Figure 2: q4_fd_optimal_step_benchmark_b.pdf
# ============================================================

# True gradient at x0=[-1, 4]:
#   df/dx1 = 2*(x1-1) + cos(x1) = 2*(-2) + cos(-1) = -4 + cos(-1)
#   df/dx2 = 10*(x2-2) = 10*2 = 20
x0_fd = np.array([-1.0, 4.0], dtype=np.float64)

cos_neg1 = np.cos(np.float64(-1.0))
true_grad = np.array([-4.0 + cos_neg1, 20.0], dtype=np.float64)
true_grad_norm = np.linalg.norm(true_grad)

print(f"True gradient at x0: {true_grad}  (norm={true_grad_norm:.6f})")

def loss_B_f64(x):
    x = np.asarray(x, dtype=np.float64)
    return (x[0] - 1.0)**2 + 5.0*(x[1] - 2.0)**2 + np.sin(x[0])

def forward_fd_grad(f, x, delta):
    x = np.asarray(x, dtype=np.float64)
    f0 = f(x)
    grad = np.zeros(len(x), dtype=np.float64)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += delta
        grad[i] = (f(xp) - f0) / delta
    return grad

def centred_fd_grad(f, x, delta):
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros(len(x), dtype=np.float64)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += delta
        xm = x.copy(); xm[i] -= delta
        grad[i] = (f(xp) - f(xm)) / (2.0 * delta)
    return grad

deltas = np.logspace(-14, 1, 200)

errors_fwd = np.zeros(len(deltas))
errors_cen = np.zeros(len(deltas))

for i, d in enumerate(deltas):
    g_fwd = forward_fd_grad(loss_B_f64, x0_fd, d)
    g_cen = centred_fd_grad(loss_B_f64, x0_fd, d)
    errors_fwd[i] = np.linalg.norm(g_fwd - true_grad)
    errors_cen[i] = np.linalg.norm(g_cen - true_grad)

# Find optimal (minimum error) deltas
idx_opt_fwd = np.argmin(errors_fwd)
idx_opt_cen = np.argmin(errors_cen)
delta_opt_fwd = deltas[idx_opt_fwd]
delta_opt_cen = deltas[idx_opt_cen]
err_opt_fwd = errors_fwd[idx_opt_fwd]
err_opt_cen = errors_cen[idx_opt_cen]

print(f"Optimal delta (forward):  {delta_opt_fwd:.3e}  (error={err_opt_fwd:.3e})")
print(f"Optimal delta (centred):  {delta_opt_cen:.3e}  (error={err_opt_cen:.3e})")

# Reference slopes for the truncation-error region
# Forward: error ~ C1 * delta^1  =>  slope 1
# Centred: error ~ C2 * delta^2  =>  slope 2
ref_delta = np.logspace(-8, -1, 50)
ref_fwd = 2.0 * ref_delta          # slope 1 reference (C * delta)
ref_cen = 5.0 * ref_delta**2       # slope 2 reference (C * delta^2)

# Markers for delta=0.05 (good) and delta=0.8 (bad) from the assignment
delta_good = 0.05
delta_bad  = 0.8

err_fwd_good = np.linalg.norm(forward_fd_grad(loss_B_f64, x0_fd, delta_good) - true_grad)
err_cen_good = np.linalg.norm(centred_fd_grad(loss_B_f64, x0_fd, delta_good) - true_grad)
err_fwd_bad  = np.linalg.norm(forward_fd_grad(loss_B_f64, x0_fd, delta_bad)  - true_grad)
err_cen_bad  = np.linalg.norm(centred_fd_grad(loss_B_f64, x0_fd, delta_bad)  - true_grad)

# ---- Plot ----
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle('Figure 2: Finite Difference Optimal Step Analysis — Benchmark B',
              fontsize=13, fontweight='bold')

col_fwd = '#d62728'
col_cen = '#1f77b4'

# --- Panel 1: Full log-log error vs delta ---
ax = axes2[0]
ax.loglog(deltas, errors_fwd, color=col_fwd, lw=1.8, label='Forward FD $O(\\delta)$')
ax.loglog(deltas, errors_cen, color=col_cen, lw=1.8, label='Centred FD $O(\\delta^2)$')

# Reference lines
ax.loglog(ref_delta, ref_fwd, 'k--', lw=1.0, alpha=0.6,
          label='slope 1 ref  ($\\propto \\delta$)')
ax.loglog(ref_delta, ref_cen, 'k:',  lw=1.0, alpha=0.6,
          label='slope 2 ref  ($\\propto \\delta^2$)')

# Optimal delta markers
ax.axvline(delta_opt_fwd, color=col_fwd, ls=':', lw=1.3,
           label=f'Opt $\\delta_{{fwd}}={delta_opt_fwd:.1e}$')
ax.axvline(delta_opt_cen, color=col_cen, ls=':', lw=1.3,
           label=f'Opt $\\delta_{{cen}}={delta_opt_cen:.1e}$')

# Assignment markers
ax.axvline(delta_good, color='green', ls='--', lw=1.3,
           label=f'$\\delta=0.05$ (good)')
ax.axvline(delta_bad, color='orange', ls='--', lw=1.3,
           label=f'$\\delta=0.8$ (bad)')

ax.set_xlabel('Step size $\\delta$')
ax.set_ylabel('$\\|\\hat{g} - g^*\\|_2$  (gradient error)')
ax.set_title('FD Error vs Step Size (Full Range)')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, which='both', alpha=0.3)

# --- Panel 2: U-shape zoom with region annotations ---
ax = axes2[1]
ax.loglog(deltas, errors_fwd, color=col_fwd, lw=1.8, label='Forward FD')
ax.loglog(deltas, errors_cen, color=col_cen, lw=1.8, label='Centred FD')

# Optimal markers (dots)
ax.plot(delta_opt_fwd, err_opt_fwd, 'o', color=col_fwd, ms=8, zorder=5,
        label=f'Opt fwd $\\delta={delta_opt_fwd:.1e}$')
ax.plot(delta_opt_cen, err_opt_cen, 's', color=col_cen, ms=8, zorder=5,
        label=f'Opt cen $\\delta={delta_opt_cen:.1e}$')

# Assignment markers
ax.axvline(delta_good, color='green', ls='--', lw=1.3, label='$\\delta=0.05$ (good)')
ax.axvline(delta_bad,  color='orange', ls='--', lw=1.3, label='$\\delta=0.8$ (bad)')

# Region annotations
ax.text(1e-12, errors_fwd[5] * 2.0, 'Round-off\ndominated',
        fontsize=8, color='gray', ha='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
ax.text(1.5, errors_fwd[-5] * 0.5, 'Truncation\ndominated',
        fontsize=8, color='gray', ha='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))

ax.set_xlabel('Step size $\\delta$')
ax.set_ylabel('$\\|\\hat{g} - g^*\\|_2$')
ax.set_title('U-Shape: Truncation vs Round-off Error')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, which='both', alpha=0.3)

fig2.tight_layout()
out2 = os.path.join(OUTDIR, 'q4_fd_optimal_step_benchmark_b.pdf')
fig2.savefig(out2, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: {out2}")
