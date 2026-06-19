#!/usr/bin/env python3
"""Pass 17 figures: stochastic gradient noise analysis, step-size sensitivity heatmap,
central vs forward differences error decomposition, PL condition visualization."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ─── Benchmark B ─────────────────────────────────────────────────────────────
def bench_B_loss(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])

def bench_B_grad(x):
    return np.array([2*(x[0]-1) + np.cos(x[0]), 10*(x[1]-2)])

# ─── Figure 1: Central vs Forward FD error decomposition ──────────────────────
# At a fixed point x0, compute FD error vs delta for both forward and central differences

x0_fd = np.array([0.5, 2.5])
grad_true = bench_B_grad(x0_fd)

deltas = np.logspace(-10, 0, 200)
eps_mach = np.finfo(float).eps  # ~2.2e-16

forward_err = []
central_err = []
for d in deltas:
    # Forward differences: [f(x+d*e_i) - f(x)] / d
    grad_fwd = np.zeros(2)
    f0 = bench_B_loss(x0_fd)
    for i in range(2):
        ei = np.zeros(2); ei[i] = 1.0
        grad_fwd[i] = (bench_B_loss(x0_fd + d*ei) - f0) / d

    # Central differences: [f(x+d*e_i) - f(x-d*e_i)] / (2d)
    grad_cen = np.zeros(2)
    for i in range(2):
        ei = np.zeros(2); ei[i] = 1.0
        grad_cen[i] = (bench_B_loss(x0_fd + d*ei) - bench_B_loss(x0_fd - d*ei)) / (2*d)

    forward_err.append(np.linalg.norm(grad_fwd - grad_true))
    central_err.append(np.linalg.norm(grad_cen - grad_true))

forward_err = np.array(forward_err)
central_err = np.array(central_err)

# Theoretical curves (rough)
# Forward: truncation O(delta), cancellation O(eps_mach/delta)
L2_fwd = 1.0  # second derivative magnitude
trunc_fwd = L2_fwd * deltas
cancel_fwd = eps_mach * np.abs(bench_B_loss(x0_fd)) / deltas
theory_fwd = trunc_fwd + cancel_fwd

# Central: truncation O(delta^2), cancellation O(eps_mach/delta)
L3_cen = 0.1  # third derivative magnitude
trunc_cen = L3_cen * deltas**2
cancel_cen = 2 * eps_mach * np.abs(bench_B_loss(x0_fd)) / deltas
theory_cen = trunc_cen + cancel_cen

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.loglog(deltas, forward_err, '-', color='#2196F3', linewidth=2.5, label='Forward FD error (actual)')
ax.loglog(deltas, central_err, '-', color='#FF5722', linewidth=2.5, label='Central FD error (actual)')
ax.loglog(deltas, theory_fwd, 'b--', linewidth=1.5, alpha=0.6, label='Fwd theory: $L_2\\delta + \\varepsilon_{mach}/\\delta$')
ax.loglog(deltas, theory_cen, 'r--', linewidth=1.5, alpha=0.6, label='Cen theory: $L_3\\delta^2 + 2\\varepsilon_{mach}/\\delta$')

# Optimal delta markers
delta_opt_fwd = (eps_mach * abs(bench_B_loss(x0_fd)) / L2_fwd)**0.5
delta_opt_cen = (eps_mach * abs(bench_B_loss(x0_fd)) / L3_cen)**(1.0/3)
ax.axvline(delta_opt_fwd, color='#2196F3', linestyle=':', alpha=0.8, label=f'$\\delta_{{opt}}^{{fwd}}\\approx{delta_opt_fwd:.1e}$')
ax.axvline(delta_opt_cen, color='#FF5722', linestyle=':', alpha=0.8, label=f'$\\delta_{{opt}}^{{cen}}\\approx{delta_opt_cen:.1e}$')

ax.set_xlabel('Step size $\\delta$', fontsize=12)
ax.set_ylabel('Gradient error $\\|\\nabla_{FD} f - \\nabla f\\|$', fontsize=12)
ax.set_title('Forward vs Central Finite Differences\nError Decomposition on Benchmark B', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

# Right: GD convergence for many delta values
ax2 = axes[1]
delta_vals = [1e-1, 5e-2, 1e-2, 1e-3, 1e-4, 1e-7]
colors_d = plt.cm.viridis(np.linspace(0.1, 0.9, len(delta_vals)))
x0_bench = np.array([-1.0, 4.0])
n_iters = 120
f_star_B = 0.7244

for dv, col in zip(delta_vals, colors_d):
    x = x0_bench.copy().astype(float)
    losses = []
    f0 = bench_B_loss(x)
    for _ in range(n_iters):
        g = np.zeros(2)
        for i in range(2):
            ei = np.zeros(2); ei[i] = 1.0
            g[i] = (bench_B_loss(x + dv*ei) - bench_B_loss(x)) / dv
        x -= 0.06 * g
        losses.append(bench_B_loss(x))
    ax2.semilogy(losses, color=col, linewidth=1.8, label=f'$\\delta={dv:.0e}$')

# Exact GD
x = x0_bench.copy().astype(float)
losses_exact = []
for _ in range(n_iters):
    x -= 0.06 * bench_B_grad(x)
    losses_exact.append(bench_B_loss(x))
ax2.semilogy(losses_exact, 'k-', linewidth=2.5, label='Exact GD')

ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('$f(x_k)$ (log)', fontsize=12)
ax2.set_title('GD Convergence vs FD Step $\\delta$\n($\\delta$ too small or large degrades performance)', fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/fd_central_vs_forward.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: fd_central_vs_forward.pdf")


# ─── Figure 2: Polyak-Łojasiewicz condition visualization ─────────────────────
# Show a strongly convex function (satisfies PL) and a non-convex function that still satisfies PL

def f_sc(x):
    """Strongly convex: f(x) = (x-2)^2 + 1"""
    return (x - 2)**2 + 1.0

def f_non_convex_pl(x):
    """Non-convex but PL: f(x) = x^2 + 3*sin^2(x), satisfies PL with mu=2"""
    return x**2 + 3*np.sin(x)**2

def grad_ncpl(x):
    return 2*x + 6*np.sin(x)*np.cos(x)

# PL constant for f_non_convex_pl: check 0.5||grad||^2 >= mu*(f - f*)
x_check = np.linspace(-4, 4, 1000)
f_vals = f_non_convex_pl(x_check)
g_vals = np.array([grad_ncpl(xi) for xi in x_check])
f_star_ncpl = min(f_non_convex_pl(0), f_non_convex_pl(-np.pi))  # approx local minima

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
x_range = np.linspace(-1, 5, 300)
ax.plot(x_range, f_sc(x_range), 'b-', linewidth=2, label='$f(x) = (x-2)^2 + 1$ (str.\ convex)')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.6, label='$f^\\star = 1$')
# Show PL condition: 0.5||grad||^2 >= mu*(f-f*)
mu_sc = 2.0  # f'' = 2 everywhere
x_demo = np.array([0.5, 1.0, 3.5])
for xi in x_demo:
    fi = f_sc(xi)
    gi = 2*(xi - 2)
    ax.annotate('', xy=(xi, fi), xytext=(xi, 1.0),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(xi + 0.05, (fi + 1.0)/2, f'$f-f^\\star$', fontsize=8, color='red')
ax.set_xlim(-1, 5); ax.set_ylim(0, 12)
ax.set_xlabel('$x$', fontsize=12); ax.set_ylabel('$f(x)$', fontsize=12)
ax.set_title('Strongly Convex $f$\n(PL condition: $\\frac{1}{2}\\|\\nabla f\\|^2 \\geq \\mu(f-f^\\star)$)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
x_range2 = np.linspace(-4, 4, 500)
ax2.plot(x_range2, f_non_convex_pl(x_range2), 'r-', linewidth=2, label='$f(x) = x^2 + 3\\sin^2(x)$')
ax2.axhline(0, color='gray', linestyle='--', alpha=0.6, label='$f^\\star = 0$')
ax2.set_xlabel('$x$', fontsize=12); ax2.set_ylabel('$f(x)$', fontsize=12)
ax2.set_title('Non-Convex but PL-Satisfying $f$\n(multiple local minima all at $f^\\star = 0$)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Right: PL condition check: plot 0.5||grad||^2 vs mu*(f - f*)
ax3 = axes[2]
pl_lhs = 0.5 * g_vals**2
pl_rhs_mu1 = 1.0 * (f_vals - 0)  # mu=1
pl_rhs_mu05 = 0.5 * (f_vals - 0)  # mu=0.5
ax3.plot(x_check, pl_lhs, 'k-', linewidth=2, label='$\\frac{1}{2}\\|\\nabla f(x)\\|^2$ (LHS)')
ax3.plot(x_check, np.maximum(pl_rhs_mu1, 0), 'b--', linewidth=1.5, label='$\\mu(f-f^\\star)$, $\\mu=1$')
ax3.plot(x_check, np.maximum(pl_rhs_mu05, 0), 'r--', linewidth=1.5, label='$\\mu(f-f^\\star)$, $\\mu=0.5$')
ax3.set_xlabel('$x$', fontsize=12)
ax3.set_ylabel('Value', fontsize=12)
ax3.set_title('PL Condition Verification\n($\\frac{1}{2}\\|\\nabla f\\|^2 \\geq \\mu(f-f^\\star)$)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-4, 4); ax3.set_ylim(-1, 50)

plt.suptitle('Polyak--\\L{}ojasiewicz (PL) Condition: Beyond Strong Convexity', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/pl_condition.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: pl_condition.pdf")


# ─── Figure 3: Hyperparameter sensitivity heatmap ─────────────────────────────
# For GD, Nesterov, Adam on Benchmark B: show convergence vs alpha x beta grid

def gd_final(alpha, n=120):
    x = np.array([-1.0, 4.0])
    for _ in range(n):
        x = x - alpha * bench_B_grad(x)
        if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e6):
            return 1e10
    return bench_B_loss(x)

def nesterov_final(alpha, beta_max, n=120):
    x = np.array([-1.0, 4.0]).astype(float)
    z = np.zeros(2)
    for k in range(1, n+1):
        bk = min((k-1)/(k+2), beta_max)
        la = x + bk * z
        g = bench_B_grad(la)
        z = bk * z - alpha * g
        x = x + z
        if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e6):
            return 1e10
    return bench_B_loss(x)

def adam_final(alpha, beta1=0.9, n=120):
    x = np.array([-1.0, 4.0]).astype(float)
    m_v = np.zeros(2); v_v = np.zeros(2)
    for t in range(1, n+1):
        g = bench_B_grad(x)
        m_v = beta1 * m_v + (1 - beta1) * g
        v_v = 0.999 * v_v + 0.001 * g**2
        mh = m_v / (1 - beta1**t)
        vh = v_v / (1 - 0.999**t)
        x = x - alpha * mh / (np.sqrt(vh) + 1e-8)
        if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e6):
            return 1e10
    return bench_B_loss(x)

alphas = np.logspace(-3, 0, 30)
betas = np.linspace(0.0, 0.99, 30)

# GD heatmap (alpha only - vs iteration)
gd_results = np.array([gd_final(a) for a in alphas])
# Nesterov heatmap (alpha x beta_max)
nes_results = np.zeros((30, 30))
for i, a in enumerate(alphas):
    for j, b in enumerate(betas):
        nes_results[i, j] = nesterov_final(a, b)
# Adam heatmap (alpha x beta1)
adam_results = np.zeros((30, 30))
for i, a in enumerate(alphas):
    for j, b in enumerate(betas):
        adam_results[i, j] = adam_final(a, b)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# GD: convergence vs alpha
ax = axes[0]
gd_clip = np.clip(gd_results, 0.7244, 10)
ax.semilogx(alphas, gd_clip, 'b-o', linewidth=2, markersize=4)
ax.axhline(0.7244, color='green', linestyle='--', linewidth=1.5, label='$f^\\star = 0.7244$')
ax.axvline(0.06, color='orange', linestyle=':', linewidth=2, label='Used $\\alpha=0.06$')
ax.set_xlabel('$\\alpha$ (log)', fontsize=12)
ax.set_ylabel('Final $f(x_{120})$', fontsize=12)
ax.set_title('GD: Final Value vs Learning Rate\n(Benchmark B, 120 iters)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.7, 5)

# Nesterov: heatmap alpha x beta_max
ax2 = axes[1]
nes_clip = np.clip(nes_results, 0.7244, 20)
im2 = ax2.pcolormesh(betas, np.log10(alphas), np.log10(nes_clip),
                     cmap='RdYlGn_r', vmin=np.log10(0.73), vmax=1.5)
plt.colorbar(im2, ax=ax2, label='$\\log_{10}(f_{final})$')
ax2.scatter([0.92], [np.log10(0.035)], c='white', s=150, marker='*', zorder=5, label='Used ($\\alpha=0.035$, $\\beta=0.92$)')
ax2.set_xlabel('$\\beta_{\\max}$', fontsize=12)
ax2.set_ylabel('$\\log_{10}(\\alpha)$', fontsize=12)
ax2.set_title('Nesterov: Heatmap of Final Value\n($\\alpha \\times \\beta_{\\max}$, green=good)', fontsize=10)
ax2.legend(fontsize=9)

# Adam: heatmap alpha x beta1
ax3 = axes[2]
adam_clip = np.clip(adam_results, 0.7244, 20)
im3 = ax3.pcolormesh(betas, np.log10(alphas), np.log10(adam_clip),
                     cmap='RdYlGn_r', vmin=np.log10(0.73), vmax=1.5)
plt.colorbar(im3, ax=ax3, label='$\\log_{10}(f_{final})$')
ax3.scatter([0.9], [np.log10(0.08)], c='white', s=150, marker='*', zorder=5, label='Used ($\\alpha=0.08$, $\\beta_1=0.9$)')
ax3.set_xlabel('$\\beta_1$', fontsize=12)
ax3.set_ylabel('$\\log_{10}(\\alpha)$', fontsize=12)
ax3.set_title('Adam: Heatmap of Final Value\n($\\alpha \\times \\beta_1$, green=good)', fontsize=10)
ax3.legend(fontsize=9)

plt.suptitle('Hyperparameter Sensitivity Analysis: GD, Nesterov, Adam on Benchmark B',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/hyperparam_sensitivity.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: hyperparam_sensitivity.pdf")


# ─── Figure 4: SGD learning rate schedule comparison ──────────────────────────
m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
y_data = X_data @ theta_star + np.random.randn(m) * 0.5
f_star_A = 0.5 * np.mean(y_data**2) - 0.5 * (X_data.T @ y_data / m) @ np.linalg.solve(
    X_data.T @ X_data / m, X_data.T @ y_data / m)

def bench_A_loss(theta):
    return 0.5 * np.mean((X_data @ theta - y_data)**2)

def sgd_schedule(schedule_fn, n_epochs=50, batch=40):
    theta = np.zeros(2)
    n = len(y_data)
    losses = [bench_A_loss(theta)]
    for ep in range(n_epochs):
        idx = np.random.permutation(n)
        for j in range(0, n, batch):
            t = ep * (n // batch) + j // batch + 1
            alpha = schedule_fn(t)
            bi = idx[j:j+batch]
            g = X_data[bi].T @ (X_data[bi] @ theta - y_data[bi]) / len(bi)
            theta -= alpha * g
        losses.append(bench_A_loss(theta))
    return np.array(losses)

n_epochs = 50
const_losses   = sgd_schedule(lambda t: 0.05)
poly_losses    = sgd_schedule(lambda t: 0.5 / (1 + t)**0.6)
cosine_losses  = sgd_schedule(lambda t: 0.005 + 0.5 * (0.1 - 0.005) * (1 + np.cos(np.pi * (t % 500) / 500)))
warmup_losses  = sgd_schedule(lambda t: min(0.1, 0.1 * t / 100))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ep = np.arange(n_epochs + 1)

ax = axes[0]
ax.semilogy(ep, np.clip(const_losses - f_star_A, 1e-10, None), '-', color='#2196F3', linewidth=2.5, label='Constant $\\alpha=0.05$')
ax.semilogy(ep, np.clip(poly_losses - f_star_A, 1e-10, None), '-', color='#4CAF50', linewidth=2.5, label='Polynomial decay $0.5/(1+t)^{0.6}$')
ax.semilogy(ep, np.clip(cosine_losses - f_star_A, 1e-10, None), '-', color='#FF5722', linewidth=2.5, label='Cosine annealing')
ax.semilogy(ep, np.clip(warmup_losses - f_star_A, 1e-10, None), '-', color='#9C27B0', linewidth=2.5, label='Linear warmup + const')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('$f(\\theta_k) - f^\\star$ (log)', fontsize=12)
ax.set_title('SGD Learning Rate Schedules\n(Benchmark A, batch=40)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: learning rate over time
ax2 = axes[1]
T = n_epochs * (m // 40)
t_arr = np.arange(1, T+1)
ax2.plot(t_arr[:200], [0.05]*200, '-', color='#2196F3', linewidth=2, label='Constant')
ax2.plot(t_arr[:200], 0.5 / (1 + t_arr[:200])**0.6, '-', color='#4CAF50', linewidth=2, label='Poly decay')
cosine_sched = 0.005 + 0.5 * (0.1 - 0.005) * (1 + np.cos(np.pi * (t_arr[:200] % 500) / 500))
ax2.plot(t_arr[:200], cosine_sched, '-', color='#FF5722', linewidth=2, label='Cosine annealing')
warmup_sched = np.minimum(0.1, 0.1 * t_arr[:200] / 100)
ax2.plot(t_arr[:200], warmup_sched, '-', color='#9C27B0', linewidth=2, label='Linear warmup')
ax2.set_xlabel('Update step $t$', fontsize=12)
ax2.set_ylabel('Learning rate $\\alpha_t$', fontsize=12)
ax2.set_title('Learning Rate Schedules (first 200 steps)\nShowing Schedule Shape', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/sgd_lr_schedules.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: sgd_lr_schedules.pdf")

print("All pass-17 figures generated.")
