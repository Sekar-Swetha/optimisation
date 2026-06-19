#!/usr/bin/env python3
"""Pass 6/7 figures: Nelder-Mead simplex evolution, SGD variance comparison, penalty ALM comparison."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

np.random.seed(42)

# ─── Benchmark functions ─────────────────────────────────────────────────────

def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

# ─── Figure 1: Nelder-Mead simplex shape evolution ────────────────────────────

def nelder_mead_with_history(f, x0, step=0.35, n_iters=160):
    n = len(x0)
    simplex = np.vstack([x0] + [x0 + step * np.eye(n)[i] for i in range(n)])
    simplex = simplex.astype(float)
    alpha_r, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    simplex_history = [simplex.copy()]
    best_history = [min(f(v) for v in simplex)]

    for _ in range(n_iters):
        fvals = np.array([f(v) for v in simplex])
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + alpha_r * (centroid - simplex[-1])
        fr = f(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
        elif fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            simplex[-1] = xe if f(xe) < fr else xr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            if f(xc) < fvals[-1]:
                simplex[-1] = xc
            else:
                best = simplex[0].copy()
                simplex = np.array([best + sigma*(v - best) for v in simplex])
        simplex_history.append(simplex.copy())
        best_history.append(f(simplex[np.argmin([f(v) for v in simplex])]))

    return simplex_history, best_history


x0_C = np.array([-1.0, 1.0])
simplex_history, best_history = nelder_mead_with_history(rosenbrock, x0_C)

# Contour grid
xg = np.linspace(-1.6, 1.6, 300)
yg = np.linspace(-0.3, 1.9, 300)
Xg, Yg = np.meshgrid(xg, yg)
Zg = (1 - Xg)**2 + 100*(Yg - Xg**2)**2

iters_show = [0, 5, 15, 40, 80, 159]
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
legend_handles = [
    Line2D([0],[0], color='red', lw=2, label='Simplex edge'),
    Line2D([0],[0], marker='*', color='b', markersize=10, linestyle='None', label='Centroid'),
    Line2D([0],[0], marker='*', color='lime', markersize=12, linestyle='None', label='True opt $(1,1)$'),
]
for plot_i, it in enumerate(iters_show):
    ax = axes[plot_i // 3][plot_i % 3]
    ax.contourf(Xg, Yg, np.log10(np.clip(Zg, 1e-6, None)), levels=25, cmap='Blues', alpha=0.35)
    ax.contour(Xg, Yg, np.log10(np.clip(Zg, 1e-6, None)), levels=25, colors='slategray', alpha=0.25, linewidths=0.4)
    simp = simplex_history[min(it, len(simplex_history)-1)]
    tri = Polygon(simp, closed=True, facecolor='orange', edgecolor='red', alpha=0.50, linewidth=1.8, zorder=4)
    ax.add_patch(tri)
    ax.plot(simp[:, 0], simp[:, 1], 'r.', markersize=7, zorder=5)
    centroid = simp.mean(axis=0)
    ax.plot(centroid[0], centroid[1], 'b*', markersize=9, zorder=5)
    ax.plot(1, 1, '*', color='lime', markersize=11, zorder=6, markeredgecolor='darkgreen', markeredgewidth=0.5)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-0.3, 1.9)
    f_best = best_history[min(it, len(best_history)-1)]
    ax.set_title(f'Iteration {it},  $f = {f_best:.4f}$', fontsize=10)
    ax.set_xlabel('$x_1$', fontsize=9); ax.set_ylabel('$x_2$', fontsize=9)
    ax.tick_params(labelsize=8)
    if plot_i == 0:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=7.5)

fig.suptitle('Nelder--Mead Simplex Shape Evolution on Rosenbrock', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/nelder_mead_simplex_evolution.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: nelder_mead_simplex_evolution.pdf")


# ─── Figure 2: SGD vs variance-reduced SGD (SVRG proxy) ──────────────────────
# Simulate on Benchmark A

m_data = 1000
np.random.seed(42)
X_data = np.random.randn(m_data, 2)
theta_star_true = np.array([3.0, 4.0])
y_data = X_data @ theta_star_true + np.random.randn(m_data)

H_mat = X_data.T @ X_data / m_data
b_vec = X_data.T @ y_data / m_data
f_star_A = 0.5 * y_data @ y_data / m_data - 0.5 * b_vec @ np.linalg.solve(H_mat, b_vec)

def loss_A(theta):
    r = X_data @ theta - y_data
    return 0.5 * np.mean(r**2)

def full_grad(theta):
    return H_mat @ theta - b_vec

def sample_grad(theta, idx):
    xi = X_data[idx]
    yi = y_data[idx]
    return xi * (xi @ theta - yi)

# Mini-batch SGD
def sgd_run(theta0, lr, batch_size, n_epochs):
    theta = theta0.copy()
    losses = [loss_A(theta)]
    for _ in range(n_epochs):
        perm = np.random.permutation(m_data)
        for start in range(0, m_data, batch_size):
            batch = perm[start:start+batch_size]
            xb = X_data[batch]; yb = y_data[batch]
            g = xb.T @ (xb @ theta - yb) / len(batch)
            theta -= lr * g
        losses.append(loss_A(theta))
    return np.array(losses)

# SVRG-style (variance reduced): full gradient snapshot every epoch
def svrg_run(theta0, lr, batch_size, n_epochs):
    theta = theta0.copy()
    losses = [loss_A(theta)]
    for _ in range(n_epochs):
        # Compute snapshot gradient
        mu = full_grad(theta)
        theta_snap = theta.copy()
        perm = np.random.permutation(m_data)
        for start in range(0, m_data, batch_size):
            batch = perm[start:start+batch_size]
            xb = X_data[batch]; yb = y_data[batch]
            g_curr = xb.T @ (xb @ theta - yb) / len(batch)
            g_snap = xb.T @ (xb @ theta_snap - yb) / len(batch)
            # Variance-reduced gradient estimate
            g_vr = g_curr - g_snap + mu
            theta -= lr * g_vr
        losses.append(loss_A(theta))
    return np.array(losses)

theta0_A = np.zeros(2)
n_epochs = 60
batch_size = 40
lr_sgd = 0.05
lr_svrg = 0.12  # SVRG can use larger step due to reduced variance

losses_sgd = sgd_run(theta0_A, lr_sgd, batch_size, n_epochs)
losses_svrg = svrg_run(theta0_A, lr_svrg, batch_size, n_epochs)
losses_full_gd = [loss_A(theta0_A)]
theta_gd = theta0_A.copy()
for _ in range(n_epochs):
    theta_gd -= 0.08 * full_grad(theta_gd)
    losses_full_gd.append(loss_A(theta_gd))
losses_full_gd = np.array(losses_full_gd)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
epochs = np.arange(n_epochs + 1)
ax.semilogy(epochs, losses_sgd - f_star_A, label=f'Mini-batch SGD ($b={batch_size}$, $\\alpha={lr_sgd}$)',
            color='steelblue', linewidth=2)
ax.semilogy(epochs, losses_svrg - f_star_A, label=f'SVRG proxy ($b={batch_size}$, $\\alpha={lr_svrg}$)',
            color='crimson', linewidth=2)
ax.semilogy(epochs, losses_full_gd - f_star_A, label='Full GD ($\\alpha=0.08$)',
            color='green', linewidth=2, linestyle='--')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('$f(\\theta_k) - f^\\star$ (log scale)', fontsize=12)
ax.set_title('SGD vs Variance-Reduced SGD (SVRG)\nBenchmark A: Linear Regression', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, n_epochs)

# Right: show variance reduction effect – plot gradient variance estimates over epochs
ax2 = axes[1]
n_trials = 30
batch = 40

def estimate_grad_variance(theta, n_trials=30, batch_size=40):
    grads = []
    for _ in range(n_trials):
        idx = np.random.choice(m_data, batch_size, replace=False)
        xb = X_data[idx]; yb = y_data[idx]
        g = xb.T @ (xb @ theta - yb) / batch_size
        grads.append(g)
    grads = np.array(grads)
    return float(np.mean(np.var(grads, axis=0)))

# Estimate variance at points along SGD trajectory
checkpoint_epochs = [0, 10, 20, 30, 40, 50, 60]
theta_check = theta0_A.copy()
sgd_vars, svrg_vars = [], []

theta_sgd_check = theta0_A.copy()
theta_svrg_check = theta0_A.copy()
epoch_list = []

for ep in range(n_epochs + 1):
    if ep in checkpoint_epochs:
        # SGD gradient variance at current theta
        var_sgd = estimate_grad_variance(theta_sgd_check)
        # SVRG variance is approximately reduced by the correlation with full gradient
        mu_snap = full_grad(theta_sgd_check)
        svrg_vars_local = []
        for _ in range(n_trials):
            idx = np.random.choice(m_data, batch_size, replace=False)
            xb = X_data[idx]; yb = y_data[idx]
            g_curr = xb.T @ (xb @ theta_svrg_check - yb) / batch_size
            g_snap = xb.T @ (xb @ theta_sgd_check - yb) / batch_size
            g_vr = g_curr - g_snap + mu_snap
            svrg_vars_local.append(g_vr)
        svrg_var = float(np.mean(np.var(svrg_vars_local, axis=0)))
        sgd_vars.append(var_sgd)
        svrg_vars.append(svrg_var)
        epoch_list.append(ep)
    # Advance one epoch
    if ep < n_epochs:
        perm = np.random.permutation(m_data)
        for start in range(0, m_data, batch_size):
            b_idx = perm[start:start+batch_size]
            xb = X_data[b_idx]; yb = y_data[b_idx]
            g = xb.T @ (xb @ theta_sgd_check - yb) / len(b_idx)
            theta_sgd_check -= lr_sgd * g
        # SVRG
        mu_s = full_grad(theta_svrg_check)
        theta_snap_s = theta_svrg_check.copy()
        perm2 = np.random.permutation(m_data)
        for start in range(0, m_data, batch_size):
            b_idx = perm2[start:start+batch_size]
            xb = X_data[b_idx]; yb = y_data[b_idx]
            g_c = xb.T @ (xb @ theta_svrg_check - yb) / len(b_idx)
            g_s = xb.T @ (xb @ theta_snap_s - yb) / len(b_idx)
            theta_svrg_check -= lr_svrg * (g_c - g_s + mu_s)

ax2.semilogy(epoch_list, sgd_vars, 'o-', color='steelblue', linewidth=2, markersize=7,
             label='SGD gradient variance')
ax2.semilogy(epoch_list, svrg_vars, 's-', color='crimson', linewidth=2, markersize=7,
             label='SVRG gradient variance')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Gradient variance (log scale)', fontsize=12)
ax2.set_title('Gradient Variance: SGD vs SVRG\n(Benchmark A, $b=40$)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/sgd_variance_reduction.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: sgd_variance_reduction.pdf")


# ─── Figure 3: ALM convergence vs penalty ─────────────────────────────────────
def bench_B_loss(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])

def bench_B_grad(x):
    return np.array([2*(x[0]-1) + np.cos(x[0]), 10*(x[1]-2)])

x0_constrained = np.array([0.2, 4.0])
f_star_B_constrained = 0.7244  # unconstrained min happens to be feasible

# Constraint: x1 >= 0.5 → g(x) = 0.5 - x1 <= 0
def constraint_violation(x):
    return max(0.0, 0.5 - x[0])

# Penalty method
def penalty_run(x0, alpha, lam, n_iters):
    x = x0.copy().astype(float)
    f_hist = [bench_B_loss(x)]
    viol_hist = [constraint_violation(x)]
    for _ in range(n_iters):
        g = bench_B_grad(x)
        # Penalty gradient: -lambda * e1 if x1 < 0.5
        if x[0] < 0.5:
            g = g + np.array([-lam, 0.0])
        x = x - alpha * g
        f_hist.append(bench_B_loss(x))
        viol_hist.append(constraint_violation(x))
    return np.array(f_hist), np.array(viol_hist)

# Augmented Lagrangian
def alm_run(x0, alpha, rho, n_iters, inner_iters=5):
    x = x0.copy().astype(float)
    nu = 0.0  # dual variable
    f_hist = [bench_B_loss(x)]
    viol_hist = [constraint_violation(x)]
    outer_iters = n_iters // inner_iters
    for _ in range(outer_iters):
        for _ in range(inner_iters):
            g = bench_B_grad(x)
            viol = 0.5 - x[0]
            # ALM gradient: nu * dg/dx + rho * max(0,g) * dg/dx
            dg = np.array([-1.0, 0.0])
            alm_grad = g + nu * dg + rho * max(0.0, viol) * dg
            x = x - alpha * alm_grad
            f_hist.append(bench_B_loss(x))
            viol_hist.append(constraint_violation(x))
        # Dual update
        nu = max(0.0, nu + rho * (0.5 - x[0]))
    # Pad to n_iters
    while len(f_hist) < n_iters + 1:
        f_hist.append(f_hist[-1])
        viol_hist.append(viol_hist[-1])
    return np.array(f_hist[:n_iters+1]), np.array(viol_hist[:n_iters+1])

n_iters = 100
lam_vals = [0.15, 1.8, 4.5]
colors_pen = ['#d62728', '#ff7f0e', '#9467bd']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for lam, col in zip(lam_vals, colors_pen):
    fh, vh = penalty_run(x0_constrained, 0.04, lam, n_iters)
    ax.semilogy(fh - f_star_B_constrained + 1e-10, color=col, linewidth=2, label=f'Penalty $\\lambda={lam}$')

# ALM
fh_alm, vh_alm = alm_run(x0_constrained, 0.04, rho=2.0, n_iters=n_iters)
ax.semilogy(fh_alm - f_star_B_constrained + 1e-10, color='blue', linewidth=2.5,
            linestyle='--', label='ALM ($\\rho=2.0$)')

# Projected GD
def proj_gd_run(x0, alpha, n_iters):
    x = x0.copy().astype(float)
    f_hist = [bench_B_loss(x)]
    viol_hist = [constraint_violation(x)]
    for _ in range(n_iters):
        x = x - alpha * bench_B_grad(x)
        x[0] = max(0.5, x[0])
        f_hist.append(bench_B_loss(x))
        viol_hist.append(constraint_violation(x))
    return np.array(f_hist), np.array(viol_hist)

fh_pgd, vh_pgd = proj_gd_run(x0_constrained, 0.08, n_iters)
ax.semilogy(fh_pgd - f_star_B_constrained + 1e-10, color='green', linewidth=2.5,
            linestyle=':', label='Projected GD ($\\alpha=0.08$)')

ax.axhline(1e-8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)', fontsize=12)
ax.set_title('Objective Convergence:\nPenalty vs ALM vs Projected GD', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
for lam, col in zip(lam_vals, colors_pen):
    fh, vh = penalty_run(x0_constrained, 0.04, lam, n_iters)
    valid = vh > 1e-12
    if valid.any():
        ax2.semilogy(np.where(valid, vh, np.nan), color=col, linewidth=2, label=f'Penalty $\\lambda={lam}$')
    else:
        ax2.plot([0, n_iters], [1e-12, 1e-12], color=col, linewidth=2, label=f'Penalty $\\lambda={lam}$ (zero)')

ax2.semilogy(np.where(vh_alm > 1e-12, vh_alm, np.nan), color='blue', linewidth=2.5,
             linestyle='--', label='ALM ($\\rho=2.0$)')
vh_pgd_plot = np.where(vh_pgd > 1e-12, vh_pgd, np.nan)
ax2.semilogy(vh_pgd_plot, color='green', linewidth=2.5, linestyle=':', label='Projected GD')
ax2.axhline(1e-4, color='gray', linestyle=':', linewidth=0.8, alpha=0.6, label='Tolerance $10^{-4}$')

ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Constraint violation $\\max(0, 0.5 - x_1)$ (log)', fontsize=12)
ax2.set_title('Constraint Violation:\nPenalty vs ALM vs Projected GD', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/penalty_alm_comparison.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: penalty_alm_comparison.pdf")

print("All pass-6 figures generated.")
