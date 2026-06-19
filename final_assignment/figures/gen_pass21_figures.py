#!/usr/bin/env python3
"""Pass 21 figures: convergence rate hierarchy chart, FISTA vs ISTA vs GD, ADMM residuals."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

np.random.seed(42)

# ─── Figure 1: Convergence rate hierarchy visualization ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: convergence curves for each algorithm family
k = np.arange(1, 301)
kappa = 20    # moderate condition number
n = 100       # finite-sum count
G = 1.0; R = 1.0  # subgradient bound

# Rates (illustrative, scaled to start at 1.0)
rate_subgrad     = (G * R) / np.sqrt(k)           # O(1/sqrt(k)) non-smooth
rate_gd          = 1.0 / k                          # O(1/k) smooth convex GD
rate_nesterov    = 1.0 / k**2                       # O(1/k^2) accelerated
rate_gd_sc       = (1 - 1/kappa)**k                 # linear O((1-1/kappa)^k)
rate_nesterov_sc = (1 - 1/np.sqrt(kappa))**k        # linear O((1-1/sqrt(kappa))^k)
rate_newton      = 0.5 ** (2**np.minimum(k, 10) / 2**10)  # super-linear approx

ax = axes[0]
ax.semilogy(k, rate_subgrad,     color='#9E9E9E', linewidth=2,   linestyle='--',  label='Subgradient  $O(1/\\sqrt{k})$', alpha=0.9)
ax.semilogy(k, rate_gd,          color='#2196F3', linewidth=2,   linestyle='-',   label='GD (convex)  $O(1/k)$', alpha=0.9)
ax.semilogy(k, rate_nesterov,    color='#4CAF50', linewidth=2.5, linestyle='-',   label='Nesterov (convex)  $O(1/k^2)$', alpha=0.9)
ax.semilogy(k, rate_gd_sc,       color='#FF9800', linewidth=2,   linestyle='--',  label=f'GD (str.\ conv., $\\kappa={kappa}$)  $O(\\rho^k)$', alpha=0.9)
ax.semilogy(k, rate_nesterov_sc, color='#F44336', linewidth=2.5, linestyle='-',   label=f'Nesterov (str.\ conv.)  $O(\\rho^k)$, better $\\rho$', alpha=0.9)
ax.semilogy(k[:15], rate_newton[:15], color='#000000', linewidth=2, marker='o', markersize=5, label='Newton  $O(\\log\\log 1/\\varepsilon)$', alpha=0.9)

ax.set_xlim(1, 300)
ax.set_ylim(1e-8, 2)
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('Optimality gap $f(x_k) - f^\\star$ (log scale)', fontsize=12)
ax.set_title('Convergence Rate Hierarchy\n(illustrative, normalised to start at 1)', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

# Add arrows indicating "better" direction
ax.annotate('faster', xy=(250, 2e-8), xytext=(180, 2e-7),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            fontsize=9, color='gray')

# Right: taxonomy chart as a table/grid
ax2 = axes[1]
ax2.axis('off')

rows = [
    # (Method, Info type, Rate, Cost/iter, Notes)
    ['Subgradient', 'Non-smooth', r'$O(1/\sqrt{k})$', r'$O(d)$', 'Optimal non-smooth'],
    ['GD (convex)', 'Smooth', r'$O(1/k)$', r'$O(d)$', '—'],
    ['Proj. GD', 'Smooth+constrained', r'$O(1/k)$', r'$O(d)+$proj', '—'],
    ['ISTA', 'Composite', r'$O(1/k)$', r'$O(d)+$prox', r'$f+\psi$'],
    ['Frank–Wolfe', 'Smooth+constrained', r'$O(1/k)$', r'$O(d)+$LMO', 'Projection-free'],
    ['Nesterov', 'Smooth', r'$O(1/k^2)$', r'$O(d)$', 'Optimal smooth'],
    ['FISTA', 'Composite', r'$O(1/k^2)$', r'$O(d)+$prox', r'Accelerated ISTA'],
    ['GD (str. conv.)', 'Smooth+s.c.', r'$O(\kappa\log 1/\varepsilon)$', r'$O(d)$', 'Linear'],
    ['Nesterov (s.c.)', 'Smooth+s.c.', r'$O(\sqrt{\kappa}\log 1/\varepsilon)$', r'$O(d)$', 'Optimal s.c.'],
    ['SVRG/SAGA', 'Finite-sum+s.c.', r'$O((n+\kappa)\log 1/\varepsilon)$', r'$O(b)$', 'VR linear'],
    ['Newton', 'Smooth+s.c.', r'$O(\log\log 1/\varepsilon)$', r'$O(d^3)$', 'Super-linear'],
    ['L-BFGS', 'Smooth+s.c.', 'super-linear (heuristic)', r'$O(md)$', 'Quasi-Newton'],
]

col_labels = ['Method', 'Setting', 'Rate', 'Cost/iter', 'Notes']
col_widths = [0.22, 0.22, 0.20, 0.12, 0.22]

# Draw header
y_start = 0.98
y_step = 0.073
x_starts = [0.01, 0.23, 0.45, 0.65, 0.77]

for ci, (label, xst) in enumerate(zip(col_labels, x_starts)):
    ax2.text(xst, y_start, label, fontsize=9, fontweight='bold',
             transform=ax2.transAxes, va='top')

ax2.axhline(y_start - 0.015, color='k', linewidth=1.2,
            xmin=0.01, xmax=0.99)

# Color rows by rate class
row_colors = {
    r'$O(1/\sqrt{k})$': '#FFE0B2',
    r'$O(1/k)$': '#E3F2FD',
    r'$O(1/k^2)$': '#E8F5E9',
    r'$O(\kappa\log 1/\varepsilon)$': '#FFF9C4',
    r'$O(\sqrt{\kappa}\log 1/\varepsilon)$': '#F3E5F5',
    r'$O((n+\kappa)\log 1/\varepsilon)$': '#F3E5F5',
    r'$O(\log\log 1/\varepsilon)$': '#FCE4EC',
    'super-linear (heuristic)': '#FCE4EC',
}

for ri, row in enumerate(rows):
    y = y_start - 0.03 - ri * y_step
    rate_key = row[2]
    bg = row_colors.get(rate_key, '#FAFAFA')
    rect = FancyBboxPatch((0.005, y - 0.055), 0.99, y_step,
                          boxstyle='round,pad=0.005', linewidth=0,
                          facecolor=bg, transform=ax2.transAxes, zorder=0)
    ax2.add_patch(rect)
    for ci, (cell, xst) in enumerate(zip(row, x_starts)):
        ax2.text(xst, y, cell, fontsize=8.5,
                 transform=ax2.transAxes, va='top')

ax2.set_title('Algorithm Taxonomy: Rate, Cost, and Setting', fontsize=11)

plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/convergence_hierarchy.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: convergence_hierarchy.pdf")


# ─── Figure 2: ISTA vs FISTA on composite Lasso-type objective ────────────────
def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

def run_ista(A, b, lam, n, L):
    """ISTA for min 0.5||Ax-b||^2 + lam||x||_1"""
    m, d = A.shape
    x = np.zeros(d)
    hist = []
    for _ in range(n):
        grad = A.T @ (A @ x - b)
        x = soft_threshold(x - grad / L, lam / L)
        hist.append(0.5 * np.linalg.norm(A @ x - b)**2 + lam * np.sum(np.abs(x)))
    return np.array(hist)

def run_fista(A, b, lam, n, L):
    """FISTA for min 0.5||Ax-b||^2 + lam||x||_1"""
    m, d = A.shape
    x = np.zeros(d); y = x.copy(); t = 1.0
    hist = []
    for _ in range(n):
        grad = A.T @ (A @ y - b)
        x_new = soft_threshold(y - grad / L, lam / L)
        t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
        y = x_new + (t - 1)/t_new * (x_new - x)
        x, t = x_new, t_new
        hist.append(0.5 * np.linalg.norm(A @ x - b)**2 + lam * np.sum(np.abs(x)))
    return np.array(hist)

def run_proxgd(A, b, lam, n, L):
    """Proximal GD with backtracking line search"""
    m, d = A.shape
    x = np.zeros(d)
    hist = []
    for _ in range(n):
        grad = A.T @ (A @ x - b)
        x = soft_threshold(x - grad / L, lam / L)
        hist.append(0.5 * np.linalg.norm(A @ x - b)**2 + lam * np.sum(np.abs(x)))
    return np.array(hist)

np.random.seed(123)
m, d = 80, 50
A = np.random.randn(m, d) / np.sqrt(m)
x_true = np.zeros(d); x_true[:8] = np.random.randn(8)  # sparse ground truth
b = A @ x_true + 0.05 * np.random.randn(m)
L = np.linalg.norm(A.T @ A, ord=2)  # Lipschitz constant
lam = 0.02
n_iters = 200

hist_ista  = run_ista(A, b, lam, n_iters, L)
hist_fista = run_fista(A, b, lam, n_iters, L)

f_star = min(hist_fista.min(), hist_ista.min())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

k_arr = np.arange(1, n_iters + 1)
ax = axes[0]
ax.semilogy(k_arr, hist_ista  - f_star, color='#2196F3', linewidth=2, label='ISTA (Proximal GD,  $O(1/k)$)')
ax.semilogy(k_arr, hist_fista - f_star, color='#4CAF50', linewidth=2.5, label='FISTA (Accelerated, $O(1/k^2)$)')

# Theoretical envelopes
C = hist_ista[0] - f_star
ax.semilogy(k_arr, C / k_arr,    'b--', linewidth=1.2, alpha=0.5, label='$O(1/k)$ envelope')
ax.semilogy(k_arr, C / k_arr**2, 'g--', linewidth=1.2, alpha=0.5, label='$O(1/k^2)$ envelope')

ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('$F(x_k) - F^\\star$ (log scale)', fontsize=12)
ax.set_title('ISTA vs FISTA on Lasso-type Objective\n$F(x) = \\frac{1}{2}\\|Ax-b\\|^2 + \\lambda\\|x\\|_1$', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: solution sparsity comparison
ax2 = axes[1]

# Run until near-convergence
hist_ista_long  = run_ista(A, b, lam, 500, L)
hist_fista_long = run_fista(A, b, lam, 500, L)

# Get final iterates by re-running (simpler approach: just show the gap curves)
k_long = np.arange(1, 501)
ax2.semilogy(k_long, hist_ista_long  - f_star, color='#2196F3', linewidth=2, label='ISTA')
ax2.semilogy(k_long, hist_fista_long - f_star, color='#4CAF50', linewidth=2.5, label='FISTA')

# Add isoaccuracy markers
target_acc = 1e-3
for label, hist, col in [('ISTA', hist_ista_long, '#2196F3'), ('FISTA', hist_fista_long, '#4CAF50')]:
    mask = (hist - f_star) < target_acc
    if mask.any():
        k_hit = np.argmax(mask) + 1
        ax2.axvline(k_hit, color=col, linestyle=':', alpha=0.7)
        ax2.text(k_hit + 5, 3e-3, f'{label}: $k={k_hit}$', color=col, fontsize=9)

ax2.axhline(target_acc, color='gray', linestyle='--', alpha=0.5, label=f'$\\varepsilon={target_acc}$ target')
ax2.set_xlabel('Iteration $k$', fontsize=12)
ax2.set_ylabel('$F(x_k) - F^\\star$ (log scale)', fontsize=12)
ax2.set_title('ISTA vs FISTA: Iterations to $\\varepsilon$-Accuracy\n(500 iterations, same Lasso problem)', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('Proximal Gradient Methods: ISTA vs FISTA', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/ista_vs_fista.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: ista_vs_fista.pdf")


# ─── Figure 3: ADMM primal/dual residuals and augmented Lagrangian convergence ─
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ADMM on: min (x-1)^2 + (z-5)^2  s.t. x = z, x >= 0.5  (z in [0.5, inf))
# Decompose: f(x) = (x-1)^2, g(z) = (z-5)^2 + indicator_{z>=0.5}
# x-update: min (x-1)^2 + rho/2 ||x - z + u||^2  -> x* = (1 + rho*(z-u))/(1+rho)
# z-update: min (z-5)^2 + rho/2 ||x - z + u||^2 + indicator_{z>=0.5}
#         -> z* = max(0.5, (5 + rho*(x+u))/(1+rho))
# u-update: u <- u + x - z

rho_vals = [0.5, 2.0, 8.0]
colors   = ['#2196F3', '#4CAF50', '#F44336']
n_admm   = 60

for rho, col in zip(rho_vals, colors):
    x = 3.0; z = 3.0; u = 0.0
    prim_res, dual_res = [], []
    f_vals = []
    for _ in range(n_admm):
        x_new = (1 + rho*(z - u)) / (1 + rho)
        z_new = max(0.5, (5 + rho*(x_new + u)) / (1 + rho))
        u    += x_new - z_new
        prim_res.append(abs(x_new - z_new))
        dual_res.append(rho * abs(z_new - z))
        f_vals.append((x_new - 1)**2 + (z_new - 5)**2)
        x, z = x_new, z_new

    k_admm = np.arange(1, n_admm + 1)
    axes[0].semilogy(k_admm, prim_res, color=col, linewidth=2, label=f'$\\rho={rho}$')
    axes[1].semilogy(k_admm, dual_res, color=col, linewidth=2, label=f'$\\rho={rho}$')
    axes[2].semilogy(k_admm, f_vals,   color=col, linewidth=2, label=f'$\\rho={rho}$')

axes[0].set_xlabel('Iteration $k$', fontsize=11)
axes[0].set_ylabel('Primal residual $\\|x_k - z_k\\|$', fontsize=11)
axes[0].set_title('ADMM Primal Residual\nvs Penalty $\\rho$', fontsize=10)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Iteration $k$', fontsize=11)
axes[1].set_ylabel('Dual residual $\\rho\\|z_k - z_{k-1}\\|$', fontsize=11)
axes[1].set_title('ADMM Dual Residual\nvs Penalty $\\rho$', fontsize=10)
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

axes[2].axhline((1.0 - 1)**2 + (5.0 - 5)**2, color='gray', linestyle='--', label='$f^\\star = 0$', alpha=0.7)
axes[2].set_xlabel('Iteration $k$', fontsize=11)
axes[2].set_ylabel('$f(x_k, z_k) = (x_k-1)^2 + (z_k-5)^2$', fontsize=11)
axes[2].set_title('ADMM Objective vs Penalty $\\rho$\n(true min $f^\\star = 0$ at $(1, 5)$)', fontsize=10)
axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

plt.suptitle('ADMM Convergence: Primal/Dual Residuals and Objective\n'
             r'$\min_x (x-1)^2 + (z-5)^2$ s.t. $x = z,\; z \geq 0.5$',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/admm_residuals.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: admm_residuals.pdf")

print("All pass-21 figures generated.")
