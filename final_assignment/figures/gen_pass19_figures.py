#!/usr/bin/env python3
"""Pass 19 figures: random search variance analysis, KKT dual function,
Newton convergence phases, mini-batch gradient variance."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(42)

# ─── Benchmark functions ─────────────────────────────────────────────────────
def bench_B_loss(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])

def bench_B_grad(x):
    return np.array([2*(x[0]-1) + np.cos(x[0]), 10*(x[1]-2)])

f_star_B = 0.7244

# ─── Figure 1: Random search variance analysis ──────────────────────────────
# Compare FD vs random search gradient estimates: variance vs #evaluations

x0 = np.array([0.5, 2.5])
g_true = bench_B_grad(x0)
delta_fd = 0.05
delta_rs = 0.05

n_trials = 500

# FD forward: uses d+1=3 evaluations, essentially deterministic
# (numerical, not stochastic; so variance comes from rounding only)
grad_fds = []
f0 = bench_B_loss(x0)
for _ in range(n_trials):
    g = np.zeros(2)
    for i in range(2):
        ei = np.zeros(2); ei[i] = 1.0
        g[i] = (bench_B_loss(x0 + delta_fd*ei) - f0) / delta_fd
    grad_fds.append(g)
grad_fds = np.array(grad_fds)

# Nesterov random search: uses 2 evaluations, stochastic
grad_rs_1shot = []  # single draw
for _ in range(n_trials):
    u = np.random.randn(2); u /= np.linalg.norm(u)
    g_hat = (2.0 / delta_rs) * (bench_B_loss(x0 + delta_rs*u) - bench_B_loss(x0 - delta_rs*u)) * u
    grad_rs_1shot.append(g_hat)
grad_rs_1shot = np.array(grad_rs_1shot)

# Random search averaged over k draws
k_avgs = [1, 2, 4, 8, 16, 32]
rs_var_vs_k = []
for k in k_avgs:
    var_components = []
    for _ in range(200):
        g_avg = np.zeros(2)
        for _ in range(k):
            u = np.random.randn(2); u /= np.linalg.norm(u)
            g_hat = (2.0 / delta_rs) * (bench_B_loss(x0 + delta_rs*u) - bench_B_loss(x0 - delta_rs*u)) * u
            g_avg += g_hat
        g_avg /= k
        var_components.append(np.linalg.norm(g_avg - g_true)**2)
    rs_var_vs_k.append(np.mean(var_components))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
# Gradient estimate cloud
ax.scatter(grad_fds[:, 0], grad_fds[:, 1], s=10, alpha=0.5, color='#2196F3',
           label=f'FD (var$\\approx${np.var(grad_fds):.2e})')
ax.scatter(grad_rs_1shot[:, 0], grad_rs_1shot[:, 1], s=5, alpha=0.3, color='#FF5722',
           label=f'Random search (var$\\approx${np.var(grad_rs_1shot):.2f})')
ax.plot(*g_true, 'k*', markersize=15, label='True $\\nabla f$', zorder=6)
ax.set_xlabel('$\\hat{g}^{(1)}$ (first component)', fontsize=11)
ax.set_ylabel('$\\hat{g}^{(2)}$ (second component)', fontsize=11)
ax.set_title('Gradient Estimate Clouds\n(500 realisations, FD vs Random Search)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
# Variance decomposition
ax2.loglog(k_avgs, rs_var_vs_k, 'o-', color='#FF5722', linewidth=2.5, markersize=8,
           label='Random search (avg $k$ draws)')
ax2.loglog(k_avgs, np.full(len(k_avgs), np.mean(np.sum((grad_fds - g_true)**2, axis=1))),
           'b--', linewidth=2, label=f'FD error (constant ≈ {np.mean(np.sum((grad_fds - g_true)**2, axis=1)):.2e})')
# O(1/k) reference
ax2.loglog(k_avgs, rs_var_vs_k[0] / np.array(k_avgs), 'k:', linewidth=1.5, label='$O(1/k)$ reference')
ax2.set_xlabel('Number of random draws $k$', fontsize=11)
ax2.set_ylabel('$\\mathbb{E}\\|\\hat{g} - \\nabla f\\|^2$', fontsize=11)
ax2.set_title('Random Search: Variance Reduction\nby Averaging $k$ Directional Estimates', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Right: error histogram
ax3 = axes[2]
rs_errors = np.linalg.norm(grad_rs_1shot - g_true, axis=1)
fd_errors = np.linalg.norm(grad_fds - g_true, axis=1)
ax3.hist(rs_errors, bins=40, density=True, alpha=0.7, color='#FF5722', label=f'Random search ($\\mu$={np.mean(rs_errors):.2f})')
ax3.hist(fd_errors, bins=40, density=True, alpha=0.9, color='#2196F3', label=f'FD error ($\\mu$={np.mean(fd_errors):.2e})')
ax3.set_xlabel('Gradient error $\\|\\hat{g} - \\nabla f\\|$', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title('Error Distribution\n(Random search is high-variance; FD is near-deterministic)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 15)

plt.suptitle('Gradient Estimation: Finite Differences vs Nesterov Random Search', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/random_search_variance.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: random_search_variance.pdf")


# ─── Figure 2: KKT dual function visualisation ────────────────────────────────
# f(x1,x2) = (x1-1)^2 + 5*(x2-2)^2 + sin(x1), constraint x1 >= 0.5
# For active constraint case x1 >= 0.7: dual q(nu) = min_x L(x,nu)

from scipy.optimize import minimize_scalar

def dual_q(nu, constraint_lb=0.7):
    """Compute dual function q(nu) = min_x [f(x) + nu*(constraint_lb - x1)]"""
    # x2 unconstrained: x2* = 2
    # x1: solve 2(x1-1) + cos(x1) - nu = 0
    def obj_x1(x1):
        return (x1-1)**2 + np.sin(x1) + nu*(constraint_lb - x1)
    res = minimize_scalar(obj_x1, bounds=(-3, 5), method='bounded')
    return (res.fun + 5*(2-2)**2, res.x)

nu_range = np.linspace(-0.5, 1.5, 200)
q_vals = []
x1_minimisers = []
for nu in nu_range:
    q_val, x1_min = dual_q(nu, 0.7)
    q_vals.append(q_val)
    x1_minimisers.append(x1_min)
q_vals = np.array(q_vals)
x1_minimisers = np.array(x1_minimisers)

# Find dual optimum
idx_opt = np.argmax(q_vals)
nu_star = nu_range[idx_opt]
q_star = q_vals[idx_opt]

# Primal optimum value
p_star_active = (0.7 - 1)**2 + np.sin(0.7) + 5*(2-2)**2
print(f"Dual optimum: nu* = {nu_star:.3f}, q* = {q_star:.4f}, p* = {p_star_active:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.plot(nu_range, q_vals, 'b-', linewidth=2.5, label='Dual function $q(\\nu)$')
ax.axhline(p_star_active, color='green', linestyle='--', linewidth=1.5,
           label=f'Primal optimum $p^\\star = {p_star_active:.4f}$')
ax.axvline(nu_star, color='red', linestyle=':', linewidth=1.5,
           label=f'Optimal $\\nu^\\star \\approx {nu_star:.3f}$')
ax.plot(nu_star, q_star, 'r*', markersize=14, zorder=5)
ax.set_xlabel('$\\nu$ (dual variable)', fontsize=12)
ax.set_ylabel('$q(\\nu)$', fontsize=12)
ax.set_title('Lagrangian Dual Function\n(active constraint $x_1 \\geq 0.7$)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(nu_star + 0.05, q_star - 0.02, f'Strong duality:\n$q^\\star = p^\\star = {q_star:.4f}$',
        fontsize=8, color='darkgreen')

# Duality gap for varying nu
ax2 = axes[1]
duality_gap = p_star_active - q_vals  # >= 0 by weak duality
ax2.semilogy(nu_range, np.clip(duality_gap, 1e-8, None), 'b-', linewidth=2.5,
             label='Duality gap $p^\\star - q(\\nu)$')
ax2.axvline(nu_star, color='red', linestyle=':', linewidth=1.5, label=f'$\\nu^\\star \\approx {nu_star:.3f}$')
ax2.set_xlabel('$\\nu$', fontsize=12)
ax2.set_ylabel('Duality gap (log)', fontsize=12)
ax2.set_title('Duality Gap $p^\\star - q(\\nu)$\n(Zero at $\\nu^\\star$ = strong duality)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Sensitivity / shadow price interpretation
ax3 = axes[2]
# Vary constraint bound b, plot optimal primal p*(b)
b_vals = np.linspace(0.3, 1.3, 100)
p_star_vals = []
for b in b_vals:
    # Constrained optimum for x1 >= b
    unconstrained_x1 = 0.582  # approx unconstrained min
    if b <= unconstrained_x1:
        # Constraint inactive
        p_star_vals.append(bench_B_loss(np.array([unconstrained_x1, 2.0])))
    else:
        # Constraint active at x1 = b
        p_star_vals.append((b-1)**2 + np.sin(b) + 5*(2-2)**2)

ax3.plot(b_vals, p_star_vals, 'b-', linewidth=2.5, label='Optimal value $p^\\star(b)$')
ax3.axvline(0.582, color='gray', linestyle='--', alpha=0.7, label='Unconstrained $x_1^\\star \\approx 0.582$')
ax3.axvline(0.7, color='red', linestyle=':', label='$b = 0.7$ (active case)')
# Shadow price annotation
b_act = 0.7
slope = (p_star_vals[int(np.searchsorted(b_vals, 0.71))] -
         p_star_vals[int(np.searchsorted(b_vals, 0.69))]) / 0.02
ax3.annotate(f'Shadow price\n$\\partial p^\\star/\\partial b \\approx {slope:.3f}$\n$= \\nu^\\star \\approx {nu_star:.3f}$',
             xy=(0.7, (0.7-1)**2 + np.sin(0.7)),
             xytext=(0.85, 0.4),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=9)
ax3.set_xlabel('Constraint bound $b$ ($x_1 \\geq b$)', fontsize=12)
ax3.set_ylabel('Optimal primal value $p^\\star(b)$', fontsize=12)
ax3.set_title('Sensitivity Analysis: $p^\\star(b)$ vs Constraint Bound\n(Shadow price = KKT multiplier $\\nu^\\star$)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.suptitle('Lagrangian Duality and Sensitivity Analysis (Q5: $x_1 \\geq b$)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/kkt_dual_function.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: kkt_dual_function.pdf")


# ─── Figure 3: Newton convergence phases (damped vs quadratic) ────────────────
# Run Newton on Benchmark B, showing the two phases

def newton_phases(x0, alpha=1.0, damping=1e-8, n=40):
    x = x0.copy().astype(float)
    history = {'x': [x.copy()], 'f': [bench_B_loss(x)], 'decrement': []}
    for _ in range(n):
        g = bench_B_grad(x)
        H = np.array([[2 - np.sin(x[0]), 0], [0, 10]])  # Hessian of bench_B
        H_reg = H + damping * np.eye(2)
        p = np.linalg.solve(H_reg, g)
        lam_sq = g @ np.linalg.solve(H_reg, g)  # Newton decrement squared
        history['decrement'].append(np.sqrt(lam_sq))

        # Damped step if decrement is large
        if np.sqrt(lam_sq) >= 0.25:
            step = 1.0 / (1 + np.sqrt(lam_sq))
        else:
            step = alpha  # full step in quadratic phase
        x = x - step * p
        history['x'].append(x.copy())
        history['f'].append(bench_B_loss(x))
    return history

x0_B = np.array([-2.0, 5.0])
hist = newton_phases(x0_B)
f_vals = np.array(hist['f'])
decrements = np.array(hist['decrement'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
k_arr = np.arange(len(f_vals))
ax.semilogy(k_arr, np.clip(f_vals - f_star_B, 1e-14, None), 'b-o', linewidth=2, markersize=5)
# Mark phase transition
phase_change = np.argmax(decrements < 0.25)
if phase_change > 0:
    ax.axvline(phase_change, color='red', linestyle='--', linewidth=1.5,
               label=f'Phase change at $k={phase_change}$\n($\\lambda={decrements[phase_change]:.4f} < 0.25$)')
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('$f(x_k) - f^\\star$ (log)', fontsize=12)
ax.set_title("Newton's Method: Damped then Quadratic Phase\n(Benchmark B, start $(-2,5)$)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.semilogy(np.arange(len(decrements)), decrements, 'r-o', linewidth=2, markersize=5,
             label='Newton decrement $\\lambda(x_k)$')
ax2.axhline(0.25, color='gray', linestyle='--', linewidth=1.5, label='Quadratic phase threshold $1/4$')
if phase_change > 0:
    ax2.axvline(phase_change, color='red', linestyle=':', linewidth=1.5)
ax2.set_xlabel('Iteration $k$', fontsize=12)
ax2.set_ylabel('Newton decrement $\\lambda(x_k)$', fontsize=12)
ax2.set_title('Newton Decrement: Damped Phase ($\\lambda \\geq 1/4$)\nvs Quadratic Phase ($\\lambda < 1/4$)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Right: error doublings in quadratic phase
ax3 = axes[2]
errs = np.clip(f_vals - f_star_B, 1e-14, None)
k_quad = phase_change if phase_change > 0 else 5
if k_quad < len(errs) - 3:
    quad_errs = errs[k_quad:]
    k_q = np.arange(len(quad_errs))
    ax3.semilogy(k_q, quad_errs, 'b-o', linewidth=2, markersize=6, label='Residual (quadratic phase)')
    # Fit quadratic convergence: e_{k+1} ~ C * e_k^2
    if len(quad_errs) > 2 and quad_errs[0] > 1e-12:
        C_est = quad_errs[1] / quad_errs[0]**2 if quad_errs[0] > 0 else 1
        quadratic_fit = quad_errs[0] * (C_est * quad_errs[0])**(2**k_q - 1)
        quadratic_fit = np.clip(quadratic_fit, 1e-14, None)
        ax3.semilogy(k_q[:min(6, len(k_q))], quadratic_fit[:min(6, len(k_q))],
                     'r--', linewidth=1.5, label='Quadratic fit $C^{2^k-1}e_0$')
ax3.set_xlabel('Iteration within quadratic phase', fontsize=12)
ax3.set_ylabel('$f(x_k) - f^\\star$ (log)', fontsize=12)
ax3.set_title("Quadratic Phase Detail\n(Digits of accuracy double each iteration)", fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.suptitle("Newton's Method: Two-Phase Convergence (Damped then Quadratic)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/newton_phases.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: newton_phases.pdf")


# ─── Figure 4: Batch size and gradient noise floor ────────────────────────────
m = 2000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
y_data = X_data @ theta_star + np.random.randn(m) * 0.5

def bench_A_loss(theta):
    return 0.5 * np.mean((X_data @ theta - y_data)**2)

H_mat = X_data.T @ X_data / m
b_vec = X_data.T @ y_data / m
f_star_A = 0.5 * np.mean(y_data**2) - 0.5 * b_vec @ np.linalg.solve(H_mat, b_vec)

def sgd_noise_floor(batch, alpha=0.05, n_epochs=60):
    n = len(y_data)
    theta = np.zeros(2)
    f_hist = [bench_A_loss(theta)]
    for ep in range(n_epochs):
        idx = np.random.permutation(n)
        for j in range(0, n, batch):
            bi = idx[j:j+batch]
            g = X_data[bi].T @ (X_data[bi] @ theta - y_data[bi]) / len(bi)
            theta -= alpha * g
        f_hist.append(bench_A_loss(theta))
    return np.array(f_hist)

batch_sizes = [1, 5, 20, 100, 500, m]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(batch_sizes)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ep = np.arange(61)

ax = axes[0]
for bs, col in zip(batch_sizes, colors):
    lbl = 'Full batch (GD)' if bs == m else f'batch={bs}'
    loss_hist = sgd_noise_floor(bs)
    ax.semilogy(ep, np.clip(loss_hist - f_star_A, 1e-12, None), color=col, linewidth=2,
                label=lbl)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('$f(\\theta_k) - f^\\star$ (log)', fontsize=12)
ax.set_title('SGD Noise Floor vs Batch Size\n(Benchmark A, $\\alpha=0.05$, 60 epochs)', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: noise floor (epoch 50-60 average) vs batch size
ax2 = axes[1]
floors = []
for bs in batch_sizes:
    losses = sgd_noise_floor(bs)
    floors.append(np.mean(losses[-10:] - f_star_A))
# Theoretical noise floor: alpha * sigma^2 / (2 * mu * b)
# sigma^2 ~ 0.5^2 = 0.25 (noise in y), L ~ 1, mu ~ 0.5
sigma2_est = 0.25
mu_est = 0.5
theoretical_floor = [0.05 * sigma2_est / (2 * mu_est * b) for b in batch_sizes]

ax2.loglog(batch_sizes, floors, 'b-o', linewidth=2.5, markersize=8, label='Empirical noise floor')
ax2.loglog(batch_sizes, theoretical_floor, 'r--', linewidth=2, label='Theory: $\\alpha\\sigma^2/(2\\mu b)$')
ax2.set_xlabel('Batch size $b$', fontsize=12)
ax2.set_ylabel('Noise floor $\\mathbb{E}[f(\\theta_k)] - f^\\star$', fontsize=12)
ax2.set_title('Noise Floor vs Batch Size\n($O(1/b)$ decay confirmed)', fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle('Mini-Batch SGD: Noise Floor Analysis (Benchmark A)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/sgd_noise_floor.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: sgd_noise_floor.pdf")

print("All pass-19 figures generated.")
