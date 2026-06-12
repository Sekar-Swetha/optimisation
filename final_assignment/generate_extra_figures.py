"""
Generate additional figures for the improved report.
These supplement the figures already produced by CS7DS2_Final_Code_25336453.py.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ============================================================
# Re-define benchmarks and optimisers (same as original code)
# ============================================================
m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
eps_noise = np.random.randn(m)
y_data = X_data @ theta_star + eps_noise

def loss_A(theta):
    r = X_data @ theta - y_data
    return 0.5 * np.mean(r**2)
def grad_A(theta):
    r = X_data @ theta - y_data
    return X_data.T @ r / m

def loss_B(x):
    return (x[0] - 1)**2 + 5*(x[1] - 2)**2 + np.sin(x[0])
def grad_B(x):
    return np.array([2*(x[0] - 1) + np.cos(x[0]), 10*(x[1] - 2)])

def loss_C(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
def grad_C(x):
    g1 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    g2 = 200*(x[1] - x[0]**2)
    return np.array([g1, g2])
def hessian_C(x):
    h11 = 2 + 1200*x[0]**2 - 400*x[1]
    h12 = -400*x[0]
    return np.array([[h11, h12], [h12, 200]])

theta0_A = np.array([0.0, 0.0])
x0_B = np.array([-1.0, 4.0])
x0_C = np.array([-1.0, 1.0])

def gradient_descent(grad_fn, loss_fn, x0, alpha, n_iters):
    x = x0.copy().astype(float)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for _ in range(n_iters):
        x = x - alpha * grad_fn(x)
        x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

def heavy_ball(grad_fn, loss_fn, x0, alpha, beta, n_iters):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        z = beta * z + alpha * g
        x = x - z
        x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

def nesterov_momentum(grad_fn, loss_fn, x0, alpha, beta_max, n_iters):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for k in range(1, n_iters + 1):
        beta_k = min((k - 1) / (k + 2), beta_max)
        lookahead = x + beta_k * z
        g = grad_fn(lookahead)
        z = beta_k * z - alpha * g
        x = x + z
        x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

def newtons_method(grad_fn, hess_fn, loss_fn, x0, alpha, n_iters, damping=1e-8):
    x = x0.copy().astype(float)
    x_hist, f_hist, update_norms = [x.copy()], [loss_fn(x)], []
    for _ in range(n_iters):
        g = grad_fn(x)
        H = hess_fn(x) if callable(hess_fn) else hess_fn
        H_reg = H + damping * np.eye(len(x))
        try:
            p = np.linalg.solve(H_reg, g)
        except np.linalg.LinAlgError:
            p = g
        update_norms.append(np.linalg.norm(alpha * p))
        x = x - alpha * p
        x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist), np.array(update_norms)

# ============================================================
# FIGURE 1: Nesterov vs Heavy Ball on Rosenbrock (direct comparison)
# Shows the benefit of look-ahead
# ============================================================
print("Generating q_heavy_vs_nesterov_C.pdf ...")

n_iters = 300
gd_C = gradient_descent(grad_C, loss_C, x0_C, 0.0012, n_iters)
hb_C = heavy_ball(grad_C, loss_C, x0_C, 0.0008, 0.86, n_iters)
nest_C = nesterov_momentum(grad_C, loss_C, x0_C, 0.0007, 0.90, n_iters)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.semilogy(gd_C[1], 'k--', lw=1.5, label=f'GD ($f_{{300}}={gd_C[1][-1]:.3f}$)')
ax.semilogy(hb_C[1], 'tab:red', lw=2, label=f'Heavy Ball ($f_{{300}}={hb_C[1][-1]:.3f}$)')
ax.semilogy(nest_C[1], 'tab:blue', lw=2, label=f'Nesterov ($f_{{300}}={nest_C[1][-1]:.4f}$)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Value (log scale)')
ax.set_title('Convergence on Rosenbrock: Momentum Methods Compared')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Convergence rate estimation: compute slope in log space
ax2 = axes[1]
k_range = np.arange(50, n_iters+1)
# Theoretical: GD ~linear, Nesterov ~1/k^2
ax2.plot(np.log10(k_range), np.log10(np.maximum(gd_C[1][50:], 1e-10)),
         'k--', lw=1.5, label='GD (empirical)')
ax2.plot(np.log10(k_range), np.log10(np.maximum(hb_C[1][50:], 1e-10)),
         'tab:red', lw=2, label='Heavy Ball (empirical)')
ax2.plot(np.log10(k_range), np.log10(np.maximum(nest_C[1][50:], 1e-10)),
         'tab:blue', lw=2, label='Nesterov (empirical)')
# Reference lines
ref_k = np.array([50, n_iters])
ax2.plot(np.log10(ref_k), np.log10(gd_C[1][50]) - 0.5*np.log10(ref_k/50),
         'gray', ls=':', lw=1, label='$O(1/k^{0.5})$ reference')
ax2.set_xlabel('$\\log_{10}$(Iteration)')
ax2.set_ylabel('$\\log_{10}$(Objective Value)')
ax2.set_title('Log-Log Convergence Profile on Rosenbrock')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('Heavy Ball vs Nesterov Acceleration: Momentum Method Comparison',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('figures/q_heavy_vs_nesterov_C.pdf', bbox_inches='tight')
plt.close()
print("  Saved.")

# ============================================================
# FIGURE 2: Finite difference error analysis as function of delta
# ============================================================
print("Generating q4_fd_error_analysis.pdf ...")

# For Benchmark B at a specific point
x_test = np.array([0.5, 2.5])
true_grad = grad_B(x_test)  # [2*(0.5-1)+cos(0.5), 10*(2.5-2)] = [-0.122, 5.0] approx

deltas = np.logspace(-8, 1, 100)
errors_grad = []
for delta in deltas:
    # Forward FD
    g_fd = np.zeros(2)
    f0 = loss_B(x_test)
    for i in range(2):
        ei = np.zeros(2); ei[i] = 1.0
        g_fd[i] = (loss_B(x_test + delta * ei) - f0) / delta
    errors_grad.append(np.linalg.norm(g_fd - true_grad))

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(deltas, errors_grad, 'b-', lw=2, label='FD gradient error $\\|g_{FD} - \\nabla f\\|$')
ax.axvline(x=0.05, color='green', ls='--', lw=1.5, label='$\\delta=0.05$ (good, used in Q4)')
ax.axvline(x=0.8, color='red', ls='--', lw=1.5, label='$\\delta=0.8$ (poor, used in Q4)')
ax.axvline(x=1e-8, color='purple', ls=':', lw=1.5, label='$\\delta=10^{-8}$ (theoretical optimum)')
# Theoretical error curves
d_plot = np.logspace(-8, 1, 50)
f_hess = 12.0  # rough second derivative bound
eps_mach = 2.2e-16
ax.loglog(d_plot, f_hess/2 * d_plot, 'gray', ls='--', alpha=0.7, label='$O(\\delta)$ truncation error')
ax.loglog(d_plot, eps_mach / d_plot * np.abs(loss_B(x_test)), 'orange', ls='--',
          alpha=0.7, label='$O(\\varepsilon_{mach}/\\delta)$ roundoff error')
ax.set_xlabel('Perturbation $\\delta$')
ax.set_ylabel('Gradient approximation error')
ax.set_title('Q4A: Finite Difference Error as a Function of $\\delta$\n(Benchmark B at $x=(0.5, 2.5)$)')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q4_fd_error_analysis.pdf', bbox_inches='tight')
plt.close()
print("  Saved.")

# ============================================================
# FIGURE 3: Summary heatmap of all methods across benchmarks
# ============================================================
print("Generating q_methods_summary_heatmap.pdf ...")

# Re-run key methods to get comparable results
# Q1 methods (120 iterations)
adagrad_results = {}
rmsprop_results = {}
hb_results = {}
polyak_results = {}
gd_results = {}

def adagrad_fn(grad_fn, loss_fn, x0, alpha0, eps, n_iters):
    x = x0.copy().astype(float)
    G = np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x); G += g**2
        eff_alpha = alpha0 / (np.sqrt(G) + eps)
        x = x - eff_alpha * g; f_hist.append(loss_fn(x))
    return np.array(f_hist)

def rmsprop_fn(grad_fn, loss_fn, x0, alpha0, beta, eps, n_iters):
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x); v = beta * v + (1 - beta) * g**2
        eff_alpha = alpha0 / (np.sqrt(v) + eps)
        x = x - eff_alpha * g; f_hist.append(loss_fn(x))
    return np.array(f_hist)

def polyak_fn(grad_fn, loss_fn, x0, f_star, eps, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        alpha_k = (loss_fn(x) - f_star) / (np.dot(g, g) + eps)
        x = x - alpha_k * g; f_hist.append(loss_fn(x))
    return np.array(f_hist)

def adam_fn(grad_fn, loss_fn, x0, alpha, beta1, beta2, eps, n_iters):
    x = x0.copy().astype(float)
    m_vec, v_vec = np.zeros_like(x), np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for t in range(1, n_iters + 1):
        g = grad_fn(x)
        m_vec = beta1 * m_vec + (1 - beta1) * g
        v_vec = beta2 * v_vec + (1 - beta2) * g**2
        m_hat = m_vec / (1 - beta1**t)
        v_hat = v_vec / (1 - beta2**t)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

H_A = X_data.T @ X_data / m

# Collect final values
results = {}
N = 150

# GD
results['GD'] = [
    gradient_descent(grad_A, loss_A, theta0_A, 0.08, N)[1][-1],
    gradient_descent(grad_B, loss_B, x0_B, 0.06, N)[1][-1],
    gradient_descent(grad_C, loss_C, x0_C, 0.0012, N)[1][-1],
]

# Polyak
results['Polyak'] = [
    polyak_fn(grad_A, loss_A, theta0_A, 0, 1e-4, N)[-1],
    polyak_fn(grad_B, loss_B, x0_B, 0, 1e-4, N)[-1],
    polyak_fn(grad_C, loss_C, x0_C, 0, 1e-3, N)[-1],
]

# Adagrad
results['Adagrad'] = [
    adagrad_fn(grad_A, loss_A, theta0_A, 1.8, 1e-5, N)[-1],
    adagrad_fn(grad_B, loss_B, x0_B, 1.2, 1e-5, N)[-1],
    adagrad_fn(grad_C, loss_C, x0_C, 0.45, 1e-5, N)[-1],
]

# RMSprop
results['RMSprop'] = [
    rmsprop_fn(grad_A, loss_A, theta0_A, 0.22, 0.9, 1e-5, N)[-1],
    rmsprop_fn(grad_B, loss_B, x0_B, 0.14, 0.9, 1e-5, N)[-1],
    rmsprop_fn(grad_C, loss_C, x0_C, 0.0035, 0.9, 1e-5, N)[-1],
]

# Heavy Ball
results['Heavy Ball'] = [
    heavy_ball(grad_A, loss_A, theta0_A, 0.045, 0.88, N)[1][-1],
    heavy_ball(grad_B, loss_B, x0_B, 0.035, 0.90, N)[1][-1],
    heavy_ball(grad_C, loss_C, x0_C, 0.0008, 0.86, N)[1][-1],
]

# Nesterov
results['Nesterov'] = [
    nesterov_momentum(grad_A, loss_A, theta0_A, 0.06, 0.90, N)[1][-1],
    nesterov_momentum(grad_B, loss_B, x0_B, 0.035, 0.92, N)[1][-1],
    nesterov_momentum(grad_C, loss_C, x0_C, 0.0007, 0.90, N)[1][-1],
]

# Adam
results['Adam'] = [
    adam_fn(grad_A, loss_A, theta0_A, 0.12, 0.82, 0.999, 1e-8, N)[-1],
    adam_fn(grad_B, loss_B, x0_B, 0.08, 0.80, 0.999, 1e-8, N)[-1],
    adam_fn(grad_C, loss_C, x0_C, 0.006, 0.80, 0.999, 1e-8, N)[-1],
]

# Newton (20 iters)
hess_A_fn = lambda x: H_A
def hessian_B(x):
    return np.array([[2 - np.sin(x[0]), 0], [0, 10]])

results["Newton's"] = [
    newtons_method(grad_A, hess_A_fn, loss_A, theta0_A, 1.0, 20)[1][-1],
    newtons_method(grad_B, hessian_B, loss_B, x0_B, 0.85, 20)[1][-1],
    newtons_method(grad_C, hessian_C, loss_C, x0_C, 0.22, 20)[1][-1],
]

methods = list(results.keys())
benchmarks = ['A (Quadratic)', 'B (Toy NN)', 'C (Rosenbrock)']
true_optima = [0.4826, 0.7244, 0.0]

# Create normalised "suboptimality" matrix: (f - f*) / (f0 - f*)
f0_vals = [loss_A(theta0_A), loss_B(x0_B), loss_C(x0_C)]

data = np.array([[results[m][i] for i in range(3)] for m in methods])
# Clamp to avoid negative due to overshoot
data_norm = np.maximum(data - np.array(true_optima), 0)
# Normalize to log scale for heatmap
data_log = np.log10(np.maximum(data_norm, 1e-12))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: actual final objective values
ax = axes[0]
im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r')
ax.set_xticks(range(3)); ax.set_xticklabels(benchmarks)
ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
ax.set_title(f'Final Objective Values After {N} Iterations\n(green=low/good, red=high/poor)')
plt.colorbar(im, ax=ax)
for i in range(len(methods)):
    for j in range(3):
        val = data[i, j]
        text = f'{val:.4f}' if val < 10 else f'{val:.1f}'
        ax.text(j, i, text, ha='center', va='center', fontsize=8,
                color='white' if data[i,j] > np.median(data[:,j]) else 'black')

# Right: log10(suboptimality) heatmap
ax = axes[1]
im2 = ax.imshow(data_log, aspect='auto', cmap='RdYlGn')
ax.set_xticks(range(3)); ax.set_xticklabels(benchmarks)
ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
ax.set_title(f'$\\log_{{10}}(f - f^\\star)$ After {N} Iterations\n(green=low suboptimality/good, red=high/poor)')
plt.colorbar(im2, ax=ax, label='$\\log_{10}(f - f^\\star)$')
for i in range(len(methods)):
    for j in range(3):
        val = data_log[i, j]
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8)

plt.suptitle('All Methods Comparison: Final Objective and Suboptimality', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('figures/q_methods_summary_heatmap.pdf', bbox_inches='tight')
plt.close()
print("  Saved.")

# ============================================================
# FIGURE 4: Effect of condition number on GD convergence rate
# ============================================================
print("Generating q1_condition_number_effect.pdf ...")

fig, ax = plt.subplots(figsize=(8, 5))

kappas = [1, 5, 10, 50, 100, 500]
colors = plt.cm.viridis(np.linspace(0, 1, len(kappas)))
n_plot = 200
k_arr = np.arange(n_plot)

for kappa, col in zip(kappas, colors):
    # Optimal fixed step: alpha = 2/(mu+L), rate = (kappa-1)/(kappa+1)
    rate = (kappa - 1) / (kappa + 1)
    # f(k) - f* = (rate)^k * (f(0) - f*)
    conv = rate**k_arr
    ax.semilogy(k_arr, conv, color=col, lw=2, label=f'$\\kappa={kappa}$ (rate $\\approx${rate:.3f})')

ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Relative suboptimality $(f(x_k)-f^\\star)/(f(x_0)-f^\\star)$')
ax.set_title('GD Convergence Rate vs. Condition Number $\\kappa = L/\\mu$\n(optimal fixed step $\\alpha=2/(\\mu+L)$)')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, n_plot])
plt.tight_layout()
plt.savefig('figures/q1_condition_number_effect.pdf', bbox_inches='tight')
plt.close()
print("  Saved.")

# ============================================================
# FIGURE 5: SGD convergence: epochs vs gradient evaluations
# ============================================================
print("Generating q2_sgd_gradient_evals.pdf ...")

def mini_batch_sgd_tracked(X, y, theta0, alpha, batch_size, n_epochs, seed=42):
    rng = np.random.RandomState(seed)
    theta = theta0.copy().astype(float)
    n = len(y)
    loss_fn_local = lambda th: 0.5 * np.mean((X @ th - y)**2)
    # Track per gradient evaluation
    grad_eval_count = 0
    grad_evals = [0]
    f_vals = [loss_fn_local(theta)]
    for _ in range(n_epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]
            r = Xb @ theta - yb
            g = Xb.T @ r / len(yb)
            theta = theta - alpha * g
            grad_eval_count += len(yb)  # count as len(yb) forward passes
        grad_evals.append(grad_eval_count)
        f_vals.append(loss_fn_local(theta))
    return np.array(f_vals), np.array(grad_evals)

sgd_b5_tracked, evals_b5 = mini_batch_sgd_tracked(X_data, y_data, theta0_A, 0.06, 5, 50)
sgd_b40_tracked, evals_b40 = mini_batch_sgd_tracked(X_data, y_data, theta0_A, 0.06, 40, 50)
sgd_b200_tracked, evals_b200 = mini_batch_sgd_tracked(X_data, y_data, theta0_A, 0.06, 200, 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.semilogy(sgd_b5_tracked, 'b-', lw=2, label=f'b=5 ({1000//5} updates/epoch)')
ax.semilogy(sgd_b40_tracked, 'r-', lw=2, label=f'b=40 ({1000//40} updates/epoch)')
ax.semilogy(sgd_b200_tracked, 'g-', lw=2, label=f'b=200 ({1000//200} updates/epoch)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Mini-Batch SGD: Convergence per Epoch\n(Benchmark A)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(evals_b5, sgd_b5_tracked, 'b-', lw=2, label='b=5')
ax.semilogy(evals_b40, sgd_b40_tracked, 'r-', lw=2, label='b=40')
ax.semilogy(evals_b200, sgd_b200_tracked, 'g-', lw=2, label='b=200')
ax.set_xlabel('Total gradient evaluations (data samples processed)')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Mini-Batch SGD: Convergence per Sample\n(Benchmark A)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Q2: Mini-Batch SGD — Epoch vs Sample-Level Convergence', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('figures/q2_sgd_gradient_evals.pdf', bbox_inches='tight')
plt.close()
print("  Saved.")

# ============================================================
# FIGURE 6: Frank-Wolfe gap evolution
# ============================================================
print("Generating q6_fw_gap.pdf ...")

X_bounds = [(0.5, 5.0), (-5.0, 10.0)]

def fw_lp_box(grad, bounds):
    z = np.zeros(len(grad))
    for i in range(len(grad)):
        z[i] = bounds[i][0] if grad[i] > 0 else bounds[i][1]
    return z

def loss_q6_interior(x):
    return (x[0] - 1)**2 + (x[1] - 5)**2
def grad_q6_interior(x):
    return np.array([2*(x[0] - 1), 2*(x[1] - 5)])

def loss_q6_boundary(x):
    return x[0]**2 + x[1]**2
def grad_q6_boundary(x):
    return np.array([2*x[0], 2*x[1]])

def frank_wolfe_with_gap(grad_fn, loss_fn, x0, bounds, beta, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    gap_hist = []
    for _ in range(n_iters):
        g = grad_fn(x)
        z = fw_lp_box(g, bounds)
        # Frank-Wolfe gap: g^T (x - z)
        gap = np.dot(g, x - z)
        gap_hist.append(gap)
        x = beta * x + (1 - beta) * z
        f_hist.append(loss_fn(x))
    return np.array(f_hist), np.array(gap_hist)

x0_fw_int = np.array([1.0, 1.0])
x0_fw_bnd = np.array([3.0, 3.0])

f_int_90, gap_int_90 = frank_wolfe_with_gap(grad_q6_interior, loss_q6_interior,
                                              x0_fw_int, X_bounds, 0.90, 180)
f_int_985, gap_int_985 = frank_wolfe_with_gap(grad_q6_interior, loss_q6_interior,
                                               x0_fw_int, X_bounds, 0.985, 180)
f_bnd, gap_bnd = frank_wolfe_with_gap(grad_q6_boundary, loss_q6_boundary,
                                       x0_fw_bnd, X_bounds, 0.93, 140)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.semilogy(np.maximum(gap_int_90, 1e-10), 'b-', lw=2,
            label='Interior $\\beta=0.90$')
ax.semilogy(np.maximum(gap_int_985, 1e-10), 'r-', lw=2,
            label='Interior $\\beta=0.985$')
ax.semilogy(np.maximum(gap_bnd, 1e-10), 'g-', lw=2,
            label='Boundary $\\beta=0.93$')
# Theoretical O(1/k) reference
k_ref = np.arange(1, 181)
ax.semilogy(k_ref, gap_int_90[0] / k_ref, 'k:', lw=1.5, label='$O(1/k)$ reference')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Frank--Wolfe Gap $g_k$ (log scale)')
ax.set_title('Q6: Frank--Wolfe Gap $g_k = \\nabla f(x_k)^T(x_k - z_k)$\n(Upper bound on suboptimality)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(np.maximum(f_int_90 - 0.0, 1e-10), 'b-', lw=2,
            label=f'Interior $\\beta=0.90$ ($f^\\star=0$)')
ax.semilogy(np.maximum(f_int_985 - 0.0, 1e-10), 'r-', lw=2,
            label=f'Interior $\\beta=0.985$ ($f^\\star=0$)')
ax.semilogy(np.maximum(f_bnd - 0.25, 1e-10), 'g-', lw=2,
            label=f'Boundary $\\beta=0.93$ ($f^\\star=0.25$)')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Suboptimality $f(x_k) - f^\\star$ (log scale)')
ax.set_title('Q6: Suboptimality $f(x_k) - f^\\star$\nvs.\ Frank--Wolfe Gap')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Q6: Frank--Wolfe Gap and Convergence', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('figures/q6_fw_gap.pdf', bbox_inches='tight')
plt.close()
print("  Saved.")

# ============================================================
# FIGURE 7: Newton convergence rate (quadratic vs linear)
# ============================================================
print("Generating q3_convergence_rates.pdf ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: GD vs Newton on Benchmark B, showing error vs iteration
gd_B = gradient_descent(grad_B, loss_B, x0_B, 0.06, 50)
newton_B = newtons_method(grad_B, hessian_B, loss_B, x0_B, 0.85, 20)

f_star_B = 0.7244
ax = axes[0]
ax.semilogy(np.maximum(gd_B[1] - f_star_B, 1e-15), 'b-o', ms=4, lw=2,
            label='GD (suboptimality)')
ax.semilogy(np.maximum(newton_B[1] - f_star_B, 1e-15), 'r-s', ms=6, lw=2,
            label="Newton (suboptimality)")
ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)')
ax.set_title("Q3: GD vs Newton's on Benchmark B\n(suboptimality $f(x_k)-f^\\star$)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: update norm plot showing linear vs quadratic decay pattern
ax = axes[1]
# For GD
gd_norms = np.array([np.linalg.norm(gd_B[0][i+1] - gd_B[0][i]) for i in range(len(gd_B[0])-1)])
ax.semilogy(gd_norms[:20], 'b-o', ms=4, lw=2, label='GD update norm $\\|x_{k+1}-x_k\\|$')
ax.semilogy(newton_B[2], 'r-s', ms=6, lw=2, label="Newton update norm")
# Reference lines
k_lin = np.arange(1, 21)
ax.semilogy(k_lin-1, gd_norms[0] * (0.85)**k_lin, 'b:', lw=1.5, alpha=0.5,
            label='Linear decay $(0.85)^k$ reference')
ax.set_xlabel('Iteration')
ax.set_ylabel('Update norm $\\|x_{k+1}-x_k\\|$ (log scale)')
ax.set_title("Q3: Update Norm Decay — Linear vs Rapid Convergence\n(Benchmark B, first 20 iterations)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle("Newton's Method: Convergence Rate Comparison with GD", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('figures/q3_convergence_rates.pdf', bbox_inches='tight')
plt.close()
print("  Saved.")

# ============================================================
# FIGURE 8: Penalty method -- effect of lambda on trajectory
# ============================================================
print("Generating q5_penalty_lambda_effect.pdf ...")

x0_q5 = np.array([0.2, 4.0])

def penalty_loss_q5(x, lam):
    return loss_B(x) + lam * max(0, -x[0] + 0.5)

def penalty_grad_q5(x, lam):
    g = grad_B(x).copy()
    if x[0] < 0.5:
        g[0] -= lam
    return g

def penalty_gd_q5(x0, alpha, lam, n_iters):
    x = x0.copy().astype(float)
    x_hist = [x.copy()]
    f_hist = [loss_B(x)]
    for _ in range(n_iters):
        g = penalty_grad_q5(x, lam)
        x = x - alpha * g
        x_hist.append(x.copy())
        f_hist.append(loss_B(x))
    return np.array(x_hist), np.array(f_hist)

lambdas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
colors_pen = plt.cm.plasma(np.linspace(0.1, 0.9, len(lambdas)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for lam, col in zip(lambdas, colors_pen):
    alpha_use = 0.03 if lam > 3 else 0.05
    _, f_hist = penalty_gd_q5(x0_q5, alpha_use, lam, 100)
    ax.semilogy(f_hist, color=col, lw=2, label=f'$\\lambda={lam}$')
ax.axhline(y=0.7244, color='k', ls='--', lw=1, label='$f^\\star = 0.7244$')
ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x_k)$ (log scale)')
ax.set_title('Q5: Penalty Method — Effect of $\\lambda$ on Convergence')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

ax = axes[1]
final_violations = []
final_fvals = []
for lam in np.logspace(-1.5, 1.5, 50):
    alpha_use = 0.02 if lam > 5 else (0.03 if lam > 2 else 0.05)
    x_hist, f_hist = penalty_gd_q5(x0_q5, alpha_use, lam, 100)
    viol = max(0, 0.5 - x_hist[-1, 0])
    final_violations.append(viol)
    final_fvals.append(f_hist[-1])

lam_range = np.logspace(-1.5, 1.5, 50)
ax.loglog(lam_range, np.maximum(final_violations, 1e-12), 'b-o', ms=3, lw=2,
          label='Final constraint violation')
ax2_right = ax.twinx()
ax2_right.semilogx(lam_range, final_fvals, 'r-s', ms=3, lw=2, label='Final $f(x)$')
ax2_right.axhline(y=0.7244, color='r', ls=':', lw=1)
ax2_right.set_ylabel('Final objective value', color='r')
ax2_right.tick_params(axis='y', labelcolor='r')
ax.set_xlabel('Penalty weight $\\lambda$')
ax.set_ylabel('Final constraint violation (log scale)', color='b')
ax.tick_params(axis='y', labelcolor='b')
ax.set_title('Q5: Penalty Method Trade-off\n($\\lambda$ effect on feasibility vs.\ optimality)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2_right.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Q5: Quadratic Penalty Method — Effect of $\\lambda$', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('figures/q5_penalty_lambda_effect.pdf', bbox_inches='tight')
plt.close()
print("  Saved.")

print("\nAll extra figures generated successfully.")
print("New figures:")
for f in ['q_heavy_vs_nesterov_C', 'q4_fd_error_analysis', 'q_methods_summary_heatmap',
          'q1_condition_number_effect', 'q2_sgd_gradient_evals', 'q6_fw_gap',
          'q3_convergence_rates', 'q5_penalty_lambda_effect']:
    print(f"  figures/{f}.pdf")
