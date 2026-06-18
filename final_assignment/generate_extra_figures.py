"""
Generate additional figures for the improved report.
Figures must be saved to final_assignment/figures/ directory.
Never overwrite existing figures.
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
# BENCHMARK FUNCTIONS (replicated from main code)
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

# ============================================================
# FIGURE 1: Comprehensive method comparison bar chart
# ============================================================
# Final objective values from the code runs
# These match the tables in the report

q1_vals = {
    'A': {'GD': 0.4826, 'Polyak': 2.7189, 'Adagrad': 0.4826, 'RMSprop': 0.5027, 'Heavy Ball': 0.4826},
    'B': {'GD': 0.7244, 'Polyak': 24.0659, 'Adagrad': 0.7244, 'RMSprop': 0.7271, 'Heavy Ball': 0.7244},
    'C': {'GD': 3.5142, 'Polyak': 0.0094, 'Adagrad': 2.2034, 'RMSprop': 3.1204, 'Heavy Ball': 1.4003}
}

q2_vals = {
    'A': {'GD': 0.4826, 'Nesterov': 0.4826, 'Adam': 0.4826},
    'B': {'GD': 0.7244, 'Nesterov': 0.7244, 'Adam': 0.7244},
    'C': {'GD': 3.388, 'Nesterov': 0.4788, 'Adam': 1.5785}
}

q3_vals = {
    'A': {'GD (80 iters)': 0.4827, "Newton (20 iters)": 0.4826},
    'B': {'GD (80 iters)': 0.7244, "Newton (20 iters)": 0.7244},
    'C': {'GD (80 iters)': 3.7319, "Newton (20 iters)": 0.6864}
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = {'A': 'Benchmark A\n(Linear Regression, $f^\\star \\approx 0.4826$)',
          'B': 'Benchmark B\n(Toy NN, $f^\\star \\approx 0.7244$)',
          'C': 'Benchmark C\n(Rosenbrock, $f^\\star = 0$)'}
colors_q1 = ['#2C7BB6', '#D7191C', '#1A9641', '#FDAE61', '#762A83']
colors_q2 = ['#2C7BB6', '#D7191C', '#1A9641']
colors_q3 = ['#2C7BB6', '#D7191C']

for idx, bname in enumerate(['A', 'B', 'C']):
    ax = axes[idx]

    all_methods = []
    all_vals = []
    all_colors = []

    # Q1 methods
    for (method, val), color in zip(q1_vals[bname].items(), colors_q1):
        all_methods.append(method)
        all_vals.append(val)
        all_colors.append(color)

    # Q2 extra methods (skip GD duplicate)
    for method, val in [('Nesterov', q2_vals[bname]['Nesterov']),
                        ('Adam', q2_vals[bname]['Adam'])]:
        all_methods.append(method)
        all_vals.append(val)
        all_colors.append('#ABDDA4' if method == 'Nesterov' else '#F4A582')

    # Q3 Newton
    all_methods.append('Newton')
    all_vals.append(q3_vals[bname]['Newton (20 iters)'])
    all_colors.append('#5E3C99')

    x_pos = np.arange(len(all_methods))
    bars = ax.bar(x_pos, all_vals, color=all_colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Add value labels on bars (clip Polyak divergence for readability)
    f_star_map = {'A': 0.4826, 'B': 0.7244, 'C': 0.0}
    for bar, val in zip(bars, all_vals):
        display_val = min(val, 5.0)
        if val > 5.0:
            ax.text(bar.get_x() + bar.get_width()/2, 0.2, f'{val:.2f}↑',
                    ha='center', va='bottom', fontsize=7, rotation=90, color='red')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Final Objective Value')
    ax.set_title(titles[bname], fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # Add f* line
    f_star = f_star_map[bname]
    if f_star > 0:
        ax.axhline(y=f_star, color='k', linestyle='--', lw=1.5, label=f'$f^\\star = {f_star}$')
        ax.legend(fontsize=8)

    # Clip y-axis for readability
    if bname in ['A', 'B']:
        ax.set_ylim(0, max(all_vals) * 1.3)
    else:
        ax.set_ylim(0, min(max(all_vals), 5.5))

plt.suptitle('Final Objective Values: All Methods Across All Benchmarks\n(lower is better; Polyak on A/B diverges due to $f^\\star$ misspecification)',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('figures/comparison_all_methods.pdf', bbox_inches='tight')
plt.close()
print("Saved: figures/comparison_all_methods.pdf")

# ============================================================
# FIGURE 2: Finite difference error analysis
# ============================================================
# Plot gradient approximation error vs delta for Benchmark B at the starting point
x_test = np.array([-1.0, 4.0])
true_grad = grad_B(x_test)

deltas = np.logspace(-14, 1, 200)
errors_fwd = []
errors_central = []

for delta in deltas:
    # Forward difference
    n = len(x_test)
    g_fwd = np.zeros(n)
    f0 = loss_B(x_test)
    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        g_fwd[i] = (loss_B(x_test + delta * ei) - f0) / delta
    errors_fwd.append(np.linalg.norm(g_fwd - true_grad))

    # Central difference
    g_central = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        g_central[i] = (loss_B(x_test + delta * ei) - loss_B(x_test - delta * ei)) / (2 * delta)
    errors_central.append(np.linalg.norm(g_central - true_grad))

errors_fwd = np.array(errors_fwd)
errors_central = np.array(errors_central)

fig, ax = plt.subplots(figsize=(9, 6))
ax.loglog(deltas, errors_fwd, 'b-', lw=2, label='Forward difference $O(\\delta)$')
ax.loglog(deltas, errors_central, 'r-', lw=2, label='Central difference $O(\\delta^2)$')

# Optimal delta markers
eps_mach = 2.2e-16
opt_fwd = np.sqrt(eps_mach)
opt_central = eps_mach**(1/3)
ax.axvline(x=opt_fwd, color='b', linestyle='--', lw=1, alpha=0.7,
           label=f'FD optimal $\\delta^* \\approx \\sqrt{{\\epsilon_{{mach}}}} \\approx 10^{{-8}}$')
ax.axvline(x=opt_central, color='r', linestyle='--', lw=1, alpha=0.7,
           label=f'CD optimal $\\delta^* \\approx \\epsilon_{{mach}}^{{1/3}} \\approx 10^{{-5}}$')

# Mark the experiment deltas
ax.axvline(x=0.05, color='green', linestyle=':', lw=2, label='$\\delta = 0.05$ (experiment: good)')
ax.axvline(x=0.8, color='orange', linestyle=':', lw=2, label='$\\delta = 0.8$ (experiment: poor)')

# Reference slopes
d_ref = np.logspace(-8, 0, 50)
ax.loglog(d_ref, 1e-7 * d_ref, 'b--', lw=1, alpha=0.4)
d_ref2 = np.logspace(-5, 0, 50)
ax.loglog(d_ref2, 1e-9 * d_ref2**2, 'r--', lw=1, alpha=0.4)

ax.set_xlabel('Step size $\\delta$')
ax.set_ylabel('Gradient approximation error $\\|\\hat{g} - \\nabla f\\|$')
ax.set_title('Finite Difference Gradient Error vs Step Size $\\delta$\n(Benchmark B at $x_0 = (-1, 4)$)')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q4_fd_error_analysis.pdf', bbox_inches='tight')
plt.close()
print("Saved: figures/q4_fd_error_analysis.pdf")

# ============================================================
# FIGURE 3: Convergence rate comparison (log-log scale)
# to reveal empirical convergence rates
# ============================================================
# Re-run key methods to get convergence history for Rosenbrock (Benchmark C)
x0_C = np.array([-1.0, 1.0])
f_star_C = 0.0

def gradient_descent(grad_fn, loss_fn, x0, alpha, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        x = x - alpha * grad_fn(x)
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

def nesterov_momentum(grad_fn, loss_fn, x0, alpha, beta_max, n_iters):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for k in range(1, n_iters + 1):
        beta_k = min((k - 1) / (k + 2), beta_max)
        lookahead = x + beta_k * z
        g = grad_fn(lookahead)
        z = beta_k * z - alpha * g
        x = x + z
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

def adam_optimiser(grad_fn, loss_fn, x0, alpha, beta1, beta2, eps, n_iters):
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

def polyak_step(grad_fn, loss_fn, x0, f_star, eps, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        alpha_k = (loss_fn(x) - f_star) / (np.dot(g, g) + eps)
        x = x - alpha_k * g
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

n_iters = 500

gd_C = gradient_descent(grad_C, loss_C, x0_C, 0.0012, n_iters)
nest_C = nesterov_momentum(grad_C, loss_C, x0_C, 0.0007, 0.90, n_iters)
adam_C = adam_optimiser(grad_C, loss_C, x0_C, 0.006, 0.80, 0.999, 1e-8, n_iters)
polyak_C = polyak_step(grad_C, loss_C, x0_C, 0.0, 1e-3, n_iters)

# Clip to avoid log(0)
eps_clip = 1e-12

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: regular log scale
ax = axes[0]
iters = np.arange(n_iters + 1)
ax.semilogy(iters, np.maximum(gd_C, eps_clip), 'k--', lw=2, label='GD ($\\alpha=0.0012$)', alpha=0.8)
ax.semilogy(iters, np.maximum(nest_C, eps_clip), 'b-', lw=2, label='Nesterov ($\\alpha=0.0007$)')
ax.semilogy(iters, np.maximum(adam_C, eps_clip), 'r-', lw=2, label='Adam ($\\alpha=0.006$)')
ax.semilogy(iters, np.maximum(polyak_C, eps_clip), 'g-', lw=2, label='Polyak ($f^\\star=0$)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Value $f(x_k)$ (log scale)')
ax.set_title('Convergence on Rosenbrock (500 iterations)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=1e-4)

# Right: log-log scale to reveal rates
ax = axes[1]
# Only plot from iteration 10 to avoid transient
start = 10
for fvals, lab, col, ls in [
    (gd_C, 'GD', 'k', '--'),
    (nest_C, 'Nesterov', 'b', '-'),
    (adam_C, 'Adam', 'r', '-'),
    (polyak_C, 'Polyak', 'g', '-')
]:
    valid = fvals[start:] > eps_clip
    if np.any(valid):
        k_vals = np.arange(start, start + len(fvals) - start)[valid]
        f_vals_valid = fvals[start:][valid]
        ax.loglog(k_vals, f_vals_valid, color=col, linestyle=ls, lw=2, alpha=0.8, label=lab)

# Reference lines
k_ref = np.logspace(1, 2.7, 100)
ax.loglog(k_ref, 50 * k_ref**(-1), 'm--', lw=1.5, alpha=0.7, label='$O(1/k)$ reference')
ax.loglog(k_ref, 50 * k_ref**(-2), 'c--', lw=1.5, alpha=0.7, label='$O(1/k^2)$ reference')

ax.set_xlabel('Iteration $k$ (log scale)')
ax.set_ylabel('Objective Value (log scale)')
ax.set_title('Log-Log Convergence Plot (Empirical Rates)')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

plt.suptitle('Extended Convergence Analysis: Rosenbrock (500 iterations)', fontsize=12)
plt.tight_layout()
plt.savefig('figures/convergence_extended_C.pdf', bbox_inches='tight')
plt.close()
print("Saved: figures/convergence_extended_C.pdf")

# ============================================================
# FIGURE 4: Newton convergence quadratic behaviour illustration
# ============================================================
# Show Newton error norm on log scale to reveal the quadratic convergence regime
x0_C = np.array([-1.0, 1.0])

def hessian_C(x):
    h11 = 2 + 1200*x[0]**2 - 400*x[1]
    h12 = -400*x[0]
    return np.array([[h11, h12], [h12, 200]])

def newtons_method(grad_fn, hess_fn, loss_fn, x0, alpha, n_iters, damping=1e-8):
    x = x0.copy().astype(float)
    x_hist = [x.copy()]
    f_hist = [loss_fn(x)]
    update_norms = []
    for _ in range(n_iters):
        g = grad_fn(x)
        H = hess_fn(x)
        H_reg = H + damping * np.eye(len(x))
        try:
            p = np.linalg.solve(H_reg, g)
        except np.linalg.LinAlgError:
            p = g
        update_norms.append(np.linalg.norm(alpha * p))
        x = x - alpha * p
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist), np.array(update_norms)

# Run Newton for more iterations to see quadratic regime
newton_long = newtons_method(grad_C, hessian_C, loss_C, x0_C, 0.22, 100, damping=1e-8)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.semilogy(newton_long[1], 'r-', lw=2, label="Newton $\\alpha=0.22$ (100 iters)")
gd_long = gradient_descent(grad_C, loss_C, x0_C, 0.001, 100)
ax.semilogy(gd_long, 'b--', lw=2, label="GD $\\alpha=0.001$ (100 iters)")
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Value (log scale)')
ax.set_title("Newton vs GD on Rosenbrock (Extended)")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
# Plot update norms on log-log scale after iteration 20 (near convergence regime)
update_norms = newton_long[2]
ax.semilogy(update_norms, 'r-o', ms=3, lw=1.5, label='Newton update magnitude $\\|\\Delta x_k\\|$')
ax.set_xlabel('Iteration')
ax.set_ylabel('Update magnitude (log scale)')
ax.set_title("Newton Update Magnitudes (Rosenbrock)")
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle("Newton's Method: Extended Analysis on Rosenbrock", fontsize=12)
plt.tight_layout()
plt.savefig('figures/q3_newton_extended.pdf', bbox_inches='tight')
plt.close()
print("Saved: figures/q3_newton_extended.pdf")

# ============================================================
# FIGURE 5: Penalty method: effect of lambda on convergence
# ============================================================
x0_q5 = np.array([0.2, 4.0])

def project_q5(x):
    x_p = x.copy()
    x_p[0] = max(0.5, x_p[0])
    return x_p

def penalty_loss_q5(x, lam):
    return loss_B(x) + lam * max(0, -x[0] + 0.5)

def penalty_grad_q5(x, lam):
    g = grad_B(x).copy()
    if x[0] < 0.5:
        g[0] -= lam
    return g

def projected_gd(x0, alpha, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_B(x)]
    for _ in range(n_iters):
        g = grad_B(x)
        x = project_q5(x - alpha * g)
        f_hist.append(loss_B(x))
    return np.array(f_hist)

def penalty_gd(x0, alpha, lam, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_B(x)]
    for _ in range(n_iters):
        g = penalty_grad_q5(x, lam)
        x = x - alpha * g
        f_hist.append(loss_B(x))
    return np.array(f_hist)

# Extended runs to show asymptotic behaviour
n = 300
pgd = projected_gd(x0_q5, 0.08, n)
pen_small = penalty_gd(x0_q5, 0.05, 0.15, n)
pen_medium = penalty_gd(x0_q5, 0.05, 1.8, n)
pen_large = penalty_gd(x0_q5, 0.03, 4.5, n)
pen_xlarge = penalty_gd(x0_q5, 0.015, 15.0, n)  # Very large lambda

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(pgd, 'k-', lw=2.5, label='Projected GD ($\\alpha=0.08$)')
ax.semilogy(pen_small, 'b--', lw=2, label='Penalty $\\lambda=0.15$ ($\\alpha=0.05$)')
ax.semilogy(pen_medium, color='orange', linestyle='--', lw=2, label='Penalty $\\lambda=1.8$ ($\\alpha=0.05$)')
ax.semilogy(pen_large, 'r--', lw=2, label='Penalty $\\lambda=4.5$ ($\\alpha=0.03$)')
ax.semilogy(pen_xlarge, 'm--', lw=2, label='Penalty $\\lambda=15.0$ ($\\alpha=0.015$)')
ax.axhline(y=0.7244, color='gray', linestyle=':', lw=1.5, label='$f^\\star \\approx 0.7244$')
ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x)$ (log scale)')
ax.set_title('Q5: Extended Convergence — Penalty vs Projected GD (300 iterations)\nHighlighting asymptotic behaviour with varying $\\lambda$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q5_penalty_extended.pdf', bbox_inches='tight')
plt.close()
print("Saved: figures/q5_penalty_extended.pdf")

# ============================================================
# FIGURE 6: SGD noise floor analysis
# ============================================================
# Theoretical noise floor for SGD as function of sigma
def mini_batch_sgd(X, y, theta0, alpha, batch_size, n_epochs, seed=42):
    rng = np.random.RandomState(seed)
    theta = theta0.copy().astype(float)
    n = len(y)
    loss_fn = lambda th: 0.5 * np.mean((X @ th - y)**2)
    epoch_losses = [loss_fn(theta)]
    for _ in range(n_epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]
            r = Xb @ theta - yb
            g = Xb.T @ r / len(yb)
            theta = theta - alpha * g
        epoch_losses.append(loss_fn(theta))
    return np.array(epoch_losses)

theta0_A = np.array([0.0, 0.0])
noise_levels = [1, 2, 3, 6, 10]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for sigma in noise_levels:
    y_noisy = X_data @ theta_star + sigma * eps_noise
    sgd = mini_batch_sgd(X_data, y_noisy, theta0_A, 0.06, 40, 60)
    ax.semilogy(sgd, lw=2, label=f'$\\sigma={sigma}$')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title('SGD (b=40) convergence for different noise levels $\\sigma$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: noise floor scaling
final_losses = []
sigmas = np.array([1, 2, 3, 6, 10])
for sigma in sigmas:
    y_noisy = X_data @ theta_star + sigma * eps_noise
    sgd = mini_batch_sgd(X_data, y_noisy, theta0_A, 0.06, 40, 100)
    final_losses.append(sgd[-1])

ax = axes[1]
ax.loglog(sigmas, final_losses, 'b-o', ms=8, lw=2, label='Observed final loss')
# Theoretical: J* ≈ 0.4826 * sigma^2 (since noise variance scales as sigma^2)
J_base = 0.4826
ax.loglog(sigmas, J_base * sigmas**2, 'r--', lw=2, label=f'$J^\\star_{{\\text{{base}}}} \\cdot \\sigma^2 = {J_base:.4f} \\cdot \\sigma^2$')
ax.set_xlabel('Noise level $\\sigma$')
ax.set_ylabel('Final loss')
ax.set_title('Noise floor scaling: $J^\\star \\propto \\sigma^2$')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

plt.suptitle('SGD Noise Analysis: Noise Floor Scales as $\\sigma^2$', fontsize=12)
plt.tight_layout()
plt.savefig('figures/q2_sgd_noise_analysis.pdf', bbox_inches='tight')
plt.close()
print("Saved: figures/q2_sgd_noise_analysis.pdf")

print("\nAll extra figures generated successfully.")
