"""
Generate additional figures for the improved report.
These complement the existing figures produced by CS7DS2_Final_Code_25336453.py
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
# BENCHMARK DEFINITIONS (replicate from original code)
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

def hessian_A_matrix():
    return X_data.T @ X_data / m

def loss_B(x):
    return (x[0] - 1)**2 + 5*(x[1] - 2)**2 + np.sin(x[0])

def grad_B(x):
    return np.array([2*(x[0] - 1) + np.cos(x[0]), 10*(x[1] - 2)])

def hessian_B(x):
    return np.array([[2 - np.sin(x[0]), 0], [0, 10]])

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

# ============================================================
# ALGORITHM IMPLEMENTATIONS
# ============================================================
def gradient_descent(grad_fn, loss_fn, x0, alpha, n_iters):
    x = x0.copy().astype(float)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for _ in range(n_iters):
        x = x - alpha * grad_fn(x)
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

def polyak_step(grad_fn, loss_fn, x0, f_star, eps, n_iters):
    x = x0.copy().astype(float)
    x_hist, f_hist, alpha_hist = [x.copy()], [loss_fn(x)], []
    for _ in range(n_iters):
        g = grad_fn(x)
        alpha_k = (loss_fn(x) - f_star) / (np.dot(g, g) + eps)
        alpha_hist.append(alpha_k)
        x = x - alpha_k * g
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist), np.array(alpha_hist)

def adagrad(grad_fn, loss_fn, x0, alpha0, eps, n_iters):
    x = x0.copy().astype(float)
    G = np.zeros_like(x)
    x_hist, f_hist, alpha_hist = [x.copy()], [loss_fn(x)], []
    for _ in range(n_iters):
        g = grad_fn(x)
        G += g**2
        eff_alpha = alpha0 / (np.sqrt(G) + eps)
        alpha_hist.append(np.mean(eff_alpha))
        x = x - eff_alpha * g
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist), np.array(alpha_hist)

def rmsprop(grad_fn, loss_fn, x0, alpha0, beta, eps, n_iters):
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    x_hist, f_hist, alpha_hist = [x.copy()], [loss_fn(x)], []
    for _ in range(n_iters):
        g = grad_fn(x)
        v = beta * v + (1 - beta) * g**2
        eff_alpha = alpha0 / (np.sqrt(v) + eps)
        alpha_hist.append(np.mean(eff_alpha))
        x = x - eff_alpha * g
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist), np.array(alpha_hist)

def heavy_ball(grad_fn, loss_fn, x0, alpha, beta, n_iters):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        z = beta * z + alpha * g
        x = x - z
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
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
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

def adam_optimiser(grad_fn, loss_fn, x0, alpha, beta1, beta2, eps, n_iters):
    x = x0.copy().astype(float)
    m_vec, v_vec = np.zeros_like(x), np.zeros_like(x)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for t in range(1, n_iters + 1):
        g = grad_fn(x)
        m_vec = beta1 * m_vec + (1 - beta1) * g
        v_vec = beta2 * v_vec + (1 - beta2) * g**2
        m_hat = m_vec / (1 - beta1**t)
        v_hat = v_vec / (1 - beta2**t)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
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
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist), np.array(update_norms)

# ============================================================
# FIGURE 1: Q1 Convergence Gap Analysis
# Show |f(x_k) - f*| to visualise convergence rates
# ============================================================
print("Generating Q1 convergence gap figure...")

# True minimums (known analytically / from code)
# Benchmark A: f* ≈ 0.4826 (the OLS minimum with noise)
# Benchmark B: f* ≈ 0.7244
# Benchmark C: f* = 0.0

# Re-run Q1 with extended iterations for better rate visualisation
N_LONG = 200

# Benchmark A
res_A_gd = gradient_descent(grad_A, loss_A, theta0_A, 0.08, N_LONG)
res_A_hb = heavy_ball(grad_A, loss_A, theta0_A, 0.045, 0.88, N_LONG)
res_A_adagrad = adagrad(grad_A, loss_A, theta0_A, 1.8, 1e-5, N_LONG)
res_A_rmsprop = rmsprop(grad_A, loss_A, theta0_A, 0.22, 0.9, 1e-5, N_LONG)

# Compute true minimum from OLS solution
H_A = hessian_A_matrix()
b_A = X_data.T @ y_data / m
theta_ols = np.linalg.solve(H_A, b_A)
f_star_A = loss_A(theta_ols)
print(f"  Benchmark A f* = {f_star_A:.6f}")

# Benchmark B: fine-tune minimum
from scipy.optimize import minimize as sp_minimize
res_opt_B = sp_minimize(loss_B, x0_B, method='L-BFGS-B')
f_star_B = res_opt_B.fun
x_star_B = res_opt_B.x
print(f"  Benchmark B f* = {f_star_B:.6f} at x* = {x_star_B}")

f_star_C = 0.0  # Known

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Benchmark A convergence gap (log scale)
ax = axes[0]
gap_gd_A = res_A_gd[1] - f_star_A
gap_hb_A = res_A_hb[1] - f_star_A
gap_ada_A = res_A_adagrad[1] - f_star_A
gap_rms_A = res_A_rmsprop[1] - f_star_A

# Only plot positive gaps
def plot_gap(ax, gap, label, color, ls='-'):
    iters = np.arange(len(gap))
    pos_mask = gap > 1e-12
    if np.any(pos_mask):
        ax.semilogy(iters[pos_mask], gap[pos_mask], color=color, linestyle=ls, lw=2, label=label)

plot_gap(ax, gap_gd_A, 'GD (baseline)', 'k', '--')
plot_gap(ax, gap_hb_A, 'Heavy Ball', 'tab:red', '-')
plot_gap(ax, gap_ada_A, 'Adagrad', 'tab:orange', '-')
plot_gap(ax, gap_rms_A, 'RMSprop', 'tab:green', '-')

# Compute slope (convergence rate) for GD on Benchmark A in linear regime
# Fit a line to log-gap in iterations 20-100
iters_fit = np.arange(20, 100)
log_gap_gd = np.log(np.maximum(gap_gd_A[20:100], 1e-15))
if np.std(log_gap_gd) > 1e-10:
    slope_gd, _ = np.polyfit(iters_fit, log_gap_gd, 1)
    ax.text(0.55, 0.85, f'GD slope ≈ {slope_gd:.4f}/iter\n(per-iter reduction $\\rho \\approx {np.exp(slope_gd):.4f}$)',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)')
ax.set_title('Q1: Convergence Gap — Benchmark A\n(Linear Regression, $f^\\star \\approx 0.4826$)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, N_LONG)

# Panel B: Benchmark C convergence gap (Rosenbrock, f* = 0)
res_C_gd = gradient_descent(grad_C, loss_C, x0_C, 0.0012, N_LONG)
res_C_hb = heavy_ball(grad_C, loss_C, x0_C, 0.0008, 0.86, N_LONG)
res_C_polyak = polyak_step(grad_C, loss_C, x0_C, 0.0, 1e-3, N_LONG)
res_C_ada = adagrad(grad_C, loss_C, x0_C, 0.45, 1e-5, N_LONG)

ax = axes[1]
plot_gap(ax, res_C_gd[1] - f_star_C, 'GD', 'k', '--')
plot_gap(ax, res_C_hb[1] - f_star_C, 'Heavy Ball', 'tab:red', '-')
plot_gap(ax, np.abs(res_C_polyak[1] - f_star_C), 'Polyak (correct $f^\\star$)', 'tab:blue', '-')
plot_gap(ax, res_C_ada[1] - f_star_C, 'Adagrad', 'tab:orange', '-')

ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)')
ax.set_title('Q1: Convergence Gap — Benchmark C\n(Rosenbrock, $f^\\star = 0$)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, N_LONG)

plt.suptitle('Q1: Convergence Rate Analysis — Error $f(x_k) - f^\\star$ vs Iteration', fontsize=13)
plt.tight_layout()
plt.savefig('figures/q1_convergence_gap.pdf', bbox_inches='tight')
plt.close()
print("  Saved figures/q1_convergence_gap.pdf")

# ============================================================
# FIGURE 2: Newton's Quadratic Convergence Evidence
# ============================================================
print("Generating Newton quadratic convergence figure...")

# Use Benchmark B where we know f* = 0.7244
# Run Newton for more iterations to see the quadratic rate
newton_B_long = newtons_method(grad_B, hessian_B, loss_B, x0_B, 0.85, 30, damping=1e-8)
gd_B_long = gradient_descent(grad_B, loss_B, x0_B, 0.06, 200)

gap_newton_B = newton_B_long[1] - f_star_B
gap_gd_B = gd_B_long[1] - f_star_B

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
iters_n = np.arange(len(gap_newton_B))
iters_gd = np.arange(len(gap_gd_B))

pos_n = gap_newton_B > 1e-15
pos_gd_B = gap_gd_B > 1e-12

ax.semilogy(iters_gd[pos_gd_B], gap_gd_B[pos_gd_B], 'b-', lw=2, label='GD (200 iters)')
ax.semilogy(iters_n[pos_n], gap_newton_B[pos_n], 'r-', lw=2, label="Newton (30 iters)")
ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)')
ax.set_title("Q3: Convergence Gap — Benchmark B\n(Newton vs GD, $f^\\star \\approx 0.7244$)")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Log-log plot of gap to reveal convergence order
ax = axes[1]
# For Newton: plot log(gap_k+1) vs log(gap_k) — if quadratic, slope = 2
log_gap_n = np.log10(np.maximum(gap_newton_B[pos_n], 1e-20))
if len(log_gap_n) > 2:
    ax.scatter(log_gap_n[:-1], log_gap_n[1:], color='red', s=50, zorder=5,
               label="Newton: $\\log_{10}|f_{k+1} - f^*|$ vs $\\log_{10}|f_k - f^*|$")
    # Reference lines for linear (slope 1) and quadratic (slope 2) convergence
    x_ref = np.linspace(log_gap_n.min(), log_gap_n.max(), 50)
    # Fit a line
    if len(log_gap_n) >= 3:
        valid = np.isfinite(log_gap_n[:-1]) & np.isfinite(log_gap_n[1:])
        if np.sum(valid) >= 2:
            slope_fit, intercept_fit = np.polyfit(log_gap_n[:-1][valid], log_gap_n[1:][valid], 1)
            ax.plot(x_ref, slope_fit * x_ref + intercept_fit, 'r--', lw=1.5,
                    label=f'Fitted slope ≈ {slope_fit:.2f}')

log_gap_gd = np.log10(np.maximum(gap_gd_B[pos_gd_B], 1e-20))
if len(log_gap_gd) > 2:
    ax.scatter(log_gap_gd[:-1:5], log_gap_gd[1::5][:len(log_gap_gd[:-1:5])], color='blue', s=20,
               alpha=0.6, label="GD (every 5th iter)")
    valid_gd = np.isfinite(log_gap_gd[:-1:5]) & np.isfinite(log_gap_gd[1::5][:len(log_gap_gd[:-1:5])])
    if np.sum(valid_gd) >= 2:
        slope_gd2, ic_gd2 = np.polyfit(log_gap_gd[:-1:5][valid_gd],
                                        log_gap_gd[1::5][:len(log_gap_gd[:-1:5])][valid_gd], 1)
        ax.plot(x_ref, slope_gd2 * x_ref + ic_gd2, 'b--', lw=1.5,
                label=f'GD slope ≈ {slope_gd2:.2f}')

ax.set_xlabel('$\\log_{10}(f(x_k) - f^\\star)$')
ax.set_ylabel('$\\log_{10}(f(x_{k+1}) - f^\\star)$')
ax.set_title('Q3: Convergence Order Analysis\n(Slope ≈ 1: linear; Slope ≈ 2: quadratic)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.plot([], [], 'k-', lw=0.5)  # invisible for legend spacing

plt.suptitle("Q3: Newton's Method — Convergence Rate Evidence", fontsize=13)
plt.tight_layout()
plt.savefig('figures/q3_newton_quadratic.pdf', bbox_inches='tight')
plt.close()
print("  Saved figures/q3_newton_quadratic.pdf")

# ============================================================
# FIGURE 3: Q2 — Comprehensive comparison with Nesterov rate
# Show that Nesterov achieves faster convergence than GD
# ============================================================
print("Generating Q2 acceleration comparison figure...")

N_Q2 = 300  # extended iterations

# Run methods with more iterations
gd_A_300 = gradient_descent(grad_A, loss_A, theta0_A, 0.08, N_Q2)
nest_A_300 = nesterov_momentum(grad_A, loss_A, theta0_A, 0.06, 0.90, N_Q2)
adam_A_300 = adam_optimiser(grad_A, loss_A, theta0_A, 0.12, 0.82, 0.999, 1e-8, N_Q2)

gd_C_300 = gradient_descent(grad_C, loss_C, x0_C, 0.0012, N_Q2)
nest_C_300 = nesterov_momentum(grad_C, loss_C, x0_C, 0.0007, 0.90, N_Q2)
adam_C_300 = adam_optimiser(grad_C, loss_C, x0_C, 0.006, 0.80, 0.999, 1e-8, N_Q2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
gap_gd_A = gd_A_300[1] - f_star_A
gap_nest_A = nest_A_300[1] - f_star_A
gap_adam_A = adam_A_300[1] - f_star_A

plot_gap(ax, gap_gd_A, 'GD (baseline)', 'k', '--')
plot_gap(ax, gap_nest_A, 'Nesterov', 'tab:blue', '-')
plot_gap(ax, gap_adam_A, 'Adam', 'tab:red', '-')

# Add reference O(1/k) and O(1/k^2) curves
k_ref = np.arange(1, N_Q2 + 2)
C1 = gap_gd_A[1] * 1.0  # calibrate
C2 = gap_nest_A[1] * 0.5
if C1 > 0 and C2 > 0:
    ax.plot(k_ref, C1 / k_ref, 'k:', lw=1.2, alpha=0.7, label='$O(1/k)$ reference')
    ax.plot(k_ref, C2 / k_ref**2, 'b:', lw=1.2, alpha=0.7, label='$O(1/k^2)$ reference')

ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)')
ax.set_title('Q2: Acceleration — Benchmark A\n(Linear Regression)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, N_Q2)

ax = axes[1]
gap_gd_C = gd_C_300[1] - f_star_C
gap_nest_C = nest_C_300[1] - f_star_C
gap_adam_C = adam_C_300[1] - f_star_C

plot_gap(ax, gap_gd_C, 'GD (baseline)', 'k', '--')
plot_gap(ax, gap_nest_C, 'Nesterov', 'tab:blue', '-')
plot_gap(ax, gap_adam_C, 'Adam', 'tab:red', '-')

ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)')
ax.set_title('Q2: Acceleration — Benchmark C\n(Rosenbrock, $f^\\star = 0$)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, N_Q2)

plt.suptitle("Q2: Momentum Methods — Convergence Gap $f(x_k) - f^\\star$ vs Iteration", fontsize=13)
plt.tight_layout()
plt.savefig('figures/q2_convergence_gap.pdf', bbox_inches='tight')
plt.close()
print("  Saved figures/q2_convergence_gap.pdf")

# ============================================================
# FIGURE 4: Benchmark Properties — Condition Number Illustration
# ============================================================
print("Generating benchmark properties figure...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Benchmark A: Hessian eigenvalue analysis
H_A = hessian_A_matrix()
eigs_A = np.linalg.eigvalsh(H_A)
L_A = eigs_A.max()
mu_A = eigs_A.min()
kappa_A = L_A / mu_A

ax = axes[0]
x1_A = np.linspace(-0.5, 6, 200)
x2_A = np.linspace(-0.5, 8, 200)
X1A, X2A = np.meshgrid(x1_A, x2_A)
ZA = np.vectorize(lambda a, b: loss_A(np.array([a, b])))(X1A, X2A)
cs = ax.contour(X1A, X2A, ZA, levels=20, cmap='viridis', alpha=0.8)
ax.plot(theta_ols[0], theta_ols[1], 'r*', ms=14, zorder=5,
        label=f'$\\theta^\\star \\approx ({theta_ols[0]:.2f}, {theta_ols[1]:.2f})$')
ax.plot(0, 0, 'k*', ms=12, label='Start $(0,0)$')
ax.set_xlabel('$\\theta_1$'); ax.set_ylabel('$\\theta_2$')
ax.set_title(f'Benchmark A: Quadratic Landscape\n$\\kappa = L/\\mu = {kappa_A:.1f}$, $L={L_A:.3f}$, $\\mu={mu_A:.3f}$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Benchmark B: varying curvature
x1_B = np.linspace(-2, 3, 300)
x2_B = np.linspace(-0.5, 5.5, 300)
X1B, X2B = np.meshgrid(x1_B, x2_B)
ZB = np.vectorize(lambda a, b: loss_B(np.array([a, b])))(X1B, X2B)

ax = axes[1]
cs = ax.contour(X1B, X2B, ZB, levels=25, cmap='viridis', alpha=0.8)
ax.plot(x_star_B[0], x_star_B[1], 'r*', ms=14, zorder=5,
        label=f'$x^\\star \\approx ({x_star_B[0]:.3f}, {x_star_B[1]:.3f})$')
ax.plot(x0_B[0], x0_B[1], 'k*', ms=12, label='Start $(-1, 4)$')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Benchmark B: Sinusoidal Perturbation\nCondition number $\\kappa \\approx 3$--$10$ (varying)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Benchmark C: Rosenbrock
x1_C = np.linspace(-1.5, 1.5, 300)
x2_C = np.linspace(-0.5, 2.0, 300)
X1C, X2C = np.meshgrid(x1_C, x2_C)
ZC = np.vectorize(lambda a, b: loss_C(np.array([a, b])))(X1C, X2C)

ax = axes[2]
cs = ax.contour(X1C, X2C, ZC, levels=np.logspace(-1, 3.5, 30), cmap='viridis', alpha=0.8)
ax.plot(1.0, 1.0, 'r*', ms=14, zorder=5, label='$x^\\star = (1, 1)$, $f^\\star = 0$')
ax.plot(x0_C[0], x0_C[1], 'k*', ms=12, label='Start $(-1, 1)$')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Benchmark C: Rosenbrock\n$\\kappa \\approx 2000$ at optimum')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.suptitle('Benchmark Problem Landscapes: Quadratic (A), Near-Quadratic (B), Ill-Conditioned (C)',
             fontsize=12)
plt.tight_layout()
plt.savefig('figures/benchmark_landscapes.pdf', bbox_inches='tight')
plt.close()
print("  Saved figures/benchmark_landscapes.pdf")

# ============================================================
# FIGURE 5: Q4 — Finite difference error as function of delta
# ============================================================
print("Generating finite difference error analysis figure...")

# For benchmark B at x0_B, compute FD error vs delta
x_test = np.array([0.5, 2.5])  # A point to evaluate at
true_grad_test = grad_B(x_test)

deltas = np.logspace(-10, 0, 100)
fd_errors = []
for delta in deltas:
    n = len(x_test)
    g_fd = np.zeros(n)
    f0 = loss_B(x_test)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        g_fd[i] = (loss_B(x_test + delta * ei) - f0) / delta
    fd_errors.append(np.linalg.norm(g_fd - true_grad_test))

fd_errors = np.array(fd_errors)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(deltas, fd_errors, 'b-', lw=2, label='FD gradient error $\\|\\hat{g} - g\\|$')
ax.axvline(0.05, color='g', linestyle='--', lw=1.5, label='$\\delta = 0.05$ (used, good)')
ax.axvline(0.8, color='r', linestyle='--', lw=1.5, label='$\\delta = 0.8$ (used, poor)')
ax.axvline(1.26e-8, color='orange', linestyle=':', lw=1.5, label='Near-optimal $\\delta \\approx 1.3\\times 10^{-8}$')

# Mark optimal delta region
opt_idx = np.argmin(fd_errors)
ax.plot(deltas[opt_idx], fd_errors[opt_idx], 'ko', ms=10, zorder=5,
        label=f'Min error at $\\delta \\approx {deltas[opt_idx]:.2e}$')

# Add reference lines
k_ref = deltas
ax.loglog(deltas[deltas > 1e-6], 2.0 * deltas[deltas > 1e-6], 'k--', lw=1, alpha=0.6,
          label='$O(\\delta)$ truncation')
machine_eps = 1e-16
f0_val = abs(loss_B(x_test))
ax.loglog(deltas[deltas < 1e-2], machine_eps * f0_val / deltas[deltas < 1e-2], 'k:', lw=1, alpha=0.6,
          label='$O(\\varepsilon_{\\mathrm{mach}}/\\delta)$ rounding')

ax.set_xlabel('Step size $\\delta$')
ax.set_ylabel('Gradient approximation error $\\|\\hat{g} - \\nabla f\\|$')
ax.set_title('Q4: Finite Difference Error vs Step Size $\\delta$\n'
             '(Benchmark B at $x = (0.5, 2.5)$)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('figures/q4_fd_error_analysis.pdf', bbox_inches='tight')
plt.close()
print("  Saved figures/q4_fd_error_analysis.pdf")

print("\nAll extra figures generated successfully.")
