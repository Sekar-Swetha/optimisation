"""
Second batch of additional figures for the improved report.
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
# Replicate benchmark definitions
# ============================================================
m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
eps_noise = np.random.randn(m)
y_data = X_data @ theta_star + eps_noise

def loss_A(theta): return 0.5*np.mean((X_data@theta - y_data)**2)
def grad_A(theta): return X_data.T@(X_data@theta - y_data)/m
def loss_B(x): return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def grad_B(x): return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])
def loss_C(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def grad_C(x):
    g1 = -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2)
    g2 = 200*(x[1]-x[0]**2)
    return np.array([g1, g2])

theta0_A = np.array([0.0, 0.0])
x0_B = np.array([-1.0, 4.0])
x0_C = np.array([-1.0, 1.0])

def gradient_descent(grad_fn, loss_fn, x0, alpha, n_iters):
    x = x0.copy().astype(float); x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for _ in range(n_iters):
        x = x - alpha*grad_fn(x); x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

def polyak_step(grad_fn, loss_fn, x0, f_star, eps, n_iters):
    x = x0.copy().astype(float); x_hist, f_hist, alpha_hist = [x.copy()], [loss_fn(x)], []
    for _ in range(n_iters):
        g = grad_fn(x); alpha_k = (loss_fn(x)-f_star)/(np.dot(g,g)+eps)
        alpha_hist.append(alpha_k); x = x - alpha_k*g
        x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist), np.array(alpha_hist)

def adagrad(grad_fn, loss_fn, x0, alpha0, eps, n_iters):
    x = x0.copy().astype(float); G = np.zeros_like(x)
    x_hist, f_hist, alpha_hist = [x.copy()], [loss_fn(x)], []
    for _ in range(n_iters):
        g = grad_fn(x); G += g**2; eff_alpha = alpha0/(np.sqrt(G)+eps)
        alpha_hist.append(np.mean(eff_alpha)); x = x - eff_alpha*g
        x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist), np.array(alpha_hist)

def heavy_ball(grad_fn, loss_fn, x0, alpha, beta, n_iters):
    x = x0.copy().astype(float); z = np.zeros_like(x)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x); z = beta*z + alpha*g; x = x - z
        x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

def nesterov_momentum(grad_fn, loss_fn, x0, alpha, beta_max, n_iters):
    x = x0.copy().astype(float); z = np.zeros_like(x)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for k in range(1, n_iters+1):
        beta_k = min((k-1)/(k+2), beta_max)
        lookahead = x + beta_k*z; g = grad_fn(lookahead)
        z = beta_k*z - alpha*g; x = x + z
        x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

def adam_optimiser(grad_fn, loss_fn, x0, alpha, beta1, beta2, eps, n_iters):
    x = x0.copy().astype(float); m_vec, v_vec = np.zeros_like(x), np.zeros_like(x)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for t in range(1, n_iters+1):
        g = grad_fn(x); m_vec = beta1*m_vec + (1-beta1)*g; v_vec = beta2*v_vec + (1-beta2)*g**2
        m_hat = m_vec/(1-beta1**t); v_hat = v_vec/(1-beta2**t)
        x = x - alpha*m_hat/(np.sqrt(v_hat)+eps)
        x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

# Known true minima
H_A = X_data.T @ X_data / m
b_A = X_data.T @ y_data / m
theta_ols = np.linalg.solve(H_A, b_A)
f_star_A = loss_A(theta_ols)
f_star_B = 0.724420
f_star_C = 0.0

# ============================================================
# FIGURE 1: Unified Q1 vs Q2 comparison bar chart (all methods, all benchmarks)
# ============================================================
print("Generating unified method comparison figure...")

N = 150  # common iteration count
# Q1 methods
gd_A = gradient_descent(grad_A, loss_A, theta0_A, 0.08, N)
poly_A = polyak_step(grad_A, loss_A, theta0_A, 0, 1e-4, N)
ada_A = adagrad(grad_A, loss_A, theta0_A, 1.8, 1e-5, N)
rms_A = adagrad(grad_A, loss_A, theta0_A, 0.22, 1e-5, N)  # approx with different alpha
hb_A = heavy_ball(grad_A, loss_A, theta0_A, 0.045, 0.88, N)
nest_A = nesterov_momentum(grad_A, loss_A, theta0_A, 0.06, 0.90, N)
adam_A = adam_optimiser(grad_A, loss_A, theta0_A, 0.12, 0.82, 0.999, 1e-8, N)

gd_C = gradient_descent(grad_C, loss_C, x0_C, 0.0012, N)
poly_C = polyak_step(grad_C, loss_C, x0_C, 0, 1e-3, N)
ada_C = adagrad(grad_C, loss_C, x0_C, 0.45, 1e-5, N)
hb_C = heavy_ball(grad_C, loss_C, x0_C, 0.0008, 0.86, N)
nest_C = nesterov_momentum(grad_C, loss_C, x0_C, 0.0007, 0.90, N)
adam_C = adam_optimiser(grad_C, loss_C, x0_C, 0.006, 0.80, 0.999, 1e-8, N)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Final values for Benchmark A
methods_A = ['GD', 'Polyak', 'Adagrad', 'Heavy Ball', 'Nesterov', 'Adam']
vals_A = [gd_A[1][-1], poly_A[1][-1], ada_A[1][-1], hb_A[1][-1], nest_A[1][-1], adam_A[1][-1]]
colors_A = ['gray', 'red', 'orange', 'blue', 'green', 'purple']

ax = axes[0]
bars = ax.bar(methods_A, np.array(vals_A) - f_star_A + 1e-15,
              color=colors_A, edgecolor='black', linewidth=0.8, alpha=0.8)
ax.axhline(0, color='k', linestyle='--', lw=1.5, alpha=0.5, label='$f^\\star$')
ax.set_yscale('log')
ax.set_ylabel('$f(x_k) - f^\\star$ after 150 iterations (log scale)')
ax.set_title('Benchmark A: Final Convergence Gap\n(Linear Regression, $\\kappa \\approx 1.12$)')
ax.tick_params(axis='x', rotation=20)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(bottom=1e-10)

# Annotate bars
for bar, val in zip(bars, vals_A):
    gap = val - f_star_A
    if gap > 1e-8:
        ax.text(bar.get_x() + bar.get_width()/2, gap*1.5,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=0)

# Final values for Benchmark C
methods_C = ['GD', 'Polyak', 'Adagrad', 'Heavy Ball', 'Nesterov', 'Adam']
vals_C = [gd_C[1][-1], poly_C[1][-1], ada_C[1][-1], hb_C[1][-1], nest_C[1][-1], adam_C[1][-1]]

ax = axes[1]
bars = ax.bar(methods_C, np.maximum(vals_C, 1e-10),
              color=colors_A, edgecolor='black', linewidth=0.8, alpha=0.8)
ax.axhline(f_star_C, color='k', linestyle='--', lw=1.5, alpha=0.5, label='$f^\\star = 0$')
ax.set_yscale('log')
ax.set_ylabel('$f(x_k)$ after 150 iterations (log scale)')
ax.set_title('Benchmark C: Final Objective\n(Rosenbrock, $\\kappa \\approx 2508$)')
ax.tick_params(axis='x', rotation=20)
ax.grid(True, axis='y', alpha=0.3)

for bar, val in zip(bars, vals_C):
    if val > 1e-5:
        ax.text(bar.get_x() + bar.get_width()/2, val*1.3,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Comparison of All Methods: Final Convergence After 150 Iterations', fontsize=13)
plt.tight_layout()
plt.savefig('figures/q_methods_comparison.pdf', bbox_inches='tight')
plt.close()
print("  Saved figures/q_methods_comparison.pdf")

# ============================================================
# FIGURE 2: Adagrad vs RMSprop: effective step-size decay analysis
# ============================================================
print("Generating Adagrad vs RMSprop step-size decay figure...")

def rmsprop(grad_fn, loss_fn, x0, alpha0, beta, eps, n_iters):
    x = x0.copy().astype(float); v = np.zeros_like(x)
    x_hist, f_hist, alpha_hist = [x.copy()], [loss_fn(x)], []
    for _ in range(n_iters):
        g = grad_fn(x); v = beta*v + (1-beta)*g**2
        eff_alpha = alpha0/(np.sqrt(v)+eps)
        alpha_hist.append(np.mean(eff_alpha)); x = x - eff_alpha*g
        x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist), np.array(alpha_hist)

N_step = 200
ada_B_step = adagrad(grad_B, loss_B, x0_B, 1.2, 1e-5, N_step)
rms_B_step = rmsprop(grad_B, loss_B, x0_B, 0.14, 0.9, 1e-5, N_step)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.semilogy(ada_B_step[2], 'b-', lw=2, label='Adagrad (mean eff.\ step)')
ax.semilogy(rms_B_step[2], 'r-', lw=2, label='RMSprop (mean eff.\ step)')

# Reference: Adagrad decays as alpha0 / sqrt(k * bar_g^2) ~ 1/sqrt(k)
k_ref = np.arange(1, N_step+1)
# Calibrate Adagrad decay
if ada_B_step[2][0] > 0:
    C_ada = ada_B_step[2][0] * np.sqrt(1)
    ax.loglog(k_ref, C_ada / np.sqrt(k_ref), 'b:', lw=1.5, alpha=0.7, label='$O(1/\\sqrt{k})$ reference')

ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Effective Step Size')
ax.set_title('Adagrad vs RMSprop: Step-Size Decay\n(Benchmark B, log-log scale)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')
ax.set_xscale('log')

ax = axes[1]
ax.semilogy(ada_B_step[1], 'b-', lw=2, label=f'Adagrad ($f_{{end}}={ada_B_step[1][-1]:.4f}$)')
ax.semilogy(rms_B_step[1], 'r-', lw=2, label=f'RMSprop ($f_{{end}}={rms_B_step[1][-1]:.4f}$)')
ax.axhline(f_star_B, color='k', linestyle='--', lw=1.5, alpha=0.7, label=f'$f^\\star \\approx {f_star_B:.4f}$')
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Value (log scale)')
ax.set_title('Adagrad vs RMSprop: Convergence\n(Benchmark B)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Adagrad vs RMSprop: Effective Step Size and Convergence', fontsize=13)
plt.tight_layout()
plt.savefig('figures/q1_adagrad_rmsprop_detail.pdf', bbox_inches='tight')
plt.close()
print("  Saved figures/q1_adagrad_rmsprop_detail.pdf")

# ============================================================
# FIGURE 3: Q5 Augmented Lagrangian comparison
# ============================================================
print("Generating Q5 augmented Lagrangian vs penalty figure...")

def loss_B(x): return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def grad_B(x): return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

x0_q5 = np.array([0.2, 4.0])

def project_q5(x):
    x_p = x.copy(); x_p[0] = max(0.5, x_p[0]); return x_p

def projected_gd_q5(x0, alpha, n_iters):
    x = x0.copy().astype(float); x_hist = [x.copy()]; f_hist = [loss_B(x)]
    for _ in range(n_iters):
        g = grad_B(x); x = project_q5(x - alpha*g)
        x_hist.append(x.copy()); f_hist.append(loss_B(x))
    return np.array(x_hist), np.array(f_hist)

def penalty_gd_q5(x0, alpha, lam, n_iters):
    x = x0.copy().astype(float)
    x_hist = [x.copy()]; f_hist = [loss_B(x)]
    for _ in range(n_iters):
        g = grad_B(x).copy()
        if x[0] < 0.5: g[0] -= lam
        x = x - alpha*g
        x_hist.append(x.copy()); f_hist.append(loss_B(x))
    return np.array(x_hist), np.array(f_hist)

# Augmented Lagrangian method
def augmented_lagrangian_q5(x0, alpha, mu_aug, n_outer, n_inner):
    """Augmented Lagrangian for min f(x) s.t. x1 >= 0.5 (g(x) = 0.5 - x1 <= 0)"""
    x = x0.copy().astype(float)
    lam = 0.0  # Lagrange multiplier
    x_hist = [x.copy()]; f_hist = [loss_B(x)]
    for outer in range(n_outer):
        for inner in range(n_inner):
            # Gradient of augmented Lagrangian: f(x) + lam*max(0, 0.5-x1) + mu/2*max(0, 0.5-x1)^2
            g = grad_B(x).copy()
            viol = max(0.0, 0.5 - x[0])
            if x[0] < 0.5:
                g[0] += -lam - mu_aug * viol
            x = x - alpha * g
            x_hist.append(x.copy()); f_hist.append(loss_B(x))
        # Update multiplier
        viol = max(0.0, 0.5 - x[0])
        lam = max(0.0, lam + mu_aug * viol)
    return np.array(x_hist), np.array(f_hist)

N_q5 = 100
pgd = projected_gd_q5(x0_q5, 0.08, N_q5)
pen_45 = penalty_gd_q5(x0_q5, 0.03, 4.5, N_q5)
pen_015 = penalty_gd_q5(x0_q5, 0.05, 0.15, N_q5)
# AL with 5 outer iterations of 20 inner = 100 total
al = augmented_lagrangian_q5(x0_q5, 0.05, 2.0, 5, 20)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.semilogy([loss_B(x) for x in pgd[0]], 'b-', lw=2, label='Projected GD')
ax.semilogy([loss_B(x) for x in pen_015[0]], 'g--', lw=2, label='Penalty $\\lambda=0.15$')
ax.semilogy([loss_B(x) for x in pen_45[0]], 'r--', lw=2, label='Penalty $\\lambda=4.5$')
ax.semilogy([loss_B(x) for x in al[0]], 'm-', lw=2, label='Aug.\ Lagrangian ($\\mu=2$)')
ax.axhline(f_star_B, color='k', linestyle=':', lw=1.5, label=f'$f^\\star={f_star_B:.4f}$')
ax.set_xlabel('Iteration'); ax.set_ylabel('$f(x)$ (log scale)')
ax.set_title('Q5: Constrained Optimisation Methods\n(Objective Value)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
def violation(x_hist): return np.array([max(0, 0.5 - x[0]) for x in x_hist])

v_pgd = violation(pgd[0])
v_pen015 = violation(pen_015[0])
v_pen45 = violation(pen_45[0])
v_al = violation(al[0])

for v, lab, c, ls in [(v_pgd, 'Projected GD', 'b', '-'),
                       (v_pen015, 'Penalty $\\lambda=0.15$', 'g', '--'),
                       (v_pen45, 'Penalty $\\lambda=4.5$', 'r', '--'),
                       (v_al, 'Aug.\ Lagrangian ($\\mu=2$)', 'm', '-')]:
    mask = v > 1e-12
    if np.any(mask):
        ax.semilogy(np.where(mask)[0], v[mask], color=c, linestyle=ls, lw=2, label=lab)
    else:
        ax.semilogy([0], [1e-16], color=c, linestyle=ls, lw=2, label=lab + ' (feasible)')
ax.set_xlabel('Iteration'); ax.set_ylabel('Constraint violation (log)')
ax.set_title('Q5: Constraint Violation\n$\\max(0, 0.5 - x_1)$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle('Q5: Comparison of Constrained Optimisation Approaches', fontsize=13)
plt.tight_layout()
plt.savefig('figures/q5_augmented_lagrangian.pdf', bbox_inches='tight')
plt.close()
print("  Saved figures/q5_augmented_lagrangian.pdf")

print("\nAll pass-2 extra figures generated successfully.")
