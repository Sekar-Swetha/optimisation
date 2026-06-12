"""
Additional figures for the report - Pass 30
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
# Benchmark A: linear regression quadratic
# ============================================================
m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
y_data = X_data @ theta_star + np.random.randn(m)
H_A = X_data.T @ X_data / m
b_A = X_data.T @ y_data / m
theta_opt = np.linalg.solve(H_A, b_A)
f_star_A = 0.5 * np.mean((X_data @ theta_opt - y_data)**2)

def loss_A(t): return 0.5 * np.mean((X_data @ t - y_data)**2)
def grad_A(t): return H_A @ t - b_A

# ============================================================
# Figure 1: CG vs GD vs Newton on Benchmark A
# ============================================================
theta0 = np.array([0.0, 0.0])

# GD (80 iterations)
def run_gd_A(n_iters, alpha=0.08):
    t = theta0.copy().astype(float)
    f_hist = [loss_A(t)]
    for _ in range(n_iters):
        t = t - alpha * grad_A(t)
        f_hist.append(loss_A(t))
    return np.array(f_hist)

# Newton (exact, single step on quadratic)
def run_newton_A(n_iters):
    t = theta0.copy().astype(float)
    f_hist = [loss_A(t)]
    for _ in range(n_iters):
        g = grad_A(t)
        t = t - np.linalg.solve(H_A, g)
        f_hist.append(loss_A(t))
    return np.array(f_hist)

# Conjugate Gradient (exact)
def run_cg_A(n_iters):
    t = theta0.copy().astype(float)
    f_hist = [loss_A(t)]
    r = grad_A(t)  # residual = A*t - b
    p = -r.copy()  # search direction
    for _ in range(n_iters):
        Ap = H_A @ p
        alpha_k = np.dot(r, r) / np.dot(p, Ap)
        t = t + alpha_k * p
        r_new = r + alpha_k * Ap
        if np.linalg.norm(r_new) < 1e-12:
            f_hist.append(loss_A(t))
            f_hist.extend([f_hist[-1]] * (n_iters - len(f_hist) + 1))
            return np.array(f_hist[:n_iters+1])
        beta_k = np.dot(r_new, r_new) / np.dot(r, r)
        p = -r_new + beta_k * p
        r = r_new
        f_hist.append(loss_A(t))
    return np.array(f_hist)

n_show = 20
f_gd = run_gd_A(n_show)
f_newton = run_newton_A(n_show)
f_cg = run_cg_A(n_show)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

k_arr = np.arange(n_show+1)

ax = axes[0]
ax.semilogy(k_arr, np.maximum(f_gd - f_star_A, 1e-15), 'k-', lw=2, label='GD ($\\alpha=0.08$)')
ax.semilogy(k_arr, np.maximum(f_newton - f_star_A, 1e-15), 'r--', lw=2, label='Newton ($\\alpha=1$, quadratic)')
ax.semilogy(k_arr, np.maximum(f_cg - f_star_A, 1e-15), 'b-.', lw=2.5, label='CG (exact 2-step)')
ax.axvline(x=2, color='b', ls=':', lw=1.5, alpha=0.7, label='CG terminates ($k=d=2$)')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'$J(\theta_k) - J^\star$ (log scale)')
ax.set_title('GD vs Newton vs CG on Benchmark A\n(Quadratic Loss, $d=2$, $\\kappa \\approx 10$)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, n_show])

ax = axes[1]
ax.semilogy(k_arr, np.maximum(f_gd - f_star_A, 1e-15), 'k-', lw=2, label='GD ($\\alpha=0.08$)')
ax.semilogy(k_arr, np.maximum(f_newton - f_star_A, 1e-15), 'r--', lw=2, label='Newton ($\\alpha=1$)')
ax.semilogy(k_arr, np.maximum(f_cg - f_star_A, 1e-15), 'b-.', lw=2.5, label='CG')
# Add theoretical GD rate
kk = np.arange(1, n_show+1)
kappa_A = np.linalg.eigvalsh(H_A)[-1] / np.linalg.eigvalsh(H_A)[0]
gd_theory = (f_gd[0] - f_star_A) * ((kappa_A - 1)/(kappa_A + 1))**(2*kk)
ax.semilogy(kk, gd_theory, 'k:', lw=1.5, alpha=0.7, label=f'GD theory: $(({kappa_A:.1f}-1)/({kappa_A:.1f}+1))^k$')
# CG theoretical rate
cg_theory = (f_gd[0] - f_star_A) * (2 * ((np.sqrt(kappa_A)-1)/(np.sqrt(kappa_A)+1))**kk)
ax.semilogy(kk, cg_theory, 'b:', lw=1.5, alpha=0.7, label=r'CG theory: $O(((\sqrt{\kappa}-1)/(\sqrt{\kappa}+1))^k)$')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'$J(\theta_k) - J^\star$ (log scale)')
ax.set_title('With Theoretical Convergence Bounds')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, n_show])

plt.suptitle('Conjugate Gradient: Exact 2-Step Termination on 2D Quadratic Benchmark A\n(Newton = 1 step; GD = 80+ steps to match CG accuracy)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q3_cg_vs_newton_vs_gd.pdf', bbox_inches='tight')
plt.close()
print("Saved q3_cg_vs_newton_vs_gd.pdf")

# ============================================================
# Figure 2: Nesterov momentum schedule and energy function
# ============================================================
def beta_schedule(k, beta_max=0.90):
    return min((k-1)/(k+2), beta_max) if k > 1 else 0.0

k_vals = np.arange(1, 121)
betas = np.array([beta_schedule(k) for k in k_vals])

# Effective per-step rate for different strategies
# Nesterov (adaptive)
# Fixed beta = 0.90
# Fixed beta = 0.70
# No momentum (GD)

# Convergence curves comparison on Benchmark C (Rosenbrock)
def loss_C(x): return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
def grad_C(x):
    dx1 = -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2)
    dx2 = 200*(x[1]-x[0]**2)
    return np.array([dx1, dx2])

x0 = np.array([-1.0, 1.0])
N = 120
alpha_nest = 0.0007

def run_nesterov_beta_max(x0, alpha, bmax, n):
    x = x0.copy().astype(float)
    z = np.zeros(2)
    f = [loss_C(x)]
    for k in range(1, n+1):
        bk = min((k-1)/(k+2), bmax)
        la = x + bk*z
        z = bk*z - alpha*grad_C(la)
        x = x + z
        f.append(loss_C(x))
    return np.array(f)

def run_nesterov_fixed_beta(x0, alpha, beta, n):
    x = x0.copy().astype(float)
    z = np.zeros(2)
    f = [loss_C(x)]
    for k in range(1, n+1):
        la = x + beta*z
        z = beta*z - alpha*grad_C(la)
        x = x + z
        f.append(loss_C(x))
    return np.array(f)

f_adaptive = run_nesterov_beta_max(x0, alpha_nest, 0.90, N)
f_fixed090 = run_nesterov_fixed_beta(x0, alpha_nest, 0.90, N)
f_fixed070 = run_nesterov_fixed_beta(x0, alpha_nest, 0.70, N)
f_fixed050 = run_nesterov_fixed_beta(x0, alpha_nest, 0.50, N)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

k_arr = np.arange(N+1)

ax = axes[0]
ax.plot(k_vals, betas, 'b-', lw=2, label=r'Adaptive: $\beta_k = (k-1)/(k+2)$, capped at 0.90')
ax.axhline(y=0.90, color='orange', ls='--', lw=1.5, label=r'Fixed $\beta = 0.90$')
ax.axhline(y=0.70, color='green', ls='--', lw=1.5, label=r'Fixed $\beta = 0.70$')
ax.axhline(y=0.50, color='red', ls='--', lw=1.5, label=r'Fixed $\beta = 0.50$')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'Momentum $\beta_k$')
ax.set_title(r'Nesterov Momentum Schedule: Adaptive vs Fixed $\beta$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.05, 1.0])

ax = axes[1]
ax.semilogy(k_arr, f_adaptive + 1e-10, 'b-', lw=2, label=r'Adaptive $\beta_k$ (capped 0.90)')
ax.semilogy(k_arr, f_fixed090 + 1e-10, 'orange', lw=1.8, ls='--', label=r'Fixed $\beta = 0.90$')
ax.semilogy(k_arr, f_fixed070 + 1e-10, 'green', lw=1.8, ls='--', label=r'Fixed $\beta = 0.70$')
ax.semilogy(k_arr, f_fixed050 + 1e-10, 'red', lw=1.8, ls='--', label=r'Fixed $\beta = 0.50$')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'$f(x_k)$ (log scale)')
ax.set_title('Effect of Momentum Schedule on Convergence\n(Benchmark C, Rosenbrock)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Nesterov Momentum: Adaptive Schedule vs Fixed $\\beta$ Comparison',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q2_nesterov_momentum_schedule.pdf', bbox_inches='tight')
plt.close()
print("Saved q2_nesterov_momentum_schedule.pdf")

print("All figures generated successfully.")
