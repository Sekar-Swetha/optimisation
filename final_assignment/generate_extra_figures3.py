"""
Additional figures for the report - Pass 8
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
# Figure 1: Rosenbrock Hessian eigenvalue spectrum along the valley
# ============================================================
def rosenbrock_hessian(x1, x2):
    H = np.array([
        [2 + 1200*x1**2 - 400*x2, -400*x1],
        [-400*x1, 200.0]
    ])
    return H

x1_vals = np.linspace(-1.0, 1.0, 200)
x2_valley = x1_vals**2  # valley floor

lam_min = []
lam_max = []
kappa = []

for x1, x2 in zip(x1_vals, x2_valley):
    H = rosenbrock_hessian(x1, x2)
    eigs = np.linalg.eigvalsh(H)
    lam_min.append(eigs[0])
    lam_max.append(eigs[1])
    kappa.append(eigs[1] / max(eigs[0], 1e-12))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.semilogy(x1_vals, lam_min, 'b-', lw=2, label=r'$\lambda_{\min}$')
ax.semilogy(x1_vals, lam_max, 'r-', lw=2, label=r'$\lambda_{\max}$')
ax.axvline(x=-1, color='k', ls='--', alpha=0.5, label='Start $x_0=(-1,1)$')
ax.axvline(x=1, color='green', ls='--', alpha=0.5, label='Optimum $x^*=(1,1)$')
ax.set_xlabel(r'$x_1$ (along valley $x_2 = x_1^2$)')
ax.set_ylabel('Eigenvalue (log scale)')
ax.set_title('Rosenbrock Hessian Eigenvalues Along Valley')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(x1_vals, kappa, 'purple', lw=2)
ax.axvline(x=-1, color='k', ls='--', alpha=0.5, label='Start $x_0=(-1,1)$')
ax.axvline(x=1, color='green', ls='--', alpha=0.5, label='Optimum $x^*=(1,1)$')
ax.axhline(y=2009, color='orange', ls=':', lw=1.5, label=r'$\kappa=2009$ (optimum)')
ax.set_xlabel(r'$x_1$ (along valley $x_2 = x_1^2$)')
ax.set_ylabel(r'Condition number $\kappa = \lambda_{\max}/\lambda_{\min}$  (log scale)')
ax.set_title(r'Condition Number Along Valley Floor')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Rosenbrock Hessian Spectrum: Why Damping is Required Globally',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q3_rosenbrock_hessian_spectrum.pdf', bbox_inches='tight')
plt.close()
print("Saved q3_rosenbrock_hessian_spectrum.pdf")

# ============================================================
# Figure 2: PL condition verification for Benchmarks A and B
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

theta_opt = np.linalg.solve(X_data.T @ X_data, X_data.T @ y_data) / m * m
f_star_A = loss_A(theta_opt)

def loss_B(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def grad_B(x):
    return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

# Find f_star_B numerically
from scipy.optimize import minimize
res = minimize(loss_B, [0.6, 2.0], method='Nelder-Mead')
f_star_B = res.fun

# Sample random points and check PL inequality: 0.5||grad||^2 >= mu*(f - f*)
np.random.seed(0)
n_pts = 500

# Benchmark A: sample near optimum
thetas = theta_opt + np.random.randn(n_pts, 2) * 2.0
lhs_A = [0.5 * np.dot(grad_A(t), grad_A(t)) for t in thetas]
rhs_A = [loss_A(t) - f_star_A for t in thetas]

# Benchmark B: sample near optimum
pts_B = np.array([[0.582, 2.0]]) + np.random.randn(n_pts, 2) * 1.0
lhs_B = [0.5 * np.dot(grad_B(p), grad_B(p)) for p in pts_B]
rhs_B = [max(0, loss_B(p) - f_star_B) for p in pts_B]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
rhs_A_arr = np.array(rhs_A)
lhs_A_arr = np.array(lhs_A)
sc = ax.scatter(rhs_A_arr, lhs_A_arr, c='blue', alpha=0.4, s=15, label='Sample points')
t_range = np.linspace(0, max(rhs_A_arr), 100)
# Estimate mu from min ratio
mu_A = np.min(lhs_A_arr / (rhs_A_arr + 1e-12))
ax.plot(t_range, mu_A * t_range, 'r-', lw=2, label=f'PL: $\\mu \\approx {mu_A:.3f}$')
ax.set_xlabel(r'$f(\theta) - f^{\star}$')
ax.set_ylabel(r'$\frac{1}{2}\|\nabla f(\theta)\|^2$')
ax.set_title('Benchmark A: PL Condition')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
rhs_B_arr = np.array(rhs_B)
lhs_B_arr = np.array(lhs_B)
sc = ax.scatter(rhs_B_arr, lhs_B_arr, c='green', alpha=0.4, s=15, label='Sample points')
t_range = np.linspace(0, max(rhs_B_arr), 100)
mu_B = np.min(lhs_B_arr / (rhs_B_arr + 1e-12))
ax.plot(t_range, mu_B * t_range, 'r-', lw=2, label=f'PL: $\\mu \\approx {mu_B:.3f}$')
ax.set_xlabel(r'$f(x) - f^{\star}$')
ax.set_ylabel(r'$\frac{1}{2}\|\nabla f(x)\|^2$')
ax.set_title('Benchmark B: PL Condition (local)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle(r'Polyak--Łojasiewicz Condition: $\frac{1}{2}\|\nabla f\|^2 \geq \mu(f - f^{\star})$',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q1_pl_condition.pdf', bbox_inches='tight')
plt.close()
print("Saved q1_pl_condition.pdf")

# ============================================================
# Figure 3: Theoretical convergence rate comparison on Rosenbrock
# ============================================================
k_vals = np.arange(1, 301)

# Theoretical curves
gd_rate = (1 - 1/2009)**np.arange(300)  # GD on Rosenbrock (starting from f0-f* ~ 1606)
f0 = 1606.0  # f(x0) - f* for Rosenbrock at (-1,1): (1-(-1))^2 + 100*(1-1)^2 = 4...
# Actual: f(-1,1) = (1-(-1))^2 + 100*(1-1)^2 = 4
f0_rosen = 4.0  # f(x0) for Rosenbrock at (-1,1) = 4

gd_theory = f0_rosen * (1 - 1/2009)**k_vals
hb_theory = f0_rosen * ((np.sqrt(2009)-1)/(np.sqrt(2009)+1))**k_vals
nesterov_theory = 2 * 1002.5 * 2.8**2 / (k_vals + 1)**2  # 2L||x0-x*||^2/(k+1)^2
# ||x0 - x*|| for Rosenbrock: ||(-1,1)-(1,1)|| = 2
nesterov_theory = 2 * 1002.5 * 4.0 / (k_vals + 1)**2

fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(k_vals, gd_theory, 'k-', lw=2, alpha=0.7, label=r'GD theory: $(1-1/\kappa)^k f_0$')
ax.semilogy(k_vals, hb_theory, 'b--', lw=2, alpha=0.7,
            label=r'Heavy Ball theory: $\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k f_0$')
ax.semilogy(k_vals, nesterov_theory, 'r-.', lw=2, alpha=0.7,
            label=r'Nesterov theory: $2L\|x_0-x^*\|^2/(k+1)^2$')

# Reference slopes
ref_k = np.array([50, 300])
ax.semilogy(ref_k, 1e2 * (ref_k/50.0)**(-2), 'r:', lw=1, alpha=0.5, label=r'$O(1/k^2)$ reference')
ax.semilogy(ref_k, 2 * (ref_k/50.0)**(-1), 'g:', lw=1, alpha=0.5, label=r'$O(1/k)$ reference')

ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'$f(x_k) - f^{\star}$ (theoretical upper bound, log scale)')
ax.set_title('Theoretical Convergence Rate Comparison on Rosenbrock ($\\kappa \\approx 2009$)')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim([1, 300])

plt.tight_layout()
plt.savefig('figures/q2_theoretical_rates.pdf', bbox_inches='tight')
plt.close()
print("Saved q2_theoretical_rates.pdf")

# ============================================================
# Figure 4: SGD batch size: bias-variance trade-off illustration
# ============================================================
m_bsz = 1000
X_bsz = np.random.randn(m_bsz, 2)
theta_star_bsz = np.array([3.0, 4.0])
y_bsz = X_bsz @ theta_star_bsz + np.random.randn(m_bsz)

def sgd_run(X, y, alpha, batch_size, n_epochs, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    theta = np.zeros(2)
    f_hist = []
    for ep in range(n_epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            batch = idx[i:i+batch_size]
            Xb, yb = X[batch], y[batch]
            g = Xb.T @ (Xb @ theta - yb) / len(batch)
            theta = theta - alpha * g
        r = X @ theta - y
        f_hist.append(0.5 * np.mean(r**2))
    return np.array(f_hist)

epochs = 60
alpha_sgd = 0.06

f_b5 = sgd_run(X_bsz, y_bsz, alpha_sgd, 5, epochs)
f_b20 = sgd_run(X_bsz, y_bsz, alpha_sgd, 20, epochs)
f_b100 = sgd_run(X_bsz, y_bsz, alpha_sgd, 100, epochs)
f_full = sgd_run(X_bsz, y_bsz, alpha_sgd, m_bsz, epochs)  # full GD

theta_opt_bsz = np.linalg.solve(X_bsz.T @ X_bsz, X_bsz.T @ y_bsz)
f_star_bsz = 0.5 * np.mean((X_bsz @ theta_opt_bsz - y_bsz)**2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ep_arr = np.arange(1, epochs+1)
ax = axes[0]
ax.semilogy(ep_arr, f_b5 - f_star_bsz + 1e-6, 'r-', lw=2, label='$b=5$ (variance floor highest)')
ax.semilogy(ep_arr, f_b20 - f_star_bsz + 1e-6, 'g-', lw=2, label='$b=20$')
ax.semilogy(ep_arr, f_b100 - f_star_bsz + 1e-6, 'b-', lw=2, label='$b=100$')
ax.semilogy(ep_arr, f_full - f_star_bsz + 1e-6, 'k--', lw=2, label='Full GD ($b=m$)')
ax.axhline(y=alpha_sgd / (2 * 0.9 * 5), color='r', ls=':', lw=1.5, alpha=0.6,
           label='Variance floor ($b=5$)')
ax.axhline(y=alpha_sgd / (2 * 0.9 * 100), color='b', ls=':', lw=1.5, alpha=0.6,
           label='Variance floor ($b=100$)')
ax.set_xlabel('Epoch')
ax.set_ylabel(r'$J(\theta) - J^{\star}$ (log scale)')
ax.set_title('SGD: Batch Size vs. Variance Floor\n(constant $\\alpha = 0.06$)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
# Per sample efficiency: x-axis = total samples processed
total_samples_b5 = ep_arr * m_bsz  # each epoch processes all m samples
total_samples_b100 = ep_arr * m_bsz
ax.semilogy(total_samples_b5, f_b5 - f_star_bsz + 1e-6, 'r-', lw=2, label='$b=5$')
ax.semilogy(total_samples_b100, f_b100 - f_star_bsz + 1e-6, 'b-', lw=2, label='$b=100$')
ax.semilogy(total_samples_b5, f_full - f_star_bsz + 1e-6, 'k--', lw=2, label='Full GD')
ax.set_xlabel('Total samples processed')
ax.set_ylabel(r'$J(\theta) - J^{\star}$ (log scale)')
ax.set_title('SGD: Sample Efficiency\n(equal total samples comparison)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('SGD Batch Size: Variance Floor and Sample Efficiency',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q2_sgd_variance_floor.pdf', bbox_inches='tight')
plt.close()
print("Saved q2_sgd_variance_floor.pdf")

print("All figures generated successfully.")
