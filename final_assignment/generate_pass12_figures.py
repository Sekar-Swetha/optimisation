"""
Pass 12: Newton quadratic convergence verification and Adagrad/RMSprop effective learning rate.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# ============================================================
# Benchmark B: Newton quadratic convergence verification
# ============================================================
def loss_B(x):
    return (x[0] - 1)**2 + 5*(x[1] - 2)**2 + np.sin(x[0])

def grad_B(x):
    return np.array([2*(x[0] - 1) + np.cos(x[0]), 10*(x[1] - 2)])

def hessian_B(x):
    return np.array([[2 - np.sin(x[0]), 0], [0, 10]])

x_star_B = np.array([0.5818, 2.0])  # approximate optimum
x0_B = np.array([-1.0, 4.0])

# Newton with alpha=1 on Benchmark B
x = x0_B.copy().astype(float)
errors = [np.linalg.norm(x - x_star_B)]
for _ in range(12):
    g = grad_B(x)
    H = hessian_B(x) + 1e-10 * np.eye(2)
    p = np.linalg.solve(H, g)
    x = x - p  # full Newton step (alpha=1)
    errors.append(np.linalg.norm(x - x_star_B))
errors = np.array(errors)

# GD with alpha=0.06 for comparison
x_gd = x0_B.copy().astype(float)
errors_gd = [np.linalg.norm(x_gd - x_star_B)]
for _ in range(50):
    x_gd = x_gd - 0.06 * grad_B(x_gd)
    errors_gd.append(np.linalg.norm(x_gd - x_star_B))
errors_gd = np.array(errors_gd)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: error norm vs iteration (log scale)
ax = axes[0]
k_newton = np.arange(len(errors))
k_gd = np.arange(len(errors_gd))
ax.semilogy(k_newton, errors, 'r-o', lw=2, markersize=6, label='Newton (full step, $\\alpha=1$)')
ax.semilogy(k_gd[:20], errors_gd[:20], 'b-', lw=2, label='GD ($\\alpha=0.06$)')
ax.set_xlabel('Iteration')
ax.set_ylabel('$\\|x_k - x^\\star\\|$ (log scale)')
ax.set_title('Newton vs GD Error Norm\n(Benchmark B)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: log-log to show convergence rate
ax = axes[1]
# Filter to iterations where error > machine epsilon
valid = errors > 1e-14
k_valid = k_newton[valid]
err_valid = errors[valid]
if len(err_valid) > 2:
    ax.loglog(k_valid[1:], err_valid[1:], 'r-o', lw=2, markersize=6, label='Newton $\\|e_k\\|$')
    # Quadratic reference: if e_{k+1} = C * e_k^2, then log e_{k+1} = log C + 2 log e_k
    # On log-log: slope 2 in log(e_{k+1}) vs log(e_k)
    ax.loglog(k_valid[1:], err_valid[0] * (err_valid[0]/err_valid[0])**k_valid[1:], 'k--', lw=1, alpha=0.5)
    # Add quadratic reference from the first non-trivial error
    # Quadratic reference: e_k = e_{k-1}^2 / e0 (passes through first two points)
    e_quad = [err_valid[0]]
    for _ in range(len(err_valid) - 1):
        next_val = e_quad[-1]**2 / err_valid[0]
        if next_val < 1e-16:
            break
        e_quad.append(next_val)
    e_quad = np.array(e_quad)
    n_q = min(len(e_quad), len(k_valid))
    ax.loglog(k_valid[:n_q], e_quad[:n_q], 'g--', lw=1.5, label='Quadratic ref $O(\\|e_{k-1}\\|^2)$')

ax.loglog(k_gd[1:20]+1, errors_gd[1:20], 'b-', lw=2, label='GD (linear rate)')
ax.set_xlabel('Iteration')
ax.set_ylabel('$\\|x_k - x^\\star\\|$')
ax.set_title('Convergence Rate Verification (log--log)\nQuadratic vs Linear')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

plt.suptitle('Q3: Newton Quadratic Convergence Verification on Benchmark B', fontsize=11)
plt.tight_layout()
plt.savefig('figures/q3_newton_quadratic.pdf', bbox_inches='tight')
plt.close()
print("Saved q3_newton_quadratic.pdf")

# ============================================================
# FIGURE 2: Effective learning rate for Adagrad vs RMSprop vs Adam
# ============================================================

def loss_A(theta):
    m = 1000
    np.random.seed(42)
    X = np.random.randn(m, 2)
    theta_star = np.array([3.0, 4.0])
    eps = np.random.randn(m)
    y = X @ theta_star + eps
    r = X @ theta - y
    return 0.5 * np.mean(r**2), X.T @ r / m  # also returns grad

# Run Adagrad, RMSprop, Adam and track effective learning rates
m = 1000
np.random.seed(42)
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
eps_noise = np.random.randn(m)
y_data = X_data @ theta_star + eps_noise

def loss_fn(theta):
    r = X_data @ theta - y_data
    return 0.5 * np.mean(r**2)

def grad_fn(theta):
    r = X_data @ theta - y_data
    return X_data.T @ r / m

n_iters = 80
theta0 = np.array([0.0, 0.0])

# Adagrad
theta = theta0.copy().astype(float)
G = np.zeros(2)
eff_lr_adagrad = []
for _ in range(n_iters):
    g = grad_fn(theta)
    G += g**2
    eff_lr = 1.8 / (np.sqrt(G) + 1e-5)
    eff_lr_adagrad.append(np.mean(eff_lr))
    theta = theta - eff_lr * g
eff_lr_adagrad = np.array(eff_lr_adagrad)

# RMSprop
theta = theta0.copy().astype(float)
v = np.zeros(2)
eff_lr_rmsprop = []
for _ in range(n_iters):
    g = grad_fn(theta)
    v = 0.9 * v + 0.1 * g**2
    eff_lr = 0.22 / (np.sqrt(v) + 1e-5)
    eff_lr_rmsprop.append(np.mean(eff_lr))
    theta = theta - eff_lr * g
eff_lr_rmsprop = np.array(eff_lr_rmsprop)

# Adam
theta = theta0.copy().astype(float)
m_mom = np.zeros(2)
v_mom = np.zeros(2)
eff_lr_adam = []
for t in range(1, n_iters + 1):
    g = grad_fn(theta)
    m_mom = 0.9 * m_mom + 0.1 * g
    v_mom = 0.999 * v_mom + 0.001 * g**2
    m_hat = m_mom / (1 - 0.9**t)
    v_hat = v_mom / (1 - 0.999**t)
    eff_lr = 0.12 / (np.sqrt(v_hat) + 1e-8)
    eff_lr_adam.append(np.mean(eff_lr))
    theta = theta - eff_lr * m_hat
eff_lr_adam = np.array(eff_lr_adam)

fig, ax = plt.subplots(figsize=(9, 5))
k = np.arange(1, n_iters + 1)
ax.semilogy(k, eff_lr_adagrad, 'b-', lw=2, label='Adagrad (monotone decay)')
ax.semilogy(k, eff_lr_rmsprop, 'g-', lw=2, label='RMSprop (stabilises near const.)')
ax.semilogy(k, eff_lr_adam, 'r-', lw=2, label='Adam (bias-corrected, stabilises)')
ax.axhline(y=0.08, color='k', lw=1.5, linestyle='--', alpha=0.7, label='GD fixed $\\alpha=0.08$')
ax.set_xlabel('Iteration')
ax.set_ylabel('Effective learning rate (mean over coordinates, log scale)')
ax.set_title('Q1/Q2: Effective Learning Rate Comparison\n(Adagrad, RMSprop, Adam on Benchmark A)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q1_effective_lr.pdf', bbox_inches='tight')
plt.close()
print("Saved q1_effective_lr.pdf")

print("\nPass 12 figures generated.")
