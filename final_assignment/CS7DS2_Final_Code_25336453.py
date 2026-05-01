"""
CS7DS2 - Optimisation Algorithms for Data Analysis - Final Assignment
Name: Swetha Sekar
Student ID: 25336453
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
# BENCHMARK DEFINITIONS
# ============================================================

# Benchmark A: Linear Regression Quadratic Loss
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

# Benchmark B: Toy Neural Network Quadratic Loss
def loss_B(x):
    return (x[0] - 1)**2 + 5*(x[1] - 2)**2 + np.sin(x[0])

def grad_B(x):
    return np.array([2*(x[0] - 1) + np.cos(x[0]), 10*(x[1] - 2)])

def hessian_B(x):
    return np.array([[2 - np.sin(x[0]), 0], [0, 10]])

# Benchmark C: Rosenbrock Function
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

# Starting points
theta0_A = np.array([0.0, 0.0])
x0_B = np.array([-1.0, 4.0])
x0_C = np.array([-1.0, 1.0])

# ============================================================
# OPTIMISER IMPLEMENTATIONS
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
# CONTOUR GRID HELPERS
# ============================================================

x1_grid_B = np.linspace(-2, 3, 400)
x2_grid_B = np.linspace(-0.5, 5.5, 400)
X1B, X2B = np.meshgrid(x1_grid_B, x2_grid_B)
ZB = np.vectorize(lambda a, b: loss_B(np.array([a, b])))(X1B, X2B)

x1_grid_C = np.linspace(-1.5, 1.5, 400)
x2_grid_C = np.linspace(-0.5, 2.0, 400)
X1C, X2C = np.meshgrid(x1_grid_C, x2_grid_C)
ZC = np.vectorize(lambda a, b: loss_C(np.array([a, b])))(X1C, X2C)

def contour_B(ax):
    ax.contour(X1B, X2B, ZB, levels=30, cmap='viridis', alpha=0.7)

def contour_C(ax):
    ax.contour(X1C, X2C, ZC, levels=np.logspace(-1, 3.5, 30), cmap='viridis', alpha=0.7)

print("="*60)
print("QUESTION 1: Adaptive Step-Size Methods")
print("="*60)

# ============================================================
# QUESTION 1
# ============================================================

# Run all Q1 methods on each benchmark
q1_results = {}
for bname, grad_fn, loss_fn, x0, params in [
    ('A', grad_A, loss_A, theta0_A, {
        'polyak': {'f_star': 0, 'eps': 1e-4},
        'adagrad': {'alpha0': 1.8, 'eps': 1e-5},
        'rmsprop': {'alpha0': 0.22, 'beta': 0.9, 'eps': 1e-5},
        'hb': {'alpha': 0.045, 'beta': 0.88},
        'gd': {'alpha': 0.08}
    }),
    ('B', grad_B, loss_B, x0_B, {
        'polyak': {'f_star': 0, 'eps': 1e-4},
        'adagrad': {'alpha0': 1.2, 'eps': 1e-5},
        'rmsprop': {'alpha0': 0.14, 'beta': 0.9, 'eps': 1e-5},
        'hb': {'alpha': 0.035, 'beta': 0.90},
        'gd': {'alpha': 0.06}
    }),
    ('C', grad_C, loss_C, x0_C, {
        'polyak': {'f_star': 0, 'eps': 1e-3},
        'adagrad': {'alpha0': 0.45, 'eps': 1e-5},
        'rmsprop': {'alpha0': 0.0035, 'beta': 0.9, 'eps': 1e-5},
        'hb': {'alpha': 0.0008, 'beta': 0.86},
        'gd': {'alpha': 0.0012}
    })
]:
    n = 120
    res = {}
    res['gd'] = gradient_descent(grad_fn, loss_fn, x0, params['gd']['alpha'], n)
    p = params['polyak']
    res['polyak'] = polyak_step(grad_fn, loss_fn, x0, p['f_star'], p['eps'], n)
    p = params['adagrad']
    res['adagrad'] = adagrad(grad_fn, loss_fn, x0, p['alpha0'], p['eps'], n)
    p = params['rmsprop']
    res['rmsprop'] = rmsprop(grad_fn, loss_fn, x0, p['alpha0'], p['beta'], p['eps'], n)
    p = params['hb']
    res['hb'] = heavy_ball(grad_fn, loss_fn, x0, p['alpha'], p['beta'], n)
    q1_results[bname] = res
    print(f"  Benchmark {bname} done")

# Q1(b): Objective vs iteration for each benchmark
for bname in ['A', 'B', 'C']:
    res = q1_results[bname]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(res['gd'][1], 'k--', lw=1.5, label='GD (baseline)')
    ax.semilogy(res['polyak'][1], lw=2, label='Polyak Step')
    ax.semilogy(res['adagrad'][1], lw=2, label='Adagrad')
    ax.semilogy(res['rmsprop'][1], lw=2, label='RMSprop')
    ax.semilogy(res['hb'][1], lw=2, label='Heavy Ball')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value (log scale)')
    title_map = {'A': 'Linear Regression', 'B': 'Toy Neural Network', 'C': 'Rosenbrock'}
    ax.set_title(f'Q1: Convergence Comparison — Benchmark {bname} ({title_map[bname]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/q1_fval_{bname}.pdf', bbox_inches='tight')
    plt.close()

# Q1(c): Contour + trajectories for B and C
for bname, contour_fn, X1g, X2g in [('B', contour_B, X1B, X2B), ('C', contour_C, X1C, X2C)]:
    res = q1_results[bname]
    fig, ax = plt.subplots(figsize=(8, 7))
    contour_fn(ax)
    labels = {'gd': 'GD', 'polyak': 'Polyak', 'adagrad': 'Adagrad', 'rmsprop': 'RMSprop', 'hb': 'Heavy Ball'}
    styles = {'gd': ('k', '--'), 'polyak': ('tab:blue', '-'), 'adagrad': ('tab:orange', '-'),
              'rmsprop': ('tab:green', '-'), 'hb': ('tab:red', '-')}
    for method in ['gd', 'polyak', 'adagrad', 'rmsprop', 'hb']:
        hist = res[method][0]
        c, ls = styles[method]
        ax.plot(hist[:, 0], hist[:, 1], color=c, linestyle=ls, lw=1.5, alpha=0.8, label=labels[method])
    x0_used = x0_B if bname == 'B' else x0_C
    ax.plot(*x0_used, 'k*', ms=14, label='Start')
    opt = [0.61, 2.0] if bname == 'B' else [1.0, 1.0]
    ax.plot(*opt, 'r*', ms=14, label='Optimum')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    title_map = {'B': 'Toy Neural Network', 'C': 'Rosenbrock'}
    ax.set_title(f'Q1: Optimisation Trajectories — Benchmark {bname} ({title_map[bname]})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'figures/q1_contour_{bname}.pdf', bbox_inches='tight')
    plt.close()

# Q1(d): Adaptive step-size evolution
for bname in ['A', 'B', 'C']:
    res = q1_results[bname]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res['polyak'][2], lw=2, label='Polyak')
    ax.plot(res['adagrad'][2], lw=2, label='Adagrad (mean eff.)')
    ax.plot(res['rmsprop'][2], lw=2, label='RMSprop (mean eff.)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Effective Step Size')
    title_map = {'A': 'Linear Regression', 'B': 'Toy Neural Network', 'C': 'Rosenbrock'}
    ax.set_title(f'Q1: Step-Size Evolution — Benchmark {bname} ({title_map[bname]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/q1_stepsize_{bname}.pdf', bbox_inches='tight')
    plt.close()

print("Q1 plots saved.")

# ============================================================
# QUESTION 2
# ============================================================
print("\n" + "="*60)
print("QUESTION 2: Momentum and Stochastic Methods")
print("="*60)

q2_results = {}
for bname, grad_fn, loss_fn, x0, params in [
    ('A', grad_A, loss_A, theta0_A, {
        'nesterov': {'alpha': 0.06, 'beta_max': 0.90},
        'adam': {'alpha': 0.12, 'beta1': 0.82, 'beta2': 0.999, 'eps': 1e-8},
        'gd': {'alpha': 0.08}
    }),
    ('B', grad_B, loss_B, x0_B, {
        'nesterov': {'alpha': 0.035, 'beta_max': 0.92},
        'adam': {'alpha': 0.08, 'beta1': 0.80, 'beta2': 0.999, 'eps': 1e-8},
        'gd': {'alpha': 0.06}
    }),
    ('C', grad_C, loss_C, x0_C, {
        'nesterov': {'alpha': 0.0007, 'beta_max': 0.90},
        'adam': {'alpha': 0.006, 'beta1': 0.80, 'beta2': 0.999, 'eps': 1e-8},
        'gd': {'alpha': 0.0012}
    })
]:
    n = 150
    res = {}
    res['gd'] = gradient_descent(grad_fn, loss_fn, x0, params['gd']['alpha'], n)
    p = params['nesterov']
    res['nesterov'] = nesterov_momentum(grad_fn, loss_fn, x0, p['alpha'], p['beta_max'], n)
    p = params['adam']
    res['adam'] = adam_optimiser(grad_fn, loss_fn, x0, p['alpha'], p['beta1'], p['beta2'], p['eps'], n)
    q2_results[bname] = res
    print(f"  Benchmark {bname} done")

# Q2(b): Objective vs iteration
for bname in ['A', 'B', 'C']:
    res = q2_results[bname]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(res['gd'][1], 'k--', lw=1.5, label='GD (baseline)')
    ax.semilogy(res['nesterov'][1], lw=2, label='Nesterov Momentum')
    ax.semilogy(res['adam'][1], lw=2, label='Adam')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value (log scale)')
    title_map = {'A': 'Linear Regression', 'B': 'Toy Neural Network', 'C': 'Rosenbrock'}
    ax.set_title(f'Q2: Convergence Comparison — Benchmark {bname} ({title_map[bname]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/q2_fval_{bname}.pdf', bbox_inches='tight')
    plt.close()

# Q2(c): Contour + trajectories for B and C
for bname, contour_fn in [('B', contour_B), ('C', contour_C)]:
    res = q2_results[bname]
    fig, ax = plt.subplots(figsize=(8, 7))
    contour_fn(ax)
    for method, lab, c in [('gd', 'GD', 'k'), ('nesterov', 'Nesterov', 'tab:blue'), ('adam', 'Adam', 'tab:red')]:
        hist = res[method][0]
        ls = '--' if method == 'gd' else '-'
        ax.plot(hist[:, 0], hist[:, 1], color=c, linestyle=ls, lw=1.5, alpha=0.8, label=lab)
    x0_used = x0_B if bname == 'B' else x0_C
    ax.plot(*x0_used, 'k*', ms=14, label='Start')
    opt = [0.61, 2.0] if bname == 'B' else [1.0, 1.0]
    ax.plot(*opt, 'r*', ms=14, label='Optimum')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    title_map = {'B': 'Toy Neural Network', 'C': 'Rosenbrock'}
    ax.set_title(f'Q2: Optimisation Trajectories — Benchmark {bname} ({title_map[bname]})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'figures/q2_contour_{bname}.pdf', bbox_inches='tight')
    plt.close()

# Q2(d): Mini-batch SGD on Benchmark A
def mini_batch_sgd(X, y, theta0, alpha, batch_size, n_epochs, seed=42):
    rng = np.random.RandomState(seed)
    theta = theta0.copy().astype(float)
    n = len(y)
    loss_fn_local = lambda th: 0.5 * np.mean((X @ th - y)**2)
    epoch_losses = [loss_fn_local(theta)]
    theta_hist = [theta.copy()]
    for _ in range(n_epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]
            r = Xb @ theta - yb
            g = Xb.T @ r / len(yb)
            theta = theta - alpha * g
        epoch_losses.append(loss_fn_local(theta))
        theta_hist.append(theta.copy())
    return np.array(epoch_losses), np.array(theta_hist)

sgd_b5 = mini_batch_sgd(X_data, y_data, theta0_A, 0.06, 5, 50)
sgd_b40 = mini_batch_sgd(X_data, y_data, theta0_A, 0.06, 40, 50)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(sgd_b5[0], lw=2, label='SGD (b=5)')
ax.semilogy(sgd_b40[0], lw=2, label='SGD (b=40)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Q2: Mini-Batch SGD — Effect of Batch Size (Benchmark A)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q2_sgd_batch.pdf', bbox_inches='tight')
plt.close()

# Q2(e): Noisy SGD
y_noisy = X_data @ theta_star + 6.0 * eps_noise

sgd_noisy_b5 = mini_batch_sgd(X_data, y_noisy, theta0_A, 0.06, 5, 50, seed=42)
sgd_noisy_b40 = mini_batch_sgd(X_data, y_noisy, theta0_A, 0.06, 40, 50, seed=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].semilogy(sgd_b5[0], lw=2, label='b=5 (low noise)')
axes[0].semilogy(sgd_b40[0], lw=2, label='b=40 (low noise)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (log scale)')
axes[0].set_title('Q2: SGD — Original Noise ($\\sigma=1$)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].semilogy(sgd_noisy_b5[0], lw=2, label='b=5 (high noise)')
axes[1].semilogy(sgd_noisy_b40[0], lw=2, label='b=40 (high noise)')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss (log scale)')
axes[1].set_title('Q2: SGD — High Noise ($\\sigma=6$)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q2_sgd_noisy.pdf', bbox_inches='tight')
plt.close()

print("Q2 plots saved.")

# ============================================================
# QUESTION 3
# ============================================================
print("\n" + "="*60)
print("QUESTION 3: Newton's Method and Local Approximation")
print("="*60)

# Q3(a): Local approximation of g(x) = x^4
x_plot = np.linspace(-0.5, 1.0, 400)
x0_approx = 0.25
g_fn = lambda x: x**4
g_prime = lambda x: 4*x**3
g_pprime = lambda x: 12*x**2

g0 = g_fn(x0_approx)
gp0 = g_prime(x0_approx)
gpp0 = g_pprime(x0_approx)

first_order = g0 + gp0 * (x_plot - x0_approx)
second_order = g0 + gp0 * (x_plot - x0_approx) + 0.5 * gpp0 * (x_plot - x0_approx)**2

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_plot, g_fn(x_plot), 'k-', lw=2.5, label='$g(x) = x^4$')
ax.plot(x_plot, first_order, 'b--', lw=2, label='First-order approximation')
ax.plot(x_plot, second_order, 'r-.', lw=2, label='Second-order approximation')
ax.plot(x0_approx, g0, 'ko', ms=10, zorder=5, label=f'$x_0 = {x0_approx}$')
ax.set_xlabel('$x$')
ax.set_ylabel('$g(x)$')
ax.set_title('Q3: Local Approximations of $g(x) = x^4$ at $x_0 = 0.25$')
ax.set_ylim(-0.1, 0.6)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q3_approximation.pdf', bbox_inches='tight')
plt.close()

# Newton on all benchmarks
H_A = hessian_A_matrix()
hess_A_fn = lambda x: H_A

q3_results = {}
for bname, grad_fn, hess_fn, loss_fn, x0, params in [
    ('A', grad_A, hess_A_fn, loss_A, theta0_A, {'gd_alpha': 0.08, 'newton_alpha': 1.0}),
    ('B', grad_B, hessian_B, loss_B, x0_B, {'gd_alpha': 0.06, 'newton_alpha': 0.85}),
    ('C', grad_C, hessian_C, loss_C, x0_C, {'gd_alpha': 0.001, 'newton_alpha': 0.22})
]:
    res = {}
    res['gd'] = gradient_descent(grad_fn, loss_fn, x0, params['gd_alpha'], 80)
    # Compute GD update norms
    gd_norms = []
    for i in range(len(res['gd'][0]) - 1):
        gd_norms.append(np.linalg.norm(res['gd'][0][i+1] - res['gd'][0][i]))
    res['gd_norms'] = np.array(gd_norms)
    res['newton'] = newtons_method(grad_fn, hess_fn, loss_fn, x0, params['newton_alpha'], 20, damping=1e-8)
    q3_results[bname] = res
    print(f"  Benchmark {bname} done")

# Q3(b,c): Convergence comparison
for bname in ['A', 'B', 'C']:
    res = q3_results[bname]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(res['gd'][1], 'b-', lw=2, label='Gradient Descent (80 iters)')
    ax.semilogy(res['newton'][1], 'r-', lw=2, label="Newton's Method (20 iters)")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value (log scale)')
    title_map = {'A': 'Linear Regression', 'B': 'Toy Neural Network', 'C': 'Rosenbrock'}
    ax.set_title(f"Q3: Newton vs GD — Benchmark {bname} ({title_map[bname]})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/q3_fval_{bname}.pdf', bbox_inches='tight')
    plt.close()

# Q3(d): Contour trajectories for B and C
for bname, contour_fn in [('B', contour_B), ('C', contour_C)]:
    res = q3_results[bname]
    fig, ax = plt.subplots(figsize=(8, 7))
    contour_fn(ax)
    ax.plot(res['gd'][0][:, 0], res['gd'][0][:, 1], 'b-o', ms=3, lw=1.5, alpha=0.8, label='GD')
    ax.plot(res['newton'][0][:, 0], res['newton'][0][:, 1], 'r-s', ms=5, lw=2, alpha=0.9, label='Newton')
    x0_used = x0_B if bname == 'B' else x0_C
    ax.plot(*x0_used, 'k*', ms=14, label='Start')
    opt = [0.61, 2.0] if bname == 'B' else [1.0, 1.0]
    ax.plot(*opt, 'g*', ms=14, label='Optimum')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    title_map = {'B': 'Toy Neural Network', 'C': 'Rosenbrock'}
    ax.set_title(f"Q3: Newton vs GD Trajectories — Benchmark {bname} ({title_map[bname]})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'figures/q3_contour_{bname}.pdf', bbox_inches='tight')
    plt.close()

# Q3(e): Update magnitude vs iteration
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, bname in enumerate(['A', 'B', 'C']):
    res = q3_results[bname]
    ax = axes[idx]
    ax.semilogy(res['gd_norms'], 'b-', lw=2, label='GD')
    ax.semilogy(res['newton'][2], 'r-', lw=2, label='Newton')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Update Magnitude (log scale)')
    title_map = {'A': 'Linear Reg.', 'B': 'Toy NN', 'C': 'Rosenbrock'}
    ax.set_title(f'Benchmark {bname} ({title_map[bname]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle('Q3: Update Magnitude vs Iteration', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/q3_update_magnitude.pdf', bbox_inches='tight')
plt.close()

print("Q3 plots saved.")

# ============================================================
# QUESTION 4
# ============================================================
print("\n" + "="*60)
print("QUESTION 4: Derivative-Free Optimisation")
print("="*60)

# Part A: Finite Difference and Nesterov Random Search on Benchmark B

def fd_gradient(loss_fn, x, delta):
    n = len(x)
    g = np.zeros(n)
    f0 = loss_fn(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        g[i] = (loss_fn(x + delta * ei) - f0) / delta
    return g

def fd_gradient_descent(loss_fn, x0, alpha, delta, n_iters):
    x = x0.copy().astype(float)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for _ in range(n_iters):
        g = fd_gradient(loss_fn, x, delta)
        x = x - alpha * g
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

def nesterov_random_search(loss_fn, x0, alpha, delta, n_iters, seed=42):
    rng = np.random.RandomState(seed)
    x = x0.copy().astype(float)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for _ in range(n_iters):
        u = rng.randn(len(x))
        u = u / np.linalg.norm(u)
        df = (loss_fn(x + delta * u) - loss_fn(x)) / delta
        x = x - alpha * df * u
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

# Run Q4 Part A
gd_exact_B = gradient_descent(grad_B, loss_B, x0_B, 0.06, 220)
fd_good_B = fd_gradient_descent(loss_B, x0_B, 0.08, 0.05, 120)
fd_poor_B = fd_gradient_descent(loss_B, x0_B, 0.08, 0.8, 120)
nrs_B = nesterov_random_search(loss_B, x0_B, 0.025, 0.08, 220)

print("  Part A done")

# Q4(a,b): Convergence comparison on B
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(gd_exact_B[1][:121], 'k-', lw=2, label='Exact GD ($\\alpha=0.06$)')
ax.semilogy(fd_good_B[1], 'b-', lw=2, label='FD GD ($\\delta=0.05$, good)')
ax.semilogy(fd_poor_B[1], 'r-', lw=2, label='FD GD ($\\delta=0.8$, poor)')
ax.semilogy(nrs_B[1], 'g-', lw=1.5, alpha=0.8, label='Nesterov Random Search')
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Value (log scale)')
ax.set_title('Q4A: Derivative Approximation Methods — Benchmark B')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q4_fval_B.pdf', bbox_inches='tight')
plt.close()

# Q4(c): Contour + trajectories for B
fig, ax = plt.subplots(figsize=(8, 7))
contour_B(ax)
ax.plot(gd_exact_B[0][:121, 0], gd_exact_B[0][:121, 1], 'k-', lw=1.5, label='Exact GD')
ax.plot(fd_good_B[0][:, 0], fd_good_B[0][:, 1], 'b-', lw=1.5, alpha=0.8, label='FD ($\\delta=0.05$)')
ax.plot(fd_poor_B[0][:, 0], fd_poor_B[0][:, 1], 'r-', lw=1.5, alpha=0.8, label='FD ($\\delta=0.8$)')
ax.plot(nrs_B[0][:, 0], nrs_B[0][:, 1], 'g-', lw=1, alpha=0.6, label='Nesterov Random')
ax.plot(*x0_B, 'k*', ms=14, label='Start')
ax.plot(0.61, 2.0, 'r*', ms=14, label='Optimum')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Q4A: Trajectories on Benchmark B (Toy NN)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('figures/q4_contour_B.pdf', bbox_inches='tight')
plt.close()

# Part B: Nelder-Mead and Grid Search on Benchmark C

def nelder_mead(loss_fn, x0, step, n_iters, alpha_r=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0.copy()
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1][i] += step

    f_vals = np.array([loss_fn(v) for v in simplex])
    centroid_hist = [np.mean(simplex, axis=0).copy()]
    f_best_hist = [np.min(f_vals)]

    for _ in range(n_iters):
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]

        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_r = centroid + alpha_r * (centroid - simplex[-1])
        f_r = loss_fn(x_r)

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
        elif f_r < f_vals[0]:
            # Expansion
            x_e = centroid + gamma * (centroid - simplex[-1])
            f_e = loss_fn(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
        else:
            if f_r < f_vals[-1]:
                # Outside contraction
                x_c = centroid + rho * (x_r - centroid)
                f_c = loss_fn(x_c)
                if f_c <= f_r:
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        f_vals[i] = loss_fn(simplex[i])
            else:
                # Inside contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                f_c = loss_fn(x_c)
                if f_c < f_vals[-1]:
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        f_vals[i] = loss_fn(simplex[i])

        centroid_hist.append(np.mean(simplex, axis=0).copy())
        f_best_hist.append(np.min(f_vals))

    return np.array(centroid_hist), np.array(f_best_hist), simplex

nm_hist, nm_fvals, nm_final = nelder_mead(loss_C, x0_C, 0.35, 160)
print("  Nelder-Mead done")

# Q4(d): Nelder-Mead trajectory on C
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
contour_C(axes[0])
axes[0].plot(nm_hist[:, 0], nm_hist[:, 1], 'b-o', ms=2, lw=1.5, alpha=0.8, label='Nelder-Mead centroid')
axes[0].plot(*x0_C, 'k*', ms=14, label='Start')
axes[0].plot(1.0, 1.0, 'r*', ms=14, label='Optimum (1,1)')
axes[0].set_xlabel('$x_1$'); axes[0].set_ylabel('$x_2$')
axes[0].set_title('Q4B: Nelder-Mead Trajectory — Rosenbrock')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.2)

axes[1].semilogy(nm_fvals, 'b-', lw=2, label='Nelder-Mead best')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Best Objective (log scale)')
axes[1].set_title('Q4B: Nelder-Mead Convergence — Rosenbrock')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q4_nelder_mead_C.pdf', bbox_inches='tight')
plt.close()

# Q4(e): Grid Search on Benchmark C
x1_gs = np.linspace(-2, 2, 55)
x2_gs = np.linspace(-1, 3, 55)
X1_GS, X2_GS = np.meshgrid(x1_gs, x2_gs)
Z_GS = np.vectorize(lambda a, b: loss_C(np.array([a, b])))(X1_GS, X2_GS)

best_idx = np.unravel_index(Z_GS.argmin(), Z_GS.shape)
best_point = np.array([X1_GS[best_idx], X2_GS[best_idx]])
best_val = Z_GS[best_idx]

# Best-so-far curve
Z_flat = Z_GS.flatten()
best_so_far = np.minimum.accumulate(Z_flat)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sampling pattern
contour_C(axes[0])
axes[0].scatter(X1_GS.flatten(), X2_GS.flatten(), s=3, c='gray', alpha=0.4, label='Grid points')
axes[0].plot(*best_point, 'r*', ms=16, zorder=5, label=f'Best: ({best_point[0]:.2f}, {best_point[1]:.2f})')
axes[0].plot(1.0, 1.0, 'g*', ms=14, zorder=5, label='True optimum (1,1)')
axes[0].set_xlabel('$x_1$'); axes[0].set_ylabel('$x_2$')
axes[0].set_title('Q4B: Grid Search Sampling — Rosenbrock')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.2)

axes[1].semilogy(best_so_far, 'b-', lw=1.5)
axes[1].set_xlabel('Grid Point Index')
axes[1].set_ylabel('Best Value So Far (log scale)')
axes[1].set_title(f'Q4B: Grid Search Best-So-Far (best $f$ = {best_val:.4f})')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q4_grid_search_C.pdf', bbox_inches='tight')
plt.close()

print("Q4 plots saved.")

# ============================================================
# QUESTION 5
# ============================================================
print("\n" + "="*60)
print("QUESTION 5: Constrained Optimisation")
print("="*60)

# Benchmark B with constraint x1 >= 0.5
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

# Projected GD
def projected_gd_q5(x0, alpha, n_iters):
    x = x0.copy().astype(float)
    x_hist = [x.copy()]
    f_hist = [loss_B(x)]
    for _ in range(n_iters):
        g = grad_B(x)
        x = project_q5(x - alpha * g)
        x_hist.append(x.copy())
        f_hist.append(loss_B(x))
    return np.array(x_hist), np.array(f_hist)

# Penalty GD
def penalty_gd_q5(x0, alpha, lam, n_iters):
    x = x0.copy().astype(float)
    x_hist = [x.copy()]
    f_hist = [penalty_loss_q5(x, lam)]
    for _ in range(n_iters):
        g = penalty_grad_q5(x, lam)
        x = x - alpha * g
        x_hist.append(x.copy())
        f_hist.append(penalty_loss_q5(x, lam))
    return np.array(x_hist), np.array(f_hist)

# Unconstrained GD
gd_uncons = gradient_descent(grad_B, loss_B, x0_q5, 0.07, 100)
pgd_q5 = projected_gd_q5(x0_q5, 0.08, 100)
pen_015 = penalty_gd_q5(x0_q5, 0.05, 0.15, 100)
pen_18 = penalty_gd_q5(x0_q5, 0.05, 1.8, 100)
pen_45 = penalty_gd_q5(x0_q5, 0.03, 4.5, 100)

print("  All Q5 methods done")

# Q5 Task 2: Objective vs iteration
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy([loss_B(x) for x in gd_uncons[0]], 'k--', lw=1.5, label='Unconstrained GD')
ax.semilogy(pgd_q5[1], 'b-', lw=2, label='Projected GD')
ax.semilogy([loss_B(x) for x in pen_015[0]], 'g-', lw=2, label='Penalty ($\\lambda=0.15$)')
ax.semilogy([loss_B(x) for x in pen_18[0]], color='orange', lw=2, label='Penalty ($\\lambda=1.8$)')
ax.semilogy([loss_B(x) for x in pen_45[0]], 'r-', lw=2, label='Penalty ($\\lambda=4.5$)')
ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x)$ (log scale)')
ax.set_title('Q5: Objective Value vs Iteration')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q5_fval.pdf', bbox_inches='tight')
plt.close()

# Q5 Task 3: Contour with feasible boundary
x1_q5 = np.linspace(-0.5, 3.0, 400)
x2_q5 = np.linspace(0.0, 5.5, 400)
X1Q5, X2Q5 = np.meshgrid(x1_q5, x2_q5)
ZQ5 = np.vectorize(lambda a, b: loss_B(np.array([a, b])))(X1Q5, X2Q5)

fig, ax = plt.subplots(figsize=(8, 7))
ax.contour(X1Q5, X2Q5, ZQ5, levels=30, cmap='viridis', alpha=0.7)
ax.axvline(x=0.5, color='red', lw=2, linestyle='--', label='$x_1 = 0.5$ (boundary)')
ax.fill_betweenx([0, 5.5], -0.5, 0.5, alpha=0.15, color='red', label='Infeasible region')

ax.plot(gd_uncons[0][:, 0], gd_uncons[0][:, 1], 'k--', lw=1.5, alpha=0.7, label='Unconstrained GD')
ax.plot(pgd_q5[0][:, 0], pgd_q5[0][:, 1], 'b-o', ms=2, lw=1.5, label='Projected GD')
ax.plot(pen_015[0][:, 0], pen_015[0][:, 1], 'g-', lw=1.5, alpha=0.8, label='Penalty $\\lambda=0.15$')
ax.plot(pen_18[0][:, 0], pen_18[0][:, 1], color='orange', lw=1.5, alpha=0.8, label='Penalty $\\lambda=1.8$')
ax.plot(pen_45[0][:, 0], pen_45[0][:, 1], 'r-', lw=1.5, alpha=0.8, label='Penalty $\\lambda=4.5$')
ax.plot(*x0_q5, 'k*', ms=14, label='Start (0.2, 4.0)')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Q5: Contour with Feasible Boundary and Trajectories')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('figures/q5_contour.pdf', bbox_inches='tight')
plt.close()

# Q5 Task 4: Constraint violation log scale
def violation(x_hist):
    return np.array([max(0, 0.5 - x[0]) for x in x_hist])

v_uncons = violation(gd_uncons[0])
v_pgd = violation(pgd_q5[0])
v_pen015 = violation(pen_015[0])
v_pen18 = violation(pen_18[0])
v_pen45 = violation(pen_45[0])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full view
ax = axes[0]
for v, lab, c, ls in [
    (v_uncons, 'Unconstrained', 'k', '--'),
    (v_pgd, 'Projected GD', 'b', '-'),
    (v_pen015, 'Penalty $\\lambda=0.15$', 'g', '-'),
    (v_pen18, 'Penalty $\\lambda=1.8$', 'orange', '-'),
    (v_pen45, 'Penalty $\\lambda=4.5$', 'r', '-')
]:
    mask = v > 0
    if np.any(mask):
        ax.semilogy(np.where(mask)[0], v[mask], color=c, linestyle=ls, lw=2, label=lab)
    else:
        ax.semilogy([0], [1e-16], color=c, linestyle=ls, lw=2, label=lab + ' (feasible)')

ax.set_xlabel('Iteration')
ax.set_ylabel('Constraint Violation (log scale)')
ax.set_title('Q5: Constraint Violation $\\max(0, 0.5 - x_1)$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Q5 Task 5: Zoomed first 30 iterations
ax = axes[1]
for v, lab, c, ls in [
    (v_uncons[:31], 'Unconstrained', 'k', '--'),
    (v_pgd[:31], 'Projected GD', 'b', '-'),
    (v_pen015[:31], 'Penalty $\\lambda=0.15$', 'g', '-'),
    (v_pen18[:31], 'Penalty $\\lambda=1.8$', 'orange', '-'),
    (v_pen45[:31], 'Penalty $\\lambda=4.5$', 'r', '-')
]:
    mask = v > 0
    if np.any(mask):
        ax.semilogy(np.where(mask)[0], v[mask], color=c, linestyle=ls, lw=2, label=lab)

ax.set_xlabel('Iteration')
ax.set_ylabel('Constraint Violation (log scale)')
ax.set_title('Q5: Constraint Violation — First 30 Iterations (Zoomed)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q5_violation.pdf', bbox_inches='tight')
plt.close()

print("Q5 plots saved.")

# ============================================================
# QUESTION 6
# ============================================================
print("\n" + "="*60)
print("QUESTION 6: Linear Programmes and Frank-Wolfe")
print("="*60)

# Feasible set X = {0.5 <= x1 <= 5, -5 <= x2 <= 10}
X_bounds = [(0.5, 5.0), (-5.0, 10.0)]

def fw_lp_box(grad, bounds):
    z = np.zeros(len(grad))
    for i in range(len(grad)):
        z[i] = bounds[i][0] if grad[i] > 0 else bounds[i][1]
    return z

# Q6(I): Linear Programme
a_lp = np.array([1, 2])
# f(x) = a^T x = x1 + 2*x2, minimize over X
# Since coefficients are positive, minimum at lower bounds: (0.5, -5)
lp_solution = np.array([0.5, -5.0])
lp_value = a_lp @ lp_solution

x1_q6 = np.linspace(0, 5.5, 300)
x2_q6 = np.linspace(-6, 11, 300)
X1Q6, X2Q6 = np.meshgrid(x1_q6, x2_q6)
ZQ6_LP = X1Q6 + 2 * X2Q6

fig, ax = plt.subplots(figsize=(8, 7))
cs = ax.contour(X1Q6, X2Q6, ZQ6_LP, levels=20, cmap='coolwarm', alpha=0.7)
ax.clabel(cs, inline=True, fontsize=8)
rect = plt.Rectangle((0.5, -5), 4.5, 15, fill=True, facecolor='lightblue', edgecolor='black',
                      alpha=0.3, lw=2, label='Feasible region $\\mathcal{X}$')
ax.add_patch(rect)
ax.plot(*lp_solution, 'r*', ms=16, zorder=5, label=f'LP optimum ({lp_solution[0]}, {lp_solution[1]})')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title(f'Q6-I: Linear Programme — $f(x) = x_1 + 2x_2$, $f^* = {lp_value}$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('figures/q6_lp.pdf', bbox_inches='tight')
plt.close()

# Q6(II): Frank-Wolfe for interior optimum
def loss_q6_interior(x):
    return (x[0] - 1)**2 + (x[1] - 5)**2

def grad_q6_interior(x):
    return np.array([2*(x[0] - 1), 2*(x[1] - 5)])

def frank_wolfe(grad_fn, loss_fn, x0, bounds, beta, n_iters):
    x = x0.copy().astype(float)
    x_hist, f_hist, z_hist = [x.copy()], [loss_fn(x)], []
    for _ in range(n_iters):
        g = grad_fn(x)
        z = fw_lp_box(g, bounds)
        z_hist.append(z.copy())
        x = beta * x + (1 - beta) * z
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist), np.array(z_hist)

x0_fw_int = np.array([1.0, 1.0])
fw_int_090 = frank_wolfe(grad_q6_interior, loss_q6_interior, x0_fw_int, X_bounds, 0.90, 180)
fw_int_0985 = frank_wolfe(grad_q6_interior, loss_q6_interior, x0_fw_int, X_bounds, 0.985, 180)

# Q6(III): Frank-Wolfe for boundary optimum
def loss_q6_boundary(x):
    return x[0]**2 + x[1]**2

def grad_q6_boundary(x):
    return np.array([2*x[0], 2*x[1]])

x0_fw_bnd = np.array([3.0, 3.0])
fw_bnd = frank_wolfe(grad_q6_boundary, loss_q6_boundary, x0_fw_bnd, X_bounds, 0.93, 140)

print("  Frank-Wolfe done")

# Q6 Task 2: FW convergence for interior
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(fw_int_090[1], 'b-', lw=2, label='FW $\\beta=0.90$')
ax.semilogy(fw_int_0985[1], 'r-', lw=2, label='FW $\\beta=0.985$')
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Value (log scale)')
ax.set_title('Q6-II: Frank-Wolfe Convergence — Interior Optimum')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q6_fw_convergence.pdf', bbox_inches='tight')
plt.close()

# Q6 Task 3: Contour + FW trajectories
ZQ6_INT = np.vectorize(lambda a, b: loss_q6_interior(np.array([a, b])))(X1Q6, X2Q6)
ZQ6_BND = np.vectorize(lambda a, b: loss_q6_boundary(np.array([a, b])))(X1Q6, X2Q6)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

ax = axes[0]
ax.contour(X1Q6, X2Q6, ZQ6_INT, levels=20, cmap='viridis', alpha=0.7)
rect1 = plt.Rectangle((0.5, -5), 4.5, 15, fill=True, facecolor='lightblue', edgecolor='black',
                       alpha=0.2, lw=2, label='$\\mathcal{X}$')
ax.add_patch(rect1)
ax.plot(fw_int_090[0][:, 0], fw_int_090[0][:, 1], 'b-o', ms=2, lw=1.5, alpha=0.8, label='FW $\\beta=0.90$')
ax.plot(fw_int_0985[0][:, 0], fw_int_0985[0][:, 1], 'r-o', ms=2, lw=1.5, alpha=0.8, label='FW $\\beta=0.985$')
ax.plot(*x0_fw_int, 'k*', ms=14, label='Start')
ax.plot(1.0, 5.0, 'g*', ms=14, label='Optimum (1,5)')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Q6-II: FW Trajectories — Interior Optimum')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

ax = axes[1]
ax.contour(X1Q6, X2Q6, ZQ6_BND, levels=20, cmap='viridis', alpha=0.7)
rect2 = plt.Rectangle((0.5, -5), 4.5, 15, fill=True, facecolor='lightblue', edgecolor='black',
                       alpha=0.2, lw=2, label='$\\mathcal{X}$')
ax.add_patch(rect2)
ax.plot(fw_bnd[0][:, 0], fw_bnd[0][:, 1], 'b-o', ms=2, lw=1.5, alpha=0.8, label='FW $\\beta=0.93$')
ax.plot(*x0_fw_bnd, 'k*', ms=14, label='Start')
ax.plot(0.5, 0.0, 'g*', ms=14, label='Constrained opt (0.5, 0)')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Q6-III: FW Trajectory — Boundary Optimum')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('figures/q6_fw_contour.pdf', bbox_inches='tight')
plt.close()

# Q6 Task 4: x_k and z_k evolution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Interior optimum, beta=0.90
ax = axes[0, 0]
ax.plot(fw_int_090[0][:, 0], 'b-', lw=1.5, label='$x_{1}^{(k)}$')
ax.plot(fw_int_090[0][:, 1], 'b--', lw=1.5, label='$x_{2}^{(k)}$')
ax.plot(range(1, len(fw_int_090[2])+1), fw_int_090[2][:, 0], 'r:', lw=1.5, label='$z_{1}^{(k)}$')
ax.plot(range(1, len(fw_int_090[2])+1), fw_int_090[2][:, 1], 'r-.', lw=1.5, label='$z_{2}^{(k)}$')
ax.set_xlabel('Iteration'); ax.set_ylabel('Value')
ax.set_title('Interior Optimum ($\\beta=0.90$)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Interior optimum, beta=0.985
ax = axes[0, 1]
ax.plot(fw_int_0985[0][:, 0], 'b-', lw=1.5, label='$x_{1}^{(k)}$')
ax.plot(fw_int_0985[0][:, 1], 'b--', lw=1.5, label='$x_{2}^{(k)}$')
ax.plot(range(1, len(fw_int_0985[2])+1), fw_int_0985[2][:, 0], 'r:', lw=1.5, label='$z_{1}^{(k)}$')
ax.plot(range(1, len(fw_int_0985[2])+1), fw_int_0985[2][:, 1], 'r-.', lw=1.5, label='$z_{2}^{(k)}$')
ax.set_xlabel('Iteration'); ax.set_ylabel('Value')
ax.set_title('Interior Optimum ($\\beta=0.985$)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Boundary optimum
ax = axes[1, 0]
ax.plot(fw_bnd[0][:, 0], 'b-', lw=1.5, label='$x_{1}^{(k)}$')
ax.plot(fw_bnd[0][:, 1], 'b--', lw=1.5, label='$x_{2}^{(k)}$')
ax.plot(range(1, len(fw_bnd[2])+1), fw_bnd[2][:, 0], 'r:', lw=1.5, label='$z_{1}^{(k)}$')
ax.plot(range(1, len(fw_bnd[2])+1), fw_bnd[2][:, 1], 'r-.', lw=1.5, label='$z_{2}^{(k)}$')
ax.set_xlabel('Iteration'); ax.set_ylabel('Value')
ax.set_title('Boundary Optimum ($\\beta=0.93$)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Convergence comparison
ax = axes[1, 1]
ax.semilogy(fw_int_090[1], 'b-', lw=2, label='Interior $\\beta=0.90$')
ax.semilogy(fw_int_0985[1], 'r-', lw=2, label='Interior $\\beta=0.985$')
ax.semilogy(fw_bnd[1], 'g-', lw=2, label='Boundary $\\beta=0.93$')
ax.set_xlabel('Iteration'); ax.set_ylabel('Objective (log)')
ax.set_title('Convergence: Interior vs Boundary')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('Q6: Evolution of $x_k$ and $z_k$ vs Iteration', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/q6_evolution.pdf', bbox_inches='tight')
plt.close()

print("Q6 plots saved.")
print("\n" + "="*60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("="*60)
