"""
Additional figures for the report - Pass 23
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
# Benchmark C: Rosenbrock
# ============================================================
def loss_C(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_C(x):
    dx1 = -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2)
    dx2 = 200*(x[1]-x[0]**2)
    return np.array([dx1, dx2])

x0 = np.array([-1.0, 1.0])
N = 300

# GD
def run_gd(x0, alpha, n):
    x = x0.copy().astype(float)
    f = [loss_C(x)]
    for _ in range(n):
        x = x - alpha * grad_C(x)
        f.append(loss_C(x))
    return np.array(f)

# Polyak
def run_polyak(x0, f_star, eps, n):
    x = x0.copy().astype(float)
    f = [loss_C(x)]
    for _ in range(n):
        g = grad_C(x)
        alpha_k = (loss_C(x) - f_star) / (np.dot(g, g) + eps)
        x = x - alpha_k * g
        f.append(loss_C(x))
    return np.array(f)

# Adagrad
def run_adagrad(x0, alpha0, eps, n):
    x = x0.copy().astype(float)
    G = np.zeros(2)
    f = [loss_C(x)]
    for _ in range(n):
        g = grad_C(x)
        G += g**2
        x = x - alpha0 / (np.sqrt(G) + eps) * g
        f.append(loss_C(x))
    return np.array(f)

# RMSprop
def run_rmsprop(x0, alpha0, beta, eps, n):
    x = x0.copy().astype(float)
    v = np.zeros(2)
    f = [loss_C(x)]
    for _ in range(n):
        g = grad_C(x)
        v = beta*v + (1-beta)*g**2
        x = x - alpha0 / (np.sqrt(v) + eps) * g
        f.append(loss_C(x))
    return np.array(f)

# Heavy Ball
def run_hb(x0, alpha, beta, n):
    x = x0.copy().astype(float)
    z = np.zeros(2)
    f = [loss_C(x)]
    for _ in range(n):
        z = beta*z + alpha*grad_C(x)
        x = x - z
        f.append(loss_C(x))
    return np.array(f)

# Nesterov
def run_nesterov(x0, alpha, beta_max, n):
    x = x0.copy().astype(float)
    z = np.zeros(2)
    f = [loss_C(x)]
    for k in range(1, n+1):
        bk = min((k-1)/(k+2), beta_max)
        la = x + bk*z
        z = bk*z - alpha*grad_C(la)
        x = x + z
        f.append(loss_C(x))
    return np.array(f)

# Adam
def run_adam(x0, alpha, b1, b2, eps, n):
    x = x0.copy().astype(float)
    m = np.zeros(2); v = np.zeros(2)
    f = [loss_C(x)]
    for t in range(1, n+1):
        g = grad_C(x)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g**2
        mh = m/(1-b1**t); vh = v/(1-b2**t)
        x = x - alpha * mh / (np.sqrt(vh) + eps)
        f.append(loss_C(x))
    return np.array(f)

# Run all methods
f_gd = run_gd(x0, 0.0012, N)
f_polyak = run_polyak(x0, 0.0, 1e-3, N)
f_adagrad = run_adagrad(x0, 0.45, 1e-5, N)
f_rmsprop = run_rmsprop(x0, 0.0035, 0.9, 1e-5, N)
f_hb = run_hb(x0, 0.0008, 0.86, N)
f_nesterov = run_nesterov(x0, 0.0007, 0.90, N)
f_adam = run_adam(x0, 0.006, 0.80, 0.999, 1e-8, N)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

k_arr = np.arange(N+1)
colors = {'GD': 'k', 'Polyak': 'purple', 'Adagrad': 'red', 'RMSprop': 'orange',
          'Heavy Ball': 'brown', 'Nesterov': 'blue', 'Adam': 'green'}
styles = {'GD': '-', 'Polyak': '--', 'Adagrad': ':', 'RMSprop': '-.',
          'Heavy Ball': '-', 'Nesterov': '-', 'Adam': '--'}

data = [('GD', f_gd), ('Polyak ($f^*=0$)', f_polyak), ('Adagrad', f_adagrad),
        ('RMSprop', f_rmsprop), ('Heavy Ball', f_hb), ('Nesterov', f_nesterov), ('Adam', f_adam)]

ax = axes[0]
for label, f_hist in data:
    key = label.split()[0].rstrip('(')
    ax.semilogy(k_arr, f_hist + 1e-10, lw=1.8, label=label)
ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'$f(x_k)$ (log scale)')
ax.set_title('All Q1+Q2 Methods: Convergence on Benchmark C\n(Full 300 iterations, log scale)')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, N])

ax = axes[1]
for label, f_hist in data:
    ax.semilogy(k_arr, f_hist + 1e-10, lw=1.8, label=label)
ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'$f(x_k)$ (log scale)')
ax.set_title('Zoomed: Iterations 100-300')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim([100, N])
ax.set_ylim([1e-3, 5.0])

plt.suptitle('Comprehensive Convergence Comparison: All First-Order Methods on Rosenbrock ($\\kappa \\approx 2009$)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q_comprehensive_convergence.pdf', bbox_inches='tight')
plt.close()
print("Saved q_comprehensive_convergence.pdf")

# ============================================================
# Figure 2: Convergence on all 3 benchmarks - comparison summary
# ============================================================
np.random.seed(42)
m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
y_data = X_data @ theta_star + np.random.randn(m)
theta_opt = np.linalg.solve(X_data.T @ X_data, X_data.T @ y_data)
f_star_A = 0.5 * np.mean((X_data @ theta_opt - y_data)**2)

def loss_A(theta): return 0.5 * np.mean((X_data @ theta - y_data)**2)
def grad_A(theta): return X_data.T @ (X_data @ theta - y_data) / m

def loss_B(x): return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def grad_B(x): return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])
f_star_B = 0.7244

x0_A = np.array([0.0, 0.0])
x0_B = np.array([-1.0, 4.0])
x0_C = np.array([-1.0, 1.0])

def run_method_all(method_fn, x0_A, x0_B, x0_C, *args):
    fA = method_fn(x0_A, *args)[:120]
    fB = method_fn(x0_B, *args)[:120]
    fC = method_fn(x0_C, *args)[:120]
    return fA, fB, fC

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
n_show = 120
k_show = np.arange(n_show+1)

methods_config = [
    ('GD', 'k-', run_gd),
    ('Nesterov', 'b-', run_nesterov),
    ('Adam', 'g--', run_adam),
]

benchmarks = [
    ('Benchmark A', f_star_A, x0_A, loss_A, grad_A),
    ('Benchmark B', f_star_B, x0_B, loss_B, grad_B),
    ('Benchmark C', 0.0, x0_C, loss_C, grad_C),
]

# Override run functions for multiple benchmarks
def run_gd_any(loss_fn, grad_fn, x0, alpha, n):
    x = x0.copy().astype(float)
    f = [loss_fn(x)]
    for _ in range(n):
        x = x - alpha * grad_fn(x)
        f.append(loss_fn(x))
    return np.array(f)

def run_nest_any(loss_fn, grad_fn, x0, alpha, bmax, n):
    x = x0.copy().astype(float); z = np.zeros_like(x)
    f = [loss_fn(x)]
    for k in range(1, n+1):
        bk = min((k-1)/(k+2), bmax)
        la = x + bk*z
        z = bk*z - alpha*grad_fn(la)
        x = x + z
        f.append(loss_fn(x))
    return np.array(f)

def run_adam_any(loss_fn, grad_fn, x0, alpha, b1, b2, eps, n):
    x = x0.copy().astype(float); mv = np.zeros_like(x); vv = np.zeros_like(x)
    f = [loss_fn(x)]
    for t in range(1, n+1):
        g = grad_fn(x)
        mv = b1*mv + (1-b1)*g; vv = b2*vv + (1-b2)*g**2
        mh = mv/(1-b1**t); vh = vv/(1-b2**t)
        x = x - alpha * mh / (np.sqrt(vh) + eps)
        f.append(loss_fn(x))
    return np.array(f)

gd_alphas = [0.08, 0.06, 0.0012]
nest_alphas = [(0.06, 0.90), (0.035, 0.92), (0.0007, 0.90)]
adam_alphas = [(0.12, 0.82), (0.08, 0.80), (0.006, 0.80)]

for col, (bname, fstar, x0b, lossfn, gradfn) in enumerate(benchmarks):
    ax_sub = axes[0][col]
    ax_log = axes[1][col]

    fgd = run_gd_any(lossfn, gradfn, x0b, gd_alphas[col], n_show)
    fnest = run_nest_any(lossfn, gradfn, x0b, nest_alphas[col][0], nest_alphas[col][1], n_show)
    fadam = run_adam_any(lossfn, gradfn, x0b, adam_alphas[col][0], adam_alphas[col][1], 0.999, 1e-8, n_show)

    ax_sub.plot(k_show, fgd, 'k-', lw=1.8, label='GD')
    ax_sub.plot(k_show, fnest, 'b-', lw=1.8, label='Nesterov')
    ax_sub.plot(k_show, fadam, 'g--', lw=1.8, label='Adam')
    ax_sub.axhline(y=fstar, color='r', ls=':', lw=1.5, alpha=0.7, label=f'$f^* \\approx {fstar:.3f}$')
    ax_sub.set_title(bname)
    ax_sub.legend(fontsize=8)
    ax_sub.set_xlabel('Iteration'); ax_sub.set_ylabel('$f(x_k)$')
    ax_sub.grid(True, alpha=0.3)

    ax_log.semilogy(k_show, np.maximum(fgd - fstar, 1e-10), 'k-', lw=1.8, label='GD')
    ax_log.semilogy(k_show, np.maximum(fnest - fstar, 1e-10), 'b-', lw=1.8, label='Nesterov')
    ax_log.semilogy(k_show, np.maximum(fadam - fstar, 1e-10), 'g--', lw=1.8, label='Adam')
    ax_log.set_xlabel('Iteration'); ax_log.set_ylabel(r'$f(x_k) - f^*$ (log)')
    ax_log.legend(fontsize=8); ax_log.grid(True, alpha=0.3)

plt.suptitle('GD vs Nesterov vs Adam: Convergence on All Three Benchmarks\n(Top: objective value; Bottom: log suboptimality)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q_gd_nesterov_adam_all_benchmarks.pdf', bbox_inches='tight')
plt.close()
print("Saved q_gd_nesterov_adam_all_benchmarks.pdf")

print("All figures generated successfully.")
