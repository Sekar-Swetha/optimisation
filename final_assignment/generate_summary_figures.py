"""
Generate additional summary figures for the routine report.
Must be run from within final_assignment/ directory.
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
# Re-define benchmarks (same as original code)
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
x0_C = np.array([-1.0, 1.0])

# ============================================================
# Re-run all methods on Benchmark C for comprehensive comparison
# ============================================================

def gradient_descent(grad_fn, loss_fn, x0, alpha, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        x = x - alpha * grad_fn(x)
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

def adagrad(grad_fn, loss_fn, x0, alpha0, eps, n_iters):
    x = x0.copy().astype(float)
    G = np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        G += g**2
        x = x - alpha0 / (np.sqrt(G) + eps) * g
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

def rmsprop(grad_fn, loss_fn, x0, alpha0, beta, eps, n_iters):
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        v = beta * v + (1 - beta) * g**2
        x = x - alpha0 / (np.sqrt(v) + eps) * g
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

def heavy_ball(grad_fn, loss_fn, x0, alpha, beta, n_iters):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        z = beta * z + alpha * g
        x = x - z
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

def adam_opt(grad_fn, loss_fn, x0, alpha, beta1, beta2, eps, n_iters):
    x = x0.copy().astype(float)
    mv, vv = np.zeros_like(x), np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for t in range(1, n_iters + 1):
        g = grad_fn(x)
        mv = beta1 * mv + (1 - beta1) * g
        vv = beta2 * vv + (1 - beta2) * g**2
        mh = mv / (1 - beta1**t)
        vh = vv / (1 - beta2**t)
        x = x - alpha * mh / (np.sqrt(vh) + eps)
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

def newtons_method(grad_fn, hess_fn, loss_fn, x0, alpha, n_iters, damping=1e-8):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        H = hess_fn(x) + damping * np.eye(len(x))
        try:
            p = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            p = g
        x = x - alpha * p
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

# Run all methods on Rosenbrock for 150 iterations (common budget)
n = 150

print("Running all methods on Benchmark C (Rosenbrock)...")

f_gd    = gradient_descent(grad_C, loss_C, x0_C, 0.0012, n)
f_poly  = polyak_step(grad_C, loss_C, x0_C, 0.0, 1e-3, n)
f_ada   = adagrad(grad_C, loss_C, x0_C, 0.45, 1e-5, n)
f_rms   = rmsprop(grad_C, loss_C, x0_C, 0.0035, 0.9, 1e-5, n)
f_hb    = heavy_ball(grad_C, loss_C, x0_C, 0.0008, 0.86, n)
f_nest  = nesterov_momentum(grad_C, loss_C, x0_C, 0.0007, 0.90, n)
f_adam  = adam_opt(grad_C, loss_C, x0_C, 0.006, 0.80, 0.999, 1e-8, n)
f_newt_20 = newtons_method(grad_C, hessian_C, loss_C, x0_C, 0.22, 20)

print("All methods done.")

# ============================================================
# FIGURE 1: Comprehensive convergence on Rosenbrock (all methods)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

iters = np.arange(n + 1)
ax.semilogy(iters, f_gd,    'k--',  lw=1.8, label='GD ($\\alpha=0.0012$)', zorder=2)
ax.semilogy(iters, f_poly,  lw=2.2, label='Polyak ($f^\\star=0$)', zorder=5)
ax.semilogy(iters, f_ada,   lw=1.8, label='Adagrad ($\\alpha_0=0.45$)', zorder=3)
ax.semilogy(iters, f_rms,   lw=1.8, label='RMSprop ($\\alpha_0=0.0035$)', zorder=3)
ax.semilogy(iters, f_hb,    lw=2.0, label='Heavy Ball ($\\beta=0.86$)', zorder=4)
ax.semilogy(iters, f_nest,  lw=2.2, label='Nesterov ($\\beta_{\\max}=0.90$)', zorder=5)
ax.semilogy(iters, f_adam,  lw=2.0, label='Adam ($\\alpha=0.006$)', zorder=4)
# Newton only runs 20 iters — plot separately with different x-axis offset marker
ax.semilogy(np.arange(21), f_newt_20, 'r-s', ms=4, lw=2.5, label="Newton ($\\alpha=0.22$, 20 iters)", zorder=6)

ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Value $f(x_k)$ (log scale)')
ax.set_title('Comprehensive Convergence Comparison: Benchmark C (Rosenbrock, $\\kappa \\approx 2504$)')
ax.legend(fontsize=9, loc='upper right', ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, n)
plt.tight_layout()
plt.savefig('figures/summary_rosenbrock_all.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figures/summary_rosenbrock_all.pdf")

# ============================================================
# FIGURE 2: 3-panel overview — best method per question on each benchmark
# ============================================================
# Q1 best: Polyak (C), Adagrad (A,B)
# Q2 best: Nesterov
# Q3 best: Newton
# Show convergence on all three benchmarks in a 1x3 grid

# Benchmark A
def loss_A_fn(theta): return 0.5 * np.mean((X_data @ theta - y_data)**2)
def grad_A_fn(theta): return X_data.T @ (X_data @ theta - y_data) / m

f_gd_A    = gradient_descent(grad_A_fn, loss_A_fn, theta0_A, 0.08, 150)
f_nest_A  = nesterov_momentum(grad_A_fn, loss_A_fn, theta0_A, 0.06, 0.90, 150)
f_adam_A  = adam_opt(grad_A_fn, loss_A_fn, theta0_A, 0.12, 0.82, 0.999, 1e-8, 150)

H_A = X_data.T @ X_data / m
def hess_A_fn(x): return H_A
f_newt_A = newtons_method(grad_A_fn, hess_A_fn, loss_A_fn, theta0_A, 1.0, 20)

# Benchmark B
def grad_B_fn(x): return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])
def loss_B_fn(x): return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def hess_B_fn(x): return np.array([[2-np.sin(x[0]), 0], [0, 10]])
x0_B = np.array([-1.0, 4.0])

f_gd_B    = gradient_descent(grad_B_fn, loss_B_fn, x0_B, 0.06, 150)
f_nest_B  = nesterov_momentum(grad_B_fn, loss_B_fn, x0_B, 0.035, 0.92, 150)
f_adam_B  = adam_opt(grad_B_fn, loss_B_fn, x0_B, 0.08, 0.80, 0.999, 1e-8, 150)
f_newt_B  = newtons_method(grad_B_fn, hess_B_fn, loss_B_fn, x0_B, 0.85, 20)

print("All benchmark data computed.")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

titles = ['Benchmark A (Linear Regression)', 'Benchmark B (Toy NN)', 'Benchmark C (Rosenbrock)']
gd_data   = [f_gd_A,   f_gd_B,   f_gd]
nest_data = [f_nest_A, f_nest_B, f_nest]
adam_data = [f_adam_A, f_adam_B, f_adam]
newt_data = [f_newt_A, f_newt_B, f_newt_20]

for i, ax in enumerate(axes):
    ax.semilogy(np.arange(151), gd_data[i],   'k--', lw=1.5, label='GD (baseline)', zorder=2)
    ax.semilogy(np.arange(151), nest_data[i],  lw=2.2, label='Nesterov', zorder=4)
    ax.semilogy(np.arange(151), adam_data[i],  lw=2.0, label='Adam', zorder=3)
    ax.semilogy(np.arange(21),  newt_data[i],  'r-s', ms=4, lw=2.5, label="Newton (20 iters)", zorder=5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$f(x_k)$ (log scale)' if i == 0 else '')
    ax.set_title(titles[i])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Best-Method Comparison: GD vs Nesterov vs Adam vs Newton Across All Benchmarks',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('figures/summary_best_methods.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figures/summary_best_methods.pdf")

# ============================================================
# FIGURE 3: Step-size decay illustration comparing GD (fixed),
#           Nesterov (effective), Adam (effective) on Benchmark C
# ============================================================
# Compute effective step sizes for Adam on C
def adam_with_steps(grad_fn, loss_fn, x0, alpha, beta1, beta2, eps, n_iters):
    x = x0.copy().astype(float)
    mv, vv = np.zeros_like(x), np.zeros_like(x)
    eff_alpha_hist = []
    for t in range(1, n_iters + 1):
        g = grad_fn(x)
        mv = beta1 * mv + (1 - beta1) * g
        vv = beta2 * vv + (1 - beta2) * g**2
        mh = mv / (1 - beta1**t)
        vh = vv / (1 - beta2**t)
        eff = alpha / (np.sqrt(vh) + eps)
        eff_alpha_hist.append(np.mean(eff))
        x = x - alpha * mh / (np.sqrt(vh) + eps)
    return np.array(eff_alpha_hist)

adam_steps_C = adam_with_steps(grad_C, loss_C, x0_C, 0.006, 0.80, 0.999, 1e-8, 150)

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0.0012, color='k', linestyle='--', lw=1.5, label='GD fixed step $\\alpha=0.0012$')
ax.plot(adam_steps_C, lw=2, label='Adam mean effective $\\alpha_k$ (coord.\ average)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Effective Step Size')
ax.set_title('Effective Step Size: GD (fixed) vs Adam (adaptive) — Benchmark C')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/summary_adam_steps_C.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figures/summary_adam_steps_C.pdf")

print("\nAll summary figures generated successfully.")
