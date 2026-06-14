"""
Generate fourth batch of additional figures:
  1. newton_damping_C.pdf    -- Newton on Rosenbrock with different alpha (damping)
  2. sgd_variance.pdf        -- gradient variance as function of batch size
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

def loss_C(x): return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
def grad_C(x):
    return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
def hessian_C(x):
    h11 = 2+1200*x[0]**2-400*x[1]; h12 = -400*x[0]
    return np.array([[h11,h12],[h12,200]])

x0_C = np.array([-1.0, 1.0])

def newtons_method(x0, alpha, n_iters, damping=1e-8):
    x = x0.copy().astype(float)
    f_hist, x_hist = [loss_C(x)], [x.copy()]
    for _ in range(n_iters):
        g = grad_C(x)
        H = hessian_C(x)
        H_reg = H + damping * np.eye(len(x))
        try:
            p = np.linalg.solve(H_reg, g)
        except:
            p = g
        # Clip step to avoid explosion
        if np.linalg.norm(alpha * p) > 10:
            p = p / np.linalg.norm(p) * 10 / alpha
        x = x - alpha * p
        f_hist.append(loss_C(x))
        x_hist.append(x.copy())
    return np.array(f_hist), np.array(x_hist)

# ============================================================
# FIGURE 1: Newton with different damping on Rosenbrock
# ============================================================
alphas = [1.0, 0.5, 0.22, 0.1]
colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']
n_iters = 50

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x1_grid = np.linspace(-1.5, 1.5, 400)
x2_grid = np.linspace(-0.5, 2.0, 400)
X1C, X2C = np.meshgrid(x1_grid, x2_grid)
ZC = np.vectorize(lambda a, b: loss_C(np.array([a, b])))(X1C, X2C)

ax = axes[0]
ax.contour(X1C, X2C, ZC, levels=np.logspace(-1, 3.5, 25), cmap='viridis', alpha=0.5)
for alpha, c in zip(alphas, colors):
    try:
        fh, xh = newtons_method(x0_C, alpha, n_iters)
        # Clip trajectory for display
        mask = np.abs(xh[:, 0]) < 3
        ax.plot(xh[mask, 0], xh[mask, 1], color=c, lw=1.5, alpha=0.8,
                label=f'$\\alpha={alpha}$, final $f={min(fh[-1], 99):.2f}$')
    except Exception as e:
        pass
ax.plot(*x0_C, 'k*', ms=14, label='Start')
ax.plot(1, 1, 'r*', ms=14, label='Optimum (1,1)')
ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-0.5, 2.0])
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title("Newton Trajectories for Different Damping $\\alpha$\n(Rosenbrock)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

ax = axes[1]
for alpha, c in zip(alphas, colors):
    try:
        fh, xh = newtons_method(x0_C, alpha, n_iters)
        fh_plot = np.clip(fh, 1e-10, 1e5)
        ax.semilogy(fh_plot, color=c, lw=2, label=f'$\\alpha={alpha}$')
    except:
        pass
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('$f(x_k)$ (log scale)', fontsize=12)
ax.set_title("Newton Convergence for Different Damping $\\alpha$\n(Rosenbrock, 50 iterations)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle("Q3: Effect of Newton Step Damping on Rosenbrock", fontsize=12)
plt.tight_layout()
plt.savefig('figures/newton_damping_C.pdf', bbox_inches='tight')
plt.close()
print("Saved: newton_damping_C.pdf")

# ============================================================
# FIGURE 2: SGD gradient variance vs batch size
# ============================================================
np.random.seed(42)
m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
eps_noise = np.random.randn(m)
y_data = X_data @ theta_star + eps_noise

# True gradient at OLS solution
theta_ols = np.linalg.lstsq(X_data, y_data, rcond=None)[0]
true_grad = X_data.T @ (X_data @ theta_ols - y_data) / m

# Estimate gradient variance for different batch sizes
batch_sizes = [1, 2, 5, 10, 20, 40, 100, 200, 500, 1000]
n_trials = 200
grad_vars = []

rng = np.random.RandomState(0)
for b in batch_sizes:
    grads = []
    for _ in range(n_trials):
        idx = rng.choice(m, b, replace=False)
        Xb, yb = X_data[idx], y_data[idx]
        r = Xb @ theta_ols - yb
        g = Xb.T @ r / b
        grads.append(g)
    grads = np.array(grads)
    # Variance of gradient norm
    grad_norm_sq = np.mean([np.sum((g - true_grad)**2) for g in grads])
    grad_vars.append(grad_norm_sq)

# Theoretical: Var[g_batch] = (1/b - 1/m) * Sigma_g
# Approximately proportional to 1/b for large m
theoretical = grad_vars[0] * (1.0 / np.array(batch_sizes))

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(batch_sizes, grad_vars, 'b-o', ms=7, lw=2, label='Empirical gradient variance')
ax.loglog(batch_sizes, theoretical / batch_sizes[0] * batch_sizes[0],
          'r--', lw=1.5, label='$O(1/b)$ reference')
# Mark Q2 batch sizes
for b_mark, col in [(5, 'green'), (40, 'orange')]:
    idx = batch_sizes.index(b_mark) if b_mark in batch_sizes else None
    if idx is not None:
        ax.axvline(b_mark, color=col, linestyle=':', lw=1.5,
                   label=f'$b={b_mark}$ (Q2 experiment)')
ax.set_xlabel('Batch size $b$', fontsize=12)
ax.set_ylabel('$\\mathbb{E}[\\|\\hat{g}_b - \\nabla J\\|^2]$\n(gradient noise power)', fontsize=11)
ax.set_title('Mini-Batch SGD: Gradient Variance vs Batch Size\n(Benchmark A, at $\\hat{\\theta}_{\\rm OLS}$, $n=200$ trials)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('figures/sgd_variance.pdf', bbox_inches='tight')
plt.close()
print("Saved: sgd_variance.pdf")

print("All fourth-batch figures generated.")
