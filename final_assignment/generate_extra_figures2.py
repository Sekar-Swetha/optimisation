"""
Additional figures for the report - Pass 4
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
# Re-define benchmarks
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

theta0_A = np.array([0.0, 0.0])
x0_B = np.array([-1.0, 4.0])
x0_C = np.array([-1.0, 1.0])

# ============================================================
# FIGURE 1: Nelder-Mead simplex evolution (snapshots)
# ============================================================
print("Generating q4_nelder_mead_simplex_evolution.pdf ...")

def nelder_mead_full(loss_fn, x0, step, n_iters, alpha_r=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0.copy()
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1][i] += step
    f_vals = np.array([loss_fn(v) for v in simplex])
    simplex_snapshots = [simplex.copy()]
    f_best_hist = [np.min(f_vals)]
    for it in range(n_iters):
        order = np.argsort(f_vals)
        simplex = simplex[order]; f_vals = f_vals[order]
        centroid = np.mean(simplex[:-1], axis=0)
        x_r = centroid + alpha_r * (centroid - simplex[-1])
        f_r = loss_fn(x_r)
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r; f_vals[-1] = f_r
        elif f_r < f_vals[0]:
            x_e = centroid + gamma * (centroid - simplex[-1])
            f_e = loss_fn(x_e)
            if f_e < f_r: simplex[-1] = x_e; f_vals[-1] = f_e
            else: simplex[-1] = x_r; f_vals[-1] = f_r
        else:
            if f_r < f_vals[-1]:
                x_c = centroid + rho * (x_r - centroid)
                f_c = loss_fn(x_c)
                if f_c <= f_r: simplex[-1] = x_c; f_vals[-1] = f_c
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        f_vals[i] = loss_fn(simplex[i])
            else:
                x_c = centroid + rho * (simplex[-1] - centroid)
                f_c = loss_fn(x_c)
                if f_c < f_vals[-1]: simplex[-1] = x_c; f_vals[-1] = f_c
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        f_vals[i] = loss_fn(simplex[i])
        if it in [0, 4, 14, 39, 79, 159]:
            simplex_snapshots.append(simplex.copy())
        f_best_hist.append(np.min(f_vals))
    return simplex_snapshots, np.array(f_best_hist)

snapshots, nm_fvals = nelder_mead_full(loss_C, x0_C, 0.35, 160)

x1_grid = np.linspace(-1.5, 1.5, 400)
x2_grid = np.linspace(-0.5, 2.0, 400)
X1G, X2G = np.meshgrid(x1_grid, x2_grid)
ZG = np.vectorize(lambda a, b: loss_C(np.array([a, b])))(X1G, X2G)

iter_labels = [0, 1, 5, 15, 40, 80, 160]
colors_snap = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.contour(X1G, X2G, ZG, levels=np.logspace(-1, 3.5, 20), cmap='gray', alpha=0.5)
for i, (snap, col, lab) in enumerate(zip(snapshots, colors_snap, iter_labels)):
    triangle = plt.Polygon(snap[:3], fill=False, edgecolor=col, linewidth=2, alpha=0.8)
    ax.add_patch(triangle)
    centroid = snap.mean(axis=0)
    ax.plot(*centroid, 'o', color=col, ms=5)
ax.plot(*x0_C, 'k*', ms=14, label='Start $(-1,1)$', zorder=5)
ax.plot(1.0, 1.0, 'r*', ms=14, label='Optimum $(1,1)$', zorder=5)
# Create proxy artists for legend
from matplotlib.lines import Line2D
proxies = [Line2D([0], [0], color=colors_snap[i], linewidth=2,
                  label=f'iter {iter_labels[i]}') for i in range(len(snapshots))]
proxies += [Line2D([0], [0], marker='*', color='k', linewidth=0, ms=10, label='Start'),
            Line2D([0], [0], marker='*', color='r', linewidth=0, ms=10, label='Optimum')]
ax.legend(handles=proxies, fontsize=8, ncol=2)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Q4B: Nelder--Mead Simplex Evolution\n(snapshots at selected iterations)')
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-0.6, 2.1)
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.semilogy(nm_fvals, 'b-', lw=2, label='Best objective value')
for it_lab in iter_labels[1:]:
    ax.axvline(x=it_lab, color='gray', ls=':', alpha=0.5)
    ax.text(it_lab, nm_fvals[min(it_lab, len(nm_fvals)-1)], f'  {it_lab}',
            va='center', fontsize=8, color='gray')
ax.set_xlabel('Iteration'); ax.set_ylabel('Best $f$ (log scale)')
ax.set_title('Q4B: Nelder--Mead Convergence\n(vertical lines = snapshot iterations)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle('Q4B: Nelder--Mead Simplex Method on Rosenbrock', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('figures/q4_nm_simplex_evolution.pdf', bbox_inches='tight')
plt.close()
print("  Saved.")

# ============================================================
# FIGURE 2: SGD learning rate schedule comparison
# ============================================================
print("Generating q2_sgd_lr_schedule.pdf ...")

def mini_batch_sgd_schedule(X, y, theta0, alpha_fn, batch_size, n_epochs, seed=42):
    rng = np.random.RandomState(seed)
    theta = theta0.copy().astype(float)
    n = len(y)
    loss_fn_local = lambda th: 0.5 * np.mean((X @ th - y)**2)
    epoch_losses = [loss_fn_local(theta)]
    step = 0
    for e in range(n_epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]
            r = Xb @ theta - yb
            g = Xb.T @ r / len(yb)
            alpha = alpha_fn(step)
            theta = theta - alpha * g
            step += 1
        epoch_losses.append(loss_fn_local(theta))
    return np.array(epoch_losses)

total_steps = 50 * (1000 // 40)  # 50 epochs, b=40

# Different schedules
constant_sgd = mini_batch_sgd_schedule(
    X_data, y_data, theta0_A,
    lambda k: 0.06, 40, 50)

step_decay_sgd = mini_batch_sgd_schedule(
    X_data, y_data, theta0_A,
    lambda k: 0.06 * (0.5 ** (k // (total_steps // 5))), 40, 50)

sqrt_decay_sgd = mini_batch_sgd_schedule(
    X_data, y_data, theta0_A,
    lambda k: 0.06 / np.sqrt(k + 1), 40, 50)

linear_decay_sgd = mini_batch_sgd_schedule(
    X_data, y_data, theta0_A,
    lambda k: max(0.06 * (1 - k / total_steps), 1e-4), 40, 50)

cosine_decay_sgd = mini_batch_sgd_schedule(
    X_data, y_data, theta0_A,
    lambda k: 0.06 * 0.5 * (1 + np.cos(np.pi * k / total_steps)), 40, 50)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(constant_sgd, 'b-', lw=2, label='Constant $\\alpha=0.06$')
ax.semilogy(step_decay_sgd, 'r-', lw=2, label='Step decay ($\\times 0.5$ every 10 epochs)')
ax.semilogy(sqrt_decay_sgd, 'g-', lw=2, label='$1/\\sqrt{k}$ decay')
ax.semilogy(linear_decay_sgd, 'm-', lw=2, label='Linear decay to $10^{-4}$')
ax.semilogy(cosine_decay_sgd, 'orange', lw=2, label='Cosine annealing')
ax.axhline(y=0.4826, color='k', ls='--', lw=1, label='$J^\\star = 0.4826$')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Q2: SGD Learning Rate Schedules Comparison\n(b=40, Benchmark A)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q2_sgd_lr_schedule.pdf', bbox_inches='tight')
plt.close()
print("  Saved.")

# ============================================================
# FIGURE 3: All methods trajectory comparison on Benchmark C
# ============================================================
print("Generating q_all_methods_rosenbrock.pdf ...")

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
        x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

def adam_fn(grad_fn, loss_fn, x0, alpha, beta1, beta2, eps, n_iters):
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

def gradient_descent(grad_fn, loss_fn, x0, alpha, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        x = x - alpha * grad_fn(x); f_hist.append(loss_fn(x))
    return np.array(f_hist)

def polyak_fn(grad_fn, loss_fn, x0, f_star, eps, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        alpha_k = (loss_fn(x) - f_star) / (np.dot(g, g) + eps)
        x = x - alpha_k * g; f_hist.append(loss_fn(x))
    return np.array(f_hist)

def adagrad_fn(grad_fn, loss_fn, x0, alpha0, eps, n_iters):
    x = x0.copy().astype(float)
    G = np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x); G += g**2
        eff_alpha = alpha0 / (np.sqrt(G) + eps)
        x = x - eff_alpha * g; f_hist.append(loss_fn(x))
    return np.array(f_hist)

def heavy_ball_fn(grad_fn, loss_fn, x0, alpha, beta, n_iters):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x); z = beta * z + alpha * g; x = x - z
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

def hessian_C(x):
    h11 = 2 + 1200*x[0]**2 - 400*x[1]
    h12 = -400*x[0]
    return np.array([[h11, h12], [h12, 200]])

def newtons_method(grad_fn, hess_fn, loss_fn, x0, alpha, n_iters, damping=1e-8):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x); H = hess_fn(x) + damping * np.eye(len(x))
        try: p = np.linalg.solve(H, g)
        except: p = g
        x = x - alpha * p; f_hist.append(loss_fn(x))
    return np.array(f_hist)

N = 300
gd_C = gradient_descent(grad_C, loss_C, x0_C, 0.0012, N)
polyak_C = polyak_fn(grad_C, loss_C, x0_C, 0, 1e-3, N)
adagrad_C = adagrad_fn(grad_C, loss_C, x0_C, 0.45, 1e-5, N)
hb_C = heavy_ball_fn(grad_C, loss_C, x0_C, 0.0008, 0.86, N)
nest_C = nesterov_momentum(grad_C, loss_C, x0_C, 0.0007, 0.90, N)[1]
adam_C = adam_fn(grad_C, loss_C, x0_C, 0.006, 0.80, 0.999, 1e-8, N)
newton_C = newtons_method(grad_C, hessian_C, loss_C, x0_C, 0.22, 20)

fig, ax = plt.subplots(figsize=(10, 6))
iters_gd = np.arange(len(gd_C))
ax.semilogy(iters_gd, gd_C, 'k--', lw=1.5, alpha=0.8, label=f'GD (f$_{{300}}$={gd_C[-1]:.2f})')
ax.semilogy(iters_gd, polyak_C, 'tab:purple', lw=2, label=f'Polyak (f$_{{300}}$={polyak_C[-1]:.4f})')
ax.semilogy(iters_gd, adagrad_C, 'tab:orange', lw=2, label=f'Adagrad (f$_{{300}}$={adagrad_C[-1]:.2f})')
ax.semilogy(iters_gd, hb_C, 'tab:red', lw=2, label=f'Heavy Ball (f$_{{300}}$={hb_C[-1]:.2f})')
ax.semilogy(iters_gd, nest_C, 'tab:blue', lw=2, label=f'Nesterov (f$_{{300}}$={nest_C[-1]:.4f})')
ax.semilogy(iters_gd, adam_C, 'tab:green', lw=2, label=f'Adam (f$_{{300}}$={adam_C[-1]:.3f})')
newton_iters = np.arange(len(newton_C))
ax.semilogy(newton_iters, newton_C, 'tab:brown', lw=2.5, marker='s', ms=6,
            label=f"Newton/20it (f$_{{20}}$={newton_C[-1]:.3f})")
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Value (log scale)')
ax.set_title('All Methods Compared on Rosenbrock Benchmark C\n(300 iterations, log scale)')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/q_all_methods_rosenbrock.pdf', bbox_inches='tight')
plt.close()
print("  Saved.")

# ============================================================
# FIGURE 4: Projection vs penalty -- visual comparison
# ============================================================
print("Generating q5_projection_vs_penalty_visual.pdf ...")

x0_q5 = np.array([0.2, 4.0])

def project_q5(x):
    x_p = x.copy(); x_p[0] = max(0.5, x_p[0]); return x_p

def projected_gd_q5_hist(x0, alpha, n_iters):
    x = x0.copy().astype(float)
    x_hist = [x.copy()]
    for _ in range(n_iters):
        g = grad_B(x); x = project_q5(x - alpha * g); x_hist.append(x.copy())
    return np.array(x_hist)

def penalty_gd_q5_hist(x0, alpha, lam, n_iters):
    x = x0.copy().astype(float)
    x_hist = [x.copy()]
    for _ in range(n_iters):
        g = grad_B(x).copy()
        if x[0] < 0.5: g[0] -= lam
        x = x - alpha * g; x_hist.append(x.copy())
    return np.array(x_hist)

pgd_hist = projected_gd_q5_hist(x0_q5, 0.08, 100)
pen18_hist = penalty_gd_q5_hist(x0_q5, 0.05, 1.8, 100)
pen45_hist = penalty_gd_q5_hist(x0_q5, 0.03, 4.5, 100)
gd_uncons_hist = np.array([x0_q5.copy()])
x_temp = x0_q5.copy().astype(float)
for _ in range(100):
    x_temp = x_temp - 0.07 * grad_B(x_temp); gd_uncons_hist = np.vstack([gd_uncons_hist, x_temp])

x1_q5 = np.linspace(-0.3, 2.5, 300)
x2_q5 = np.linspace(0.5, 5.0, 300)
X1Q5, X2Q5 = np.meshgrid(x1_q5, x2_q5)
ZQ5 = np.vectorize(lambda a, b: loss_B(np.array([a, b])))(X1Q5, X2Q5)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, hist, title, color in [
    (axes[0], pgd_hist, 'Projected GD ($\\alpha=0.08$)', 'tab:blue'),
    (axes[1], pen45_hist, 'Penalty Method ($\\lambda=4.5$, $\\alpha=0.03$)', 'tab:red'),
]:
    ax.contour(X1Q5, X2Q5, ZQ5, levels=25, cmap='viridis', alpha=0.6)
    ax.axvline(x=0.5, color='red', lw=2, linestyle='--', label='$x_1 = 0.5$ boundary')
    ax.fill_betweenx([0.5, 5.0], -0.3, 0.5, alpha=0.12, color='red', label='Infeasible')
    # Plot gradient steps explicitly
    n_show = min(30, len(hist)-1)
    for i in range(n_show):
        ax.annotate('', xy=hist[i+1], xytext=hist[i],
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))
    ax.plot(hist[n_show:, 0], hist[n_show:, 1], color=color, lw=1.5, alpha=0.5)
    ax.plot(*x0_q5, 'k*', ms=14, label=f'Start $(0.2, 4.0)$', zorder=5)
    ax.plot(0.582, 2.0, 'g*', ms=14, label='Constrained opt $(0.582, 2.0)$', zorder=5)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title(f'Q5: {title}')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.3, 2.5); ax.set_ylim(0.5, 5.0)

plt.suptitle('Q5: Projected GD vs.\ Penalty Method — Trajectory Comparison with Arrows',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('figures/q5_projection_vs_penalty_visual.pdf', bbox_inches='tight')
plt.close()
print("  Saved.")

print("\nAll pass-4 figures generated.")
