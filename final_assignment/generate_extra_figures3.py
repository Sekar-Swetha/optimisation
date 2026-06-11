"""
Third batch of additional figures: Frank-Wolfe with decaying step and convergence bound,
plus mini-batch SGD per-gradient-evaluation efficiency plot.
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
# Benchmark B definition
# ============================================================
def loss_B(x): return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def grad_B(x): return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

# ============================================================
# Figure 1: Frank-Wolfe with decaying schedule vs fixed beta
# Interior optimum: f(x) = (x1-1)^2 + (x2-5)^2 on [0.5,5]x[-5,10]
# ============================================================
def loss_fw_interior(x): return (x[0]-1)**2 + (x[1]-5)**2
def grad_fw_interior(x): return np.array([2*(x[0]-1), 2*(x[1]-5)])

bounds = [(0.5, 5.0), (-5.0, 10.0)]
x0_fw = np.array([1.0, 1.0])
f_star_interior = 0.0
L_interior = 2.0  # Hessian of (x1-1)^2+(x2-5)^2 is 2I
D_sq = (5.0-0.5)**2 + (10.0-(-5.0))**2  # = 20.25 + 225 = 245.25

def frank_wolfe_fixed(grad_fn, loss_fn, x0, bounds, beta, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        z = np.array([bounds[i][0] if g[i] > 0 else bounds[i][1] for i in range(len(g))])
        x = beta * x + (1 - beta) * z
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

def frank_wolfe_decaying(grad_fn, loss_fn, x0, bounds, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    for k in range(n_iters):
        g = grad_fn(x)
        z = np.array([bounds[i][0] if g[i] > 0 else bounds[i][1] for i in range(len(g))])
        gamma_k = 2.0 / (k + 2)  # Standard decaying step
        x = (1 - gamma_k) * x + gamma_k * z
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

n_fw = 300
f_fixed_90 = frank_wolfe_fixed(grad_fw_interior, loss_fw_interior, x0_fw, bounds, 0.90, n_fw)
f_fixed_985 = frank_wolfe_fixed(grad_fw_interior, loss_fw_interior, x0_fw, bounds, 0.985, n_fw)
f_decaying = frank_wolfe_decaying(grad_fw_interior, loss_fw_interior, x0_fw, bounds, n_fw)

# Theoretical bound: 2*L*D^2 / (k+2)
k_arr = np.arange(n_fw + 1)
fw_bound = 2 * L_interior * D_sq / (k_arr + 2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: objective value
ax = axes[0]
ax.semilogy(k_arr, np.maximum(f_fixed_90 - f_star_interior, 1e-10),
            'b-', label=r'Fixed $\beta=0.90$', linewidth=1.5)
ax.semilogy(k_arr, np.maximum(f_fixed_985 - f_star_interior, 1e-10),
            'g-', label=r'Fixed $\beta=0.985$', linewidth=1.5)
ax.semilogy(k_arr, np.maximum(f_decaying - f_star_interior, 1e-10),
            'r-', label=r'Decaying $\gamma_k=2/(k+2)$', linewidth=2.0)
ax.semilogy(k_arr[1:], fw_bound[1:],
            'k--', label=r'$2LD^2/(k+2)$ bound', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'$f(x_k) - f^\star$ (log scale)')
ax.set_title('Frank--Wolfe: Interior Optimum\nDecaying vs Fixed Step')
ax.legend(fontsize=9)
ax.set_ylim([1e-5, None])
ax.grid(True, alpha=0.3)

# Right: log-log to show O(1/k) slope
k_plot = k_arr[2:]
gap_decaying = np.maximum(f_decaying[2:] - f_star_interior, 1e-12)
ax2 = axes[1]
ax2.loglog(k_plot, gap_decaying, 'r-', label=r'Decaying $\gamma_k=2/(k+2)$', linewidth=2.0)
ax2.loglog(k_plot, fw_bound[2:], 'k--', label=r'$O(1/k)$ bound', linewidth=1.5, alpha=0.7)
# Reference O(1/k) slope
ref_k = np.array([10, 300])
ax2.loglog(ref_k, 5.0/ref_k, 'm:', label=r'$O(1/k)$ reference', linewidth=1.5)
ax2.set_xlabel('Iteration $k$ (log scale)')
ax2.set_ylabel(r'$f(x_k) - f^\star$ (log scale)')
ax2.set_title('Frank--Wolfe Rate: Log--Log\nDecaying Schedule')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/q6_fw_decaying.pdf', bbox_inches='tight')
plt.close()
print("Saved figures/q6_fw_decaying.pdf")

# ============================================================
# Figure 2: Benchmark A SGD efficiency – loss vs cumulative gradient evaluations
# ============================================================
m = 1000
np.random.seed(42)
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
eps_noise = np.random.randn(m)
y_data = X_data @ theta_star + eps_noise

def loss_A(theta): return 0.5*np.mean((X_data@theta - y_data)**2)
def grad_A_batch(theta, idx): return X_data[idx].T@(X_data[idx]@theta - y_data[idx]) / len(idx)
def grad_A(theta): return X_data.T@(X_data@theta - y_data)/m

f_star_A = 0.4826

n_epochs = 50
alpha_sgd = 0.06

def sgd_run(batch_size, alpha, seed=0):
    rng = np.random.RandomState(seed)
    theta = np.array([0.0, 0.0])
    f_hist = [loss_A(theta)]
    grad_evals = [0]
    total_evals = 0
    for e in range(n_epochs):
        perm = rng.permutation(m)
        for i in range(0, m, batch_size):
            idx = perm[i:i+batch_size]
            g = grad_A_batch(theta, idx)
            theta = theta - alpha * g
            total_evals += len(idx)
        f_hist.append(loss_A(theta))
        grad_evals.append(total_evals)
    return np.array(grad_evals), np.array(f_hist)

# Full GD for reference
theta_gd = np.array([0.0, 0.0])
f_gd = [loss_A(theta_gd)]
evals_gd = [0]
for ep in range(n_epochs):
    theta_gd = theta_gd - alpha_sgd * grad_A(theta_gd)
    f_gd.append(loss_A(theta_gd))
    evals_gd.append((ep+1) * m)

evals_b5, f_b5 = sgd_run(5, alpha_sgd)
evals_b40, f_b40 = sgd_run(40, alpha_sgd)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: per epoch
ax = axes[0]
ax.semilogy(range(len(f_b5)), np.abs(np.array(f_b5) - f_star_A) + 1e-12,
            'b-', label='SGD $b=5$', linewidth=1.5)
ax.semilogy(range(len(f_b40)), np.abs(np.array(f_b40) - f_star_A) + 1e-12,
            'g-', label='SGD $b=40$', linewidth=1.5)
ax.semilogy(range(len(f_gd)), np.abs(np.array(f_gd) - f_star_A) + 1e-12,
            'r--', label='Full GD', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel(r'$|J(\theta) - J^\star|$ (log)')
ax.set_title('SGD: Loss vs Epoch\n(each epoch = full data pass)')
ax.legend(); ax.grid(True, alpha=0.3)

# Right: per gradient evaluation
ax = axes[1]
ax.semilogy(evals_b5, np.abs(np.array(f_b5) - f_star_A) + 1e-12,
            'b-', label='SGD $b=5$', linewidth=1.5)
ax.semilogy(evals_b40, np.abs(np.array(f_b40) - f_star_A) + 1e-12,
            'g-', label='SGD $b=40$', linewidth=1.5)
ax.semilogy(evals_gd, np.abs(np.array(f_gd) - f_star_A) + 1e-12,
            'r--', label='Full GD', linewidth=1.5)
ax.set_xlabel('Cumulative gradient evaluations')
ax.set_ylabel(r'$|J(\theta) - J^\star|$ (log)')
ax.set_title('SGD: Loss vs Gradient Evaluations\n(computational efficiency view)')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/q2_sgd_efficiency.pdf', bbox_inches='tight')
plt.close()
print("Saved figures/q2_sgd_efficiency.pdf")

# ============================================================
# Figure 3: Q1 Polyak convergence analysis – step size vs iteration on Benchmark C
# Shows WHY Polyak works when f* is known: step decreases smoothly as f(x_k) -> 0
# ============================================================
def polyak_step_fn(grad_fn, loss_fn, x0, f_star, eps, n_iters):
    x = x0.copy().astype(float)
    x_hist, f_hist, alpha_hist = [x.copy()], [loss_fn(x)], []
    for _ in range(n_iters):
        g = grad_fn(x)
        alpha_k = (loss_fn(x) - f_star) / (np.dot(g, g) + eps)
        alpha_hist.append(alpha_k)
        x = x - alpha_k * g
        x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist), np.array(alpha_hist)

def gradient_descent(grad_fn, loss_fn, x0, alpha, n_iters):
    x = x0.copy().astype(float); x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for _ in range(n_iters):
        x = x - alpha*grad_fn(x); x_hist.append(x.copy()); f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

x0_C = np.array([-1.0, 1.0])
def loss_C(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def grad_C(x):
    g1 = -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2)
    g2 = 200*(x[1]-x[0]**2)
    return np.array([g1, g2])

f_star_C = 0.0
_, f_polyak_C, alpha_polyak_C = polyak_step_fn(grad_C, loss_C, x0_C, f_star_C, 1e-3, 200)
_, f_gd_C = gradient_descent(grad_C, loss_C, x0_C, 0.0012, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.semilogy(range(len(f_polyak_C)), np.maximum(f_polyak_C - f_star_C, 1e-12),
            'r-', label='Polyak ($f^\\star=0$)', linewidth=2)
ax.semilogy(range(len(f_gd_C)), np.maximum(f_gd_C - f_star_C, 1e-12),
            'b--', label='GD ($\\alpha=0.0012$)', linewidth=1.5)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$f(x_k) - f^\star$ (log)')
ax.set_title('Benchmark~C: Polyak vs GD\nObjective gap (200 iterations)')
ax.legend(); ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(range(len(alpha_polyak_C)), alpha_polyak_C, 'r-', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Polyak step size $\\alpha_k$')
ax2.set_title('Polyak Step Size Evolution\nBenchmark C (decreases as $f(x_k) \\to 0$)')
ax2.grid(True, alpha=0.3)
# Annotate the smooth decrease
ax2.annotate('Smooth decay:\n' + r'$\alpha_k = (f(x_k) - 0)/\|\nabla f\|^2 \to 0$',
             xy=(100, alpha_polyak_C[100]), xytext=(130, alpha_polyak_C[30]),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=9)

plt.tight_layout()
plt.savefig('figures/q1_polyak_analysis.pdf', bbox_inches='tight')
plt.close()
print("Saved figures/q1_polyak_analysis.pdf")

print("All figures generated successfully.")
