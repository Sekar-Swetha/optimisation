"""Pass 3 supplementary figures for CS7DS2 report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ================================================================
# BENCHMARK DEFINITIONS
# ================================================================
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

x0_C = np.array([-1.0, 1.0])

# ================================================================
# FIGURE 1: Q4 Apples-to-apples comparison on Benchmark C
# Plots BEST f vs TOTAL FUNCTION EVALUATIONS (not iterations)
# ================================================================

# GD on Rosenbrock (1 FE per iter for function value only; gradient is "free")
# Treating each iteration as 1 FE for the function
def gradient_descent_count(grad_fn, loss_fn, x0, alpha, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    fe_hist = [0]
    fe_count = 1  # initial evaluation
    for _ in range(n_iters):
        x = x - alpha * grad_fn(x)
        fe_count += 1  # 1 FE per iter (gradient is analytic)
        f_hist.append(loss_fn(x))
        fe_hist.append(fe_count)
    return np.array(fe_hist), np.array(f_hist)

# FD GD on Rosenbrock (d+1 = 3 FEs per iter for gradient approximation)
def fd_gradient_descent_count(loss_fn, x0, alpha, delta, n_iters):
    d = len(x0)
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    fe_hist = [0]
    fe_count = 1
    for _ in range(n_iters):
        f0 = loss_fn(x)
        fe_count += 1
        g = np.zeros(d)
        for i in range(d):
            ei = np.zeros(d); ei[i] = 1.0
            g[i] = (loss_fn(x + delta * ei) - f0) / delta
            fe_count += 1
        x = x - alpha * g
        f_hist.append(loss_fn(x))
        fe_hist.append(fe_count)
    return np.array(fe_hist), np.array(f_hist)

# Nesterov Random Search on Rosenbrock (2 FEs per iter)
def nrs_count(loss_fn, x0, alpha, delta, n_iters, seed=99):
    rng = np.random.RandomState(seed)
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    fe_hist = [0]
    fe_count = 1
    for _ in range(n_iters):
        u = rng.randn(len(x))
        u = u / np.linalg.norm(u)
        f0 = loss_fn(x)
        f1 = loss_fn(x + delta * u)
        fe_count += 2
        df = (f1 - f0) / delta
        x = x - alpha * df * u
        f_hist.append(loss_fn(x))
        fe_hist.append(fe_count)
    return np.array(fe_hist), np.array(f_hist)

# Nelder-Mead function evaluation count
def nelder_mead_count(loss_fn, x0, step, n_iters, alpha_r=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0.copy()
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1][i] += step
    f_vals = np.array([loss_fn(v) for v in simplex])
    fe_count = n + 1
    f_best_hist = [np.min(f_vals)]
    fe_hist = [fe_count]

    for _ in range(n_iters):
        order = np.argsort(f_vals)
        simplex = simplex[order]; f_vals = f_vals[order]
        centroid = np.mean(simplex[:-1], axis=0)
        x_r = centroid + alpha_r * (centroid - simplex[-1])
        f_r = loss_fn(x_r); fe_count += 1
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r; f_vals[-1] = f_r
        elif f_r < f_vals[0]:
            x_e = centroid + gamma * (centroid - simplex[-1])
            f_e = loss_fn(x_e); fe_count += 1
            if f_e < f_r: simplex[-1] = x_e; f_vals[-1] = f_e
            else: simplex[-1] = x_r; f_vals[-1] = f_r
        else:
            if f_r < f_vals[-1]:
                x_c = centroid + rho * (x_r - centroid)
                f_c = loss_fn(x_c); fe_count += 1
                if f_c <= f_r: simplex[-1] = x_c; f_vals[-1] = f_c
                else:
                    for i in range(1, n+1):
                        simplex[i] = simplex[0]+sigma*(simplex[i]-simplex[0])
                        f_vals[i] = loss_fn(simplex[i]); fe_count += 1
            else:
                x_c = centroid + rho * (simplex[-1] - centroid)
                f_c = loss_fn(x_c); fe_count += 1
                if f_c < f_vals[-1]: simplex[-1] = x_c; f_vals[-1] = f_c
                else:
                    for i in range(1, n+1):
                        simplex[i] = simplex[0]+sigma*(simplex[i]-simplex[0])
                        f_vals[i] = loss_fn(simplex[i]); fe_count += 1
        f_best_hist.append(np.min(f_vals))
        fe_hist.append(fe_count)
    return np.array(fe_hist), np.array(f_best_hist)

# Grid search FE count
x1_gs = np.linspace(-2, 2, 55)
x2_gs = np.linspace(-1, 3, 55)
X1_GS, X2_GS = np.meshgrid(x1_gs, x2_gs)
Z_flat = np.array([loss_C(np.array([X1_GS.flat[i], X2_GS.flat[i]])) for i in range(3025)])
best_so_far = np.minimum.accumulate(Z_flat)
fe_grid = np.arange(1, 3026)

# Run all methods
fe_gd, f_gd = gradient_descent_count(grad_C, loss_C, x0_C, 0.001, 500)
fe_fd, f_fd = fd_gradient_descent_count(loss_C, x0_C, 0.06, 0.05, 300)
fe_nrs, f_nrs = nrs_count(loss_C, x0_C, 0.0008, 0.05, 600)
fe_nm, f_nm = nelder_mead_count(loss_C, x0_C, 0.35, 160)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(fe_gd, f_gd, 'b-', lw=2, label='GD (exact grad, $\\alpha=0.001$)')
ax.semilogy(fe_fd, f_fd, 'g-', lw=2, label='FD GD ($\\delta=0.05$, $\\alpha=0.06$, 3 FE/iter)')
ax.semilogy(fe_nrs, f_nrs, 'm-', lw=1.5, alpha=0.8, label='Nesterov Random Search (2 FE/iter)')
ax.semilogy(fe_nm, f_nm, 'r-', lw=2, label='Nelder--Mead ($\\leq$3 FE/iter)')
ax.semilogy(fe_grid, best_so_far, 'k-', lw=1.5, alpha=0.7, label='Grid Search (3025 total FE)')
ax.set_xlabel('Total function evaluations')
ax.set_ylabel('Best $f$ found (log scale)')
ax.set_title('Q4: Method Comparison on Rosenbrock — Per Function Evaluation')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 3100)
plt.tight_layout()
plt.savefig('figures/supp_q4_fe_comparison.pdf', bbox_inches='tight')
plt.close()
print('Generated supp_q4_fe_comparison.pdf')

# ================================================================
# FIGURE 2: Frank-Wolfe gap over iterations (Q6 interior case)
# Shows the FW gap as a certificate of suboptimality
# ================================================================

def loss_q6_interior(x):
    return (x[0] - 1)**2 + (x[1] - 5)**2

def grad_q6_interior(x):
    return np.array([2*(x[0] - 1), 2*(x[1] - 5)])

X_bounds = [(0.5, 5.0), (-5.0, 10.0)]

def frank_wolfe_with_gap(grad_fn, loss_fn, x0, bounds, beta, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    gap_hist = []
    for _ in range(n_iters):
        g = grad_fn(x)
        z = np.array([bounds[i][0] if g[i] > 0 else bounds[i][1] for i in range(len(g))])
        fw_gap = np.dot(g, x - z)  # = nabla_f^T (x - z) >= 0
        gap_hist.append(fw_gap)
        x = beta * x + (1 - beta) * z
        f_hist.append(loss_fn(x))
    return np.array(f_hist), np.array(gap_hist)

x0_fw = np.array([1.0, 1.0])
f_090, gap_090 = frank_wolfe_with_gap(grad_q6_interior, loss_q6_interior, x0_fw, X_bounds, 0.90, 180)
f_0985, gap_0985 = frank_wolfe_with_gap(grad_q6_interior, loss_q6_interior, x0_fw, X_bounds, 0.985, 180)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.semilogy(f_090, 'b-', lw=2, label='$\\beta=0.90$, $f(x_k)$')
ax.semilogy(f_0985, 'r-', lw=2, label='$\\beta=0.985$, $f(x_k)$')
ax.semilogy(gap_090, 'b--', lw=1.5, alpha=0.7, label='$\\beta=0.90$, FW gap')
ax.semilogy(gap_0985, 'r--', lw=1.5, alpha=0.7, label='$\\beta=0.985$, FW gap')
ax.set_xlabel('Iteration')
ax.set_ylabel('Value (log scale)')
ax.set_title('Q6: $f(x_k)$ and Frank--Wolfe Gap $\\nabla f^T(x-z)$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# The FW gap is an upper bound on f(x) - f*
ax = axes[1]
f_star = 0.0
iters_fw = np.arange(1, 181)
ax.loglog(iters_fw, gap_090, 'b-', lw=2, label='FW gap $\\beta=0.90$')
ax.loglog(iters_fw, gap_0985, 'r-', lw=2, label='FW gap $\\beta=0.985$')
ax.loglog(iters_fw, f_090[1:], 'b--', lw=1.5, alpha=0.7, label='$f(x_k)$, $\\beta=0.90$')
ax.loglog(iters_fw, f_0985[1:], 'r--', lw=1.5, alpha=0.7, label='$f(x_k)$, $\\beta=0.985$')
# Reference O(1/k)
ax.loglog(iters_fw, 1500.0/iters_fw, 'k--', lw=1, alpha=0.5, label='$O(1/k)$ reference')
ax.set_xlabel('Iteration $k$ (log scale)')
ax.set_ylabel('Gap / objective (log scale)')
ax.set_title('Log-Log: FW Gap vs Iteration')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('figures/supp_fw_gap.pdf', bbox_inches='tight')
plt.close()
print('Generated supp_fw_gap.pdf')

# ================================================================
# FIGURE 3: Heavy Ball vs GD on Benchmark C — momentum effect
# Shows how momentum helps on ill-conditioned problems
# ================================================================
def loss_B(x):
    return (x[0] - 1)**2 + 5*(x[1] - 2)**2 + np.sin(x[0])
def grad_B(x):
    return np.array([2*(x[0] - 1) + np.cos(x[0]), 10*(x[1] - 2)])

x0_B = np.array([-1.0, 4.0])
f_star_B = 0.7244

# Run GD and Heavy Ball with different beta values
alphas_hb = [0.0, 0.5, 0.80, 0.90]
colors = ['k', 'blue', 'green', 'red']
n = 120

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
# GD baseline
x = x0_B.copy().astype(float)
gd_f = [loss_B(x)]
for _ in range(n):
    x = x - 0.035 * grad_B(x)
    gd_f.append(loss_B(x))
ax.semilogy(gd_f, 'k-', lw=2, label='GD (no momentum, $\\beta=0$)')

for beta, color in zip([0.5, 0.80, 0.90], ['blue', 'green', 'red']):
    x = x0_B.copy().astype(float)
    z = np.zeros(2)
    hb_f = [loss_B(x)]
    for _ in range(n):
        g = grad_B(x)
        z = beta * z + 0.035 * g
        x = x - z
        hb_f.append(loss_B(x))
    ax.semilogy(hb_f, color=color, lw=2, label=f'Heavy Ball $\\beta={beta}$')

ax.set_xlabel('Iteration')
ax.set_ylabel('Objective value (log scale)')
ax.set_title('Effect of Momentum $\\beta$ on Heavy Ball (Bench. B)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Convergence gap vs iteration (shows speedup ratio)
ax = axes[1]
x = x0_B.copy().astype(float)
gd_f = [loss_B(x)]
for _ in range(n):
    x = x - 0.035 * grad_B(x)
    gd_f.append(loss_B(x))
gd_gap = np.array(gd_f) - f_star_B
mask = gd_gap > 1e-10
iters = np.arange(len(gd_f))
ax.semilogy(iters[mask], gd_gap[mask], 'k-', lw=2, label='GD gap')

for beta, color in zip([0.5, 0.80, 0.90], ['blue', 'green', 'red']):
    x = x0_B.copy().astype(float)
    z = np.zeros(2)
    hb_f = [loss_B(x)]
    for _ in range(n):
        g = grad_B(x)
        z = beta * z + 0.035 * g
        x = x - z
        hb_f.append(loss_B(x))
    hb_gap = np.array(hb_f) - f_star_B
    mask = hb_gap > 1e-10
    ax.semilogy(iters[mask], hb_gap[mask], color=color, lw=2, label=f'Heavy Ball $\\beta={beta}$')

ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x_k) - f^*$ (log scale)')
ax.set_title('Function-Value Gap: GD vs Heavy Ball (Bench. B)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/supp_hb_momentum.pdf', bbox_inches='tight')
plt.close()
print('Generated supp_hb_momentum.pdf')

print('\nAll pass 3 figures done.')
