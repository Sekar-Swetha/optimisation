"""Pass 4 figure generation: LR schedules, FW gap, NM simplex, oracle comparison."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
FIGDIR = 'figures'

# ── colour palette ────────────────────────────────────────────────────────────
BLUE   = '#1f77b4'
ORANGE = '#ff7f0e'
GREEN  = '#2ca02c'
RED    = '#d62728'
PURPLE = '#9467bd'
BROWN  = '#8c564b'
PINK   = '#e377c2'
GREY   = '#7f7f7f'

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark B
def f_B(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])

def grad_B(x):
    return np.array([2*(x[0]-1) + np.cos(x[0]), 10*(x[1]-2)])

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Learning-rate schedule comparison for SGD on Benchmark A
# ─────────────────────────────────────────────────────────────────────────────
print("Generating q2_lr_schedule.pdf ...")

np.random.seed(42)
m, n = 1000, 2
theta_star = np.array([3.0, 4.0])
X = np.random.randn(m, n)
eps_noise = np.random.randn(m)
y = X @ theta_star + eps_noise
f_star = np.linalg.lstsq(X, y, rcond=None)[1][0] / m  # residual / m
theta_ols = np.linalg.lstsq(X, y, rcond=None)[0]

def loss_A(theta):
    r = X @ theta - y
    return 0.5 * np.dot(r, r) / m

def grad_A(theta):
    return X.T @ (X @ theta - y) / m

def sgd_epoch(theta0, alpha_schedule, n_epochs=60, batch=40):
    """Run mini-batch SGD with given per-epoch alpha schedule."""
    theta = theta0.copy()
    f_hist = [loss_A(theta)]
    for ep in range(n_epochs):
        alpha = alpha_schedule(ep)
        idx = np.random.permutation(m)
        for start in range(0, m, batch):
            b_idx = idx[start:start+batch]
            g = X[b_idx].T @ (X[b_idx] @ theta - y[b_idx]) / batch
            theta = theta - alpha * g
        f_hist.append(loss_A(theta))
    return np.array(f_hist)

theta0 = np.zeros(2)
n_epochs = 60
alpha0 = 0.12

schedules = {
    'Constant $\\alpha_0$':        lambda ep: alpha0,
    'Step decay (×0.5 / 20 ep.)': lambda ep: alpha0 * (0.5 ** (ep // 20)),
    'Linear decay':                lambda ep: alpha0 * (1 - ep / n_epochs),
    'Cosine annealing':            lambda ep: 0.5 * alpha0 * (1 + np.cos(np.pi * ep / n_epochs)),
    'Polynomial ($t^{-0.5}$)':    lambda ep: alpha0 / np.sqrt(ep + 1),
}
colours = [BLUE, ORANGE, GREEN, RED, PURPLE]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
for (name, sched), col in zip(schedules.items(), colours):
    fh = sgd_epoch(theta0, sched, n_epochs, batch=40)
    ax.semilogy(np.arange(len(fh)), fh - f_star, color=col, lw=1.8, label=name)
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('$J(\\theta) - J^{\\star}$', fontsize=11)
ax.set_title('SGD convergence: learning-rate schedules\n(Benchmark A, batch=40)', fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.35)

ax = axes[1]
ep_range = np.arange(n_epochs)
for (name, sched), col in zip(schedules.items(), colours):
    alphas = [sched(ep) for ep in ep_range]
    ax.plot(ep_range, alphas, color=col, lw=1.8, label=name)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Learning rate $\\alpha_k$', fontsize=11)
ax.set_title('Learning-rate schedule comparison', fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/q2_lr_schedule.pdf', bbox_inches='tight')
plt.close()
print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Frank-Wolfe duality gap evolution
# ─────────────────────────────────────────────────────────────────────────────
print("Generating q6_fw_gap.pdf ...")

bounds = [(0.5, 5.0), (-5.0, 10.0)]

def f_int(x):   # interior optimum at (1, 5)
    return (x[0]-1)**2 + (x[1]-5)**2

def grad_int(x):
    return np.array([2*(x[0]-1), 2*(x[1]-5)])

def f_bnd(x):   # boundary optimum at (0.5, 0)
    return x[0]**2 + x[1]**2

def grad_bnd(x):
    return np.array([2*x[0], 2*x[1]])

def frank_wolfe_gap(grad_fn, loss_fn, x0, bnds, beta, n_iters):
    x = x0.copy().astype(float)
    f_hist, gap_hist = [loss_fn(x)], []
    for _ in range(n_iters):
        g = grad_fn(x)
        z = np.array([bnds[i][0] if g[i] > 0 else bnds[i][1] for i in range(len(g))])
        gap = float(g @ (x - z))
        gap_hist.append(max(gap, 1e-12))
        x = beta * x + (1 - beta) * z
        f_hist.append(loss_fn(x))
    return np.array(f_hist), np.array(gap_hist)

x0_int = np.array([1.0, 1.0])  # interior start
x0_bnd = np.array([3.0, 3.0])  # boundary start

n_fw = 180
fh_90,  gh_90  = frank_wolfe_gap(grad_int, f_int, x0_int, bounds, 0.90,  n_fw)
fh_985, gh_985 = frank_wolfe_gap(grad_int, f_int, x0_int, bounds, 0.985, n_fw)
fh_bnd, gh_bnd = frank_wolfe_gap(grad_bnd, f_bnd, x0_bnd, bounds, 0.93,  140)

# Compute gap at specific iterations for annotation
iters_ann = [10, 50, 100, 150, 179]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
k_arr = np.arange(1, len(gh_90)+1)
ax.semilogy(k_arr, gh_90,  color=BLUE,   lw=1.8, label='$\\beta=0.90$, gap $G_k$')
ax.semilogy(k_arr, fh_90[1:], color=BLUE, lw=1.4, ls='--', label='$\\beta=0.90$, $f(x_k)-f^\\star$')
ax.semilogy(k_arr, gh_985, color=ORANGE, lw=1.8, label='$\\beta=0.985$, gap $G_k$')
ax.semilogy(k_arr, fh_985[1:], color=ORANGE, lw=1.4, ls='--', label='$\\beta=0.985$, $f(x_k)-f^\\star$')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('Value (log scale)', fontsize=11)
ax.set_title('FW gap vs optimality gap\n(Interior optimum $(1,5)$)', fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.35)
ax.set_xlim(1, n_fw)

ax = axes[1]
k_bnd = np.arange(1, len(gh_bnd)+1)
ax.semilogy(k_bnd, gh_bnd,    color=GREEN, lw=1.8, label='$\\beta=0.93$, gap $G_k$')
ax.semilogy(k_bnd, fh_bnd[1:] - 0.25, color=GREEN, lw=1.4, ls='--', label='$f(x_k)-f^\\star$')
# Add theoretical O(1/k) reference
k_ref = np.linspace(2, 140, 200)
C_ref = gh_bnd[0] * 2.0  # scale
ax.semilogy(k_ref, C_ref / k_ref, color=GREY, lw=1.4, ls=':', label='$O(1/k)$ reference')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('Value (log scale)', fontsize=11)
ax.set_title('FW gap vs optimality gap\n(Boundary optimum $(0.5,0)$)', fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.35)

ax = axes[2]
# Numerical example: gap values at selected iterations
iters_show = [1, 5, 10, 20, 50, 100, 150, 179]
gaps_90  = [gh_90[i-1]  for i in iters_show]
gaps_985 = [gh_985[i-1] for i in iters_show]
x_pos = np.arange(len(iters_show))
w = 0.35
bars1 = ax.bar(x_pos - w/2, gaps_90,  w, color=BLUE,   alpha=0.8, label='$\\beta=0.90$')
bars2 = ax.bar(x_pos + w/2, gaps_985, w, color=ORANGE, alpha=0.8, label='$\\beta=0.985$')
ax.set_yscale('log')
ax.set_xticks(x_pos)
ax.set_xticklabels([str(i) for i in iters_show], fontsize=9)
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('Gap $G_k$ (log scale)', fontsize=11)
ax.set_title('Gap values at selected iterations\n(Interior optimum, numerical)', fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.35, axis='y')
# Add numerical annotations for final iterations
for bar, val in zip(bars1[-2:], gaps_90[-2:]):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.5,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7)
for bar, val in zip(bars2[-2:], gaps_985[-2:]):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.5,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/q6_fw_gap.pdf', bbox_inches='tight')
plt.close()
print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Nelder-Mead simplex evolution on Rosenbrock
# ─────────────────────────────────────────────────────────────────────────────
print("Generating q4_nm_simplex_evolution.pdf ...")

def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def nm_with_simplex_history(f, x0, step=0.35, n_iters=160):
    """Minimal Nelder-Mead implementation that records simplex at each step."""
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        v = x0.copy()
        v[i] += step
        simplex.append(v)
    simplex = np.array(simplex, dtype=float)

    simplex_history = [simplex.copy()]
    f_hist = [min(f(v) for v in simplex)]

    alpha_r, gamma_e, rho_c, sigma_s = 1.0, 2.0, 0.5, 0.5

    for _ in range(n_iters):
        # Sort
        order = np.argsort([f(v) for v in simplex])
        simplex = simplex[order]
        fvals = np.array([f(v) for v in simplex])

        centroid = simplex[:-1].mean(axis=0)

        # Reflection
        x_r = centroid + alpha_r * (centroid - simplex[-1])
        f_r = f(x_r)

        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma_e * (x_r - centroid)
            if f(x_e) < f_r:
                simplex[-1] = x_e
            else:
                simplex[-1] = x_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
        else:
            # Contraction
            x_c = centroid + rho_c * (simplex[-1] - centroid)
            if f(x_c) < fvals[-1]:
                simplex[-1] = x_c
            else:
                # Shrink
                simplex[1:] = simplex[0] + sigma_s * (simplex[1:] - simplex[0])

        simplex_history.append(simplex.copy())
        f_hist.append(f(simplex[0]))

    return simplex_history, np.array(f_hist)

x0_C = np.array([-1.0, 1.0])
simplex_hist, f_nm = nm_with_simplex_history(rosenbrock, x0_C, step=0.35, n_iters=160)

# Create grid for contour
x1g = np.linspace(-1.5, 1.5, 300)
x2g = np.linspace(-0.5, 2.5, 300)
X1g, X2g = np.meshgrid(x1g, x2g)
Zg = (1 - X1g)**2 + 100*(X2g - X1g**2)**2

# Select specific iterations to show the simplex
show_iters = [0, 5, 15, 40, 80, 160]
colours_iters = [BLUE, ORANGE, GREEN, RED, PURPLE, BROWN]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
cs = ax.contourf(X1g, X2g, np.log10(Zg + 1e-6), levels=20, cmap='coolwarm', alpha=0.5)
ax.contour(X1g, X2g, np.log10(Zg + 1e-6), levels=20, colors='white', linewidths=0.4, alpha=0.6)
plt.colorbar(cs, ax=ax, label='$\\log_{10}(f+10^{-6})$', pad=0.02)

for it, col in zip(show_iters, colours_iters):
    s = simplex_hist[it]
    # close the triangle
    tri = np.vstack([s, s[0]])
    ax.plot(tri[:, 0], tri[:, 1], '-', color=col, lw=2.0,
            label=f'Iter {it}', zorder=5)
    ax.fill(s[:, 0], s[:, 1], alpha=0.25, color=col)

# Trace centroid path
centroids = np.array([s.mean(axis=0) for s in simplex_hist])
ax.plot(centroids[:, 0], centroids[:, 1], 'k--', lw=1.0, alpha=0.5, label='Centroid path')
ax.plot(1, 1, 'r*', ms=14, zorder=10, label='Optimum $(1,1)$')
ax.plot(x0_C[0], x0_C[1], 'kx', ms=10, mew=2, zorder=10, label='Start $(-1,1)$')
ax.set_xlabel('$x_1$', fontsize=11)
ax.set_ylabel('$x_2$', fontsize=11)
ax.set_title('Nelder--Mead simplex evolution\n(Rosenbrock, 160 iterations)', fontsize=10)
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)

ax = axes[1]
ax.semilogy(np.arange(len(f_nm)), f_nm, color=BLUE, lw=1.8, label='Best $f$ in simplex')

# Add annotations at the show_iters
for it, col in zip(show_iters[1:], colours_iters[1:]):
    ax.axvline(it, color=col, ls='--', lw=1.0, alpha=0.7)
    ax.text(it + 1, f_nm[it] * 1.5, f'{it}', color=col, fontsize=8)

# Compute simplex sizes (diameter of simplex) over time
sizes = [np.max(np.linalg.norm(s - s.mean(axis=0), axis=1)) for s in simplex_hist]
ax2 = ax.twinx()
ax2.semilogy(np.arange(len(sizes)), sizes, color=ORANGE, lw=1.4, ls=':', alpha=0.8)
ax2.set_ylabel('Simplex diameter', color=ORANGE, fontsize=10)
ax2.tick_params(axis='y', labelcolor=ORANGE)

ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('$f(\\mathrm{best\\ vertex})$', fontsize=11)
ax.set_title('Convergence and simplex size\nvs iteration', fontsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/q4_nm_simplex_evolution.pdf', bbox_inches='tight')
plt.close()
print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Oracle-type comparison – iterations to achieve ε-accuracy on C
# ─────────────────────────────────────────────────────────────────────────────
print("Generating oracle_comparison.pdf ...")

# Simulate each oracle type on Rosenbrock
def gd_rosenbrock(x0, alpha, n_iters):
    x = x0.copy().astype(float)
    hist = [rosenbrock(x)]
    for _ in range(n_iters):
        g = np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
                       200*(x[1]-x[0]**2)])
        x = x - alpha * g
        hist.append(rosenbrock(x))
    return np.array(hist)

def nesterov_rosenbrock(x0, alpha, beta_max, n_iters):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    hist = [rosenbrock(x)]
    for k in range(1, n_iters+1):
        beta_k = min((k-1)/(k+2), beta_max)
        look = x + beta_k * z
        g = np.array([-2*(1-look[0]) - 400*look[0]*(look[1]-look[0]**2),
                       200*(look[1]-look[0]**2)])
        z = beta_k * z - alpha * g
        x = x + z
        hist.append(rosenbrock(x))
    return np.array(hist)

def newton_rosenbrock(x0, alpha, n_iters):
    x = x0.copy().astype(float)
    hist = [rosenbrock(x)]
    for _ in range(n_iters):
        g = np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
                       200*(x[1]-x[0]**2)])
        H = np.array([[2 + 1200*x[0]**2 - 400*x[1], -400*x[0]],
                       [-400*x[0], 200.0]])
        H_reg = H + 1e-6 * np.eye(2)
        try:
            p = np.linalg.solve(H_reg, g)
        except:
            p = g
        x = x - alpha * p
        hist.append(rosenbrock(x))
    return np.array(hist)

def nrs_rosenbrock(x0, alpha, delta, n_iters):
    x = x0.copy().astype(float)
    hist = [rosenbrock(x)]
    for _ in range(n_iters):
        u = np.random.randn(2)
        u /= np.linalg.norm(u)
        g_hat = (rosenbrock(x + delta*u) - rosenbrock(x)) / delta * u
        x = x - alpha * g_hat
        hist.append(rosenbrock(x))
    return np.array(hist)

x0_C = np.array([-1.0, 1.0])
N_MAX = 500

fh_gd    = gd_rosenbrock(x0_C, 0.0012, N_MAX)
fh_nes   = nesterov_rosenbrock(x0_C, 0.0007, 0.90, N_MAX)
fh_newt  = newton_rosenbrock(x0_C, 0.22, 20)  # only 20 steps
fh_nrs   = nrs_rosenbrock(x0_C, 0.002, 0.01, N_MAX)
fh_nm    = f_nm  # from above (160 iters)

# Count function evaluations
# GD:    1 grad = 1+d evals = 3 evals per iter
# Nes:   1 grad per iter = 3 evals
# Newton: grad (3 evals) + 4 Hessian evals (central diff in 2D) = 7 evals
# NRS:   2 evals per iter
# NM:    up to d+2=4 evals per iter on average ~2.5 for 2D
fe_gd   = np.arange(len(fh_gd))   * 3    # n+1 per step
fe_nes  = np.arange(len(fh_nes))  * 3
fe_newt = np.arange(len(fh_newt)) * 7    # grad+hessian
fe_nrs  = np.arange(len(fh_nrs))  * 2
fe_nm   = np.arange(len(fh_nm))   * 2.5  # approx

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: function value vs iterations
ax = axes[0]
ax.semilogy(np.arange(len(fh_gd)),   fh_gd,  color=BLUE,   lw=1.8, label='GD (1st-order, $\\alpha=0.0012$)')
ax.semilogy(np.arange(len(fh_nes)),  fh_nes, color=GREEN,  lw=1.8, label='Nesterov (1st-order)')
ax.semilogy(np.arange(len(fh_newt)), fh_newt,color=RED,    lw=2.2, label='Newton (2nd-order, 20 iters)', marker='o', ms=5, markevery=2)
ax.semilogy(np.arange(len(fh_nrs)),  fh_nrs, color=ORANGE, lw=1.4, label='NRS (0th-order)', alpha=0.8)
ax.semilogy(np.arange(len(fh_nm)),   fh_nm,  color=PURPLE, lw=1.4, label='Nelder--Mead (0th-order)')
ax.set_xlabel('Iterations $k$', fontsize=11)
ax.set_ylabel('$f(x_k)$ (log scale)', fontsize=11)
ax.set_title('Oracle type comparison on Rosenbrock\n(Benchmark C, $f^\\star=0$)', fontsize=10)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, 300)
ax.set_ylim(1e-2, 20)

# Right: function value vs function evaluations
ax = axes[1]
ax.semilogy(fe_gd[:301],   fh_gd[:301],  color=BLUE,   lw=1.8, label='GD (3 FE/iter)')
ax.semilogy(fe_nes[:301],  fh_nes[:301], color=GREEN,  lw=1.8, label='Nesterov (3 FE/iter)')
ax.semilogy(fe_newt,       fh_newt,      color=RED,    lw=2.2, label='Newton (7 FE/iter)', marker='o', ms=5)
ax.semilogy(fe_nrs[:301],  fh_nrs[:301], color=ORANGE, lw=1.4, label='NRS (2 FE/iter)', alpha=0.8)
ax.semilogy(fe_nm,         fh_nm,        color=PURPLE, lw=1.4, label='Nelder--Mead ($\\approx$2.5 FE/iter)')
ax.set_xlabel('Function evaluations', fontsize=11)
ax.set_ylabel('$f(x_k)$ (log scale)', fontsize=11)
ax.set_title('Oracle type comparison (cost-adjusted)\nvs function evaluations', fontsize=10)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, 900)
ax.set_ylim(1e-2, 20)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/oracle_comparison.pdf', bbox_inches='tight')
plt.close()
print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Adam warmup / bias-correction visualisation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating q2_adam_bias_correction.pdf ...")

def adam_run(x0, alpha, beta1, beta2, eps, n_iters, warmup=False, warmup_steps=20):
    x = x0.copy().astype(float)
    m_v, v_v = np.zeros_like(x), np.zeros_like(x)
    hist = [f_B(x)]
    for t in range(1, n_iters+1):
        g = grad_B(x)
        m_v = beta1 * m_v + (1 - beta1) * g
        v_v = beta2 * v_v + (1 - beta2) * g**2
        m_hat = m_v / (1 - beta1**t)
        v_hat = v_v / (1 - beta2**t)
        if warmup:
            eff_alpha = alpha * min(1.0, t / warmup_steps)
        else:
            eff_alpha = alpha
        x = x - eff_alpha * m_hat / (np.sqrt(v_hat) + eps)
        hist.append(f_B(x))
    return np.array(hist)

x0_B = np.array([-1.0, 4.0])
N_ADAM = 150

# Compare: with/without bias correction, with/without warmup
configs = [
    ('Adam (bias-corrected, no warmup)', 0.08, 0.80, 0.999, False),
    ('Adam (warmup 20 steps)',           0.12, 0.80, 0.999, True),
]

# Also show moment evolution (v_t for slow vs fast beta2)
beta2_vals = [0.9, 0.99, 0.999]
t_range = np.arange(1, N_ADAM+1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
for name, alpha, beta1, beta2, warmup in configs:
    fh = adam_run(x0_B, alpha, beta1, beta2, 1e-8, N_ADAM, warmup)
    ax.semilogy(np.arange(len(fh)), fh - 0.7244, lw=1.8, label=name)
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('$f(x_k) - f^\\star$', fontsize=11)
ax.set_title('Adam variants on Benchmark B', fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.35)

ax = axes[1]
# Bias correction factor (1 - beta2^t) for different beta2
for b2, col in zip(beta2_vals, [BLUE, ORANGE, RED]):
    corr = 1 - b2**t_range
    ax.plot(t_range, corr, color=col, lw=1.8, label=f'$\\beta_2 = {b2}$')
ax.axhline(1, color='k', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel('Iteration $t$', fontsize=11)
ax.set_ylabel('Bias-correction factor $(1-\\beta_2^t)$', fontsize=11)
ax.set_title('Bias correction warm-up speed\nfor different $\\beta_2$', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.35)
ax.set_xlim(1, 100)
ax.set_ylim(0, 1.05)
ax.fill_between(t_range[:20], 0, 1 - 0.999**t_range[:20],
                alpha=0.15, color=RED, label='Slow warm-up region')

ax = axes[2]
# Effective LR = alpha * sqrt(1-beta2^t) / sqrt(v_hat) approximation
# Show that without bias correction, effective LR is alpha * (1-beta2^t)^{1/2} / sigma
# With bias correction, it converges to alpha/sigma
sigma = 1.0  # normalised
for b2, col in zip(beta2_vals, [BLUE, ORANGE, RED]):
    eff_lr_no_corr = 0.08 * np.sqrt(1 - b2**t_range) / sigma
    eff_lr_corr    = 0.08 * np.ones_like(t_range) / sigma
    ax.plot(t_range, eff_lr_no_corr, color=col, lw=1.8, label=f'Without correction $\\beta_2={b2}$')
ax.axhline(0.08, color='k', ls='--', lw=1.4, label='With bias correction (any $\\beta_2$)')
ax.set_xlabel('Iteration $t$', fontsize=11)
ax.set_ylabel('Effective LR (normalised)', fontsize=11)
ax.set_title('Bias correction effect on effective\nlearning rate (early iterations)', fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.35)
ax.set_xlim(1, 80)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/q2_adam_bias_correction.pdf', bbox_inches='tight')
plt.close()
print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Cross-benchmark convergence heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("Generating cross_benchmark_heatmap.pdf ...")

# Final objective values for each method on each benchmark (from results tables)
methods = ['GD', 'Polyak', 'Adagrad', 'RMSprop', 'Heavy Ball',
           'Nesterov', 'Adam',
           'Newton',
           'FD GD δ=0.05', 'NRS', 'Nelder-Mead']
bench_A = [0.4826, 2.7189, 0.4826, 0.5027, 0.4826,
           0.4826, 0.4826,
           0.4826,
           np.nan, np.nan, np.nan]
bench_B = [0.7244, 24.0659, 0.7244, 0.7271, 0.7244,
           0.7244, 0.7244,
           0.7244,
           0.7244, 0.85, np.nan]  # NRS approx
bench_C = [3.5142, 0.0094, 2.2034, 3.1204, 1.4003,
           0.4788, 1.5785,
           0.6864,
           np.nan, np.nan, 2.1]  # NM approximate for C

# Normalise: for each benchmark, compute relative to best known
f_star_A, f_star_B, f_star_C = 0.4826, 0.7244, 0.0
vals_A = np.array([v - f_star_A if not np.isnan(v) else np.nan for v in bench_A])
vals_B = np.array([v - f_star_B if not np.isnan(v) else np.nan for v in bench_B])
vals_C = np.array([v - f_star_C if not np.isnan(v) else np.nan for v in bench_C])

# Stack into matrix: rows = methods, cols = benchmarks
mat = np.column_stack([vals_A, vals_B, vals_C])
mat_log = np.log10(np.maximum(mat, 1e-6))

fig, ax = plt.subplots(figsize=(7, 7))
# Use a masked array for NaN
mat_masked = np.ma.array(mat_log, mask=np.isnan(mat_log))
cmap = plt.cm.RdYlGn_r
cmap.set_bad('lightgrey')
im = ax.imshow(mat_masked, cmap=cmap, aspect='auto', vmin=-3, vmax=2)
plt.colorbar(im, ax=ax, label='$\\log_{10}(f - f^\\star)$', pad=0.02)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Bench. A\n($\\kappa\\approx1.12$)',
                     'Bench. B\n($\\kappa\\approx3.5$--6.9)',
                     'Bench. C\n($\\kappa\\approx2000$)'], fontsize=10)
ax.set_yticks(np.arange(len(methods)))
ax.set_yticklabels(methods, fontsize=10)
ax.set_title('Cross-benchmark performance heatmap\n$\\log_{10}(f-f^\\star)$ (green = better)', fontsize=11)

# Annotate cells
for i in range(len(methods)):
    for j, (raw_val, bmark) in enumerate(zip([bench_A, bench_B, bench_C], ['A','B','C'])):
        v = raw_val[i]
        if np.isnan(v):
            ax.text(j, i, 'N/A', ha='center', va='center', fontsize=8.5, color='grey')
        else:
            gap = max(v - [f_star_A, f_star_B, f_star_C][j], 1e-6)
            lv = np.log10(gap)
            txt_col = 'white' if lv > 1.5 or lv < -2 else 'black'
            ax.text(j, i, f'{gap:.4f}', ha='center', va='center', fontsize=8, color=txt_col, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/cross_benchmark_heatmap.pdf', bbox_inches='tight')
plt.close()
print("  done.")

print("\nAll Pass 4 figures generated successfully.")
