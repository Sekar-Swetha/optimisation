"""Pass 6 figures: Nelder-Mead simplex evolution, Q1 convergence rates,
augmented Lagrangian comparison, deeper rate analysis."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os, sys

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(42)

# ── Rosenbrock and derivatives ──────────────────────────────────────────────
def rosen(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosen_grad(x):
    g0 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    g1 = 200*(x[1] - x[0]**2)
    return np.array([g0, g1])

# ── Benchmark B function ─────────────────────────────────────────────────────
def bench_b(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])

def bench_b_grad(x):
    return np.array([2*(x[0]-1) + np.cos(x[0]), 10*(x[1]-2)])

# ── Optimisers ────────────────────────────────────────────────────────────────
def gradient_descent(grad_fn, loss_fn, x0, alpha, n):
    x = x0.copy().astype(float)
    hist = [loss_fn(x)]
    for _ in range(n):
        x -= alpha * grad_fn(x)
        hist.append(loss_fn(x))
    return np.array(hist)

def adagrad(grad_fn, loss_fn, x0, alpha0, eps, n):
    x = x0.copy().astype(float)
    G = np.zeros_like(x)
    hist = [loss_fn(x)]
    for _ in range(n):
        g = grad_fn(x)
        G += g**2
        x -= alpha0 / (np.sqrt(G) + eps) * g
        hist.append(loss_fn(x))
    return np.array(hist)

def heavy_ball(grad_fn, loss_fn, x0, alpha, beta, n):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    hist = [loss_fn(x)]
    for _ in range(n):
        g = grad_fn(x)
        z = beta * z + alpha * g
        x -= z
        hist.append(loss_fn(x))
    return np.array(hist)

def nesterov(grad_fn, loss_fn, x0, alpha, beta_max, n):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    hist = [loss_fn(x)]
    for k in range(1, n+1):
        bk = min((k-1)/(k+2), beta_max)
        la = x + bk * z
        g = grad_fn(la)
        z = bk * z - alpha * g
        x += z
        hist.append(loss_fn(x))
    return np.array(hist)

def rmsprop(grad_fn, loss_fn, x0, alpha0, beta, eps, n):
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    hist = [loss_fn(x)]
    for _ in range(n):
        g = grad_fn(x)
        v = beta * v + (1-beta) * g**2
        x -= alpha0 / (np.sqrt(v) + eps) * g
        hist.append(loss_fn(x))
    return np.array(hist)

# ============================================================
# Figure 1: Nelder-Mead Simplex Evolution on Rosenbrock
# ============================================================
def nelder_mead_tracked(loss_fn, x0, step=0.35, n_iters=160):
    """Return (x_hist, simplex_snapshots, op_names) with simplex tracked."""
    # initial simplex
    d = len(x0)
    simplex = np.zeros((d+1, d))
    simplex[0] = x0.copy()
    for i in range(d):
        v = x0.copy()
        v[i] += step
        simplex[i+1] = v

    fvals = np.array([loss_fn(s) for s in simplex])
    x_hist = [simplex[np.argmin(fvals)].copy()]
    snapshots = []  # (iteration, simplex copy, operation name)
    snap_iters = [0, 5, 15, 40, 79, 139]

    alpha_r, gamma_e, rho_c, sigma_s = 1.0, 2.0, 0.5, 0.5

    for it in range(n_iters):
        if it in snap_iters:
            snapshots.append((it, simplex.copy(), fvals.copy()))

        order = np.argsort(fvals)
        simplex = simplex[order]; fvals = fvals[order]
        best, worst, sw = simplex[0], simplex[-1], simplex[-2]
        centroid = simplex[:-1].mean(axis=0)

        # reflection
        xr = centroid + alpha_r * (centroid - worst)
        fr = loss_fn(xr)
        if fr < fvals[0]:
            # try expansion
            xe = centroid + gamma_e * (xr - centroid)
            fe = loss_fn(xe)
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr; fvals[-1] = fr
        else:
            # contraction
            if fr < fvals[-1]:
                xc = centroid + rho_c * (xr - centroid)
                fc = loss_fn(xc)
                if fc < fr:
                    simplex[-1] = xc; fvals[-1] = fc
                else:
                    # shrink
                    for i in range(1, d+1):
                        simplex[i] = simplex[0] + sigma_s*(simplex[i]-simplex[0])
                        fvals[i] = loss_fn(simplex[i])
            else:
                xc = centroid + rho_c * (worst - centroid)
                fc = loss_fn(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc; fvals[-1] = fc
                else:
                    for i in range(1, d+1):
                        simplex[i] = simplex[0] + sigma_s*(simplex[i]-simplex[0])
                        fvals[i] = loss_fn(simplex[i])

        x_hist.append(simplex[0].copy())

    # snapshots are captured during the loop above; no extra needed

    return np.array(x_hist), snapshots

x0_C = np.array([-1.0, 1.0])
x_nm, snapshots = nelder_mead_tracked(rosen, x0_C)

# Contour grid
xx = np.linspace(-1.6, 1.5, 300)
yy = np.linspace(-0.5, 2.0, 300)
XX, YY = np.meshgrid(xx, yy)
ZZ = (1-XX)**2 + 100*(YY - XX**2)**2
levels = np.logspace(-0.5, 4, 22)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
titles_it = [s[0] for s in snapshots]

colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(snapshots)))

for idx, (it, simp, fv) in enumerate(snapshots):
    ax = axes[idx]
    ax.contourf(XX, YY, ZZ, levels=levels, cmap='YlOrRd', alpha=0.35)
    ax.contour(XX, YY, ZZ, levels=levels, colors='gray', linewidths=0.4, alpha=0.5)
    # Trajectory up to this point
    traj = x_nm[:it+1]
    ax.plot(traj[:,0], traj[:,1], 'b-', linewidth=0.8, alpha=0.6)
    ax.plot(x_nm[0,0], x_nm[0,1], 'bs', markersize=6)
    ax.plot(1, 1, 'g*', markersize=10, zorder=5)
    # Draw simplex
    tri = plt.Polygon(simp, fill=False, edgecolor=colours[idx], linewidth=2.5)
    ax.add_patch(tri)
    # Mark vertices
    order = np.argsort(fv)
    ax.plot(simp[order[0],0], simp[order[0],1], 'go', ms=7)   # best
    ax.plot(simp[order[-1],0], simp[order[-1],1], 'rx', ms=9, mew=2)  # worst
    f_best = fv.min()
    ax.set_title(f'Iter {it}: $f_\\mathrm{{best}} = {f_best:.4f}$', fontsize=10)
    ax.set_xlim(-1.6, 1.5); ax.set_ylim(-0.5, 2.0)
    ax.set_xlabel('$x_1$', fontsize=9); ax.set_ylabel('$x_2$', fontsize=9)
    ax.tick_params(labelsize=8)

# Legend patch
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker='o', color='g', linestyle='', label='Best vertex'),
       Line2D([0],[0], marker='x', color='r', linestyle='', ms=8, mew=2, label='Worst vertex'),
       Line2D([0],[0], marker='*', color='g', linestyle='', ms=10, label='True optimum $(1,1)$'),
       mpatches.Patch(facecolor='none', edgecolor='purple', label='Simplex')]
axes[-1].legend(handles=leg, loc='lower right', fontsize=8)

plt.suptitle('Nelder--Mead Simplex Evolution on Rosenbrock ($x_0 = (-1,1)$)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q4_nelder_mead_evolution.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q4_nelder_mead_evolution.pdf")

# ============================================================
# Figure 2: Q1 Log-Log Convergence Rate Comparison (Rosenbrock)
# ============================================================
x0_C = np.array([-1.0, 1.0])
N = 500  # more iterations for clear rate asymptote

f_gd  = gradient_descent(rosen_grad, rosen, x0_C, 0.0012, N)
f_ag  = adagrad(rosen_grad, rosen, x0_C, 0.45, 1e-5, N)
f_hb  = heavy_ball(rosen_grad, rosen, x0_C, 0.0008, 0.86, N)
f_rm  = rmsprop(rosen_grad, rosen, x0_C, 0.0035, 0.9, 1e-5, N)
f_nes = nesterov(rosen_grad, rosen, x0_C, 0.0007, 0.90, N)

# Also run Polyak with correct f_star=0
def polyak(grad_fn, loss_fn, x0, f_star, eps, n):
    x = x0.copy().astype(float)
    hist = [loss_fn(x)]
    for _ in range(n):
        g = grad_fn(x)
        alpha_k = (loss_fn(x) - f_star) / (np.dot(g,g) + eps)
        x -= alpha_k * g
        hist.append(loss_fn(x))
    return np.array(hist)

f_pk  = polyak(rosen_grad, rosen, x0_C, 0.0, 1e-3, N)

ks = np.arange(1, N+1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: log-log suboptimality
ax = axes[0]
eps_floor = 1e-10
for hist, label, color, ls in [
    (f_gd,  'GD',          'steelblue', '-'),
    (f_ag,  'Adagrad',     'orange',    '--'),
    (f_hb,  'Heavy Ball',  'green',     '-.'),
    (f_rm,  'RMSprop',     'red',       ':'),
    (f_nes, 'Nesterov',    'purple',    '-'),
    (f_pk,  'Polyak',      'brown',     '--'),
]:
    sub = np.maximum(hist[1:], eps_floor)
    ax.loglog(ks, sub, color=color, linestyle=ls, linewidth=1.6, label=label)

# Reference lines (anchored at k=1 to the GD value)
k_ref = np.logspace(0, np.log10(N), 200)
c1 = f_gd[1]
c2 = f_nes[1]
ax.loglog(k_ref, c1 / k_ref, 'k--', linewidth=1.0, alpha=0.7, label=r'$O(1/k)$ ref')
ax.loglog(k_ref, c2 / k_ref**2, 'k:', linewidth=1.0, alpha=0.7, label=r'$O(1/k^2)$ ref')

ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('$f(x_k)$', fontsize=11)
ax.set_title('Q1 Log--Log Convergence on Rosenbrock', fontsize=11)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1, N); ax.set_ylim(1e-4, 10)

# Right: convergence in first 120 iterations (semi-log)
ax2 = axes[1]
k120 = np.arange(121)
for hist, label, color, ls in [
    (f_gd,  'GD',          'steelblue', '-'),
    (f_ag,  'Adagrad',     'orange',    '--'),
    (f_hb,  'Heavy Ball',  'green',     '-.'),
    (f_rm,  'RMSprop',     'red',       ':'),
    (f_nes, 'Nesterov',    'purple',    '-'),
    (f_pk,  'Polyak',      'brown',     '--'),
]:
    ax2.semilogy(k120, np.maximum(hist[:121], eps_floor),
                 color=color, linestyle=ls, linewidth=1.6, label=label)
ax2.set_xlabel('Iteration $k$', fontsize=11)
ax2.set_ylabel('$f(x_k)$', fontsize=11)
ax2.set_title('Q1 Convergence (120 iters, semi-log)', fontsize=11)
ax2.legend(fontsize=9, ncol=2)
ax2.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q1_convergence_rates.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q1_convergence_rates.pdf")

# ============================================================
# Figure 3: Augmented Lagrangian vs Penalty vs PGD (Q5)
# ============================================================
def bench_b_loss(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])

def bench_b_grad_fn(x):
    return np.array([2*(x[0]-1) + np.cos(x[0]), 10*(x[1]-2)])

x0_constrained = np.array([0.2, 4.0])
n_iters_q5 = 120

# Projected GD
def pgd(grad_fn, loss_fn, x0, alpha, n, lb=0.5):
    x = x0.copy().astype(float)
    hist = [loss_fn(x)]
    for _ in range(n):
        z = x - alpha * grad_fn(x)
        x = np.array([max(lb, z[0]), z[1]])
        hist.append(loss_fn(x))
    return np.array(hist)

# Penalty method
def penalty_method(grad_fn, loss_fn, x0, alpha, lam, n, lb=0.5):
    x = x0.copy().astype(float)
    hist = [loss_fn(x)]
    for _ in range(n):
        g = grad_fn(x)
        if x[0] < lb:
            g = g + np.array([-lam, 0.0])
        x -= alpha * g
        hist.append(loss_fn(x))
    return np.array(hist)

# Augmented Lagrangian method
def augmented_lagrangian(grad_fn, loss_fn, x0, alpha, lam0, rho, n, lb=0.5, mu_update_freq=10):
    """Augmented Lagrangian: F(x,mu) = f(x) + mu*c(x) + rho/2 * max(0,c(x))^2
       where c(x) = lb - x[0] (violation is c(x) > 0)."""
    x = x0.copy().astype(float)
    mu = 0.0  # dual variable (multiplier estimate)
    hist = [loss_fn(x)]
    for k in range(n):
        c = lb - x[0]  # constraint violation (positive = infeasible)
        c_pos = max(0.0, c)
        # gradient of augmented Lagrangian
        g = grad_fn(x)
        # d/dx c(x) = -e_1
        g_aug = g + (-1) * (mu + rho * c_pos) * np.array([1.0, 0.0])
        x -= alpha * g_aug
        # update multiplier every mu_update_freq iterations
        if (k+1) % mu_update_freq == 0:
            c_new = lb - x[0]
            mu = max(0.0, mu + rho * c_new)
        hist.append(loss_fn(x))
    return np.array(hist)

# Violation tracking
def pgd_violation(grad_fn, loss_fn, x0, alpha, n, lb=0.5):
    x = x0.copy().astype(float)
    viols = [max(0, lb - x[0])]
    for _ in range(n):
        z = x - alpha * grad_fn(x)
        x = np.array([max(lb, z[0]), z[1]])
        viols.append(max(0, lb - x[0]))
    return np.array(viols)

def penalty_violation(grad_fn, loss_fn, x0, alpha, lam, n, lb=0.5):
    x = x0.copy().astype(float)
    viols = [max(0, lb - x[0])]
    for _ in range(n):
        g = grad_fn(x)
        if x[0] < lb:
            g = g + np.array([-lam, 0.0])
        x -= alpha * g
        viols.append(max(0, lb - x[0]))
    return np.array(viols)

def al_violation(grad_fn, loss_fn, x0, alpha, lam0, rho, n, lb=0.5, mu_update_freq=10):
    x = x0.copy().astype(float)
    mu = 0.0
    viols = [max(0, lb - x[0])]
    for k in range(n):
        c = lb - x[0]
        c_pos = max(0.0, c)
        g = grad_fn(x)
        g_aug = g + (-1) * (mu + rho * c_pos) * np.array([1.0, 0.0])
        x -= alpha * g_aug
        if (k+1) % mu_update_freq == 0:
            c_new = lb - x[0]
            mu = max(0.0, mu + rho * c_new)
        viols.append(max(0, lb - x[0]))
    return np.array(viols)

pgd_hist  = pgd(bench_b_grad_fn, bench_b_loss, x0_constrained, 0.06, n_iters_q5)
pen1_hist = penalty_method(bench_b_grad_fn, bench_b_loss, x0_constrained, 0.04, 1.0, n_iters_q5)
pen5_hist = penalty_method(bench_b_grad_fn, bench_b_loss, x0_constrained, 0.03, 4.5, n_iters_q5)
al_hist   = augmented_lagrangian(bench_b_grad_fn, bench_b_loss, x0_constrained, 0.05, 0, 2.0, n_iters_q5)

pgd_viol  = pgd_violation(bench_b_grad_fn, bench_b_loss, x0_constrained, 0.06, n_iters_q5)
pen1_viol = penalty_violation(bench_b_grad_fn, bench_b_loss, x0_constrained, 0.04, 1.0, n_iters_q5)
pen5_viol = penalty_violation(bench_b_grad_fn, bench_b_loss, x0_constrained, 0.03, 4.5, n_iters_q5)
al_viol   = al_violation(bench_b_grad_fn, bench_b_loss, x0_constrained, 0.05, 0, 2.0, n_iters_q5)

ks = np.arange(n_iters_q5+1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogy(ks, pgd_hist,  'b-',  lw=2,   label='PGD ($\\alpha=0.06$)')
ax.semilogy(ks, pen1_hist, 'r--', lw=1.8, label='Penalty $\\lambda=1$ ($\\alpha=0.04$)')
ax.semilogy(ks, pen5_hist, 'g:',  lw=1.8, label='Penalty $\\lambda=4.5$ ($\\alpha=0.03$)')
ax.semilogy(ks, al_hist,   'm-.',  lw=1.8, label='Aug.\ Lagrangian ($\\rho=2, \\alpha=0.05$)')
ax.axhline(0.7244, color='k', linestyle=':', linewidth=1.0, alpha=0.6, label='$f^\\star = 0.7244$')
ax.set_xlabel('Iteration', fontsize=11); ax.set_ylabel('$f(x_k)$', fontsize=11)
ax.set_title('Objective Convergence: Constrained Methods', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax2 = axes[1]
eps_floor = 1e-15
ax2.semilogy(ks, np.maximum(pgd_viol,  eps_floor), 'b-',  lw=2,   label='PGD')
ax2.semilogy(ks, np.maximum(pen1_viol, eps_floor), 'r--', lw=1.8, label='Penalty $\\lambda=1$')
ax2.semilogy(ks, np.maximum(pen5_viol, eps_floor), 'g:',  lw=1.8, label='Penalty $\\lambda=4.5$')
ax2.semilogy(ks, np.maximum(al_viol,   eps_floor), 'm-.', lw=1.8, label='Aug.\ Lagrangian')
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Constraint violation $c(x_k) = \\max(0, 0.5 - x_1)$', fontsize=11)
ax2.set_title('Feasibility: Constraint Violation', fontsize=11)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.suptitle('Constrained Optimisation: PGD vs Penalty vs Augmented Lagrangian', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q5_augmented_lagrangian.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q5_augmented_lagrangian.pdf")

# ============================================================
# Figure 4: Newton Quadratic Convergence Demonstration
# ============================================================
# For g(x) = x^4 starting at x0 = 0.25, track iterates
def g(x): return x**4
def g_prime(x): return 4*x**3
def g_double(x): return 12*x**2

x_iter = [0.25]
for _ in range(6):
    xk = x_iter[-1]
    xnew = xk - g_prime(xk) / g_double(xk)
    x_iter.append(xnew)

x_iter = np.array(x_iter)
errors = np.abs(x_iter)  # x* = 0

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
xs = np.linspace(-0.05, 0.35, 300)
ax.plot(xs, g(xs), 'k-', linewidth=2, label='$g(x) = x^4$')

colors_it = plt.cm.Blues(np.linspace(0.3, 0.95, 6))
for i in range(6):
    xk = x_iter[i]
    # second-order approx
    quad_x = np.linspace(xk - 0.15, xk + 0.15, 200)
    quad_y = g(xk) + g_prime(xk)*(quad_x - xk) + 0.5*g_double(xk)*(quad_x-xk)**2
    ax.plot(quad_x, np.clip(quad_y, -0.001, None), '--', color=colors_it[i],
            linewidth=1.2, alpha=0.8)
    ax.plot(xk, g(xk), 'o', color=colors_it[i], markersize=8)
    ax.annotate(f'$x_{i}={xk:.4f}$', (xk, g(xk)), textcoords='offset points',
                xytext=(5, 5 + 12*i), fontsize=7.5, color=colors_it[i])

ax.plot(0, 0, 'r*', markersize=12, label='$x^\\star = 0$')
ax.set_xlim(-0.05, 0.35); ax.set_ylim(-0.001, 0.01)
ax.set_xlabel('$x$', fontsize=11); ax.set_ylabel('$g(x)$', fontsize=11)
ax.set_title('Newton Iterates on $g(x) = x^4$ from $x_0 = 0.25$', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax2 = axes[1]
k_it = np.arange(len(errors))
ax2.semilogy(k_it, errors, 'b-o', linewidth=2, markersize=7, label='$|x_k - x^\\star|$')
# Theoretical quadratic bound: for g(x)=x^4, g'''= 24x, g'' = 12x^2
# M = max|g'''|/|g''|^{1/2} near x*=0 is not well-defined (degenerate at x*=0)
# Show the empirical doubling of correct digits
for i in range(1, len(errors)):
    if errors[i] > 1e-15:
        rate = np.log10(errors[i]) / np.log10(errors[i-1]) if errors[i-1] > 0 else 0
        ax2.annotate(f'rate≈{rate:.1f}',
                     (i, errors[i]), textcoords='offset points',
                     xytext=(6, 2), fontsize=8, color='darkblue')

ax2.set_xlabel('Newton iteration $k$', fontsize=11)
ax2.set_ylabel('$|x_k - x^\\star| = |x_k|$', fontsize=11)
ax2.set_title('Quadratic Convergence of Newton on $g(x)=x^4$', fontsize=11)
ax2.legend(fontsize=10); ax2.grid(True, which='both', alpha=0.3)
ax2.set_xticks(k_it)

# Annotate: note x*=0 is degenerate (g''(0)=0), so rate is super-quadratic at start
ax2.text(0.98, 0.05, 'Note: $g\'\'(x^\\star)=0$ (degenerate);\nbehaviour is super-quadratic',
         transform=ax2.transAxes, ha='right', va='bottom', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q3_newton_quadratic.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q3_newton_quadratic.pdf")

# ============================================================
# Figure 5: Adam vs Nesterov: why Adam misses O(1/k^2) rate
# ============================================================
def adam(grad_fn, loss_fn, x0, alpha, b1, b2, eps, n):
    x = x0.copy().astype(float)
    m = np.zeros_like(x); v = np.zeros_like(x)
    hist = [loss_fn(x)]
    for t in range(1, n+1):
        g = grad_fn(x)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g**2
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        x -= alpha * mh / (np.sqrt(vh) + eps)
        hist.append(loss_fn(x))
    return np.array(hist)

x0_B = np.array([-1.0, 4.0])
N_B = 300
f_star_B = 0.7244

f_nes_B  = nesterov(bench_b_grad, bench_b, x0_B, 0.035, 0.92, N_B)
f_adam_B = adam(bench_b_grad, bench_b, x0_B, 0.08, 0.82, 0.999, 1e-8, N_B)
f_gd_B   = gradient_descent(bench_b_grad, bench_b, x0_B, 0.06, N_B)

sub_nes  = np.maximum(f_nes_B[1:]  - f_star_B, 1e-12)
sub_adam = np.maximum(f_adam_B[1:] - f_star_B, 1e-12)
sub_gd   = np.maximum(f_gd_B[1:]  - f_star_B, 1e-12)

ks_B = np.arange(1, N_B+1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.loglog(ks_B, sub_nes,  'purple',    linewidth=2, label='Nesterov')
ax.loglog(ks_B, sub_adam, 'darkorange', linewidth=2, label='Adam')
ax.loglog(ks_B, sub_gd,   'steelblue', linewidth=2, label='GD')

k_ref = np.logspace(0, np.log10(N_B), 200)
c_nes = sub_nes[0]
c_gd  = sub_gd[0]
ax.loglog(k_ref, c_gd  / k_ref,     'b--', linewidth=1.0, alpha=0.7, label=r'$O(1/k)$')
ax.loglog(k_ref, c_nes / k_ref**2,  'p--', linewidth=1.0, alpha=0.7, label=r'$O(1/k^2)$', color='purple')

ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('$f(x_k) - f^\\star$', fontsize=11)
ax.set_title('Convergence Rate: Nesterov vs Adam vs GD\n(Benchmark B, log--log)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1, N_B)

# Right: effective step size over time
def nesterov_steps(grad_fn, loss_fn, x0, alpha, beta_max, n):
    x = x0.copy().astype(float); z = np.zeros_like(x)
    steps = []
    for k in range(1, n+1):
        bk = min((k-1)/(k+2), beta_max)
        la = x + bk * z
        g = grad_fn(la)
        z = bk * z - alpha * g
        steps.append(np.linalg.norm(alpha * g))
        x += z
    return np.array(steps)

def adam_steps(grad_fn, loss_fn, x0, alpha, b1, b2, eps, n):
    x = x0.copy().astype(float)
    m = np.zeros_like(x); v = np.zeros_like(x)
    steps = []
    for t in range(1, n+1):
        g = grad_fn(x)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g**2
        mh = m/(1-b1**t); vh = v/(1-b2**t)
        effective = alpha * mh / (np.sqrt(vh) + eps)
        steps.append(np.linalg.norm(effective))
        x -= effective
    return np.array(steps)

nes_steps  = nesterov_steps(bench_b_grad, bench_b, x0_B, 0.035, 0.92, N_B)
adam_steps_ = adam_steps(bench_b_grad, bench_b, x0_B, 0.08, 0.82, 0.999, 1e-8, N_B)
gd_steps   = np.array([np.linalg.norm(0.06 * bench_b_grad(x0_B))])  # constant

ax2 = axes[1]
ax2.semilogy(np.arange(1, N_B+1), nes_steps,   'purple',    linewidth=1.8, label='Nesterov $\\|\\alpha g_k\\|$')
ax2.semilogy(np.arange(1, N_B+1), adam_steps_,  'darkorange', linewidth=1.8, label='Adam effective step $\\|\\hat{m}_t/\\sqrt{\\hat{v}_t}\\| \\cdot \\alpha$')
ax2.set_xlabel('Iteration $k$', fontsize=11)
ax2.set_ylabel('Effective step norm', fontsize=11)
ax2.set_title('Effective Step Size: Nesterov vs Adam\n(Benchmark B)', fontsize=10)
ax2.legend(fontsize=9); ax2.grid(True, which='both', alpha=0.3)
ax2.text(0.5, 0.95, 'Adam step is bounded: $\\alpha/(1-\\beta_1)$\nNesterov step decays to 0 at convergence',
         transform=ax2.transAxes, ha='center', va='top', fontsize=8.5,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle("Why Adam Doesn't Achieve $O(1/k^2)$: Adaptive Scaling vs Momentum", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q2_adam_vs_nesterov_rates.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q2_adam_vs_nesterov_rates.pdf")

# ============================================================
# Figure 6: FW Gap as Convergence Certificate (Q6 detailed)
# ============================================================
def fw_gap_detailed(grad_fn, loss_fn, x0, bounds, beta, n):
    x = x0.copy().astype(float)
    fvals, gaps, z_hist = [loss_fn(x)], [], []
    x_hist = [x.copy()]
    for _ in range(n):
        g = grad_fn(x)
        z = np.array([bounds[i][0] if g[i] > 0 else bounds[i][1] for i in range(len(g))])
        gap = np.dot(g, x - z)
        gaps.append(gap)
        z_hist.append(z.copy())
        x = beta * x + (1-beta) * z
        fvals.append(loss_fn(x))
        x_hist.append(x.copy())
    return np.array(fvals), np.array(gaps), np.array(z_hist)

def fw_interior_grad(x):
    return np.array([2*(x[0]-1), 2*(x[1]-5)])
def fw_interior_loss(x):
    return (x[0]-1)**2 + (x[1]-5)**2

bounds = [(0.5, 5.0), (-5.0, 10.0)]
x0_fw = np.array([1.0, 1.0])
N_fw = 200

fv90,  gaps90,  _ = fw_gap_detailed(fw_interior_grad, fw_interior_loss, x0_fw, bounds, 0.90, N_fw)
fv985, gaps985, _ = fw_gap_detailed(fw_interior_grad, fw_interior_loss, x0_fw, bounds, 0.985, N_fw)

f_star_fw = 0.0
sub90  = np.maximum(fv90[1:]  - f_star_fw, 1e-10)
sub985 = np.maximum(fv985[1:] - f_star_fw, 1e-10)

ks_fw = np.arange(1, N_fw+1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Left: semi-log convergence + gap
ax = axes[0]
ax.semilogy(ks_fw, np.maximum(fv90[1:],  1e-10), 'b-',  lw=2,   label='$f(x_k)$, $\\beta=0.90$')
ax.semilogy(ks_fw, np.maximum(fv985[1:], 1e-10), 'r-',  lw=2,   label='$f(x_k)$, $\\beta=0.985$')
ax.semilogy(ks_fw, np.maximum(gaps90,    1e-10), 'b--', lw=1.5, label='FW gap $g_k$, $\\beta=0.90$')
ax.semilogy(ks_fw, np.maximum(gaps985,   1e-10), 'r--', lw=1.5, label='FW gap $g_k$, $\\beta=0.985$')
ax.set_xlabel('Iteration $k$'); ax.set_ylabel('Value')
ax.set_title('FW Objective and Gap (semi-log)', fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Centre: log-log rate
ax2 = axes[1]
ax2.loglog(ks_fw, sub90,  'b-', lw=2, label='$\\beta=0.90$')
ax2.loglog(ks_fw, sub985, 'r-', lw=2, label='$\\beta=0.985$')
k_ref2 = np.logspace(0, np.log10(N_fw), 200)
ax2.loglog(k_ref2, sub90[0] / k_ref2, 'k--', lw=1.0, alpha=0.6, label='$O(1/k)$ ref')
ax2.loglog(k_ref2, sub90[0] / k_ref2**2, 'k:', lw=1.0, alpha=0.6, label='$O(1/k^2)$ ref')
ax2.set_xlabel('Iteration $k$'); ax2.set_ylabel('$f(x_k) - f^\\star$')
ax2.set_title('FW Convergence Rate (log--log)', fontsize=10)
ax2.legend(fontsize=8); ax2.grid(True, which='both', alpha=0.3)

# Right: gap / suboptimality ratio (should be >= 1 always)
ratio90  = np.maximum(gaps90,  1e-10) / sub90
ratio985 = np.maximum(gaps985, 1e-10) / sub985
ax3 = axes[2]
ax3.semilogy(ks_fw, ratio90,  'b-', lw=2, label='$g_k / (f(x_k)-f^\\star)$, $\\beta=0.90$')
ax3.semilogy(ks_fw, ratio985, 'r-', lw=2, label='$g_k / (f(x_k)-f^\\star)$, $\\beta=0.985$')
ax3.axhline(1.0, color='k', linestyle='--', lw=1.2, label='Ratio = 1 (tight bound)')
ax3.set_xlabel('Iteration $k$')
ax3.set_ylabel('$g_k\\ /\\ (f(x_k) - f^\\star)$')
ax3.set_title('FW Gap Tightness: $g_k \\geq f(x_k)-f^\\star$', fontsize=10)
ax3.legend(fontsize=8); ax3.grid(True, which='both', alpha=0.3)
ax3.text(0.5, 0.06, 'Always $\\geq 1$: gap is a\nvalid convergence certificate',
         transform=ax3.transAxes, ha='center', fontsize=8.5,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Frank--Wolfe: Convergence Rate and Gap Certificate Analysis', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q6_fw_gap_detailed.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q6_fw_gap_detailed.pdf")

# ============================================================
# Figure 7: Heavy Ball Stability Region
# ============================================================
# For quadratic f with eigenvalue pair [mu, L],
# HB converges iff both eigenvalues of the 2x2 iteration matrix are inside unit circle.
# Condition: 0 < alpha < 2(1+beta)/L  AND  beta < 1
# Optimal convergence rate: ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2 per step
# Show: rate vs beta for fixed alpha, and stability boundary in (alpha, beta) space

mu, L = 0.399, 1001.6  # Rosenbrock at optimum
kappa = L / mu

# Exact spectral radius of HB iteration matrix for quadratic with eigenvalue lambda
def hb_spectral_radius(alpha, beta, lam):
    """Spectral radius of [[1-alpha*lam + beta, -beta],[1,0]]."""
    a = 1 - alpha*lam + beta
    b = -beta
    # char poly: z^2 - a*z - b = 0 => z^2 - a*z + beta = 0 (since -b = beta)
    disc = a**2 - 4*beta
    if disc >= 0:
        z1 = (a + np.sqrt(disc))/2
        z2 = (a - np.sqrt(disc))/2
        return max(abs(z1), abs(z2))
    else:
        # complex roots, radius = sqrt(beta)
        return np.sqrt(beta)

# Stability region in (alpha, beta) space for two eigenvalues mu and L
alpha_grid = np.linspace(1e-4, 4.0/L, 200)
beta_grid  = np.linspace(0, 0.995, 200)
A, B = np.meshgrid(alpha_grid, beta_grid)
rho_mu = np.vectorize(hb_spectral_radius)(A, B, mu)
rho_L  = np.vectorize(hb_spectral_radius)(A, B, L)
rho_max = np.maximum(rho_mu, rho_L)  # worst-case over eigenvalues

# Optimal point: beta* = ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2, alpha* = (1-sqrt(beta*))^2/mu
beta_opt = ((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**2
alpha_opt = (1 - np.sqrt(beta_opt))**2 / mu

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
im = ax.contourf(alpha_grid * 1000, beta_grid, rho_max,
                 levels=[0, 0.5, 0.8, 0.95, 1.0, 1.5],
                 cmap='RdYlGn_r')
cs = ax.contour(alpha_grid * 1000, beta_grid, rho_max,
                levels=[1.0], colors='red', linewidths=2)
plt.colorbar(im, ax=ax, label='Spectral radius $\\rho$')
ax.clabel(cs, fmt='$\\rho=1$ (stability boundary)', fontsize=8)
ax.plot(alpha_opt * 1000, beta_opt, 'w*', markersize=14,
        label=f'Optimal $(\\alpha^\\star, \\beta^\\star) = ({alpha_opt*1000:.3f}\\times10^{{-3}}, {beta_opt:.3f})$')
ax.set_xlabel('$\\alpha \\times 10^3$', fontsize=11)
ax.set_ylabel('$\\beta$', fontsize=11)
ax.set_title(f'Heavy Ball Stability Region (Rosenbrock $\\kappa\\approx{kappa:.0f}$)', fontsize=10)
ax.legend(fontsize=8, loc='upper right')

# Right: convergence rate along optimal alpha* for different beta
ax2 = axes[1]
betas = np.linspace(0.01, 0.999, 300)
rho_opt_alpha = np.array([max(
    hb_spectral_radius(alpha_opt, b, mu),
    hb_spectral_radius(alpha_opt, b, L)
) for b in betas])

# Nesterov rate for comparison (not directly comparable but indicative)
# For quadratic f, Nesterov with alpha=1/L has rate (1-sqrt(mu/L))^k
nes_rate = 1 - np.sqrt(mu/L)

ax2.semilogy(betas, rho_opt_alpha, 'b-', lw=2, label='HB rate at $\\alpha^\\star$')
ax2.axvline(beta_opt, color='r', linestyle='--', lw=1.5,
            label=f'Optimal $\\beta^\\star={beta_opt:.3f}$')
ax2.axhline(nes_rate, color='g', linestyle=':', lw=1.5,
            label=f'GD rate at $\\alpha=1/L$: $1-1/\\sqrt{{\\kappa}}\\approx{nes_rate:.4f}$')
ax2.axhline((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1), color='purple', linestyle='-.',
            lw=1.5,
            label=f'HB optimal rate: $(\\sqrt{{\\kappa}}-1)/(\\sqrt{{\\kappa}}+1)\\approx{(np.sqrt(kappa)-1)/(np.sqrt(kappa)+1):.4f}$')
ax2.set_xlabel('$\\beta$', fontsize=11)
ax2.set_ylabel('Per-iteration convergence rate $\\rho$', fontsize=11)
ax2.set_title(f'HB Rate vs $\\beta$ at $\\alpha = \\alpha^\\star = {alpha_opt:.4e}$\n(Rosenbrock, $\\kappa\\approx{kappa:.0f}$)', fontsize=10)
ax2.legend(fontsize=8); ax2.grid(True, which='both', alpha=0.3)
ax2.set_xlim(0, 1); ax2.set_ylim(1e-4, 1.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q1_hb_stability.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q1_hb_stability.pdf")

print("All Pass 6 figures generated successfully.")
