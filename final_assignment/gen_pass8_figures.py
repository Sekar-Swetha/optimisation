"""Pass 8 figure generation: additional comparative and analysis figures."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
FIGDIR = 'figures'
BLUE, ORANGE, GREEN, RED, PURPLE = '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'
BROWN, GREY = '#8c564b', '#7f7f7f'

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark functions
# ─────────────────────────────────────────────────────────────────────────────
def rosenbrock(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def f_int(x): return (x[0]-1)**2 + (x[1]-5)**2
def grad_int(x): return np.array([2*(x[0]-1), 2*(x[1]-5)])

def f_bnd(x): return x[0]**2 + x[1]**2
def grad_bnd(x): return np.array([2*x[0], 2*x[1]])

bounds = [(0.5, 5.0), (-5.0, 10.0)]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: FW open-loop vs fixed-beta comparison
# ─────────────────────────────────────────────────────────────────────────────
print("Generating q6_fw_openloop.pdf ...")

def frank_wolfe_openloop(grad_fn, loss_fn, x0, bnds, n_iters):
    """FW with open-loop schedule gamma_k = 2/(k+2)."""
    x = x0.copy().astype(float)
    f_hist, gap_hist = [loss_fn(x)], []
    for k in range(n_iters):
        g = grad_fn(x)
        z = np.array([bnds[i][0] if g[i] > 0 else bnds[i][1] for i in range(len(g))])
        gap = float(g @ (x - z))
        gap_hist.append(max(gap, 1e-12))
        gamma = 2.0 / (k + 2)
        x = (1 - gamma) * x + gamma * z
        f_hist.append(loss_fn(x))
    return np.array(f_hist), np.array(gap_hist)

def frank_wolfe_fixed(grad_fn, loss_fn, x0, bnds, beta, n_iters):
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

x0_int = np.array([1.0, 1.0])
x0_bnd = np.array([3.0, 3.0])
N = 300

# Interior case
fh_ol,   gh_ol   = frank_wolfe_openloop(grad_int, f_int, x0_int, bounds, N)
fh_085,  gh_085  = frank_wolfe_fixed(grad_int, f_int, x0_int, bounds, 0.985, N)
fh_090,  gh_090  = frank_wolfe_fixed(grad_int, f_int, x0_int, bounds, 0.90,  N)

# Boundary case
fh_ol_b,  gh_ol_b  = frank_wolfe_openloop(grad_bnd, f_bnd, x0_bnd, bounds, N)
fh_093_b, gh_093_b = frank_wolfe_fixed(grad_bnd, f_bnd, x0_bnd, bounds, 0.93, N)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Top-left: Interior convergence
ax = axes[0, 0]
k_arr = np.arange(N+1)
ax.semilogy(k_arr, fh_ol,  color=GREEN,  lw=2.0, label='Open-loop $\\gamma_k = 2/(k+2)$')
ax.semilogy(k_arr, fh_085, color=ORANGE, lw=1.8, label='Fixed $\\beta = 0.985$')
ax.semilogy(k_arr, fh_090, color=BLUE,   lw=1.8, label='Fixed $\\beta = 0.90$')
# Theoretical O(1/k) bound for open-loop
L_int = 2.0  # L-smooth constant for f(x)=(x1-1)^2+(x2-5)^2
D_int = np.sqrt((5-0.5)**2 + (10-(-5))**2)  # box diameter
k_ref = np.arange(2, N+1)
ax.semilogy(k_ref, 2*L_int*D_int**2 / (k_ref+2), color=GREY, lw=1.2, ls='--',
            label=f'$2LD^2/(k+2)$ bound ($L={L_int}$, $D\\approx{D_int:.0f}$)')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('$f(x_k)$ (log scale)', fontsize=11)
ax.set_title('Interior optimum: convergence comparison\n$f(x) = (x_1-1)^2 + (x_2-5)^2$', fontsize=10)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, N)
ax.set_ylim(1e-4, 20)

# Top-right: Interior duality gap
ax = axes[0, 1]
k_gap = np.arange(1, N+1)
ax.semilogy(k_gap, gh_ol,  color=GREEN,  lw=2.0, label='Open-loop gap $G_k$')
ax.semilogy(k_gap, gh_085, color=ORANGE, lw=1.8, label='Fixed $\\beta=0.985$ gap')
ax.semilogy(k_gap, gh_090, color=BLUE,   lw=1.8, label='Fixed $\\beta=0.90$ gap')
ax.semilogy(k_ref, 2*L_int*D_int**2 / k_ref, color=GREY, lw=1.2, ls='--', label='$O(1/k)$ ref.')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('Duality gap $G_k$ (log scale)', fontsize=11)
ax.set_title('Interior optimum: gap comparison\n(open-loop vs fixed $\\beta$)', fontsize=10)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.35)
ax.set_xlim(1, N)

# Bottom-left: Boundary convergence
ax = axes[1, 0]
ax.semilogy(k_arr, fh_ol_b - 0.25,  color=GREEN,  lw=2.0, label='Open-loop $\\gamma_k$')
ax.semilogy(k_arr, fh_093_b - 0.25, color=ORANGE, lw=1.8, label='Fixed $\\beta=0.93$')
L_bnd = 2.0; D_bnd = D_int
ax.semilogy(k_ref, 2*L_bnd*D_bnd**2 / (k_ref+2), color=GREY, lw=1.2, ls='--', label='$2LD^2/(k+2)$ bound')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)', fontsize=11)
ax.set_title('Boundary optimum: convergence comparison\n$f(x) = x_1^2 + x_2^2$, $f^\\star = 0.25$', fontsize=10)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, N)

# Bottom-right: Step size schedule comparison
ax = axes[1, 1]
k_sched = np.arange(0, N)
gamma_ol = 2.0 / (k_sched + 2)
ax.plot(k_sched, 1 - gamma_ol,      color=GREEN,  lw=2.0, label='Open-loop $\\beta_k = k/(k+2)$')
ax.axhline(0.985, color=ORANGE, lw=1.8, ls='--', label='Fixed $\\beta = 0.985$')
ax.axhline(0.90,  color=BLUE,   lw=1.8, ls=':',  label='Fixed $\\beta = 0.90$')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('Equivalent $\\beta_k$ (retain fraction)', fontsize=11)
ax.set_title('Step size (retain fraction) comparison\n$\\beta_k = 1 - \\gamma_k$', fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, 150)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/q6_fw_openloop.pdf', bbox_inches='tight')
plt.close()
print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: GD convergence phases on Rosenbrock
# ─────────────────────────────────────────────────────────────────────────────
print("Generating q1_rosenbrock_phases.pdf ...")

def gd_rosen(x0, alpha, n_iters):
    x = x0.copy().astype(float)
    hist = [rosenbrock(x)]
    xh = [x.copy()]
    for _ in range(n_iters):
        g = np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
                       200*(x[1]-x[0]**2)])
        x = x - alpha * g
        hist.append(rosenbrock(x))
        xh.append(x.copy())
    return np.array(xh), np.array(hist)

x0_C = np.array([-1.0, 1.0])
xh, fh = gd_rosen(x0_C, 0.0012, 500)

# Identify valley entry: when x2 ≈ x1^2 (within threshold)
valley_mask = np.abs(xh[:, 1] - xh[:, 0]**2) < 0.05
valley_entry = np.argmax(valley_mask) if valley_mask.any() else 500

x1g = np.linspace(-1.3, 1.3, 250)
x2g = np.linspace(-0.2, 1.8, 250)
X1g, X2g = np.meshgrid(x1g, x2g)
Zg = (1-X1g)**2 + 100*(X2g-X1g**2)**2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
cs = ax.contourf(X1g, X2g, np.log10(Zg+1e-6), levels=25, cmap='Blues', alpha=0.6)
ax.contour(X1g, X2g, np.log10(Zg+1e-6), levels=25, colors='white', linewidths=0.4, alpha=0.5)
plt.colorbar(cs, ax=ax, label='$\\log_{10}(f+10^{-6})$', pad=0.02)
# Phase 1: rapid descent
p1 = min(valley_entry, len(xh)-1)
ax.plot(xh[:p1+1, 0], xh[:p1+1, 1], '-', color=ORANGE, lw=2.0, label=f'Phase 1: descent (0-{p1})')
# Phase 2: valley traversal
ax.plot(xh[p1:, 0], xh[p1:, 1], '-', color=GREEN, lw=1.4, label=f'Phase 2: valley ({p1}+)')
# Parabola x2=x1^2
x1_par = np.linspace(-1, 1.2, 200)
ax.plot(x1_par, x1_par**2, 'r--', lw=1.5, alpha=0.7, label='Valley: $x_2=x_1^2$')
ax.plot(1, 1, 'r*', ms=14, zorder=10, label='Optimum')
ax.plot(x0_C[0], x0_C[1], 'kx', ms=10, mew=2, label='Start $(-1,1)$')
ax.set_xlabel('$x_1$', fontsize=11)
ax.set_ylabel('$x_2$', fontsize=11)
ax.set_title('GD on Rosenbrock: two phases\n(500 iterations, $\\alpha=0.0012$)', fontsize=10)
ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-0.2, 1.8)

ax = axes[1]
iters = np.arange(len(fh))
ax.semilogy(iters[:p1+1], fh[:p1+1], color=ORANGE, lw=2.0, label='Phase 1: rapid descent')
ax.semilogy(iters[p1:], fh[p1:],    color=GREEN,  lw=2.0, label='Phase 2: slow valley traversal')
ax.axvline(p1, color='k', ls='--', lw=1.0, alpha=0.7, label=f'Valley entry $k\\approx{p1}$')
# Reference rates
k_ref2 = np.arange(1, len(fh)+1)
ref_iters = np.arange(10, min(200, len(fh)))
ax.semilogy(ref_iters, fh[10] * (1 - 1/2000)**np.arange(len(ref_iters)),
            color='grey', lw=1.0, ls=':', label='$(1-1/\\kappa)^k$ rate')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('$f(x_k)$ (log scale)', fontsize=11)
ax.set_title('GD on Rosenbrock: convergence phases\n(Phase 1 fast, Phase 2 slow)', fontsize=10)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, 500)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/q1_rosenbrock_phases.pdf', bbox_inches='tight')
plt.close()
print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Condition number analysis per algorithm
# ─────────────────────────────────────────────────────────────────────────────
print("Generating q5_alm_comparison.pdf ...")

def f_B_constrained(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])

def grad_B_constrained(x):
    return np.array([2*(x[0]-1) + np.cos(x[0]), 10*(x[1]-2)])

# Projected GD
def projected_gd_run(x0, alpha, lb, n_iters):
    x = x0.copy()
    hist = [f_B_constrained(x)]
    viol = [max(0, lb - x[0])]
    for _ in range(n_iters):
        g = grad_B_constrained(x)
        x = x - alpha * g
        x[0] = max(lb, x[0])
        hist.append(f_B_constrained(x))
        viol.append(max(0, lb - x[0]))
    return np.array(hist), np.array(viol)

# Penalty method
def penalty_gd_run(x0, alpha, lam, lb, n_iters):
    x = x0.copy()
    hist = [f_B_constrained(x)]
    viol = [max(0, lb - x[0])]
    for _ in range(n_iters):
        g = grad_B_constrained(x).copy()
        if x[0] < lb:
            g[0] -= lam
        x = x - alpha * g
        hist.append(f_B_constrained(x))
        viol.append(max(0, lb - x[0]))
    return np.array(hist), np.array(viol)

# Augmented Lagrangian (simple version)
def alm_run(x0, rho, lb, n_iters_outer=10, n_iters_inner=10):
    x = x0.copy().astype(float)
    mu = 0.0
    hist_f = [f_B_constrained(x)]
    hist_viol = [max(0, lb - x[0])]
    alpha_inner = 0.05
    for outer in range(n_iters_outer):
        for _ in range(n_iters_inner):
            viol = max(0.0, lb - x[0])
            g = grad_B_constrained(x).copy()
            if x[0] < lb:
                g[0] += mu * (-1) + rho * (x[0] - lb) * (-1) * (-1)
                g[0] -= mu + rho * viol
            x = x - alpha_inner * g
            hist_f.append(f_B_constrained(x))
            hist_viol.append(max(0, lb - x[0]))
        # Dual update
        mu = max(0, mu + rho * max(0, lb - x[0]))
    return np.array(hist_f), np.array(hist_viol)

x0_c = np.array([0.2, 4.0])
lb = 0.5
N_c = 100

fh_pgd,  vh_pgd  = projected_gd_run(x0_c, 0.08, lb, N_c)
fh_p045, vh_p045 = penalty_gd_run(x0_c, 0.05, 0.15, lb, N_c)
fh_p18,  vh_p18  = penalty_gd_run(x0_c, 0.05, 1.8,  lb, N_c)
fh_p45,  vh_p45  = penalty_gd_run(x0_c, 0.03, 4.5,  lb, N_c)
fh_alm,  vh_alm  = alm_run(x0_c, rho=5.0, lb=lb, n_iters_outer=10, n_iters_inner=10)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
iters_pgd = np.arange(len(fh_pgd))
ax.semilogy(iters_pgd, np.abs(fh_pgd - 0.7244) + 1e-6, color=BLUE, lw=2.0, label='Projected GD')
ax.semilogy(np.arange(len(fh_p045)), np.abs(fh_p045 - 0.7244) + 1e-6, color=ORANGE, lw=1.8, label='Penalty $\\lambda=0.15$')
ax.semilogy(np.arange(len(fh_p18)), np.abs(fh_p18 - 0.7244) + 1e-6, color=GREEN, lw=1.8, label='Penalty $\\lambda=1.8$')
ax.semilogy(np.arange(len(fh_p45)), np.abs(fh_p45 - 0.7244) + 1e-6, color=RED, lw=1.8, label='Penalty $\\lambda=4.5$')
ax.semilogy(np.arange(len(fh_alm)), np.abs(fh_alm - 0.7244) + 1e-6, color=PURPLE, lw=2.0, ls='--', label='ALM ($\\rho=5$)')
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('$|f(x_k) - f^\\star|$ (log)', fontsize=11)
ax.set_title('Q5: Optimality gap comparison\nProjected GD vs Penalty vs ALM', fontsize=10)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, 100)

ax = axes[1]
ax.semilogy(iters_pgd + 1, vh_pgd + 1e-9, color=BLUE, lw=2.0, label='Projected GD')
ax.semilogy(np.arange(len(vh_p045))+1, vh_p045+1e-9, color=ORANGE, lw=1.8, label='Penalty $\\lambda=0.15$')
ax.semilogy(np.arange(len(vh_p18))+1,  vh_p18+1e-9,  color=GREEN,  lw=1.8, label='Penalty $\\lambda=1.8$')
ax.semilogy(np.arange(len(vh_p45))+1,  vh_p45+1e-9,  color=RED,    lw=1.8, label='Penalty $\\lambda=4.5$')
ax.semilogy(np.arange(len(vh_alm))+1,  vh_alm+1e-9,  color=PURPLE, lw=2.0, ls='--', label='ALM ($\\rho=5$)')
ax.axhline(1e-9, color='k', lw=0.7, ls=':', alpha=0.5)
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Constraint violation $\\max(0, 0.5-x_1)$', fontsize=11)
ax.set_title('Q5: Constraint violation\n(log scale; Proj. GD achieves zero immediately)', fontsize=10)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.35)
ax.set_xlim(1, 100)
ax.set_ylim(1e-10, 1)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/q5_alm_comparison.pdf', bbox_inches='tight')
plt.close()
print("  done.")

print("\nAll Pass 8 figures generated successfully.")
