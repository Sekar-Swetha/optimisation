"""Pass 12 figures: NRS vs exact GD rate comparison, penalty convergence proof,
Frank-Wolfe LP vertex selection, all-methods benchmark comparison."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(42)

# ============================================================
# Figure 1: NRS vs exact GD rate (varying dimension d)
# ============================================================
# Show how NRS convergence degrades with dimension d
# For d=2: NRS mean rate ≈ (1/d)*||grad|| = 0.5 GD rate
# For d=5, 10, 20: progressively slower

def bench_b_loss(x): return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def bench_b_grad(x): return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

# Generic convex quadratic with dimension d for NRS
def make_quadratic(d, kappa=5.0):
    L = kappa; mu = 1.0
    # Hessian: diag(mu, ..., mu, L) so condition number = L/mu = kappa
    H_diag = np.concatenate([np.ones(d-1)*mu, [L]])
    def loss(x): return 0.5 * np.dot(x * H_diag, x)
    def grad(x): return H_diag * x
    return loss, grad

def gd(gfn, lfn, x0, alpha, n):
    x = x0.copy().astype(float); h = [lfn(x)]
    for _ in range(n):
        x -= alpha * gfn(x); h.append(lfn(x))
    return np.array(h)

def nrs(gfn, lfn, x0, alpha, delta, n, seed=42):
    np.random.seed(seed)
    x = x0.copy().astype(float); d = len(x); h = [lfn(x)]
    for _ in range(n):
        u = np.random.randn(d); u /= np.linalg.norm(u)
        g_est = (lfn(x + delta*u) - lfn(x)) / delta * u
        x -= alpha * g_est; h.append(lfn(x))
    return np.array(h)

N = 300
dims = [2, 5, 10, 20]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(dims)))
for d, color in zip(dims, colors):
    loss, grad = make_quadratic(d, kappa=5.0)
    x0 = np.ones(d)
    fstar = 0.0

    # GD optimal alpha = 2/(mu+L) = 2/6 = 0.333
    alpha_gd = 2.0 / (1.0 + 5.0)
    h_gd = gd(grad, loss, x0, alpha_gd, N)

    # NRS: alpha = alpha_gd/d (effective scale), delta = 0.01
    h_nrs_list = []
    for seed in range(10):
        h = nrs(grad, loss, x0, alpha_gd/(d), 0.01, N, seed=seed)
        h_nrs_list.append(h)
    h_nrs_mean = np.mean(h_nrs_list, axis=0)

    ax.semilogy(np.arange(N+1), np.maximum(h_gd - fstar, 1e-10), '-',
                color=color, lw=2, label=f'GD $d={d}$')
    ax.semilogy(np.arange(N+1), np.maximum(h_nrs_mean - fstar, 1e-10), '--',
                color=color, lw=1.5, label=f'NRS $d={d}$ (mean 10 seeds)')

ax.set_xlabel('Iteration $k$'); ax.set_ylabel('$f(x_k) - f^\\star$')
ax.set_title('GD vs Nesterov Random Search\nConvergence for Various Dimensions $d$', fontsize=10)
ax.legend(fontsize=7.5, ncol=2); ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(0, N)

# Right: NRS effective convergence rate ratio vs d
ax2 = axes[1]
# For NRS, effective rate ≈ (1/d) * GD rate
# In oracle complexity: NRS needs O(d/epsilon) gradient-equivalent steps
# GD needs O(1/epsilon) (ignoring log for simplicity)
d_range = np.arange(1, 101)
# Cost per gradient equivalent: NRS uses 2 FE vs d+1 FE for FD gradient
cost_nrs_per_iter = 2  # function evaluations
cost_fd_per_iter = d_range + 1
cost_exact_per_iter = 1  # if gradient oracle available

# Iterations to epsilon accuracy (relative to GD)
iter_ratio_nrs = d_range  # NRS needs d times more iterations than GD
iter_ratio_fd = 1  # FD has same per-iteration progress as exact GD
fe_ratio_nrs = iter_ratio_nrs * cost_nrs_per_iter  # function evaluations
fe_ratio_fd  = iter_ratio_fd * cost_fd_per_iter    # function evaluations

ax2.loglog(d_range, fe_ratio_nrs, 'r-', lw=2,
           label='NRS: $2d$ FE / GD iteration\n(needs $d\\times$ more iters)')
ax2.loglog(d_range, fe_ratio_fd, 'b--', lw=2,
           label='FD GD: $(d+1)$ FE / iteration\n(same per-iter progress as GD)')
ax2.axhline(1, color='k', linestyle=':', lw=1, label='Exact GD: 1 FE / iter (oracle)')
ax2.set_xlabel('Dimension $d$'); ax2.set_ylabel('Function evaluations\n(relative to exact GD iterations)')
ax2.set_title('Oracle Complexity: FE to Match Exact GD\n($\\kappa = 5$, ignore log factors)', fontsize=10)
ax2.legend(fontsize=8.5); ax2.grid(True, which='both', alpha=0.3)
ax2.text(0.95, 0.05, 'For $d=2$:\nNRS: 4 FE, FD: 3 FE per GD iter',
         transform=ax2.transAxes, ha='right', va='bottom', fontsize=8.5,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Derivative-Free Oracle Complexity: NRS and FD GD vs Exact GD', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q4_oracle_complexity.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q4_oracle_complexity.pdf")

# ============================================================
# Figure 2: Frank-Wolfe LP vertex selection dynamics
# ============================================================
# Show how z_k changes as x_k moves

def fw_run_detailed(grad_fn, loss_fn, x0, bounds, beta, n):
    x = x0.copy().astype(float)
    xs, zs, fvals = [x.copy()], [], [loss_fn(x)]
    for _ in range(n):
        g = grad_fn(x)
        z = np.array([bounds[i][0] if g[i] > 0 else bounds[i][1] for i in range(len(g))])
        zs.append(z.copy())
        x = beta * x + (1-beta) * z
        xs.append(x.copy())
        fvals.append(loss_fn(x))
    return np.array(xs), np.array(zs), np.array(fvals)

# Interior optimum: f(x) = (x1-1)^2 + (x2-5)^2
def fw_interior_grad(x): return np.array([2*(x[0]-1), 2*(x[1]-5)])
def fw_interior_loss(x): return (x[0]-1)**2 + (x[1]-5)**2

bounds = [(0.5, 5.0), (-5.0, 10.0)]
vertices = np.array([[0.5, -5], [5, -5], [0.5, 10], [5, 10]])  # 4 corners

x0_fw = np.array([1.0, 1.0])
N_fw = 100

xs90,  zs90,  fv90  = fw_run_detailed(fw_interior_grad, fw_interior_loss, x0_fw, bounds, 0.90,  N_fw)
xs985, zs985, fv985 = fw_run_detailed(fw_interior_grad, fw_interior_loss, x0_fw, bounds, 0.985, N_fw)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: trajectory
ax = axes[0]
XX = np.linspace(0.5, 5, 200); YY = np.linspace(-5, 10, 200)
Xg, Yg = np.meshgrid(XX, YY)
Fg = (Xg-1)**2 + (Yg-5)**2
ax.contourf(Xg, Yg, Fg, levels=15, cmap='YlOrRd', alpha=0.35)
ax.contour(Xg, Yg, Fg, levels=15, colors='gray', linewidths=0.4, alpha=0.5)
ax.plot([0.5, 5, 5, 0.5, 0.5], [-5, -5, 10, 10, -5], 'k-', lw=2, label='Feasible set $\\mathcal{X}$')
ax.plot(xs90[:,0], xs90[:,1], 'b-o', ms=3, lw=1.5, label='$x_k$, $\\beta=0.90$')
ax.plot(xs985[:,0], xs985[:,1], 'r-s', ms=3, lw=1.5, label='$x_k$, $\\beta=0.985$')
ax.plot(1, 5, 'g*', ms=12, zorder=5, label='Optimum $(1,5)$')
ax.plot(x0_fw[0], x0_fw[1], 'ks', ms=8, zorder=5, label='Start $(1,1)$')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('FW Trajectories (Interior Optimum)', fontsize=10)
ax.legend(fontsize=7.5); ax.set_aspect('equal')

# Middle: z_k component evolution over time
ax2 = axes[1]
ks_fw = np.arange(N_fw)
ax2.plot(ks_fw, zs90[:,0], 'b-', lw=1.5, label='$z_k[1]$ ($x_1$ LMO), $\\beta=0.90$')
ax2.plot(ks_fw, zs90[:,1], 'b--', lw=1.5, label='$z_k[2]$ ($x_2$ LMO), $\\beta=0.90$')
ax2.plot(ks_fw, zs985[:,0], 'r-', lw=1.5, alpha=0.7, label='$z_k[1]$, $\\beta=0.985$')
ax2.plot(ks_fw, zs985[:,1], 'r--', lw=1.5, alpha=0.7, label='$z_k[2]$, $\\beta=0.985$')
ax2.axhline(0.5, color='gray', linestyle=':', lw=1, label='$x_1$ lower bound 0.5')
ax2.axhline(5, color='gray', linestyle=':', lw=1, label='$x_1$ upper 5')
ax2.axhline(-5, color='gray', linestyle='--', lw=1, label='$x_2$ lower -5')
ax2.axhline(10, color='gray', linestyle='--', lw=1, label='$x_2$ upper 10')
ax2.set_xlabel('Iteration $k$')
ax2.set_ylabel('LP vertex component $z_{k,i}$')
ax2.set_title('Frank--Wolfe LP Vertex $z_k$ vs Iteration\n(Interior Optimum)', fontsize=10)
ax2.legend(fontsize=7.5, ncol=2); ax2.grid(True, alpha=0.3)

# Right: distribution of which vertex was selected
vertex_names = ['$(0.5,-5)$', '$(5,-5)$', '$(0.5,10)$', '$(5,10)$']

def which_vertex(zs, vertices):
    counts = np.zeros(4)
    for z in zs:
        dists = [np.sum((z - v)**2) for v in vertices]
        counts[np.argmin(dists)] += 1
    return counts

c90  = which_vertex(zs90, vertices)
c985 = which_vertex(zs985, vertices)

ax3 = axes[2]
x_pos = np.arange(4)
w = 0.35
ax3.bar(x_pos - w/2, c90,  width=w, color='blue', alpha=0.7, label='$\\beta=0.90$')
ax3.bar(x_pos + w/2, c985, width=w, color='red',  alpha=0.7, label='$\\beta=0.985$')
ax3.set_xticks(x_pos); ax3.set_xticklabels(vertex_names, fontsize=8.5)
ax3.set_xlabel('LP Vertex'); ax3.set_ylabel('Times selected as $z_k$')
ax3.set_title(f'LP Vertex Selection Frequency\n({N_fw} FW iterations, interior opt.)', fontsize=10)
ax3.legend(fontsize=9); ax3.grid(True, axis='y', alpha=0.3)
ax3.text(0.5, 0.95,
         'Near $(1,5)$: grad direction alternates\nbetween $(+,+)$, $(+,-)$, etc.\n'
         'causing vertex oscillation',
         transform=ax3.transAxes, ha='center', va='top', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Frank--Wolfe LP Vertex Selection: Interior Optimum ($\\beta=0.90$ vs $\\beta=0.985$)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q6_fw_vertex_dynamics.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q6_fw_vertex_dynamics.pdf")

# ============================================================
# Figure 3: Grid search scalability (curse of dimensionality)
# ============================================================
dims = np.arange(1, 12)
resolutions = [10, 55, 100]  # number of grid points per dimension

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for N_g, ls, marker in [(10, '-', 'o'), (55, '--', 's'), (100, ':', '^')]:
    total_evals = N_g ** dims
    # Stop when > 10^15
    valid = total_evals <= 1e15
    ax.semilogy(dims[valid], total_evals[valid], ls+marker, ms=5, lw=2,
                label=f'$N={N_g}$ pts/dim')
ax.axhline(1e6, color='g', linestyle=':', lw=1, label='$10^6$ (feasible)')
ax.axhline(1e9, color='orange', linestyle=':', lw=1, label='$10^9$ (slow)')
ax.axhline(1e12, color='r', linestyle=':', lw=1, label='$10^{12}$ (intractable)')
ax.set_xlabel('Dimension $d$')
ax.set_ylabel('Total function evaluations $N^d$')
ax.set_title('Grid Search: Curse of Dimensionality\nTotal evaluations $N^d$ vs dimension', fontsize=10)
ax.legend(fontsize=8.5); ax.grid(True, which='both', alpha=0.3)
ax.set_xticks(dims[::2])

# Right: grid resolution vs accuracy (1D analogy extended to d dims)
# Resolution error ~ sqrt(d) * (range/N): worst-case distance to nearest grid point
ranges = 4  # total range per dimension (e.g., [-2, 2])
ax2 = axes[1]
N_range = np.arange(2, 100)
for d, color in zip([1, 2, 5, 10], ['blue', 'green', 'orange', 'red']):
    grid_spacing = ranges / N_range
    max_error = np.sqrt(d) * grid_spacing  # worst-case L2 distance to nearest grid point
    ax2.loglog(N_range, max_error, color=color, lw=2, label=f'$d={d}$')

ax2.set_xlabel('Grid points per dimension $N$')
ax2.set_ylabel('Max error $\\sqrt{d} \\cdot (\\mathrm{range}/N)$')
ax2.set_title('Grid Search Accuracy vs Resolution\n(worst-case distance to nearest grid point)', fontsize=10)
ax2.legend(fontsize=9); ax2.grid(True, which='both', alpha=0.3)
ax2.text(0.95, 0.95, 'For $d=2$, $N=55$:\nmax error $= \\sqrt{2} \\cdot 4/55 \\approx 0.10$\n'
         '(matches assignment grid)',
         transform=ax2.transAxes, ha='right', va='top', fontsize=8.5,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Grid Search: Scalability and Accuracy Analysis', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q4_grid_search_analysis.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q4_grid_search_analysis.pdf")

print("All Pass 12 figures generated successfully.")
