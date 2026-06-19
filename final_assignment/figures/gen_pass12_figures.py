#!/usr/bin/env python3
"""Pass 12 figures: all-methods contour on Rosenbrock, FW vertex jumping visualization."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

np.random.seed(42)

def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

# ─── Figure 1: All methods trajectory comparison on Rosenbrock ────────────────
def run_gd(x0, alpha, n):
    x = x0.copy().astype(float)
    traj = [x.copy()]
    for _ in range(n):
        x -= alpha * grad_rosenbrock(x)
        traj.append(x.copy())
    return np.array(traj)

def run_heavy_ball(x0, alpha, beta, n):
    x = x0.copy().astype(float); z = np.zeros(2)
    traj = [x.copy()]
    for _ in range(n):
        g = grad_rosenbrock(x)
        z = beta * z + alpha * g
        x = x - z
        traj.append(x.copy())
    return np.array(traj)

def run_nesterov(x0, alpha, beta_max, n):
    x = x0.copy().astype(float); z = np.zeros(2)
    traj = [x.copy()]
    for k in range(1, n+1):
        beta_k = min((k-1)/(k+2), beta_max)
        la = x + beta_k * z
        g = grad_rosenbrock(la)
        z = beta_k * z - alpha * g
        x = x + z
        traj.append(x.copy())
    return np.array(traj)

def run_polyak(x0, f_star, eps, n):
    x = x0.copy().astype(float)
    traj = [x.copy()]
    for _ in range(n):
        g = grad_rosenbrock(x)
        a = (rosenbrock(x) - f_star) / (np.dot(g, g) + eps)
        x = x - a * g
        traj.append(x.copy())
    return np.array(traj)

def run_adagrad(x0, alpha0, eps, n):
    x = x0.copy().astype(float); G = np.zeros(2)
    traj = [x.copy()]
    for _ in range(n):
        g = grad_rosenbrock(x)
        G += g**2
        x = x - alpha0 / (np.sqrt(G) + eps) * g
        traj.append(x.copy())
    return np.array(traj)

def run_adam(x0, alpha, b1, b2, eps, n):
    x = x0.copy().astype(float); m_v = np.zeros(2); v_v = np.zeros(2)
    traj = [x.copy()]
    for t in range(1, n+1):
        g = grad_rosenbrock(x)
        m_v = b1 * m_v + (1-b1) * g
        v_v = b2 * v_v + (1-b2) * g**2
        mh = m_v/(1-b1**t); vh = v_v/(1-b2**t)
        x = x - alpha * mh/(np.sqrt(vh)+eps)
        traj.append(x.copy())
    return np.array(traj)

def run_newton(x0, alpha, damping, n):
    x = x0.copy().astype(float)
    traj = [x.copy()]
    for _ in range(n):
        g = grad_rosenbrock(x)
        x1, x2 = x
        H = np.array([
            [-2 + 400*(3*x1**2 - x2), -400*x1],
            [-400*x1, 200]
        ])
        H_reg = H + damping * np.eye(2)
        try:
            p = np.linalg.solve(H_reg, g)
        except:
            p = g
        x = x - alpha * p
        traj.append(x.copy())
    return np.array(traj)

x0 = np.array([-1.0, 1.0])
n_iters = 150

traj_gd     = run_gd(x0, 0.0012, n_iters)
traj_hb     = run_heavy_ball(x0, 0.0008, 0.86, n_iters)
traj_nes    = run_nesterov(x0, 0.0007, 0.90, n_iters)
traj_poly   = run_polyak(x0, 0.0, 1e-3, n_iters)
traj_ada    = run_adagrad(x0, 0.45, 1e-5, n_iters)
traj_adam   = run_adam(x0, 0.006, 0.80, 0.999, 1e-8, n_iters)
traj_newton = run_newton(x0, 0.22, 1e-8, 20)

# Contour
xg = np.linspace(-1.6, 1.6, 400)
yg = np.linspace(-0.3, 1.9, 400)
Xg, Yg = np.meshgrid(xg, yg)
Zg = (1 - Xg)**2 + 100*(Yg - Xg**2)**2

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

method_data = [
    (traj_gd,     'GD ($\\alpha=0.0012$)',           '#2196F3', '-',  1.5),
    (traj_hb,     'Heavy Ball ($\\alpha=0.0008$, $\\beta=0.86$)', '#FF5722', '--', 1.8),
    (traj_nes,    'Nesterov ($\\alpha=0.0007$, $\\beta_{max}=0.90$)', '#4CAF50', '-',  2.0),
    (traj_poly,   'Polyak ($f^\\star=0$)',           '#9C27B0', '-',  1.5),
    (traj_ada,    'Adagrad ($\\alpha_0=0.45$)',      '#795548', '--', 1.5),
    (traj_adam,   'Adam ($\\alpha=0.006$)',           '#FF9800', '-',  1.5),
    (traj_newton, 'Newton ($\\alpha=0.22$, 20 iters)', '#000000', 'o-', 2.0),
]

for ax_idx, (xlim, ylim, title_sfx) in enumerate([
        ([-1.6, 1.6], [-0.3, 1.9], '(full view)'),
        ([-0.1, 1.3], [0.0, 1.6], '(zoomed near optimum)')]):
    ax = axes[ax_idx]
    ax.contourf(Xg, Yg, np.log10(np.clip(Zg, 1e-6, None)), levels=25, cmap='Blues', alpha=0.3)
    ax.contour(Xg, Yg, np.log10(np.clip(Zg, 1e-6, None)), levels=25, colors='gray', alpha=0.2, linewidths=0.4)

    for traj, label, col, ls, lw in method_data:
        tx, ty = traj[:, 0], traj[:, 1]
        if ax_idx == 0:
            ax.plot(tx, ty, ls, color=col, linewidth=lw, label=label, alpha=0.85)
            ax.plot(tx[0], ty[0], 'o', color=col, markersize=6, zorder=5)
            ax.plot(tx[-1], ty[-1], 's', color=col, markersize=6, zorder=5)
        else:
            # Zoom — clip trajectories to view
            mask = (tx >= xlim[0]) & (tx <= xlim[1]) & (ty >= ylim[0]) & (ty <= ylim[1])
            if mask.any():
                # Plot only visible portion
                ax.plot(tx[mask], ty[mask], ls, color=col, linewidth=lw, alpha=0.85)

    ax.plot(1, 1, '*', color='gold', markersize=16, zorder=6, markeredgecolor='k', markeredgewidth=0.8)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel('$x_1$', fontsize=12); ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(f'All Methods: Rosenbrock Trajectories {title_sfx}', fontsize=11)
    ax.grid(True, alpha=0.2)

axes[0].legend(loc='upper right', fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/all_trajectories_rosenbrock.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: all_trajectories_rosenbrock.pdf")


# ─── Figure 2: Frank-Wolfe vertex jumping visualization ──────────────────────
def fw_run_with_lmo(x0, bounds, beta, n_iters):
    """Run FW and return iterates + LMO vertices."""
    x = x0.copy().astype(float)
    x_hist, z_hist, gap_hist = [x.copy()], [], []

    def f(v):
        return (v[0]-1)**2 + (v[1]-5)**2  # interior optimum at (1, 5)

    def grad_f(v):
        return np.array([2*(v[0]-1), 2*(v[1]-5)])

    for _ in range(n_iters):
        g = grad_f(x)
        z = np.array([bounds[i][0] if g[i] > 0 else bounds[i][1] for i in range(2)])
        gap = float(g @ (x - z))
        gap_hist.append(gap)
        z_hist.append(z.copy())
        x = beta * x + (1 - beta) * z
        x_hist.append(x.copy())
    return np.array(x_hist), np.array(z_hist), np.array(gap_hist)

bounds = [(0.5, 5.0), (-5.0, 10.0)]
x0_fw = np.array([1.0, 1.0])
n_fw = 80

traj_fw90, z90, gap90 = fw_run_with_lmo(x0_fw, bounds, 0.90, n_fw)
traj_fw985, z985, gap985 = fw_run_with_lmo(x0_fw, bounds, 0.985, n_fw)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: trajectory + LMO vertices (beta=0.90)
ax = axes[0][0]
# Draw feasible box
box_x = [0.5, 5.0, 5.0, 0.5, 0.5]
box_y = [-5.0, -5.0, 10.0, 10.0, -5.0]
ax.plot(box_x, box_y, 'k-', linewidth=2, label='Feasible box $\\mathcal{X}$')
ax.fill(box_x, box_y, alpha=0.05, color='blue')

# Draw trajectory
ax.plot(traj_fw90[:, 0], traj_fw90[:, 1], 'b-', linewidth=1.5, alpha=0.7, label='Iterate $x_k$')
ax.plot(traj_fw90[:, 0], traj_fw90[:, 1], 'b.', markersize=4, alpha=0.5)

# Draw LMO vertex arrows for first 20 iters
for i in range(0, min(15, n_fw)):
    xk = traj_fw90[i]
    zk = z90[i]
    ax.annotate('', xy=zk, xytext=xk,
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8, alpha=0.4))

ax.scatter(z90[:20, 0], z90[:20, 1], c='red', s=30, zorder=5, alpha=0.6, label='LMO vertex $z_k$ (first 20)')
ax.plot(1, 5, '*', color='gold', markersize=14, zorder=6, markeredgecolor='k', label='Optimum $(1, 5)$')
ax.set_xlim(-0.5, 6); ax.set_ylim(-6, 11)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title(f'FW Trajectory + LMO Vertices ($\\beta=0.90$)\n(first 15 LMO arrows shown)', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Top-right: same for beta=0.985
ax2 = axes[0][1]
ax2.plot(box_x, box_y, 'k-', linewidth=2)
ax2.fill(box_x, box_y, alpha=0.05, color='blue')
ax2.plot(traj_fw985[:, 0], traj_fw985[:, 1], 'b-', linewidth=1.5, alpha=0.7, label='Iterate $x_k$')
ax2.plot(traj_fw985[:, 0], traj_fw985[:, 1], 'b.', markersize=4, alpha=0.5)
for i in range(0, min(15, n_fw)):
    xk = traj_fw985[i]
    zk = z985[i]
    ax2.annotate('', xy=zk, xytext=xk,
                 arrowprops=dict(arrowstyle='->', color='red', lw=0.8, alpha=0.4))
ax2.scatter(z985[:20, 0], z985[:20, 1], c='red', s=30, zorder=5, alpha=0.6)
ax2.plot(1, 5, '*', color='gold', markersize=14, zorder=6, markeredgecolor='k')
ax2.set_xlim(-0.5, 6); ax2.set_ylim(-6, 11)
ax2.set_xlabel('$x_1$'); ax2.set_ylabel('$x_2$')
ax2.set_title(f'FW Trajectory + LMO Vertices ($\\beta=0.985$)\n(smaller step, less vertex jumping)', fontsize=10)
ax2.grid(True, alpha=0.2)

# Bottom-left: LMO vertex coordinates over iterations
ax3 = axes[1][0]
its = np.arange(n_fw)
ax3.plot(its, z90[:, 0], 'r-', linewidth=2, label='$z_k^{(1)}$ ($x_1$ coord, $\\beta=0.90$)')
ax3.plot(its, z90[:, 1], 'b-', linewidth=2, label='$z_k^{(2)}$ ($x_2$ coord, $\\beta=0.90$)')
ax3.plot(its, z985[:, 0], 'r--', linewidth=1.5, alpha=0.7, label='$z_k^{(1)}$ ($\\beta=0.985$)')
ax3.plot(its, z985[:, 1], 'b--', linewidth=1.5, alpha=0.7, label='$z_k^{(2)}$ ($\\beta=0.985$)')
ax3.axhline(1, color='gray', linestyle=':', alpha=0.5)  # optimum x1
ax3.axhline(5, color='gray', linestyle=':', alpha=0.5)  # optimum x2
ax3.set_xlabel('Iteration', fontsize=11)
ax3.set_ylabel('LMO coordinate $z_k^{(i)}$', fontsize=11)
ax3.set_title('LMO Vertex Coordinates vs Iteration\n(showing vertex jumping pattern)', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Bottom-right: FW gap comparison
ax4 = axes[1][1]
k_arr = np.arange(1, n_fw+1)
ax4.semilogy(k_arr, gap90, '-', color='#2196F3', linewidth=2, label='FW gap $G_k$ ($\\beta=0.90$)')
ax4.semilogy(k_arr, gap985, '-', color='#FF5722', linewidth=2, label='FW gap $G_k$ ($\\beta=0.985$)')
# O(1/k) reference (use standard step gamma_k = 2/(k+2))
L_fw = 2.0; D2 = (4.5)**2 + (15)**2  # diameter of X
ow_bound = 4*L_fw*D2 / (k_arr + 2)
ax4.semilogy(k_arr, ow_bound, 'k--', linewidth=1.5, alpha=0.7, label='$4LD^2/(k+2)$ bound')
ax4.set_xlabel('Iteration $k$', fontsize=11)
ax4.set_ylabel('FW gap $G_k = \\nabla f(x_k)^T(x_k - z_k)$', fontsize=11)
ax4.set_title('Frank--Wolfe Gap vs Theoretical $O(1/k)$ Bound', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.suptitle('Frank--Wolfe Algorithm: Vertex Jumping and Gap Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/fw_vertex_jumping.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: fw_vertex_jumping.pdf")

print("All pass-12 figures generated.")
