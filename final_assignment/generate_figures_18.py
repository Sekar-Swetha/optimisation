"""
Generate additional figures for the 18th improvement cycle.
Saves to final_assignment/figures/ - never overwrites existing files.
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
# Figure 1: Frank-Wolfe duality gap evolution
# Interior optimum: f(x) = (x1-1)^2 + (x2-5)^2, x0=(1,1)
# Boundary optimum: f(x) = x1^2 + x2^2, x0=(3,3)
# feasible set: x1 in [0.5,5], x2 in [-5,10]
# ============================================================
out1 = 'figures/fw_duality_gap.pdf'
if not os.path.exists(out1):

    def fw_interior(n_iters=180, beta=0.985):
        """Interior optimum: f(x)=(x1-1)^2+(x2-5)^2, x* = (1,5)"""
        x1_lo, x1_hi = 0.5, 5.0
        x2_lo, x2_hi = -5.0, 10.0

        def f(x):
            return (x[0]-1)**2 + (x[1]-5)**2

        def grad_f(x):
            return np.array([2*(x[0]-1), 2*(x[1]-5)])

        def lmo(g):
            # Minimise g^T z over box
            z1 = x1_lo if g[0] > 0 else x1_hi
            z2 = x2_lo if g[1] > 0 else x2_hi
            return np.array([z1, z2])

        x = np.array([1.0, 1.0])
        fvals = [f(x)]
        gaps = []

        for k in range(n_iters):
            g = grad_f(x)
            z = lmo(g)
            gap = g @ (x - z)  # duality gap
            gaps.append(gap)
            x = beta * x + (1 - beta) * z
            fvals.append(f(x))

        return np.array(fvals), np.array(gaps)

    def fw_boundary(n_iters=140, beta=0.93):
        """Boundary optimum: f(x)=x1^2+x2^2, x* constrained = (0.5,0)"""
        x1_lo, x1_hi = 0.5, 5.0
        x2_lo, x2_hi = -5.0, 10.0

        def f(x):
            return x[0]**2 + x[1]**2

        def grad_f(x):
            return np.array([2*x[0], 2*x[1]])

        def lmo(g):
            z1 = x1_lo if g[0] > 0 else x1_hi
            z2 = x2_lo if g[1] > 0 else x2_hi
            return np.array([z1, z2])

        x = np.array([3.0, 3.0])
        fvals = [f(x)]
        gaps = []

        for k in range(n_iters):
            g = grad_f(x)
            z = lmo(g)
            gap = g @ (x - z)
            gaps.append(gap)
            x = beta * x + (1 - beta) * z
            fvals.append(f(x))

        return np.array(fvals), np.array(gaps)

    fvals_int, gaps_int = fw_interior(180, 0.985)
    fvals_bnd, gaps_bnd = fw_boundary(140, 0.93)

    # Theoretical FW bound for interior: 2 * L * D^2 / (k+2)
    # L = 2 for f=(x1-1)^2+(x2-5)^2; D = sqrt((5-0.5)^2+(10-(-5))^2) = sqrt(245.25) ~ 15.66
    L_int = 2.0
    D2_int = (5.0 - 0.5)**2 + (10.0 - (-5.0))**2  # = 20.25 + 225 = 245.25
    k_arr_int = np.arange(180)
    fw_bound_int = 2 * L_int * D2_int / (k_arr_int + 2)

    # For boundary: L = 2 for f=x1^2+x2^2
    L_bnd = 2.0
    D2_bnd = D2_int
    k_arr_bnd = np.arange(140)
    fw_bound_bnd = 2 * L_bnd * D2_bnd / (k_arr_bnd + 2)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Interior: objective
    ax = axes[0, 0]
    ax.semilogy(np.arange(181), np.maximum(fvals_int, 1e-12), 'b-', lw=2, label='$f(x_k)$ interior ($\\beta=0.985$)')
    ax.axhline(0.0, color='k', ls='--', lw=1, alpha=0.5, label='$f^\\star = 0$')
    ax.set_xlabel('Iteration $k$')
    ax.set_ylabel('$f(x_k)$ (log scale)')
    ax.set_title('Interior Optimum: Objective')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Interior: duality gap
    ax2 = axes[0, 1]
    ax2.semilogy(k_arr_int, np.maximum(gaps_int, 1e-12), 'b-', lw=2, label='FW gap $g_k$ (interior)')
    ax2.semilogy(k_arr_int, fw_bound_int, 'r--', lw=1.5, label=f'Bound $2LD^2/(k+2)$ ($L={L_int}, D^2\\approx{D2_int:.0f}$)')
    ax2.set_xlabel('Iteration $k$')
    ax2.set_ylabel('Duality gap $g_k$ (log scale)')
    ax2.set_title('Interior Optimum: Frank--Wolfe Gap')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Boundary: objective
    ax3 = axes[1, 0]
    ax3.semilogy(np.arange(141), np.maximum(fvals_bnd - 0.25, 1e-12), 'g-', lw=2, label='$f(x_k) - f^\\star$ boundary ($\\beta=0.93$)')
    ax3.set_xlabel('Iteration $k$')
    ax3.set_ylabel('$f(x_k) - f^\\star$ (log scale)')
    ax3.set_title('Boundary Optimum: Suboptimality')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Boundary: duality gap
    ax4 = axes[1, 1]
    ax4.semilogy(k_arr_bnd, np.maximum(gaps_bnd, 1e-12), 'g-', lw=2, label='FW gap $g_k$ (boundary)')
    ax4.semilogy(k_arr_bnd, fw_bound_bnd, 'r--', lw=1.5, label=f'Bound $2LD^2/(k+2)$')
    ax4.set_xlabel('Iteration $k$')
    ax4.set_ylabel('Duality gap $g_k$ (log scale)')
    ax4.set_title('Boundary Optimum: Frank--Wolfe Gap')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Frank--Wolfe Duality Gap vs Theoretical Bound\n'
                 'Interior optimum: $f=(x_1-1)^2+(x_2-5)^2$; '
                 'Boundary optimum: $f=x_1^2+x_2^2$', fontsize=11)
    fig.tight_layout()
    fig.savefig(out1, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out1}')
else:
    print(f'Skipping {out1} (already exists)')


# ============================================================
# Figure 2: Nesterov theoretical vs empirical convergence on Rosenbrock
# Shows the gap between the worst-case O(1/k^2) bound and empirical results
# ============================================================
out2 = 'figures/nesterov_theory_vs_empirical.pdf'
if not os.path.exists(out2):

    def rosenbrock(x):
        return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

    def grad_rosen(x):
        return np.array([
            -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
            200*(x[1]-x[0]**2)
        ])

    x0 = np.array([-1.0, 1.0])
    x_star = np.array([1.0, 1.0])
    R = np.linalg.norm(x0 - x_star)  # = 2.0

    # Run Nesterov for 500 iterations
    n_iters = 500
    alpha_nes = 0.0007
    beta_max = 0.90

    x = x0.copy()
    z = np.zeros(2)
    fvals_nes = [rosenbrock(x)]

    for k in range(1, n_iters + 1):
        beta_k = min((k-1)/(k+2), beta_max)
        x_tilde = x + beta_k * z
        g = grad_rosen(x_tilde)
        z_new = beta_k * z - alpha_nes * g
        x = x + z_new
        z = z_new
        fvals_nes.append(rosenbrock(x))

    fvals_nes = np.array(fvals_nes)

    # Run GD for 500 iterations
    alpha_gd = 0.0012
    x = x0.copy()
    fvals_gd = [rosenbrock(x)]
    for _ in range(n_iters):
        g = grad_rosen(x)
        x = x - alpha_gd * g
        fvals_gd.append(rosenbrock(x))
    fvals_gd = np.array(fvals_gd)

    # Theoretical bounds
    # L ≈ λ_max(H) ≈ 1002 at the starting point (conservative estimate)
    L_rosen = 1002.0
    k_arr = np.arange(n_iters + 1)
    # Nesterov bound: 2L ||x0-x*||^2 / (k+1)^2
    nes_bound = 2 * L_rosen * R**2 / (k_arr + 1)**2
    nes_bound[0] = rosenbrock(x0)  # set k=0 to actual f0
    # GD linear bound: f0 * rho^k with rho = (kappa-1)/(kappa+1) ≈ 0.9992
    kappa = 2504.0
    rho_gd = (kappa - 1) / (kappa + 1)
    gd_linear_bound = rosenbrock(x0) * rho_gd**k_arr

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.semilogy(k_arr, np.maximum(fvals_nes, 1e-10), 'b-', lw=2, label='Nesterov (empirical)')
    ax.semilogy(k_arr, np.maximum(fvals_gd, 1e-10), 'k-', lw=2, label='GD (empirical)')
    ax.semilogy(k_arr[1:], np.maximum(nes_bound[1:], 1e-10), 'b--', lw=1.5, alpha=0.7,
                label=f'Nesterov bound $2L\\|x_0-x^\\star\\|^2/(k+1)^2$\n$L={L_rosen:.0f}$, $R={R:.1f}$')
    ax.semilogy(k_arr[1:], np.maximum(gd_linear_bound[1:], 1e-10), 'k--', lw=1.5, alpha=0.7,
                label=f'GD bound $f_0\\rho^k$, $\\rho\\approx{rho_gd:.4f}$')
    ax.set_xlabel('Iteration $k$')
    ax.set_ylabel('$f(x_k)$ (log scale)')
    ax.set_title('Semi-log: Nesterov vs GD (Rosenbrock, 500 iters)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, 500)
    ax.grid(True, alpha=0.3)

    # Log-log comparison
    ax2 = axes[1]
    k_plot = k_arr[1:]
    ax2.loglog(k_plot, np.maximum(fvals_nes[1:], 1e-10), 'b-', lw=2, label='Nesterov (empirical)')
    ax2.loglog(k_plot, np.maximum(fvals_gd[1:], 1e-10), 'k-', lw=2, label='GD (empirical)')
    # Reference lines
    c_nes = fvals_nes[50] * 50**2  # calibrate at k=50
    c_gd = fvals_gd[50] * 50  # calibrate at k=50
    ax2.loglog(k_plot, c_nes / k_plot**2, 'c--', lw=1.5, label='$O(1/k^2)$ reference')
    ax2.loglog(k_plot, c_gd / k_plot, 'm--', lw=1.5, label='$O(1/k)$ reference')
    ax2.set_xlabel('Iteration $k$ (log scale)')
    ax2.set_ylabel('$f(x_k)$ (log scale)')
    ax2.set_title('Log-log: Empirical Convergence Rates')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')

    fig.suptitle('Nesterov vs GD: Theoretical Bounds vs Empirical Convergence on Rosenbrock\n'
                 '($\\kappa \\approx 2504$, $x_0 = (-1,1)$)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out2, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out2}')
else:
    print(f'Skipping {out2} (already exists)')


print('Done.')
