"""
Generate additional figures for the 11th improvement cycle.
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
# Figure 1: Learning rate scheduling comparison
# ============================================================
out1 = 'figures/lr_scheduling_comparison.pdf'
if not os.path.exists(out1):
    T = 200
    t = np.arange(T)
    alpha_max = 0.1
    alpha_min = 1e-4
    T_warm = 20

    # (a) Constant
    lr_const = np.full(T, alpha_max)

    # (b) Linear warmup then constant
    lr_warmup = np.where(t < T_warm, alpha_max * t / T_warm, alpha_max)

    # (c) Step decay (gamma=0.1 every 60 iters)
    step = 60
    gamma = 0.316  # 0.1^(1/3) ~ three decays
    lr_step = alpha_max * (gamma ** (t // step))

    # (d) Cosine annealing
    lr_cosine = alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + np.cos(np.pi * t / T))

    # (e) Warmup + cosine annealing
    t_eff = np.clip(t - T_warm, 0, T - T_warm)
    T_cos = T - T_warm
    lr_warmup_cosine = np.where(
        t < T_warm,
        alpha_max * t / max(T_warm, 1),
        alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + np.cos(np.pi * t_eff / T_cos))
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(t, lr_const, 'k-', lw=1.5, label='Constant $\\alpha = 0.1$')
    ax.plot(t, lr_warmup, 'b--', lw=1.5, label='Linear warmup ($T_{\\rm warm}=20$)')
    ax.plot(t, lr_step, 'r-', lw=1.5, label='Step decay ($\\gamma=0.316$, step=60)')
    ax.plot(t, lr_cosine, 'g-', lw=1.5, label='Cosine annealing')
    ax.plot(t, lr_warmup_cosine, 'm:', lw=2, label='Warmup + cosine (AdamW)')
    ax.axvline(T_warm, color='gray', ls=':', lw=1, label='End of warmup')
    ax.set_xlabel('Iteration $t$')
    ax.set_ylabel('Learning rate $\\alpha_t$')
    ax.set_title('Learning Rate Schedules (linear scale)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, T-1)
    ax.set_ylim(0, alpha_max * 1.1)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.semilogy(t, np.maximum(lr_const, 1e-10), 'k-', lw=1.5, label='Constant')
    ax2.semilogy(t, np.maximum(lr_warmup, 1e-10), 'b--', lw=1.5, label='Linear warmup')
    ax2.semilogy(t, np.maximum(lr_step, 1e-10), 'r-', lw=1.5, label='Step decay')
    ax2.semilogy(t, np.maximum(lr_cosine, 1e-10), 'g-', lw=1.5, label='Cosine annealing')
    ax2.semilogy(t, np.maximum(lr_warmup_cosine, 1e-10), 'm:', lw=2, label='Warmup + cosine')
    ax2.axvline(T_warm, color='gray', ls=':', lw=1, label='End of warmup')
    ax2.set_xlabel('Iteration $t$')
    ax2.set_ylabel('Learning rate $\\alpha_t$ (log scale)')
    ax2.set_title('Learning Rate Schedules (log scale)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_xlim(0, T-1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Common Learning Rate Schedules: Warmup, Step Decay, Cosine Annealing', fontsize=11)
    fig.tight_layout()
    fig.savefig(out1, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out1}')
else:
    print(f'Skipping {out1} (already exists)')


# ============================================================
# Figure 2: Diagonal vs full-matrix preconditioning on Rosenbrock
# ============================================================
out2 = 'figures/preconditioning_rosenbrock.pdf'
if not os.path.exists(out2):
    def rosenbrock(x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad_rosen(x):
        return np.array([
            -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0]**2)
        ])

    def hess_rosen(x):
        return np.array([
            [1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
            [-400 * x[0], 200]
        ])

    x0 = np.array([-1.0, 1.0])
    n_iters = 80
    alpha_gd = 0.00095

    # Method 1: GD (no preconditioning)
    x = x0.copy()
    traj_gd = [x.copy()]
    fvals_gd = [rosenbrock(x)]
    for _ in range(n_iters):
        g = grad_rosen(x)
        x = x - alpha_gd * g
        traj_gd.append(x.copy())
        fvals_gd.append(rosenbrock(x))

    # Method 2: Diagonal preconditioned GD
    # P = diag(H)^{-1} at starting point
    H0 = hess_rosen(x0)
    alpha_diag = 0.85
    x = x0.copy()
    traj_diag = [x.copy()]
    fvals_diag = [rosenbrock(x)]
    for _ in range(n_iters):
        g = grad_rosen(x)
        H = hess_rosen(x)
        diag_H = np.diag(H)
        diag_H_safe = np.maximum(diag_H, 0.1)
        step = g / diag_H_safe
        x = x - alpha_diag * step
        traj_diag.append(x.copy())
        fvals_diag.append(rosenbrock(x))

    # Method 3: Full Newton preconditioning (damped)
    alpha_newton = 0.22
    lam = 1e-8
    x = x0.copy()
    traj_newton = [x.copy()]
    fvals_newton = [rosenbrock(x)]
    for _ in range(n_iters):
        g = grad_rosen(x)
        H = hess_rosen(x) + lam * np.eye(2)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = g
        x = x - alpha_newton * step
        traj_newton.append(x.copy())
        fvals_newton.append(rosenbrock(x))

    traj_gd = np.array(traj_gd)
    traj_diag = np.array(traj_diag)
    traj_newton = np.array(traj_newton)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: convergence curves
    ax = axes[0]
    iters = np.arange(n_iters + 1)
    ax.semilogy(iters, fvals_gd, 'k-', lw=1.5, label='GD (no preconditioning)')
    ax.semilogy(iters, np.maximum(fvals_diag, 1e-16), 'b--', lw=1.5, label='Diag-precond. GD')
    ax.semilogy(iters, np.maximum(fvals_newton, 1e-16), 'r-', lw=1.5, label='Newton (full Hessian)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$f(x_k)$ (log scale)')
    ax.set_title('Preconditioning on Rosenbrock')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: trajectories
    ax2 = axes[1]
    x1_grid = np.linspace(-1.5, 1.5, 200)
    x2_grid = np.linspace(-0.5, 2.0, 200)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    F = 100 * (X2 - X1**2)**2 + (1 - X1)**2
    levels = np.logspace(-1, 3.5, 18)
    ax2.contour(X1, X2, F, levels=levels, colors='gray', alpha=0.4, linewidths=0.7)
    ax2.plot(traj_gd[:, 0], traj_gd[:, 1], 'k-o', ms=2, lw=1.2,
             label='GD', zorder=3)
    ax2.plot(traj_diag[:, 0], traj_diag[:, 1], 'b--s', ms=2, lw=1.2,
             label='Diag-precond.', zorder=3)
    ax2.plot(traj_newton[:, 0], traj_newton[:, 1], 'r-^', ms=2, lw=1.2,
             label='Newton', zorder=3)
    ax2.plot(1, 1, 'g*', ms=12, zorder=5, label='Optimum $(1,1)$')
    ax2.plot(x0[0], x0[1], 'kv', ms=8, zorder=5, label='Start $(-1,1)$')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('Trajectories (80 iterations)')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-0.5, 2.0)
    ax2.grid(True, alpha=0.2)

    fig.suptitle('Effect of Preconditioning on Rosenbrock ($\\kappa \\approx 2504$)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out2, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out2}')
else:
    print(f'Skipping {out2} (already exists)')

print('Done.')
