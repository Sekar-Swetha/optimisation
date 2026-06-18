"""
Generate additional figures for the 17th improvement cycle.
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
# Figure 1: Newton convergence phases on Rosenbrock
# (damped linear phase vs quadratic convergence phase)
# ============================================================
out1 = 'figures/newton_convergence_phases.pdf'
if not os.path.exists(out1):
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
    n_iters = 120
    alpha_newton = 0.22
    lam = 1e-8

    x = x0.copy()
    fvals = [rosenbrock(x)]
    newton_decrements = []
    dist_to_opt = [np.linalg.norm(x - np.array([1.0, 1.0]))]

    for _ in range(n_iters):
        g = grad_rosen(x)
        H = hess_rosen(x) + lam * np.eye(2)
        try:
            p = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            p = g
        # Newton decrement
        nd = np.sqrt(g @ np.linalg.solve(H, g))
        newton_decrements.append(nd)
        x = x - alpha_newton * p
        fvals.append(rosenbrock(x))
        dist_to_opt.append(np.linalg.norm(x - np.array([1.0, 1.0])))

    fvals = np.array(fvals)
    dist_to_opt = np.array(dist_to_opt)
    newton_decrements = np.array(newton_decrements)

    # Identify transition to quadratic regime
    # quadratic regime: consecutive ratio of errors e_{k+1}/e_k^2 is approximately constant
    eps = 1e-10
    iters = np.arange(n_iters + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: convergence of f
    ax = axes[0]
    ax.semilogy(iters, np.maximum(fvals, 1e-12), 'b-', lw=2, label='Newton ($\\alpha=0.22$)')
    ax.axvline(30, color='orange', ls='--', lw=1.5, label='Est. linear$\\to$quadratic transition')
    ax.set_xlabel('Iteration $k$')
    ax.set_ylabel('$f(x_k)$ (log scale)')
    ax.set_title('Objective Value')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Middle: Newton decrement
    ax2 = axes[1]
    ax2.semilogy(np.arange(n_iters), newton_decrements + 1e-16, 'r-', lw=2)
    ax2.axhline(1.0, color='gray', ls=':', lw=1.5, label='$\\lambda(x)=1$ (quadratic regime threshold)')
    ax2.set_xlabel('Iteration $k$')
    ax2.set_ylabel('Newton decrement $\\lambda(x_k)$')
    ax2.set_title('Newton Decrement')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Right: distance to optimum
    ax3 = axes[2]
    ax3.semilogy(iters, np.maximum(dist_to_opt, 1e-12), 'g-', lw=2)
    # Show quadratic convergence reference: if quadratic from some k0, error^2
    k0 = 60
    if dist_to_opt[k0] < 1:
        ref_k = np.arange(k0, n_iters + 1)
        ref_err = dist_to_opt[k0] ** (2 ** (ref_k - k0))
        valid = ref_err > 1e-15
        ax3.semilogy(ref_k[valid], ref_err[valid], 'r--', lw=1.5, label='Quadratic ref')
    ax3.set_xlabel('Iteration $k$')
    ax3.set_ylabel('$\\|x_k - x^\\star\\|$')
    ax3.set_title('Distance to Optimum')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.suptitle('Newton Convergence Phases on Rosenbrock (120 iterations, $\\alpha=0.22$)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out1, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out1}')
else:
    print(f'Skipping {out1} (already exists)')


# ============================================================
# Figure 2: Adagrad vs RMSprop effective step size comparison
# ============================================================
out2 = 'figures/adagrad_rmsprop_effective_lr.pdf'
if not os.path.exists(out2):
    np.random.seed(42)

    # Simulate gradient sequence on Rosenbrock
    def rosenbrock(x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad_rosen(x):
        return np.array([
            -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0]**2)
        ])

    x0 = np.array([-1.0, 1.0])

    # Run GD for 150 iters and record gradients
    x = x0.copy()
    grads = []
    for _ in range(150):
        g = grad_rosen(x)
        grads.append(g.copy())
        x = x - 0.0012 * g
    grads = np.array(grads)

    # Adagrad effective step (alpha0=0.45)
    alpha0_ada = 0.45
    eps = 1e-5
    G = np.zeros(2)
    ada_eff = []
    for g in grads:
        G = G + g**2
        eff = alpha0_ada / (np.sqrt(G) + eps)
        ada_eff.append(eff.copy())
    ada_eff = np.array(ada_eff)

    # RMSprop effective step (alpha0=0.0035, beta=0.9)
    alpha0_rms = 0.0035
    beta_rms = 0.9
    v = np.zeros(2)
    rms_eff = []
    for g in grads:
        v = beta_rms * v + (1 - beta_rms) * g**2
        eff = alpha0_rms / (np.sqrt(v) + eps)
        rms_eff.append(eff.copy())
    rms_eff = np.array(rms_eff)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    t = np.arange(len(grads))

    for i, (coord, label) in enumerate([(0, '$x_1$ component'), (1, '$x_2$ component')]):
        ax = axes[i]
        ax.semilogy(t, ada_eff[:, coord], 'b-', lw=1.5, label=f'Adagrad ($\\alpha_0={alpha0_ada}$)')
        ax.semilogy(t, rms_eff[:, coord], 'r--', lw=1.5, label=f'RMSprop ($\\alpha_0={alpha0_rms}$)')
        ax.set_xlabel('Iteration $t$')
        ax.set_ylabel(f'Effective step ({label})')
        ax.set_title(f'Effective Learning Rate: {label}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Adagrad vs RMSprop Effective Step Sizes on Benchmark C (Rosenbrock)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out2, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out2}')
else:
    print(f'Skipping {out2} (already exists)')

print('Done.')
