"""Generate Pass 26 figures: efficiency comparison and LR sensitivity."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

np.random.seed(42)

# ─── Benchmarks ────────────────────────────────────────────────────────────────
m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
eps_noise = np.random.randn(m)
y_data = X_data @ theta_star + eps_noise

def loss_C(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_C(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

def hess_C(x):
    return np.array([
        [2 + 1200*x[0]**2 - 400*x[1], -400*x[0]],
        [-400*x[0], 200]
    ])

x0_C = np.array([-1.0, 1.0])

# ─── Algorithm implementations with full history ───────────────────────────────

def run_gd(x0, alpha, n):
    x = x0.copy().astype(float)
    hist = [loss_C(x)]
    ge = [0]      # cumulative gradient evaluations (not counting f(x0))
    for k in range(n):
        x = x - alpha * grad_C(x)
        hist.append(loss_C(x))
        ge.append(ge[-1] + 1)   # 1 gradient eval per step
    return np.array(hist), np.array(ge)

def run_nesterov(x0, alpha, bmax, n):
    x = x0.copy().astype(float); z = np.zeros_like(x)
    hist = [loss_C(x)]; ge = [0]
    for k in range(1, n+1):
        bk = min((k-1)/(k+2), bmax)
        la = x + bk*z
        gr = grad_C(la)
        z = bk*z - alpha*gr; x = x + z
        hist.append(loss_C(x))
        ge.append(ge[-1] + 1)   # 1 gradient eval per step
    return np.array(hist), np.array(ge)

def run_hb(x0, alpha, beta, n):
    x = x0.copy().astype(float); z = np.zeros_like(x)
    hist = [loss_C(x)]; ge = [0]
    for _ in range(n):
        gr = grad_C(x); z = beta*z + alpha*gr; x = x - z
        hist.append(loss_C(x))
        ge.append(ge[-1] + 1)
    return np.array(hist), np.array(ge)

def run_adam(x0, alpha, b1, b2, eps, n):
    x = x0.copy().astype(float); mv = np.zeros_like(x); vv = np.zeros_like(x)
    hist = [loss_C(x)]; ge = [0]
    for t in range(1, n+1):
        gr = grad_C(x); mv = b1*mv+(1-b1)*gr; vv = b2*vv+(1-b2)*gr**2
        mh = mv/(1-b1**t); vh = vv/(1-b2**t)
        x = x - alpha*mh/(np.sqrt(vh)+eps)
        hist.append(loss_C(x))
        ge.append(ge[-1] + 1)
    return np.array(hist), np.array(ge)

def run_newton(x0, alpha, lam, n):
    """Each Newton step costs 1 grad + 1 Hessian (d^2 multiplications).
    For d=2: Hessian costs 4 partial derivatives ~ 4 function evals.
    We count each Newton step as d+1 = 3 'gradient-equivalent units' for d=2:
    1 gradient (2 partials) + 1 Hessian (4 second partials) → ~3 gradient evals.
    """
    d = len(x0)
    x = x0.copy().astype(float)
    hist = [loss_C(x)]; ge = [0]
    for _ in range(n):
        gr = grad_C(x)
        H = hess_C(x) + lam*np.eye(d)
        x = x - alpha*np.linalg.solve(H, gr)
        hist.append(loss_C(x))
        # Newton step cost: 1 gradient + d partial Hessian rows = d+1 gradient evals
        ge.append(ge[-1] + (d + 1))
    return np.array(hist), np.array(ge)

def run_polyak(x0, fstar, eps, n):
    x = x0.copy().astype(float)
    hist = [loss_C(x)]; ge = [0]
    for _ in range(n):
        gr = grad_C(x)
        ak = (loss_C(x) - fstar) / (np.dot(gr,gr) + eps)
        x = x - ak*gr
        hist.append(loss_C(x))
        ge.append(ge[-1] + 1)
    return np.array(hist), np.array(ge)

N = 150

gd_f, gd_ge = run_gd(x0_C, 0.0012, N)
nes_f, nes_ge = run_nesterov(x0_C, 0.0007, 0.90, N)
hb_f, hb_ge = run_hb(x0_C, 0.0008, 0.86, N)
adam_f, adam_ge = run_adam(x0_C, 0.006, 0.80, 0.999, 1e-8, N)
poly_f, poly_ge = run_polyak(x0_C, 0.0, 1e-3, N)
newt_f, newt_ge = run_newton(x0_C, 0.22, 1e-8, 80)  # 80 Newton steps = 240 grad evals

# ─── Figure 1: Efficiency comparison (convergence vs gradient evals) ──────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Computational Efficiency: Convergence vs Gradient-Evaluation Cost\n'
             '(Benchmark C, Rosenbrock, $\\kappa \\approx 2504$)', fontsize=13, y=1.01)

colors = {
    'GD': '#2196F3',
    'Nesterov': '#4CAF50',
    'Heavy Ball': '#FF9800',
    'Adam': '#9C27B0',
    'Polyak': '#F44336',
    'Newton (×3/step)': '#795548',
}

ax = axes[0]
ax.semilogy(gd_ge,   gd_f,   '-',  color=colors['GD'],         lw=2.0, label=f'GD ($f_{{150}}={gd_f[-1]:.3f}$)')
ax.semilogy(nes_ge,  nes_f,  '-',  color=colors['Nesterov'],   lw=2.0, label=f'Nesterov ($f_{{150}}={nes_f[-1]:.3f}$)')
ax.semilogy(hb_ge,   hb_f,   '--', color=colors['Heavy Ball'], lw=1.8, label=f'Heavy Ball ($f_{{150}}={hb_f[-1]:.3f}$)')
ax.semilogy(adam_ge, adam_f, '--', color=colors['Adam'],       lw=1.8, label=f'Adam ($f_{{150}}={adam_f[-1]:.3f}$)')
ax.semilogy(poly_ge, poly_f, ':',  color=colors['Polyak'],     lw=2.0, label=f'Polyak ($f_{{150}}={poly_f[-1]:.3f}$)')
ax.semilogy(newt_ge, newt_f, 's-', color=colors['Newton (×3/step)'], lw=1.8, ms=4,
            label=f'Newton (80 steps, ×3 cost)')

ax.set_xlabel('Cumulative gradient evaluations', fontsize=11)
ax.set_ylabel('$f(x_k)$ (log scale)', fontsize=11)
ax.set_title('Convergence vs gradient evaluation count\n(Newton step = 3 grad-evals for $d=2$)', fontsize=10)
ax.legend(fontsize=8.5, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 160)

# Right panel: convergence vs iterations (for direct comparison)
ax2 = axes[1]
iters = np.arange(N+1)
ax2.semilogy(iters, gd_f,   '-',  color=colors['GD'],         lw=2.0, label='GD')
ax2.semilogy(iters, nes_f,  '-',  color=colors['Nesterov'],   lw=2.0, label='Nesterov NAG')
ax2.semilogy(iters, hb_f,   '--', color=colors['Heavy Ball'], lw=1.8, label='Heavy Ball')
ax2.semilogy(iters, adam_f, '--', color=colors['Adam'],       lw=1.8, label='Adam')
ax2.semilogy(iters, poly_f, ':',  color=colors['Polyak'],     lw=2.0, label='Polyak')
ax2.semilogy(np.arange(81), newt_f, 's-', color=colors['Newton (×3/step)'],
             lw=1.8, ms=4, label='Newton (20 steps shown)')

ax2.set_xlabel('Iterations', fontsize=11)
ax2.set_ylabel('$f(x_k)$ (log scale)', fontsize=11)
ax2.set_title('Same data vs iterations\n(per-step cost not accounted for)', fontsize=10)
ax2.legend(fontsize=8.5, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 160)

plt.tight_layout()
plt.savefig('figures/q_efficiency_comparison.pdf', bbox_inches='tight')
plt.close()
print("Saved figures/q_efficiency_comparison.pdf")

# ─── Figure 2: Learning rate / step size sensitivity ──────────────────────────

alpha_gd_vals   = [0.0003, 0.0006, 0.0009, 0.0012, 0.0015, 0.0018]
alpha_nes_vals  = [0.0003, 0.0005, 0.0007, 0.0009, 0.0011, 0.0013]
alpha_adam_vals = [0.001, 0.002, 0.004, 0.006, 0.008, 0.010]

N_sens = 150
n_alpha = len(alpha_gd_vals)
cmap = plt.cm.viridis

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle('Learning Rate Sensitivity on Benchmark C (Rosenbrock, 150 iterations)',
             fontsize=13, y=1.01)

# GD sensitivity
ax = axes[0]
final_gd = []
for i, a in enumerate(alpha_gd_vals):
    try:
        f_hist, _ = run_gd(x0_C, a, N_sens)
        if np.all(np.isfinite(f_hist)):
            col = cmap(i / (n_alpha - 1))
            ax.semilogy(f_hist, color=col, lw=1.8, label=f'$\\alpha={a:.4f}$')
            final_gd.append((a, f_hist[-1]))
        else:
            final_gd.append((a, np.inf))
    except:
        final_gd.append((a, np.inf))
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('$f(x_k)$ (log scale)', fontsize=11)
ax.set_title('GD: step size sensitivity', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.axhline(y=gd_f[-1], color='gray', ls=':', alpha=0.5, label='baseline')

# Nesterov sensitivity
ax = axes[1]
final_nes = []
for i, a in enumerate(alpha_nes_vals):
    try:
        f_hist, _ = run_nesterov(x0_C, a, 0.90, N_sens)
        if np.all(np.isfinite(f_hist)):
            col = cmap(i / (n_alpha - 1))
            ax.semilogy(f_hist, color=col, lw=1.8, label=f'$\\alpha={a:.4f}$')
            final_nes.append((a, f_hist[-1]))
        else:
            final_nes.append((a, np.inf))
    except:
        final_nes.append((a, np.inf))
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('$f(x_k)$ (log scale)', fontsize=11)
ax.set_title('Nesterov NAG: step size sensitivity', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Adam sensitivity
ax = axes[2]
final_adam = []
for i, a in enumerate(alpha_adam_vals):
    try:
        f_hist, _ = run_adam(x0_C, a, 0.80, 0.999, 1e-8, N_sens)
        if np.all(np.isfinite(f_hist)):
            col = cmap(i / (n_alpha - 1))
            ax.semilogy(f_hist, color=col, lw=1.8, label=f'$\\alpha={a:.3f}$')
            final_adam.append((a, f_hist[-1]))
        else:
            final_adam.append((a, np.inf))
    except:
        final_adam.append((a, np.inf))
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('$f(x_k)$ (log scale)', fontsize=11)
ax.set_title('Adam: learning rate sensitivity', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/q_lr_sensitivity.pdf', bbox_inches='tight')
plt.close()
print("Saved figures/q_lr_sensitivity.pdf")

# Print numerical summary
print("\n=== GD final values vs alpha ===")
for a, fv in final_gd:
    print(f"  alpha={a:.4f}: f={fv:.4f}")
print("=== Nesterov final values vs alpha ===")
for a, fv in final_nes:
    print(f"  alpha={a:.4f}: f={fv:.4f}")
print("=== Adam final values vs alpha ===")
for a, fv in final_adam:
    print(f"  alpha={a:.3f}: f={fv:.4f}")
