"""
Generate additional figures for the improved report (second batch):
  1. summary_bar_chart.pdf     -- all methods on all benchmarks (bar chart)
  2. fd_error_vs_delta.pdf     -- FD approximation error as function of delta
  3. adagrad_vs_rmsprop.pdf    -- effective step decay comparison
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
# FIGURE 1: Summary bar chart of final objective values
# ============================================================
methods = ['GD', 'Polyak', 'Adagrad', 'RMSprop', 'Heavy\nBall', 'Nesterov', 'Adam', 'Newton']
bench_A = [0.4826, 2.7189, 0.4826, 0.5027, 0.4826, 0.4826, 0.4826, 0.4826]
bench_B = [0.7244, 24.0659, 0.7244, 0.7271, 0.7244, 0.7244, 0.7244, 0.7244]
bench_C = [3.5142, 0.0094, 2.2034, 3.1204, 1.4003, 0.4788, 1.5785, 0.6864]

x_pos = np.arange(len(methods))
width = 0.25

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Benchmark A
ax = axes[0]
bars = ax.bar(x_pos, bench_A, width=0.6, color=['steelblue']*8, alpha=0.8, edgecolor='black', lw=0.5)
ax.axhline(y=0.4826, color='red', linestyle='--', lw=1.5, label='$J^\\star \\approx 0.4826$')
ax.set_xticks(x_pos); ax.set_xticklabels(methods, fontsize=8)
ax.set_ylabel('Final objective value')
ax.set_title('Benchmark A: Linear Regression')
ax.set_ylim([0, 3.5])
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
# Highlight polyak bar in red
bars[1].set_color('tomato')

# Benchmark B - log scale because Polyak blows up
ax = axes[1]
bench_B_plot = bench_B.copy(); bench_B_plot[1] = min(bench_B_plot[1], 10)  # clip Polyak
bars = ax.bar(x_pos, bench_B_plot, width=0.6, color=['steelblue']*8, alpha=0.8, edgecolor='black', lw=0.5)
bars[1].set_color('tomato')
ax.axhline(y=0.7244, color='red', linestyle='--', lw=1.5, label='$f^\\star \\approx 0.7244$')
ax.set_xticks(x_pos); ax.set_xticklabels(methods, fontsize=8)
ax.set_ylabel('Final objective value (clipped at 10)')
ax.set_title('Benchmark B: Toy Neural Network')
ax.set_ylim([0, 12])
ax.annotate('Polyak:\n24.1\n(diverged)', xy=(1, 10), xytext=(1.5, 10.5),
            fontsize=7, color='tomato', ha='center',
            arrowprops=dict(arrowstyle='->', color='tomato', lw=1))
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

# Benchmark C
ax = axes[2]
colors_C = ['steelblue', 'green', 'steelblue', 'steelblue', 'steelblue', 'royalblue', 'steelblue', 'darkorange']
bars = ax.bar(x_pos, bench_C, width=0.6, color=colors_C, alpha=0.85, edgecolor='black', lw=0.5)
ax.axhline(y=0.0, color='red', linestyle='--', lw=1.5, label='$f^\\star = 0$')
ax.set_xticks(x_pos); ax.set_xticklabels(methods, fontsize=8)
ax.set_ylabel('Final objective value')
ax.set_title('Benchmark C: Rosenbrock')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
# Annotate bars
for bar, val in zip(bars, bench_C):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

plt.suptitle('Summary: Final Objective Values After Fixed Iteration Budget\n'
             '(Q1: 120 iters; Q2: 150 iters; Q3: 20 Newton iters vs 80 GD iters)',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('figures/summary_bar_chart.pdf', bbox_inches='tight')
plt.close()
print("Saved: summary_bar_chart.pdf")

# ============================================================
# FIGURE 2: Finite Difference approximation error vs delta
# For g(x) = x^4 at x0 = 0.25 (used in Q4A)
# ============================================================
def g(x): return x**4
def g_exact_deriv(x): return 4*x**3

x0_fd = 0.25
exact_g_prime = g_exact_deriv(x0_fd)  # 4 * 0.25^3 = 0.0625

deltas = np.logspace(-12, 0, 200)
fd_forward_err = np.abs((g(x0_fd + deltas) - g(x0_fd)) / deltas - exact_g_prime)
fd_central_err = np.abs((g(x0_fd + deltas/2) - g(x0_fd - deltas/2)) / deltas - exact_g_prime)

# Theoretical bounds (O(delta) truncation, O(eps_mach/delta) rounding)
eps_mach = np.finfo(float).eps  # ~2.2e-16
trunc_fwd  = deltas * abs(g_exact_deriv(x0_fd)) / 2  # ~= delta * 0.0313
round_fwd  = 2 * eps_mach * abs(g(x0_fd)) / deltas
trunc_cen  = (deltas**2) * 12 * x0_fd / 6  # O(delta^2) for central
round_cen  = 2 * eps_mach * abs(g(x0_fd)) / deltas

fig, ax = plt.subplots(figsize=(9, 6))
ax.loglog(deltas, fd_forward_err,  'b-', lw=2, label='Forward FD error')
ax.loglog(deltas, fd_central_err,  'r-', lw=2, label='Central FD error')
ax.loglog(deltas, trunc_fwd, 'b:', lw=1.2, label='$O(\\delta)$ truncation (forward)')
ax.loglog(deltas, trunc_cen, 'r:', lw=1.2, label='$O(\\delta^2)$ truncation (central)')
ax.loglog(deltas, round_fwd, 'b--', lw=1.2, label='$O(\\varepsilon_{\\rm mach}/\\delta)$ rounding')

# Mark optimal deltas
delta_opt_fwd = np.sqrt(eps_mach * abs(g(x0_fd)) / abs(g_exact_deriv(x0_fd)))
delta_opt_cen = (eps_mach * abs(g(x0_fd)) / abs(g_exact_deriv(x0_fd)))**(1/3)
ax.axvline(delta_opt_fwd, color='b', linestyle='-.', lw=1, alpha=0.7,
           label=f'Optimal $\\delta_{{\\rm fwd}} \\approx {delta_opt_fwd:.1e}$')
ax.axvline(delta_opt_cen, color='r', linestyle='-.', lw=1, alpha=0.7,
           label=f'Optimal $\\delta_{{\\rm cen}} \\approx {delta_opt_cen:.1e}$')

# Mark the Q4 deltas used
ax.axvline(0.05, color='g', linestyle='--', lw=1.5, alpha=0.8, label='$\\delta=0.05$ (Q4 good)')
ax.axvline(0.80, color='orange', linestyle='--', lw=1.5, alpha=0.8, label='$\\delta=0.8$ (Q4 poor)')

ax.set_xlabel('$\\delta$ (perturbation size)', fontsize=12)
ax.set_ylabel('Absolute approximation error', fontsize=12)
ax.set_title("Finite Difference Approximation Error for $g(x)=x^4$ at $x_0=0.25$\n"
             "(Forward vs Central Differences)", fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim([1e-12, 1.2])
plt.tight_layout()
plt.savefig('figures/fd_error_vs_delta.pdf', bbox_inches='tight')
plt.close()
print("Saved: fd_error_vs_delta.pdf")

# ============================================================
# FIGURE 3: Adagrad vs RMSprop effective step over long run
# on Benchmark B — showing Adagrad's monotone decay to near-zero
# ============================================================
def loss_B(x): return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def grad_B(x): return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

x0_B = np.array([-1.0, 4.0])
n_long = 500

def adagrad_steps(grad_fn, x0, alpha0, eps, n_iters):
    x = x0.copy().astype(float); G = np.zeros_like(x)
    eff_steps = []
    for _ in range(n_iters):
        g = grad_fn(x); G += g**2
        eff = alpha0 / (np.sqrt(G) + eps)
        eff_steps.append(eff.copy())
        x = x - eff * g
    return np.array(eff_steps)

def rmsprop_steps(grad_fn, x0, alpha0, beta, eps, n_iters):
    x = x0.copy().astype(float); v = np.zeros_like(x)
    eff_steps = []
    for _ in range(n_iters):
        g = grad_fn(x); v = beta*v + (1-beta)*g**2
        eff = alpha0 / (np.sqrt(v) + eps)
        eff_steps.append(eff.copy())
        x = x - eff * g
    return np.array(eff_steps)

ada_eff = adagrad_steps(grad_B, x0_B, 1.2, 1e-5, n_long)
rms_eff = rmsprop_steps(grad_B, x0_B, 0.14, 0.9, 1e-5, n_long)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogy(ada_eff[:, 0], 'b-', lw=1.5, label='Adagrad $x_1$ eff. step')
ax.semilogy(ada_eff[:, 1], 'b--', lw=1.5, label='Adagrad $x_2$ eff. step')
ax.semilogy(rms_eff[:, 0], 'r-', lw=1.5, label='RMSprop $x_1$ eff. step')
ax.semilogy(rms_eff[:, 1], 'r--', lw=1.5, label='RMSprop $x_2$ eff. step')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Effective step size (log scale)', fontsize=12)
ax.set_title('Adagrad vs RMSprop: Per-Coordinate Effective Step\n(Benchmark B, 500 iterations)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
# Show ratio of x1/x2 effective steps -- coordinate adaptation
ax.plot(ada_eff[:, 0] / ada_eff[:, 1], 'b-', lw=1.5, label='Adagrad: step$_{x_1}$ / step$_{x_2}$')
ax.plot(rms_eff[:, 0] / rms_eff[:, 1], 'r-', lw=1.5, label='RMSprop: step$_{x_1}$ / step$_{x_2}$')
ax.axhline(1.0, color='gray', linestyle='--', lw=1, label='Isotropic baseline')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Step ratio $\\alpha_{x_1} / \\alpha_{x_2}$')
ax.set_title('Coordinate Adaptation: Step-Size Ratio\n(ratio $> 1$: more steps in $x_1$ direction)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle('Q1: Adagrad vs RMSprop Effective Step-Size Analysis (Benchmark B)', fontsize=12)
plt.tight_layout()
plt.savefig('figures/adagrad_vs_rmsprop_steps.pdf', bbox_inches='tight')
plt.close()
print("Saved: adagrad_vs_rmsprop_steps.pdf")

print("All second-batch additional figures generated.")
