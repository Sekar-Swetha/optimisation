"""
Generate supplementary figures for the CS7DS2 Final Report.
These add value beyond the main figures:
  - q2_convergence_rate.pdf: log-log convergence rate plot (Q2 Rosenbrock)
  - q6_fw_gap.pdf:           Frank-Wolfe duality gap vs iteration (Q6)
  - q1_method_comparison.pdf: summary bar chart across all Q1 benchmarks
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ── Benchmark definitions (must match main code) ──────────────────────────

m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
eps_noise = np.random.randn(m)
y_data = X_data @ theta_star + eps_noise

def loss_C(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_C(x):
    g1 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    g2 = 200*(x[1] - x[0]**2)
    return np.array([g1, g2])

x0_C = np.array([-1.0, 1.0])

# ── Optimiser implementations (minimal) ───────────────────────────────────

def gradient_descent(grad_fn, loss_fn, x0, alpha, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        x = x - alpha * grad_fn(x)
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

def nesterov_momentum(grad_fn, loss_fn, x0, alpha, beta_max, n_iters):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for k in range(1, n_iters + 1):
        beta_k = min((k - 1) / (k + 2), beta_max)
        lookahead = x + beta_k * z
        g = grad_fn(lookahead)
        z = beta_k * z - alpha * g
        x = x + z
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

def adam_optimiser(grad_fn, loss_fn, x0, alpha, beta1, beta2, eps, n_iters):
    x = x0.copy().astype(float)
    m_vec, v_vec = np.zeros_like(x), np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for t in range(1, n_iters + 1):
        g = grad_fn(x)
        m_vec = beta1 * m_vec + (1 - beta1) * g
        v_vec = beta2 * v_vec + (1 - beta2) * g**2
        m_hat = m_vec / (1 - beta1**t)
        v_hat = v_vec / (1 - beta2**t)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

# ── Figure 1: Q2 convergence rate log-log plot on Rosenbrock ──────────────
#  Shows empirical O(1/k) vs O(1/k^2) rates vs GD / Nesterov / Adam

print("Generating q2_convergence_rate.pdf ...")
N = 300
gd_C    = gradient_descent(grad_C, loss_C, x0_C, 0.0012, N)
nes_C   = nesterov_momentum(grad_C, loss_C, x0_C, 0.0007, 0.90, N)
adam_C  = adam_optimiser(grad_C, loss_C, x0_C, 0.006, 0.80, 0.999, 1e-8, N)

# Approximate f* for Rosenbrock (= 0) — shift by small epsilon for log scale
f_star_C = 0.0
iters = np.arange(1, N + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: raw convergence
ax = axes[0]
ax.semilogy(gd_C,   'k--', lw=1.5, label='GD ($\\alpha=0.0012$)')
ax.semilogy(nes_C,  'b-',  lw=2,   label='Nesterov ($\\alpha=0.0007$)')
ax.semilogy(adam_C, 'r-',  lw=2,   label='Adam ($\\alpha=0.006$)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective $f(x_k)$ (log scale)')
ax.set_title('Q2: Rosenbrock Convergence (semi-log)')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: log-log plot with O(1/k) and O(1/k^2) reference lines
ax = axes[1]
eps_shift = 1e-6  # avoid log(0)

gd_err   = np.maximum(gd_C[1:],   eps_shift)
nes_err  = np.maximum(nes_C[1:],  eps_shift)
adam_err = np.maximum(adam_C[1:], eps_shift)

ax.loglog(iters, gd_err,   'k--', lw=1.5, label='GD')
ax.loglog(iters, nes_err,  'b-',  lw=2,   label='Nesterov')
ax.loglog(iters, adam_err, 'r-',  lw=2,   label='Adam')

# Reference lines
k_ref = iters[10:]  # skip first few
c1 = gd_err[10] * k_ref[0]          # O(1/k) slope through GD at k=10
c2 = nes_err[10] * k_ref[0]**2      # O(1/k^2) slope through Nesterov at k=10
ax.loglog(k_ref, c1 / k_ref,     'k:', lw=1.5, alpha=0.6, label='$O(1/k)$ ref')
ax.loglog(k_ref, c2 / k_ref**2,  'b:', lw=1.5, alpha=0.6, label='$O(1/k^2)$ ref')

ax.set_xlabel('Iteration $k$ (log scale)')
ax.set_ylabel('Objective $f(x_k)$ (log scale)')
ax.set_title('Q2: Log-log Convergence Rate Analysis')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Q2: Convergence Rate Comparison on Rosenbrock (300 iterations)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('figures/q2_convergence_rate.pdf', bbox_inches='tight')
plt.close()
print("  Saved q2_convergence_rate.pdf")

# ── Figure 2: Frank-Wolfe duality gap evolution ────────────────────────────
#  For interior optimum f(x) = (x1-1)^2 + (x2-5)^2 on box [0.5,5]x[-5,10]

print("Generating q6_fw_gap.pdf ...")

X_bounds = [(0.5, 5.0), (-5.0, 10.0)]

def loss_fw_int(x):
    return (x[0]-1)**2 + (x[1]-5)**2

def grad_fw_int(x):
    return np.array([2*(x[0]-1), 2*(x[1]-5)])

def fw_lp_box(g, bounds):
    return np.array([bounds[i][0] if g[i] > 0 else bounds[i][1]
                     for i in range(len(g))])

def frank_wolfe_with_gap(grad_fn, loss_fn, x0, bounds, beta, n_iters):
    x = x0.copy().astype(float)
    f_hist, gap_hist = [loss_fn(x)], []
    for _ in range(n_iters):
        g = grad_fn(x)
        z = fw_lp_box(g, bounds)
        # Frank-Wolfe gap = g^T (x - z) >= 0  (primal duality gap upper bound)
        fw_gap = g @ (x - z)
        gap_hist.append(max(fw_gap, 1e-16))
        x = beta * x + (1 - beta) * z
        f_hist.append(loss_fn(x))
    return np.array(f_hist), np.array(gap_hist)

# Standard step size gamma_k = 2/(k+2)
def frank_wolfe_standard(grad_fn, loss_fn, x0, bounds, n_iters):
    x = x0.copy().astype(float)
    f_hist, gap_hist = [loss_fn(x)], []
    for k in range(n_iters):
        gamma_k = 2.0 / (k + 2)
        g = grad_fn(x)
        z = fw_lp_box(g, bounds)
        fw_gap = g @ (x - z)
        gap_hist.append(max(fw_gap, 1e-16))
        x = (1 - gamma_k) * x + gamma_k * z
        f_hist.append(loss_fn(x))
    return np.array(f_hist), np.array(gap_hist)

x0_fw_int = np.array([1.0, 1.0])
fw_090_f, fw_090_gap  = frank_wolfe_with_gap(grad_fw_int, loss_fw_int, x0_fw_int, X_bounds, 0.90,  180)
fw_985_f, fw_985_gap  = frank_wolfe_with_gap(grad_fw_int, loss_fw_int, x0_fw_int, X_bounds, 0.985, 180)
fw_std_f, fw_std_gap  = frank_wolfe_standard(grad_fw_int, loss_fw_int, x0_fw_int, X_bounds, 180)

iters_fw = np.arange(1, 181)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.semilogy(iters_fw, fw_090_gap,  'b-',  lw=2, label='FW gap ($\\beta=0.90$)')
ax.semilogy(iters_fw, fw_985_gap,  'r-',  lw=2, label='FW gap ($\\beta=0.985$)')
ax.semilogy(iters_fw, fw_std_gap,  'g-',  lw=2, label='FW gap ($\\gamma_k=2/(k+2)$)')
# Reference O(1/k) line
c_ref = fw_std_gap[0]
ax.loglog(iters_fw, c_ref / iters_fw, 'k:', lw=1.5, alpha=0.6, label='$O(1/k)$ reference')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Frank--Wolfe Gap $g_k$ (log scale)')
ax.set_title('Q6: Frank--Wolfe Duality Gap Evolution')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(fw_090_f,  'b-',  lw=2, label='$f(x_k)$ ($\\beta=0.90$)')
ax.semilogy(fw_985_f,  'r-',  lw=2, label='$f(x_k)$ ($\\beta=0.985$)')
ax.semilogy(fw_std_f,  'g-',  lw=2, label='$f(x_k)$ ($\\gamma_k=2/(k+2)$)')
ax.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='$f^\\star = 0$')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Objective $f(x_k)$ (log scale)')
ax.set_title('Q6: FW Objective — Fixed $\\beta$ vs Standard Step')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Q6: Frank--Wolfe Analysis — Interior Optimum $f(x)=(x_1-1)^2+(x_2-5)^2$',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('figures/q6_fw_gap.pdf', bbox_inches='tight')
plt.close()
print("  Saved q6_fw_gap.pdf")

# ── Figure 3: Q1 method summary comparison bar chart ─────────────────────
#  All Q1 methods, all benchmarks, final f-values (grouped bars)

print("Generating q1_method_comparison.pdf ...")

# Values from Table tab:q1_final (120 iterations)
methods   = ['GD', 'Polyak', 'Adagrad', 'RMSprop', 'Heavy Ball']
bench_A   = [0.4826,  2.7189,  0.4826,   0.5027,    0.4826]
bench_B   = [0.7244, 24.0659,  0.7244,   0.7271,    0.7244]
bench_C   = [3.5142,  0.0094,  2.2034,   3.1204,    1.4003]
f_star    = {'A': 0.4826, 'B': 0.7244, 'C': 0.0}

x_pos = np.arange(len(methods))
width = 0.28

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
colors = ['#4C72B0', '#DD8452', '#55A868']

for ax, (bench_name, vals, fstar), title in zip(
    axes,
    [('A', bench_A, 0.4826), ('B', bench_B, 0.7244), ('C', bench_C, 0.0)],
    ['Benchmark A\n(Linear Regression)', 'Benchmark B\n(Toy Neural Net)', 'Benchmark C\n(Rosenbrock)']
):
    # Cap diverged Polyak values for display
    display_vals = [min(v, 5.0) if bench_name != 'B' else min(v, 35.0) for v in vals]
    bars = ax.bar(x_pos, display_vals, width=0.6, color=colors[0], alpha=0.7, edgecolor='black', lw=0.8)
    ax.axhline(y=fstar, color='red', linestyle='--', lw=2, label=f'$f^* = {fstar}$')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Final objective value $f(x_{120})$')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    # Annotate bars
    for bar_obj, val in zip(bars, vals):
        txt = f'{val:.3f}' if val < 30 else f'{val:.1f}†'
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2,
                bar_obj.get_height() + 0.02 * max(display_vals),
                txt, ha='center', va='bottom', fontsize=8)

axes[1].set_ylabel('')
axes[2].set_ylabel('')
plt.suptitle('Q1: Final Objective Values After 120 Iterations — All Methods, All Benchmarks\n'
             '(† clipped for Polyak on B: true value = 24.07)', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('figures/q1_method_comparison.pdf', bbox_inches='tight')
plt.close()
print("  Saved q1_method_comparison.pdf")

# ── Figure 4: Benchmark B Hessian condition number vs x1 ─────────────────
print("Generating q3_hessian_condition.pdf ...")

x1_range = np.linspace(-2, 3, 500)
H11 = 2 - np.sin(x1_range)  # first eigenvalue (H22 = 10 is fixed)
kappa = 10.0 / H11           # condition number

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(x1_range, H11, 'b-', lw=2, label='$H_{11}(x_1) = 2 - \\sin(x_1)$')
ax.axhline(y=10, color='r', linestyle='--', lw=1.5, label='$H_{22} = 10$ (constant)')
ax.axvline(x=-1.0, color='gray', linestyle=':', lw=1.5, alpha=0.8, label='Starting $x_1=-1$')
ax.axvline(x=0.582, color='green', linestyle=':', lw=1.5, alpha=0.8, label='Optimal $x_1 \\approx 0.582$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('Eigenvalue')
ax.set_title('Benchmark B: Hessian Eigenvalues vs $x_1$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 3)

ax = axes[1]
ax.plot(x1_range, kappa, 'r-', lw=2, label='Condition number $\\kappa(x_1)$')
ax.axvline(x=-1.0, color='gray', linestyle=':', lw=1.5, alpha=0.8, label='Starting $x_1=-1$')
ax.axvline(x=0.582, color='green', linestyle=':', lw=1.5, alpha=0.8, label='Optimal $x_1\\approx 0.582$')
# Annotate specific points
for x_pt, label in [(-1.0, 'start\n$\\kappa\\approx3.5$'), (0.582, 'optimum\n$\\kappa\\approx6.9$')]:
    k_val = 10 / (2 - np.sin(x_pt))
    ax.plot(x_pt, k_val, 'ko', ms=8, zorder=5)
    ax.annotate(label, xy=(x_pt, k_val), xytext=(x_pt + 0.3, k_val + 0.5),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))
ax.set_xlabel('$x_1$')
ax.set_ylabel('Condition number $\\kappa = \\lambda_{\\max}/\\lambda_{\\min}$')
ax.set_title('Benchmark B: Condition Number vs $x_1$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 3)
ax.set_ylim(0, 15)

plt.suptitle('Benchmark B: Hessian Structure (diagonal, $x_2$-independent)',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('figures/q3_hessian_condition.pdf', bbox_inches='tight')
plt.close()
print("  Saved q3_hessian_condition.pdf")

print("\nAll supplementary figures generated successfully.")
