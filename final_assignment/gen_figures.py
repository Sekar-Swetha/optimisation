"""Generate additional figures for the improved report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
os.chdir('/home/user/optimisation/final_assignment')
os.makedirs('figures', exist_ok=True)

# ================================================================
# Figure 1: Newton iterations on g(x) = x^4 - LINEAR convergence
# (Degenerate case: g''(0)=0, so Newton does NOT converge quadratically)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_plot = np.linspace(-0.02, 0.30, 400)
g_fn = lambda x: x**4
g_prime = lambda x: 4*x**3
g_pprime = lambda x: 12*x**2

# Newton iterations from x0 = 0.25
# x_{k+1} = x_k - (4x_k^3)/(12x_k^2) = x_k - x_k/3 = (2/3)x_k  (LINEAR rate 2/3)
x_iter = [0.25]
for _ in range(5):
    xi = x_iter[-1]
    xi_next = xi - g_prime(xi) / g_pprime(xi)
    x_iter.append(xi_next)

print('Newton iterates on x^4:', ['{:.8f}'.format(v) for v in x_iter])
linear_rates = [x_iter[i+1]/x_iter[i] for i in range(len(x_iter)-1)]
print('Linear convergence rates:', ['{:.4f}'.format(r) for r in linear_rates])

ax = axes[0]
ax.plot(x_plot, g_fn(x_plot), 'k-', lw=2.5, label=r'$g(x) = x^4$')
colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(x_iter)-1))
for i in range(len(x_iter) - 1):
    xi = x_iter[i]
    g0 = g_fn(xi); gp0 = g_prime(xi); gpp0 = g_pprime(xi)
    quad = g0 + gp0 * (x_plot - xi) + 0.5 * gpp0 * (x_plot - xi)**2
    ax.plot(x_plot, quad, '--', color=colors[i], alpha=0.75, lw=1.5,
            label='Step {} quadratic model'.format(i))
    ax.axvline(xi, color=colors[i], linestyle=':', lw=1.2, alpha=0.6)
    ax.plot(xi, g_fn(xi), 'o', color=colors[i], ms=9, zorder=5)
ax.axvline(0, color='red', lw=1.5, linestyle='--', alpha=0.7, label='Optimum $x^*=0$')
ax.set_xlim(-0.01, 0.28)
ax.set_ylim(-0.0015, 0.006)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$g(x)$')
ax.set_title(r'Newton Steps on $g(x)=x^4$ from $x_0=0.25$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Convergence plot -- showing linear (not quadratic) rate
errors = [abs(xi) for xi in x_iter]
k_arr = np.arange(len(errors))
linear_fit = 0.25 * (2.0/3.0)**k_arr

ax2 = axes[1]
ax2.semilogy(k_arr, errors, 'b-o', lw=2, ms=9, label=r'$|x_k|$ (Newton iterates)')
ax2.semilogy(k_arr, linear_fit, 'r--', lw=1.8,
             label=r'Linear fit: $|x_k| = 0.25 \cdot (2/3)^k$')
ax2.set_xlabel('Iteration $k$')
ax2.set_ylabel(r'$|x_k - x^*|$ (log scale)')
ax2.set_title(r'Convergence Rate: Newton on $g(x)=x^4$')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.text(0.35, 0.7,
         'Linear convergence (rate $= 2/3$)\nbecause $g\'\'(x^*) = 0$:\nHessian singular at optimum.',
         transform=ax2.transAxes, fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

plt.tight_layout()
plt.savefig('figures/q3_newton_iterations.pdf', bbox_inches='tight')
plt.close()
print('Saved q3_newton_iterations.pdf')

# ================================================================
# Figure 2: FD error analysis vs delta
# ================================================================
m_data = 1000
X_data = np.random.randn(m_data, 2)
theta_star_val = np.array([3.0, 4.0])
eps_noise = np.random.randn(m_data)
y_data = X_data @ theta_star_val + eps_noise

def loss_B(x):
    return (x[0] - 1)**2 + 5*(x[1] - 2)**2 + np.sin(x[0])

def grad_B(x):
    return np.array([2*(x[0] - 1) + np.cos(x[0]), 10*(x[1] - 2)])

x_test = np.array([-1.0, 4.0])
g_true = grad_B(x_test)

deltas = np.logspace(-12, 0, 200)
fd_errors = []
for d in deltas:
    g_fd = np.zeros(2)
    f0 = loss_B(x_test)
    for i in range(2):
        ei = np.zeros(2)
        ei[i] = d
        g_fd[i] = (loss_B(x_test + ei) - f0) / d
    fd_errors.append(np.linalg.norm(g_fd - g_true))

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(deltas, fd_errors, 'b-', lw=2, label=r'FD error $\|\hat{g} - \nabla f\|$')
ax.axvline(0.05, color='green', lw=2, linestyle='--', label=r'$\delta=0.05$ (good, used in code)')
ax.axvline(0.8, color='red', lw=2, linestyle='--', label=r'$\delta=0.8$ (poor, used in code)')
optimal_delta = np.sqrt(np.finfo(float).eps)
ax.axvline(optimal_delta, color='purple', lw=1.5, linestyle=':',
           label=r'$\delta \approx \sqrt{\varepsilon_{\rm mach}} \approx 10^{-8}$ (optimal)')

# Region labels
ax.fill_between([1e-12, optimal_delta], [1e-16, 1e-16], [1e2, 1e2],
                alpha=0.08, color='orange', label='Round-off dominated')
ax.fill_between([optimal_delta, 10.0], [1e-16, 1e-16], [1e2, 1e2],
                alpha=0.05, color='blue', label='Truncation dominated')

ax.set_xlabel(r'Step size $\delta$')
ax.set_ylabel('Gradient Approximation Error')
ax.set_title(r'Q4: FD Gradient Error vs $\delta$ on Benchmark B')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim([1e-12, 2])
ax.set_ylim([1e-10, 100])
plt.tight_layout()
plt.savefig('figures/q4_fd_error.pdf', bbox_inches='tight')
plt.close()
print('Saved q4_fd_error.pdf')

# ================================================================
# Figure 3: Q1 Summary bar chart of final objective values
# ================================================================
methods = ['GD\n(base)', 'Polyak', 'Adagrad', 'RMSprop', 'Heavy\nBall']
bench_A = [0.4826, 2.7189, 0.4826, 0.5027, 0.4826]
bench_B = [0.7244, 24.0659, 0.7244, 0.7271, 0.7244]
bench_C = [3.5142, 0.0094, 2.2034, 3.1204, 1.4003]

x = np.arange(len(methods))
clrs = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
configs = [
    (bench_A, 'Benchmark A (Linear Regression)', 0.6, [0.4826, None]),
    (bench_B, 'Benchmark B (Toy Neural Network)', 2.0, [0.7244, None]),
    (bench_C, 'Benchmark C (Rosenbrock)', 5.0, [None, None]),
]
for ax, (vals, title, ylim, refs) in zip(axes, configs):
    bars = ax.bar(x, np.clip(vals, 0, ylim), width=0.6, color=clrs, alpha=0.85, edgecolor='white')
    # Mark best
    best_idx = np.argmin(vals)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Final Objective Value')
    ax.set_title('Q1: ' + title)
    ax.set_ylim(0, ylim * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, vals)):
        disp = '{:.4f}'.format(val) if val < ylim else '{:.2f}'.format(val)
        ax.text(bar.get_x() + bar.get_width() / 2., min(val, ylim) + ylim * 0.01,
                disp, ha='center', va='bottom', fontsize=8, rotation=0,
                color='red' if val > ylim else 'black')

plt.suptitle('Q1: Final Objective Values After 120 Iterations (gold border = best per benchmark)',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('figures/q1_summary_bar.pdf', bbox_inches='tight')
plt.close()
print('Saved q1_summary_bar.pdf')

# ================================================================
# Figure 4: KKT / Active set illustration for Q5
# ================================================================
x1_q5 = np.linspace(-0.2, 2.5, 300)
x2_q5 = np.linspace(0.5, 4.5, 300)
X1Q5, X2Q5 = np.meshgrid(x1_q5, x2_q5)
ZQ5 = (X1Q5 - 1)**2 + 5*(X2Q5 - 2)**2 + np.sin(X1Q5)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: KKT illustration — constrained vs unconstrained minimum
ax = axes[0]
cs = ax.contour(X1Q5, X2Q5, ZQ5, levels=np.linspace(0.7, 10, 25), cmap='viridis', alpha=0.7)
ax.axvline(x=0.5, color='red', lw=2.5, linestyle='--', label=r'Constraint $x_1 = 0.5$')
ax.fill_betweenx([0.5, 4.5], -0.2, 0.5, alpha=0.15, color='red')
ax.plot(0.582, 2.0, 'g*', ms=16, zorder=6, label=r'Unconstrained min $(0.582, 2)$')
ax.plot(0.582, 2.0, 'go', ms=16, zorder=5, fillstyle='none', lw=2)
ax.text(0.6, 1.8, r'$x_1^* = 0.582 > 0.5$', fontsize=9, color='green',
        bbox=dict(facecolor='white', alpha=0.7))
ax.text(0.55, 1.7, 'Constraint inactive\nat unconstrained min', fontsize=8, color='green')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_title('Q5: Constrained Optimisation Geometry\n(Benchmark B, constraint $x_1 \\geq 0.5$)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_xlim(-0.2, 2.5)
ax.set_ylim(0.5, 4.5)

# Right: Penalty landscape for different lambda
x1_pen = np.linspace(-0.5, 3.0, 300)
x2_fixed = 2.0
penalty_vals_015 = [(x1 - 1)**2 + 5*(x2_fixed - 2)**2 + np.sin(x1) + 0.15 * max(0, 0.5 - x1)
                    for x1 in x1_pen]
penalty_vals_18 = [(x1 - 1)**2 + 5*(x2_fixed - 2)**2 + np.sin(x1) + 1.8 * max(0, 0.5 - x1)
                   for x1 in x1_pen]
penalty_vals_45 = [(x1 - 1)**2 + 5*(x2_fixed - 2)**2 + np.sin(x1) + 4.5 * max(0, 0.5 - x1)
                   for x1 in x1_pen]
orig_vals = [(x1 - 1)**2 + 5*(x2_fixed - 2)**2 + np.sin(x1) for x1 in x1_pen]

ax2 = axes[1]
ax2.plot(x1_pen, orig_vals, 'k-', lw=2, label=r'$f(x)$ (unconstrained)')
ax2.plot(x1_pen, penalty_vals_015, 'g--', lw=2, label=r'$F_{0.15}(x)$')
ax2.plot(x1_pen, penalty_vals_18, 'orange', lw=2, linestyle='-.', label=r'$F_{1.8}(x)$')
ax2.plot(x1_pen, penalty_vals_45, 'r-', lw=2, label=r'$F_{4.5}(x)$')
ax2.axvline(0.5, color='red', lw=2, linestyle='--', alpha=0.5, label=r'$x_1 = 0.5$')
ax2.fill_betweenx([0, 20], -0.5, 0.5, alpha=0.1, color='red')
ax2.set_xlabel(r'$x_1$ (at $x_2 = 2$)')
ax2.set_ylabel(r'Penalised objective $F_\lambda(x_1, 2)$')
ax2.set_title(r'Q5: Penalty Function $F_\lambda = f + \lambda \max(0, 0.5-x_1)$ vs $\lambda$')
ax2.set_ylim(0, 8)
ax2.set_xlim(-0.5, 3.0)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/q5_penalty_landscape.pdf', bbox_inches='tight')
plt.close()
print('Saved q5_penalty_landscape.pdf')

print('\nAll additional figures generated successfully!')
print('New figures:')
print('  figures/q3_newton_iterations.pdf')
print('  figures/q4_fd_error.pdf')
print('  figures/q1_summary_bar.pdf')
print('  figures/q5_penalty_landscape.pdf')

# ================================================================
# Figure 5: Comprehensive Rosenbrock convergence comparison
# All methods from all 6 questions on Benchmark C
# ================================================================
np.random.seed(42)

def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def grad_rosenbrock(x):
    return np.array([
        -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
        200*(x[1]-x[0]**2)
    ])

def hess_rosenbrock(x):
    return np.array([
        [2 + 1200*x[0]**2 - 400*x[1], -400*x[0]],
        [-400*x[0], 200]
    ])

x0 = np.array([-1.0, 1.0])
n_gd = 120

# GD
def gd_rosen(x0, alpha, n):
    x = x0.copy(); hist = [rosenbrock(x)]
    for _ in range(n):
        x = x - alpha * grad_rosenbrock(x)
        hist.append(rosenbrock(x))
    return np.array(hist)

# Heavy Ball
def heavyball_rosen(x0, alpha, beta, n):
    x = x0.copy(); z = np.zeros(2); hist = [rosenbrock(x)]
    for _ in range(n):
        z = beta*z + alpha*grad_rosenbrock(x)
        x = x - z
        hist.append(rosenbrock(x))
    return np.array(hist)

# Nesterov
def nesterov_rosen(x0, alpha, beta_max, n):
    x = x0.copy(); z = np.zeros(2); hist = [rosenbrock(x)]
    for k in range(1, n+1):
        bk = min((k-1)/(k+2), beta_max)
        lookahead = x + bk*z
        g = grad_rosenbrock(lookahead)
        z = bk*z - alpha*g
        x = x + z
        hist.append(rosenbrock(x))
    return np.array(hist)

# Polyak (correct f*=0)
def polyak_rosen(x0, n):
    x = x0.copy(); hist = [rosenbrock(x)]
    for _ in range(n):
        g = grad_rosenbrock(x)
        f = rosenbrock(x)
        alpha_k = f / (np.dot(g,g) + 1e-3)
        x = x - alpha_k*g
        hist.append(rosenbrock(x))
    return np.array(hist)

# Newton (damped)
def newton_rosen(x0, alpha, n):
    x = x0.copy(); hist = [rosenbrock(x)]
    for _ in range(n):
        g = grad_rosenbrock(x)
        H = hess_rosenbrock(x) + 1e-8*np.eye(2)
        try:
            p = np.linalg.solve(H, g)
        except:
            p = g
        x = x - alpha*p
        hist.append(rosenbrock(x))
    return np.array(hist)

# Adagrad
def adagrad_rosen(x0, alpha0, n):
    x = x0.copy(); G = np.zeros(2); hist = [rosenbrock(x)]
    for _ in range(n):
        g = grad_rosenbrock(x)
        G += g**2
        x = x - alpha0/(np.sqrt(G)+1e-5)*g
        hist.append(rosenbrock(x))
    return np.array(hist)

f_gd = gd_rosen(x0, 0.0012, n_gd)
f_hb = heavyball_rosen(x0, 0.0008, 0.86, n_gd)
f_polyak = polyak_rosen(x0, n_gd)
f_adagrad = adagrad_rosen(x0, 0.45, n_gd)
f_nesterov = nesterov_rosen(x0, 0.0007, 0.90, 150)
f_newton = newton_rosen(x0, 0.22, 20)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: convergence curves
ax = axes[0]
iters_gd = np.arange(len(f_gd))
iters_nest = np.arange(len(f_nesterov))
iters_newton = np.arange(len(f_newton))

ax.semilogy(iters_gd, f_gd+1e-16, 'k-', lw=2, label=f'GD (f={f_gd[-1]:.3f})')
ax.semilogy(iters_gd, f_hb+1e-16, 'b--', lw=2, label=f'Heavy Ball (f={f_hb[-1]:.3f})')
ax.semilogy(iters_gd, f_adagrad+1e-16, 'orange', lw=2, label=f'Adagrad (f={f_adagrad[-1]:.3f})')
ax.semilogy(iters_gd, f_polyak+1e-16, 'g-', lw=2, label=f'Polyak (f={f_polyak[-1]:.4f})')
ax.semilogy(iters_nest, f_nesterov+1e-16, 'r-.', lw=2, label=f'Nesterov (f={f_nesterov[-1]:.3f})')
ax.semilogy(iters_newton, f_newton+1e-16, 'm:', lw=2.5, ms=8, marker='o',
            markevery=5, label=f'Newton (f={f_newton[-1]:.3f})')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'$f(x_k)$ (log scale)')
ax.set_title('Benchmark C (Rosenbrock): All Methods Compared')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

# Right: table of final values
methods_names = ['GD (Q1)', 'Adagrad (Q1)', 'Heavy Ball (Q1)', 'Polyak (Q1)',
                 'Nesterov (Q2)', 'Newton (Q3)']
final_vals = [f_gd[-1], f_adagrad[-1], f_hb[-1], f_polyak[-1], f_nesterov[-1], f_newton[-1]]
clrs2 = ['black', 'orange', 'blue', 'green', 'red', 'purple']
budgets = [120, 120, 120, 120, 150, 20]

ax2 = axes[1]
positions = np.arange(len(methods_names))
bars = ax2.bar(positions, np.clip(final_vals, 0, 4.0), color=clrs2, alpha=0.8, edgecolor='white')
best_idx = np.argmin(final_vals)
bars[best_idx].set_edgecolor('gold'); bars[best_idx].set_linewidth(2.5)
ax2.set_xticks(positions)
ax2.set_xticklabels(methods_names, rotation=20, fontsize=8)
ax2.set_ylabel('Final Objective Value')
ax2.set_title('Final $f$ on Rosenbrock\n(gold = best; budget shown)')
ax2.set_ylim(0, 4.2)
ax2.grid(True, alpha=0.3, axis='y')
for bar, val, bud in zip(bars, final_vals, budgets):
    disp = f'{val:.3f}\n({bud}it)' if val < 4.0 else f'>{4.0:.1f}\n({bud}it)'
    ax2.text(bar.get_x() + bar.get_width()/2., min(val,4.0) + 0.05, disp,
             ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig('figures/all_methods_rosenbrock.pdf', bbox_inches='tight')
plt.close()
print('Saved figures/all_methods_rosenbrock.pdf')

# ================================================================
# Figure 6: FW duality gap g_k vs iteration (boundary case)
# Confirms g_k >= f(x_k) - f* (valid certificate) and O(1/k) rate
# ================================================================
np.random.seed(42)

def f_q6_boundary(x):
    return x[0]**2 + x[1]**2

def grad_q6_boundary(x):
    return 2.0 * x

bounds_q6 = [(0.5, 5.0), (-5.0, 10.0)]
f_star_boundary = 0.25  # constrained min at (0.5, 0)

x0_fw = np.array([3.0, 3.0])
beta_fw = 0.93
n_fw = 141

x = x0_fw.copy()
iters_fw, f_vals_fw, gap_vals_fw = [], [], []

for k in range(n_fw):
    iters_fw.append(k)
    fk = f_q6_boundary(x)
    f_vals_fw.append(fk)
    g = grad_q6_boundary(x)
    # LMO over box
    z = np.array([bounds_q6[i][0] if g[i] > 0 else bounds_q6[i][1]
                  for i in range(len(g))])
    fw_gap = float(np.dot(g, x - z))  # g_k = grad f(x_k)^T (x_k - z_k)
    gap_vals_fw.append(fw_gap)
    x = beta_fw * x + (1 - beta_fw) * z

iters_fw = np.array(iters_fw, dtype=float)
f_vals_fw = np.array(f_vals_fw)
gap_vals_fw = np.array(gap_vals_fw)
subopt_fw = f_vals_fw - f_star_boundary  # f(x_k) - f*

# Theoretical O(1/k) bound: 2*L*C^2 / (k+2)
L_fw = 2.0
C2_fw = (5.0 - 0.5)**2 + (10.0 - (-5.0))**2  # 20.25 + 225 = 245.25
bound_fw = 2.0 * L_fw * C2_fw / (iters_fw + 2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.semilogy(iters_fw, subopt_fw + 1e-16, 'b-', lw=2, label=r'$f(x_k) - f^*$ (actual suboptimality)')
ax.semilogy(iters_fw, gap_vals_fw + 1e-16, 'r--', lw=2, label=r'FW gap $g_k = \nabla f(x_k)^\top(x_k - z_k)$')
ax.semilogy(iters_fw[1:], bound_fw[1:], 'k:', lw=1.8, label=r'Theoretical bound $2LC^2/(k+2) = 980/(k+2)$')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Value (log scale)')
ax.set_title(r'Q6: FW Duality Gap vs Suboptimality ($\beta=0.93$, boundary case)')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.text(0.5, 0.5, r'$f(x_k)-f^* \leq g_k$ always holds',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

# Right panel: ratio g_k / (f(x_k) - f*) -- should be >= 1
ax2 = axes[1]
ratio = gap_vals_fw / (subopt_fw + 1e-16)
ax2.plot(iters_fw, ratio, 'g-', lw=2, label=r'$g_k \,/\, (f(x_k) - f^*)$')
ax2.axhline(1.0, color='red', lw=1.5, linestyle='--', label='Lower bound = 1 (certificate validity)')
ax2.set_xlabel('Iteration $k$')
ax2.set_ylabel(r'Ratio $g_k / (f(x_k) - f^*)$')
ax2.set_title(r'Q6: Gap $g_k$ as Duality Certificate (ratio $\geq 1$ always)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 5)

plt.tight_layout()
plt.savefig('figures/q6_fw_gap.pdf', bbox_inches='tight')
plt.close()
print('Saved figures/q6_fw_gap.pdf')
