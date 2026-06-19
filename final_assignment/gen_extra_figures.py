"""Generate additional figures for the improved report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

# ── Benchmark setup ──────────────────────────────────────────────────────────
m = 1000
X_A = np.random.randn(m, 2)
theta_true = np.array([3.0, 4.0])
eps_noise  = np.random.randn(m)
y_A = X_A @ theta_true + eps_noise
H_A = X_A.T @ X_A / m

loss_A  = lambda t: 0.5/m * np.sum((X_A @ t - y_A)**2)
grad_A  = lambda t: X_A.T @ (X_A @ t - y_A) / m
theta0_A = np.array([0.0, 0.0])

loss_B  = lambda x: (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
grad_B  = lambda x: np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])
x0_B    = np.array([-1.0, 4.0])

loss_C  = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
grad_C  = lambda x: np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
x0_C    = np.array([-1.0, 1.0])

fstar_A = 0.4826   # approx noise floor
fstar_B = 0.7244
fstar_C = 0.0

def gd_run(grad_fn, loss_fn, x0, alpha, n):
    x = x0.copy().astype(float)
    hist = [loss_fn(x)]
    for _ in range(n):
        x -= alpha * grad_fn(x)
        hist.append(loss_fn(x))
    return np.array(hist)

# ── Figure 1: GD convergence vs theoretical linear rate (all benchmarks) ─────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

configs = [
    ('A', grad_A, loss_A, theta0_A, 0.08, fstar_A,
     r'$\kappa \approx 1.12$, $L \approx 1.03$', 120,
     'Benchmark A\n(Linear Regression)'),
    ('B', grad_B, loss_B, x0_B, 0.06, fstar_B,
     r'$\kappa \approx 3.5$--$6.9$, $L = 10$', 150,
     'Benchmark B\n(Toy Neural Network)'),
    ('C', grad_C, loss_C, x0_C, 0.0012, fstar_C,
     r'$\kappa \approx 2508$, $L \approx 1002$', 150,
     'Benchmark C\n(Rosenbrock)'),
]

# Theoretical linear rate rho = 1 - alpha*mu (where mu is smallest eigenvalue)
# For A: mu=0.925, L=1.033, alpha=0.08 → rho = (1 - 0.08*0.925) = 0.926
# For B: mu=1 (smallest eigenvalue at minimum is 1.45, but varies; use 1.0), alpha=0.06 → rho ~ 0.94
# For C: alpha=0.0012, L≈1002, mu very small near start, use L-based rho: rho = 1 - 1/(2000)

rho_theory = [
    (1 - 0.08 * 0.925),   # A: alpha * mu = 0.08 * 0.925
    (1 - 0.06 * 1.45),    # B: alpha * mu = 0.06 * 1.45
    (1 - 0.0012 * 0.4),   # C: alpha * mu = 0.0012 * 0.4
]

for ax, (bname, gf, lf, x0, alpha, fstar, kappa_str, n, title), rho in zip(axes, configs, rho_theory):
    hist = gd_run(gf, lf, x0, alpha, n)
    excess = np.maximum(hist - fstar, 1e-12)
    k = np.arange(len(excess))

    ax.semilogy(k, excess, 'b-', lw=2, label='GD empirical')

    # Theoretical bound: (1-mu/L)^k * [f(x0) - f*]
    theory = excess[0] * rho**k
    ax.semilogy(k, theory, 'r--', lw=1.5, alpha=0.8,
                label=fr'$\rho^k$, $\rho={rho:.3f}$')

    ax.set_xlabel('Iteration $k$', fontsize=11)
    ax.set_ylabel(r'$f(x_k) - f^{\star}$', fontsize=11)
    ax.set_title(title + f'\n{kappa_str}', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, n])

fig.suptitle('Gradient Descent: Empirical vs Theoretical Linear Convergence Rate',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figures/q_gd_linear_rate.pdf', bbox_inches='tight')
plt.close()
print("Saved q_gd_linear_rate.pdf")


# ── Figure 2: Nesterov momentum schedule beta_k vs fixed Heavy Ball beta ───────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

k_vals = np.arange(1, 151)
beta_k = (k_vals - 1) / (k_vals + 2)   # Nesterov schedule, unbounded

ax = axes[0]
ax.plot(k_vals, np.minimum(beta_k, 0.90), 'b-', lw=2,
        label=r'Nesterov: $\beta_k=\min\!\left(\frac{k-1}{k+2},\,0.90\right)$')
ax.axhline(0.90, color='orange', lw=1.5, ls='--', label=r'Heavy Ball: $\beta=0.90$ (fixed)')
ax.axhline(0.86, color='green', lw=1.5, ls=':', label=r'Heavy Ball: $\beta=0.86$ (C)')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel(r'Momentum $\beta_k$', fontsize=11)
ax.set_title('Momentum Schedule Comparison', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.05, 1.0])

# Show the convergence improvement of increasing beta_k on Benchmark C
ax2 = axes[1]

def nesterov_run(gf, lf, x0, alpha, beta_max, n):
    x = x0.copy().astype(float); z = np.zeros_like(x)
    hist = [lf(x)]
    for k in range(1, n+1):
        bk = min((k-1)/(k+2), beta_max)
        la = x + bk*z; g = gf(la)
        z = bk*z - alpha*g; x = x+z
        hist.append(lf(x))
    return np.array(hist)

def hb_run(gf, lf, x0, alpha, beta, n):
    x = x0.copy().astype(float); z = np.zeros_like(x)
    hist = [lf(x)]
    for _ in range(n):
        z = beta*z + alpha*gf(x); x = x-z
        hist.append(lf(x))
    return np.array(hist)

h_nest = nesterov_run(grad_C, loss_C, x0_C, 0.0007, 0.90, 150)
h_hb   = hb_run(grad_C, loss_C, x0_C, 0.0008, 0.86, 150)
h_gd   = gd_run(grad_C, loss_C, x0_C, 0.0012, 150)

k = np.arange(len(h_nest))
ax2.semilogy(k, np.maximum(h_nest, 1e-12), 'b-', lw=2, label=f'Nesterov $\\beta_k$, final $f={h_nest[-1]:.3f}$')
ax2.semilogy(np.arange(len(h_hb)), np.maximum(h_hb, 1e-12), 'g-', lw=2,
             label=f'Heavy Ball $\\beta=0.86$, final $f={h_hb[-1]:.3f}$')
ax2.semilogy(np.arange(len(h_gd)), np.maximum(h_gd, 1e-12), 'k--', lw=1.5,
             label=f'GD, final $f={h_gd[-1]:.3f}$')
ax2.set_xlabel('Iteration $k$', fontsize=11)
ax2.set_ylabel(r'$f(x_k)$', fontsize=11)
ax2.set_title('Convergence on Benchmark C (Rosenbrock)', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle('Nesterov Increasing-Momentum vs Heavy Ball Fixed-Momentum', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q2_momentum_comparison.pdf', bbox_inches='tight')
plt.close()
print("Saved q2_momentum_comparison.pdf")


# ── Figure 3: FD gradient error: direction corruption vs delta ────────────────
x_test = np.array([0.0, 2.5])   # point on Rosenbrock where gradients are large
g_exact = grad_C(x_test)

deltas = np.logspace(-8, 1, 200)
angle_errors = []
magnitude_errors = []

for delta in deltas:
    g_fd = np.array([
        (loss_C(x_test + delta*np.array([1,0])) - loss_C(x_test)) / delta,
        (loss_C(x_test + delta*np.array([0,1])) - loss_C(x_test)) / delta
    ])
    cos_sim = np.dot(g_exact, g_fd) / (np.linalg.norm(g_exact) * np.linalg.norm(g_fd) + 1e-20)
    cos_sim = np.clip(cos_sim, -1, 1)
    angle_errors.append(np.degrees(np.arccos(cos_sim)))
    magnitude_errors.append(abs(np.linalg.norm(g_fd) - np.linalg.norm(g_exact)) / np.linalg.norm(g_exact))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.semilogx(deltas, angle_errors, 'b-', lw=2)
ax.axvline(0.05, color='g', lw=2, ls='--', label=r'$\delta = 0.05$ (good)')
ax.axvline(0.8, color='r', lw=2, ls='--', label=r'$\delta = 0.8$ (poor)')
ax.axvline(1e-8, color='purple', lw=2, ls=':', label=r'$\delta_{\rm opt} \approx 10^{-8}$')
ax.set_xlabel(r'Perturbation $\delta$', fontsize=11)
ax.set_ylabel('Gradient direction error (degrees)', fontsize=11)
ax.set_title('FD Gradient Direction Error on Rosenbrock\n(at $x=(0,2.5)$)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])

ax = axes[1]
ax.loglog(deltas, magnitude_errors, 'b-', lw=2)
ax.axvline(0.05, color='g', lw=2, ls='--', label=r'$\delta = 0.05$')
ax.axvline(0.8, color='r', lw=2, ls='--', label=r'$\delta = 0.8$')
ax.axvline(1e-8, color='purple', lw=2, ls=':', label=r'$\delta_{\rm opt}$')
ax.set_xlabel(r'Perturbation $\delta$', fontsize=11)
ax.set_ylabel('Relative gradient magnitude error', fontsize=11)
ax.set_title('FD Gradient Magnitude Error on Rosenbrock', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle('Finite Difference Gradient Quality vs Perturbation Size $\\delta$',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q4_fd_error_analysis.pdf', bbox_inches='tight')
plt.close()
print("Saved q4_fd_error_analysis.pdf")

print("All extra figures generated successfully.")
