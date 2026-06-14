"""
Generate theoretical convergence bound comparison figure for Benchmark C (Rosenbrock).
Shows actual observed convergence alongside theoretical O(1/k), O(1/k^2) bounds.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

def loss_C(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def grad_C(x): return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
def hessian_C(x): return np.array([[2+1200*x[0]**2-400*x[1],-400*x[0]],[-400*x[0],200]])

x0 = np.array([-1.0, 1.0])
f0 = loss_C(x0)  # = 4.0
# Distance to optimum (1,1)
dist0 = np.linalg.norm(x0 - np.array([1.0, 1.0]))  # = 2.0
L = 1001.6  # smoothness constant at optimum
kappa = 2504.0  # condition number

# Run algorithms
def gd(n, a):
    x=x0.copy(); fh=[loss_C(x)]
    for _ in range(n): x=x-a*grad_C(x); fh.append(loss_C(x))
    return np.array(fh)

def nesterov(n, a, bmax):
    x=x0.copy(); z=np.zeros(2); fh=[loss_C(x)]
    for k in range(1,n+1):
        bk=min((k-1)/(k+2), bmax); la=x+bk*z; g=grad_C(la); z=bk*z-a*g; x=x+z; fh.append(loss_C(x))
    return np.array(fh)

def polyak(n, fs, eps):
    x=x0.copy(); fh=[loss_C(x)]
    for _ in range(n):
        g=grad_C(x); ak=(loss_C(x)-fs)/(np.dot(g,g)+eps); x=x-ak*g; fh.append(loss_C(x))
    return np.array(fh)

def newton_d(n, a):
    x=x0.copy(); fh=[loss_C(x)]
    for _ in range(n):
        g=grad_C(x); H=hessian_C(x)+1e-8*np.eye(2); x=x-a*np.linalg.solve(H,g); fh.append(loss_C(x))
    return np.array(fh)

N_max = 200
f_gd = gd(N_max, 0.0012)
f_nes = nesterov(N_max, 0.0007, 0.90)
f_poly = polyak(N_max, 0, 1e-3)
f_newt = newton_d(50, 0.22)

# Theoretical bounds (shifted to same start f0=4.0)
k_arr = np.arange(1, N_max+1)

# GD linear bound: f0 * (1 - 1/kappa)^k  (valid for convex quadratic, μ=0.4, L=1001.6)
# This underestimates since we don't have global strong convexity
# Use empirical per-step rate estimate from first 150 steps
rho_gd = (1 - 1/kappa)  # ≈ 0.9996
theory_gd = f0 * rho_gd**k_arr

# Nesterov O(1/k^2) bound: 2L||x0-x*||^2 / k^2 (requires alpha=1/L, convex)
# Scale to match starting value
theory_nes_raw = 2 * L * dist0**2 / k_arr**2
# But this bound starts very high (= 2*L*dist0^2 at k=1 ≈ 8013) so normalize to start at f0
# Actually the bound does not need to start at f0; for k=1: 2*1001.6*4/1 = 8013 >> f0
# Just show it as-is since it is a valid upper bound (and it's above f0 for small k)

# Polyak bound: ||x_k - x*||^2 ≤ ||x0-x*||^2 / (1 + mu*k * ||x0-x*||^2) for strongly convex
# For convex only: convergence rate in ||x||^2 is ||x0||^2/(k+1) approximately
# ||x_k - x*||^2 ≤ dist0^2 / (k+1) implies f - f* ≤ (L/2) dist0^2 / (k+1) = 2*L*dist0^2/(k+1)
# But this isn't the right bound for Polyak. The Polyak bound is on ||x||^2:
# ||x_{k+1}-x*||^2 ≤ ||x_k-x*||^2 - (f(x_k)-f*)^2/||g||^2
# For quadratic with mu-SC: ||x_k-x*||^2 ≤ (1-mu/L)^k ||x0-x*||^2
# Since f - f* ≤ (L/2)||x-x*||^2: f(x_k) - f* ≤ (L/2)(1-mu/L)^k dist0^2
# For Rosenbrock: mu≈0.4, L≈1001.6, rho = (1-0.4/1001.6) ≈ 0.9996 (essentially same as GD...)
# Polyak's theoretical rate for quadratics is actually the same as GD for SC functions
# The practical advantage comes from better step sizes near optimum

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: All methods + bounds
ax = axes[0]
ax.semilogy(f_gd[:N_max+1], 'k--', lw=1.5, label=f'GD observed (α=0.0012)', alpha=0.8)
ax.semilogy(range(N_max+1), f_nes[:N_max+1], 'r-', lw=2, label='Nesterov observed (α=0.0007)', alpha=0.9)
ax.semilogy(range(N_max+1), f_poly[:N_max+1], 'b-', lw=2, label='Polyak observed (f*=0, ε=1e-3)')
ax.semilogy(range(51), f_newt[:51], 'g-o', lw=2, ms=4, label='Newton observed (α=0.22)')

# Theoretical bounds
ax.semilogy(k_arr, theory_gd + 1e-8, 'k:', lw=1.5, alpha=0.6,
            label=f'GD theory: $f_0(1-1/\\kappa)^k$, $\\kappa$={kappa:.0f}')
ax.semilogy(k_arr, np.maximum(theory_nes_raw, 1e-8), 'r:', lw=1.5, alpha=0.6,
            label='Nesterov theory: $2L\\|x_0-x^*\\|^2/k^2$')
ax.axhline(0, color='gold', lw=1.5, ls='--', label='$f^*=0$')

ax.set_xlabel('Iteration')
ax.set_ylabel('Objective $f(x_k)$ (log scale)')
ax.set_title('Benchmark C: Observed vs Theoretical Convergence Rates')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 200)
ax.set_ylim(1e-4, 1e4)

# Panel 2: GD theory vs observed in detail (log-log to see linear convergence rate)
ax = axes[1]
k_plot = np.arange(1, 151)
f_gd_vals = f_gd[1:151]
f_nes_vals = f_nes[1:151]

ax.loglog(k_plot, f_gd_vals, 'k--', lw=2, label='GD observed')
ax.loglog(k_plot, f_nes_vals, 'r-', lw=2, label='Nesterov observed')
ax.loglog(k_plot, np.maximum(theory_nes_raw[:150], 1e-8), 'r:', lw=1.5, alpha=0.7,
          label='Nesterov $O(1/k^2)$ theory')
ax.loglog(k_plot, f0*np.ones_like(k_plot)*0.9996**k_plot, 'k:', lw=1.5, alpha=0.7,
          label='GD $(1-1/\\kappa)^k$ theory')

# Mark where theory becomes tighter than observation for Nesterov
# Find crossover
for i, (obs, thr) in enumerate(zip(f_nes_vals, theory_nes_raw[:150])):
    if thr < obs:
        ax.axvline(k_plot[i], color='r', ls='--', lw=1, alpha=0.5)
        ax.text(k_plot[i]+2, 1, f'k={k_plot[i]}\ntheory\nbecomes\ntight', fontsize=7, color='r')
        break

ax.set_xlabel('Iteration $k$ (log scale)')
ax.set_ylabel('Objective (log scale)')
ax.set_title('Log-log: GD vs Nesterov — theoretical rate comparison')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

fig.tight_layout()
fig.savefig('figures/q_theory_bounds.pdf', bbox_inches='tight')
plt.close()

# Print theory vs observed ratios at key iterations
print("GD theory vs observed:")
for k in [50, 100, 150]:
    pred = f0 * (1-1/kappa)**k
    obs = f_gd[k]
    print(f"  k={k}: theory={pred:.4f}, observed={obs:.4f}, ratio={obs/pred:.2f}")

print("\nNesterov theory vs observed:")
for k in [50, 100, 150]:
    pred = 2*L*dist0**2/k**2
    obs = f_nes[k]
    print(f"  k={k}: theory={pred:.4f}, observed={obs:.4f} (theory {'above' if pred>obs else 'below'})")

print(f"\nPolyak final (200 iters): f={f_poly[-1]:.6f}")
print(f"Newton final (50 iters): f={f_newt[-1]:.6f}")
print("Saved q_theory_bounds.pdf")
