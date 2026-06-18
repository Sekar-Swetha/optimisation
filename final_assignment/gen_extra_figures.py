"""
Extra figures for the final report — run from final_assignment/ directory.
Generates: q4_fd_error.pdf, q3_newton_rate.pdf, q_summary_comparison.pdf
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ----------------------------------------------------------------
# Benchmark definitions (mirror from main code)
# ----------------------------------------------------------------
def loss_B(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])

def grad_B(x):
    return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

def loss_C(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def grad_C(x):
    g1 = -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2)
    g2 = 200*(x[1]-x[0]**2)
    return np.array([g1, g2])

def hessian_C(x):
    h11 = 2 + 1200*x[0]**2 - 400*x[1]
    h12 = -400*x[0]
    return np.array([[h11, h12],[h12, 200]])

x0_B = np.array([-1.0, 4.0])
x0_C = np.array([-1.0, 1.0])

# ----------------------------------------------------------------
# Figure 1: Finite-difference truncation error vs delta
# ----------------------------------------------------------------
def exact_grad_B(x):
    return grad_B(x)

def fd_grad_B(x, delta):
    n = len(x)
    g = np.zeros(n)
    f0 = loss_B(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        g[i] = (loss_B(x + delta*ei) - f0) / delta
    return g

def central_fd_grad_B(x, delta):
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        g[i] = (loss_B(x + delta*ei) - loss_B(x - delta*ei)) / (2*delta)
    return g

x_test = np.array([0.5, 1.5])
exact = exact_grad_B(x_test)

deltas = np.logspace(-12, 0, 200)
err_fwd = []
err_cen = []
for d in deltas:
    g_fwd = fd_grad_B(x_test, d)
    g_cen = central_fd_grad_B(x_test, d)
    err_fwd.append(np.linalg.norm(g_fwd - exact))
    err_cen.append(np.linalg.norm(g_cen - exact))

err_fwd = np.array(err_fwd)
err_cen = np.array(err_cen)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(deltas, err_fwd, 'b-', lw=2, label='Forward FD: $O(\\delta)$ truncation')
ax.loglog(deltas, err_cen, 'r-', lw=2, label='Central FD: $O(\\delta^2)$ truncation')

# Mark theoretical optimal delta regions
ax.axvline(1e-8, color='blue', ls='--', alpha=0.7, label=r'Fwd optimal: $\delta \approx \sqrt{\varepsilon_{\rm mach}} \approx 10^{-8}$')
ax.axvline(1e-5, color='red', ls='--', alpha=0.7, label=r'Cen optimal: $\delta \approx \varepsilon_{\rm mach}^{1/3} \approx 10^{-5}$')

# Slope reference lines
d_ref = np.array([1e-4, 1e-2])
ax.loglog(d_ref, 3e-8 * d_ref / 1e-4, 'b:', alpha=0.5, label='Slope 1')
ax.loglog(d_ref, 3e-8 * (d_ref/1e-4)**2, 'r:', alpha=0.5, label='Slope 2')

ax.set_xlabel(r'Finite-difference step size $\delta$')
ax.set_ylabel(r'Gradient error $\|\hat{g} - g\|_2$')
ax.set_title('Q4: Finite-Difference Gradient Error vs Step Size\n(Benchmark B, evaluated at $x=(0.5,\,1.5)$)')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('figures/q4_fd_error.pdf', bbox_inches='tight')
plt.close()
print("Saved q4_fd_error.pdf")

# ----------------------------------------------------------------
# Figure 2: Newton's method — quadratic convergence demonstration
# ----------------------------------------------------------------
def newtons_method(grad_fn, hess_fn, loss_fn, x0, alpha, n_iters, damping=1e-8):
    x = x0.copy().astype(float)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        H = hess_fn(x) if callable(hess_fn) else hess_fn
        H_reg = H + damping * np.eye(len(x))
        try:
            p = np.linalg.solve(H_reg, g)
        except np.linalg.LinAlgError:
            p = g
        x = x - alpha * p
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

def gradient_descent(grad_fn, loss_fn, x0, alpha, n_iters):
    x = x0.copy().astype(float)
    x_hist, f_hist = [x.copy()], [loss_fn(x)]
    for _ in range(n_iters):
        x = x - alpha * grad_fn(x)
        x_hist.append(x.copy())
        f_hist.append(loss_fn(x))
    return np.array(x_hist), np.array(f_hist)

# Run Newton and GD on Rosenbrock
x_star_C = np.array([1.0, 1.0])
nh, nf = newtons_method(grad_C, hessian_C, loss_C, x0_C, 0.22, 40, damping=1e-8)
gh, gf = gradient_descent(grad_C, loss_C, x0_C, 0.001, 200)

dist_newton = np.linalg.norm(nh - x_star_C, axis=1)
dist_gd = np.linalg.norm(gh - x_star_C, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogy(dist_gd[:81], 'b-', lw=2, label='GD (linear rate)')
ax.semilogy(dist_newton[:41], 'r-', lw=2, label="Newton (quadratic rate)")
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\|x_k - x^*\|_2$ (log scale)')
ax.set_title("Q3: Distance to Optimum — Newton vs GD (Rosenbrock)")
ax.legend()
ax.grid(True, alpha=0.3)

# Log-log of residuals to show convergence order
ax = axes[1]
dist_n_pos = dist_newton[dist_newton > 1e-14]
its_n = np.arange(len(dist_n_pos))
ax.loglog(its_n[1:]+1, dist_n_pos[1:], 'r-o', ms=4, lw=2, label="Newton $\|x_k - x^*\|$")
# Show quadratic trend: if e_{k+1} ~ C e_k^2, then log e_{k+1} = log C + 2 log e_k
if len(dist_n_pos) > 5:
    # Fit line in log-log (residual vs previous residual)
    y = np.log10(dist_n_pos[2:])
    x_fit = np.log10(dist_n_pos[1:-1])
    # Linear fit
    slope, intercept = np.polyfit(x_fit, y, 1)
    x_range = np.linspace(x_fit.min(), x_fit.max(), 50)
    ax.loglog(10**x_range, 10**(slope*x_range + intercept), 'k--',
              label=f'Fit slope ≈ {slope:.2f} (ideal=2)')
ax.set_xlabel(r'Iteration $k$')
ax.set_ylabel(r'$\|x_k - x^*\|_2$')
ax.set_title("Q3: Newton Convergence Order (Log-Log)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('figures/q3_newton_rate.pdf', bbox_inches='tight')
plt.close()
print("Saved q3_newton_rate.pdf")

# ----------------------------------------------------------------
# Figure 3: KKT analysis for Q5 — penalty parameter effect on
#           constrained solution quality
# ----------------------------------------------------------------
# Re-run penalty GD for various lambda values and record final x1 position
def loss_B_fn(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def grad_B_fn(x):
    return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

def penalty_gd(x0, alpha, lam, n_iters):
    x = x0.copy().astype(float)
    f_hist = []
    x1_hist = []
    for _ in range(n_iters):
        g = grad_B_fn(x).copy()
        if x[0] < 0.5:
            g[0] -= lam
        x = x - alpha * g
        f_hist.append(loss_B_fn(x))
        x1_hist.append(x[0])
    return np.array(f_hist), np.array(x1_hist)

x0_q5 = np.array([0.2, 4.0])
lambdas = np.logspace(-2, 2, 40)
final_x1 = []
final_f = []
final_viol = []

for lam in lambdas:
    alpha_use = min(0.05, 0.15/lam)  # scale alpha down for large lambda
    alpha_use = max(alpha_use, 0.005)
    fh, x1h = penalty_gd(x0_q5, alpha_use, lam, 200)
    final_x1.append(x1h[-1])
    # Compute actual objective (not penalised)
    # Need last x; re-run to get x
    x = x0_q5.copy().astype(float)
    for _ in range(200):
        g = grad_B_fn(x).copy()
        if x[0] < 0.5:
            g[0] -= lam
        x = x - alpha_use * g
    final_x1[-1] = x[0]
    final_f.append(loss_B_fn(x))
    final_viol.append(max(0.0, 0.5 - x[0]))

final_x1 = np.array(final_x1)
final_f = np.array(final_f)
final_viol = np.array(final_viol)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogx(lambdas, final_viol, 'r-o', ms=4, lw=2)
ax.axhline(0, color='k', ls='--', alpha=0.5)
ax.set_xlabel(r'Penalty weight $\lambda$')
ax.set_ylabel(r'Constraint violation $\max(0,\; 0.5 - x_1)$')
ax.set_title('Q5: Constraint Violation vs Penalty Weight\n(after 200 iterations)')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogx(lambdas, final_f, 'b-o', ms=4, lw=2, label=r'Penalty solution $f(x_\lambda)$')
ax.axhline(0.7244, color='g', ls='--', lw=1.5, label=r'Constrained opt $f^* \approx 0.7244$')
ax.set_xlabel(r'Penalty weight $\lambda$')
ax.set_ylabel(r'Objective value $f(x)$')
ax.set_title('Q5: Objective Value vs Penalty Weight')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Q5: Effect of Penalty Parameter on Solution Quality', fontsize=12)
plt.tight_layout()
plt.savefig('figures/q5_penalty_analysis.pdf', bbox_inches='tight')
plt.close()
print("Saved q5_penalty_analysis.pdf")

# ----------------------------------------------------------------
# Figure 4: Frank–Wolfe duality gap demonstration
# ----------------------------------------------------------------
def loss_fw_interior(x):
    return (x[0]-1)**2 + (x[1]-5)**2

def grad_fw_interior(x):
    return np.array([2*(x[0]-1), 2*(x[1]-5)])

X_bounds = [(0.5, 5.0), (-5.0, 10.0)]

def fw_lp_box(grad, bounds):
    z = np.zeros(len(grad))
    for i in range(len(grad)):
        z[i] = bounds[i][0] if grad[i] > 0 else bounds[i][1]
    return z

def frank_wolfe_gap(grad_fn, loss_fn, x0, bounds, beta, n_iters):
    x = x0.copy().astype(float)
    f_hist, gap_hist = [loss_fn(x)], []
    for _ in range(n_iters):
        g = grad_fn(x)
        z = fw_lp_box(g, bounds)
        gap = np.dot(g, x - z)   # duality gap = <∇f, x - z>
        gap_hist.append(max(0.0, gap))
        x = beta * x + (1 - beta) * z
        f_hist.append(loss_fn(x))
    return np.array(f_hist), np.array(gap_hist)

x0_fw = np.array([1.0, 1.0])
fh_090, gh_090 = frank_wolfe_gap(grad_fw_interior, loss_fw_interior, x0_fw, X_bounds, 0.90, 200)
fh_0985, gh_0985 = frank_wolfe_gap(grad_fw_interior, loss_fw_interior, x0_fw, X_bounds, 0.985, 200)

# Theoretical O(1/k) bound starting from iteration 1
k = np.arange(1, 201)
C_ref_090 = gh_090[0]  # scale from first gap
C_ref_0985 = gh_0985[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogy(fh_090, 'b-', lw=2, label=r'$f(x_k)$, $\beta=0.90$')
ax.semilogy(fh_0985, 'r-', lw=2, label=r'$f(x_k)$, $\beta=0.985$')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Objective value (log scale)')
ax.set_title('Q6: Frank–Wolfe Objective Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(gh_090, 'b-', lw=2, label=r'Duality gap, $\beta=0.90$')
ax.semilogy(gh_0985, 'r-', lw=2, label=r'Duality gap, $\beta=0.985$')
# O(1/k) reference
ax.semilogy(k, C_ref_090 / k, 'b--', alpha=0.5, lw=1, label=r'$O(1/k)$ reference')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'Duality gap $\langle \nabla f(x_k),\; x_k - z_k \rangle$ (log)')
ax.set_title('Q6: Frank–Wolfe Duality Gap')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/q6_fw_duality_gap.pdf', bbox_inches='tight')
plt.close()
print("Saved q6_fw_duality_gap.pdf")

print("\nAll extra figures generated successfully.")
