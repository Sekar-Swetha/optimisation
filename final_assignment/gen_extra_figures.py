"""
Generate additional figures for the improved report.
Produces two new figures not in the original set:
  1. cross_comparison_rosenbrock.pdf  - all best methods vs Rosenbrock
  2. newton_convergence_rate.pdf      - log ||x_k - x*|| to illustrate quadratic convergence
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ----- Benchmark definitions (identical to main code) -----
def loss_C(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_C(x):
    g1 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    g2 = 200*(x[1] - x[0]**2)
    return np.array([g1, g2])

def hessian_C(x):
    h11 = 2 + 1200*x[0]**2 - 400*x[1]
    h12 = -400*x[0]
    return np.array([[h11, h12], [h12, 200]])

x0_C = np.array([-1.0, 1.0])
x_star_C = np.array([1.0, 1.0])

# ----- Optimiser implementations -----
def gradient_descent(grad_fn, loss_fn, x0, alpha, n_iters):
    x = x0.copy().astype(float)
    f_hist, x_hist = [loss_fn(x)], [x.copy()]
    for _ in range(n_iters):
        x = x - alpha * grad_fn(x)
        f_hist.append(loss_fn(x))
        x_hist.append(x.copy())
    return np.array(f_hist), np.array(x_hist)

def polyak_step(grad_fn, loss_fn, x0, f_star, eps, n_iters):
    x = x0.copy().astype(float)
    f_hist, x_hist = [loss_fn(x)], [x.copy()]
    for _ in range(n_iters):
        g = grad_fn(x)
        alpha_k = (loss_fn(x) - f_star) / (np.dot(g, g) + eps)
        x = x - alpha_k * g
        f_hist.append(loss_fn(x))
        x_hist.append(x.copy())
    return np.array(f_hist), np.array(x_hist)

def adagrad(grad_fn, loss_fn, x0, alpha0, eps, n_iters):
    x = x0.copy().astype(float)
    G = np.zeros_like(x)
    f_hist, x_hist = [loss_fn(x)], [x.copy()]
    for _ in range(n_iters):
        g = grad_fn(x)
        G += g**2
        x = x - alpha0 / (np.sqrt(G) + eps) * g
        f_hist.append(loss_fn(x))
        x_hist.append(x.copy())
    return np.array(f_hist), np.array(x_hist)

def heavy_ball(grad_fn, loss_fn, x0, alpha, beta, n_iters):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    f_hist, x_hist = [loss_fn(x)], [x.copy()]
    for _ in range(n_iters):
        g = grad_fn(x)
        z = beta * z + alpha * g
        x = x - z
        f_hist.append(loss_fn(x))
        x_hist.append(x.copy())
    return np.array(f_hist), np.array(x_hist)

def nesterov_momentum(grad_fn, loss_fn, x0, alpha, beta_max, n_iters):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    f_hist, x_hist = [loss_fn(x)], [x.copy()]
    for k in range(1, n_iters + 1):
        beta_k = min((k - 1) / (k + 2), beta_max)
        lookahead = x + beta_k * z
        g = grad_fn(lookahead)
        z = beta_k * z - alpha * g
        x = x + z
        f_hist.append(loss_fn(x))
        x_hist.append(x.copy())
    return np.array(f_hist), np.array(x_hist)

def adam_optimiser(grad_fn, loss_fn, x0, alpha, beta1, beta2, eps, n_iters):
    x = x0.copy().astype(float)
    m_vec, v_vec = np.zeros_like(x), np.zeros_like(x)
    f_hist, x_hist = [loss_fn(x)], [x.copy()]
    for t in range(1, n_iters + 1):
        g = grad_fn(x)
        m_vec = beta1 * m_vec + (1 - beta1) * g
        v_vec = beta2 * v_vec + (1 - beta2) * g**2
        m_hat = m_vec / (1 - beta1**t)
        v_hat = v_vec / (1 - beta2**t)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        f_hist.append(loss_fn(x))
        x_hist.append(x.copy())
    return np.array(f_hist), np.array(x_hist)

def newtons_method(grad_fn, hess_fn, loss_fn, x0, alpha, n_iters, damping=1e-8):
    x = x0.copy().astype(float)
    f_hist, x_hist = [loss_fn(x)], [x.copy()]
    for _ in range(n_iters):
        g = grad_fn(x)
        H = hess_fn(x) if callable(hess_fn) else hess_fn
        H_reg = H + damping * np.eye(len(x))
        try:
            p = np.linalg.solve(H_reg, g)
        except np.linalg.LinAlgError:
            p = g
        x = x - alpha * p
        f_hist.append(loss_fn(x))
        x_hist.append(x.copy())
    return np.array(f_hist), np.array(x_hist)

# ----- Run all methods on Rosenbrock -----
n_main = 120
gd_f, gd_x   = gradient_descent(grad_C, loss_C, x0_C, 0.0012, n_main)
pol_f, pol_x  = polyak_step(grad_C, loss_C, x0_C, 0, 1e-3, n_main)
ada_f, ada_x  = adagrad(grad_C, loss_C, x0_C, 0.45, 1e-5, n_main)
hb_f,  hb_x   = heavy_ball(grad_C, loss_C, x0_C, 0.0008, 0.86, n_main)
nes_f, nes_x  = nesterov_momentum(grad_C, loss_C, x0_C, 0.0007, 0.90, n_main)
adm_f, adm_x  = adam_optimiser(grad_C, loss_C, x0_C, 0.006, 0.80, 0.999, 1e-8, n_main)
nwt_f, nwt_x  = newtons_method(grad_C, hessian_C, loss_C, x0_C, 0.22, 20)

# ============================================================
# FIGURE 1: Cross-question Rosenbrock comparison (objective value)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(gd_f,  'k--',  lw=1.5, label='GD (baseline, $\\alpha=0.0012$)')
ax.semilogy(pol_f, color='purple', lw=2, label='Polyak ($f^*=0$, $\\varepsilon=10^{-3}$)')
ax.semilogy(ada_f, color='tab:orange', lw=2, label='Adagrad ($\\alpha_0=0.45$)')
ax.semilogy(hb_f,  color='tab:red',    lw=2, label='Heavy Ball ($\\alpha=0.0008$, $\\beta=0.86$)')
ax.semilogy(nes_f, color='tab:blue',   lw=2, label='Nesterov ($\\alpha=0.0007$, $\\beta_{\\max}=0.90$)')
ax.semilogy(adm_f, color='tab:cyan',   lw=2, label='Adam ($\\alpha=0.006$, $\\beta_1=0.80$)')
# Newton: only 20 iters, mark start of trajectory
ax.semilogy(range(0, len(nwt_f)), nwt_f, 'g-o', lw=2.5, ms=5,
            label="Newton ($\\alpha=0.22$, 20 iters)")
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('$f(x_k)$ (log scale)', fontsize=12)
ax.set_title('Cross-Method Comparison on Rosenbrock $f(x)=(1-x_1)^2+100(x_2-x_1^2)^2$',
             fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, n_main])
plt.tight_layout()
plt.savefig('figures/cross_comparison_rosenbrock.pdf', bbox_inches='tight')
plt.close()
print("Saved: cross_comparison_rosenbrock.pdf")

# ============================================================
# FIGURE 2: Newton quadratic vs GD linear convergence rate
# (log distance to optimum vs iteration, log-log to reveal order)
# ============================================================
# Use Benchmark B where x* is analytically known
def loss_B(x):
    return (x[0] - 1)**2 + 5*(x[1] - 2)**2 + np.sin(x[0])

def grad_B(x):
    return np.array([2*(x[0] - 1) + np.cos(x[0]), 10*(x[1] - 2)])

def hessian_B(x):
    return np.array([[2 - np.sin(x[0]), 0], [0, 10]])

x0_B = np.array([-1.0, 4.0])
x_star_B = np.array([0.582438, 2.0])

gd_B_f,  gd_B_x  = gradient_descent(grad_B, loss_B, x0_B, 0.06, 60)
nwt_B_f, nwt_B_x = newtons_method(grad_B, hessian_B, loss_B, x0_B, 0.85, 20)

dist_gd  = np.array([np.linalg.norm(x - x_star_B) for x in gd_B_x])
dist_nwt = np.array([np.linalg.norm(x - x_star_B) for x in nwt_B_x])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: linear vs log-scale (distance to optimum)
ax = axes[0]
ax.semilogy(dist_gd,  'b-o', ms=4, lw=1.8, label='GD ($\\alpha=0.06$)')
ax.semilogy(dist_nwt, 'r-s', ms=6, lw=2.0, label="Newton ($\\alpha=0.85$)")
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('$\\|x_k - x^*\\|$ (log scale)', fontsize=12)
ax.set_title('Distance to Optimum vs Iteration\n(Benchmark B, semi-log scale)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right: log-log scale to distinguish linear vs quadratic convergence
ax = axes[1]
# Only plot where distance > machine epsilon
eps_floor = 1e-14
gd_valid  = dist_gd[dist_gd > eps_floor]
nwt_valid = dist_nwt[dist_nwt > eps_floor]
ax.loglog(np.arange(1, len(gd_valid)+1),  gd_valid,  'b-o', ms=4, lw=1.8,
          label='GD (linear rate)')
ax.loglog(np.arange(1, len(nwt_valid)+1), nwt_valid, 'r-s', ms=6, lw=2.0,
          label='Newton (quadratic rate)')
# Reference lines
k_ref = np.array([1, 60])
ax.loglog(k_ref, 5 * k_ref**(-1.0), 'b:', lw=1.2, label='$O(1/k)$ reference')
ax.loglog(k_ref[:2], [0.5, 0.5 * 0.001], 'r:', lw=1.2, label='$O(c^{2^k})$ reference')
ax.set_xlabel('Iteration (log scale)', fontsize=12)
ax.set_ylabel('$\\|x_k - x^*\\|$ (log scale)', fontsize=12)
ax.set_title('Log-Log: Linear vs Quadratic Convergence\n(Benchmark B)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle("Q3: Newton's Method vs GD — Convergence Rate Comparison", fontsize=13)
plt.tight_layout()
plt.savefig('figures/newton_convergence_rate.pdf', bbox_inches='tight')
plt.close()
print("Saved: newton_convergence_rate.pdf")

print("All additional figures generated.")
