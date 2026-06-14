"""
Generate third batch of additional figures:
  1. convergence_order_B.pdf  -- log(f_k - f*) vs log(k) to show O(1/k^2) vs O(1/k)
  2. penalty_vs_lambda.pdf    -- penalty method sensitivity to lambda
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
# Benchmark B definitions
# ============================================================
def loss_B(x): return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def grad_B(x): return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

x0_B = np.array([-1.0, 4.0])
# True minimum (solved numerically)
x1_star = 0.582438
f_star_B = (x1_star-1)**2 + 5*(2-2)**2 + np.sin(x1_star)  # ~0.7244

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

def adam(grad_fn, loss_fn, x0, alpha, b1, b2, eps, n_iters):
    x = x0.copy().astype(float)
    m = np.zeros_like(x); v = np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for t in range(1, n_iters+1):
        g = grad_fn(x); m = b1*m+(1-b1)*g; v = b2*v+(1-b2)*g**2
        mh = m/(1-b1**t); vh = v/(1-b2**t)
        x = x - alpha*mh/(np.sqrt(vh)+eps)
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

def heavy_ball(grad_fn, loss_fn, x0, alpha, beta, n_iters):
    x = x0.copy().astype(float); z = np.zeros_like(x)
    f_hist = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x); z = beta*z + alpha*g; x = x - z
        f_hist.append(loss_fn(x))
    return np.array(f_hist)

# ============================================================
# FIGURE 1: Convergence order on Benchmark B
# log(f_k - f*) vs iteration, and log(f_k - f*) vs log(k)
# ============================================================
n_iters = 300
gd_f   = gradient_descent(grad_B, loss_B, x0_B, 0.06, n_iters)
nes_f  = nesterov_momentum(grad_B, loss_B, x0_B, 0.035, 0.92, n_iters)
adm_f  = adam(grad_B, loss_B, x0_B, 0.08, 0.80, 0.999, 1e-8, n_iters)
hb_f   = heavy_ball(grad_B, loss_B, x0_B, 0.035, 0.90, n_iters)

# Compute excess objective
floor = f_star_B
eps_floor = 1e-14
gd_excess   = np.maximum(gd_f - floor, eps_floor)
nes_excess  = np.maximum(nes_f - floor, eps_floor)
adm_excess  = np.maximum(adm_f - floor, eps_floor)
hb_excess   = np.maximum(hb_f - floor, eps_floor)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: semi-log plot (iteration vs excess objective)
ax = axes[0]
ax.semilogy(gd_excess,  'k--', lw=1.5, label='GD ($\\alpha=0.06$)')
ax.semilogy(hb_excess,  'tab:red',  lw=1.5, label='Heavy Ball ($\\alpha=0.035$, $\\beta=0.90$)')
ax.semilogy(nes_excess, 'tab:blue', lw=2,   label='Nesterov ($\\alpha=0.035$, $\\beta_{\\max}=0.92$)')
ax.semilogy(adm_excess, 'tab:green', lw=2,  label='Adam ($\\alpha=0.08$)')
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)', fontsize=12)
ax.set_title('Excess Objective vs Iteration\n(Benchmark B, $f^\\star = 0.7244$)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xlim([0, 300])

# Right: log-log plot (log k vs log excess) to read convergence order
ax = axes[1]
k_arr = np.arange(1, n_iters + 1)
# Only plot from iteration 10 onwards (avoid transient)
start = 10
ax.loglog(k_arr[start:], gd_excess[start+1:],  'k--', lw=1.5, label='GD')
ax.loglog(k_arr[start:], hb_excess[start+1:],  'tab:red',  lw=1.5, label='Heavy Ball')
ax.loglog(k_arr[start:], nes_excess[start+1:], 'tab:blue', lw=2,   label='Nesterov')
ax.loglog(k_arr[start:], adm_excess[start+1:], 'tab:green', lw=2,  label='Adam')
# Reference lines
k_ref = np.array([20, 300])
ax.loglog(k_ref, 50 * k_ref**(-1.0), 'gray', lw=1, linestyle=':', label='$O(1/k)$ slope')
ax.loglog(k_ref, 500 * k_ref**(-2.0), 'gray', lw=1, linestyle='--', label='$O(1/k^2)$ slope')
ax.set_xlabel('Iteration $k$ (log scale)', fontsize=12)
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)', fontsize=12)
ax.set_title('Log-Log Plot: Convergence Order\n(Slope $= -p$ means $O(1/k^p)$ rate)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

plt.suptitle('Q2: Convergence Rate Analysis on Benchmark B', fontsize=12)
plt.tight_layout()
plt.savefig('figures/convergence_order_B.pdf', bbox_inches='tight')
plt.close()
print("Saved: convergence_order_B.pdf")

# ============================================================
# FIGURE 2: Penalty method sensitivity to lambda
# on Benchmark B (Q5), showing final objective vs lambda
# ============================================================
def loss_B(x): return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def grad_B(x): return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

def penalty_gd(x0, alpha, lam, n_iters):
    x = x0.copy().astype(float)
    f_hist = [loss_B(x)]
    viol_hist = [max(0, 0.5 - x[0])]
    for _ in range(n_iters):
        g = grad_B(x).copy()
        if x[0] < 0.5:
            g[0] -= lam
        x = x - alpha * g
        f_hist.append(loss_B(x))
        viol_hist.append(max(0, 0.5 - x[0]))
    return np.array(f_hist), np.array(viol_hist)

x0_q5 = np.array([0.2, 4.0])
lambdas = np.logspace(-1.5, 1.5, 30)

final_f = []
final_viol = []
for lam in lambdas:
    # Use small step size to maintain stability for large lambda
    alpha = min(0.05, 1.8 / (10 + lam))
    fh, vh = penalty_gd(x0_q5, alpha, lam, 200)
    final_f.append(fh[-1])
    final_viol.append(vh[-1])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogx(lambdas, final_f, 'b-o', ms=5, lw=2)
ax.axhline(f_star_B, color='red', linestyle='--', lw=1.5, label=f'$f^\\star \\approx {f_star_B:.4f}$ (constrained min)')
ax.axvline(4.5, color='gray', linestyle=':', lw=1.5, label='$\\lambda = 4.5$ (Q5 large)')
ax.axvline(0.15, color='gray', linestyle='--', lw=1.5, label='$\\lambda = 0.15$ (Q5 small)')
ax.set_xlabel('Penalty weight $\\lambda$ (log scale)', fontsize=12)
ax.set_ylabel('Final $f(x)$ after 200 iterations')
ax.set_title('Penalty Method: Final Objective vs $\\lambda$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.loglog(lambdas, np.maximum(final_viol, 1e-10), 'r-o', ms=5, lw=2)
ax.axvline(4.5, color='gray', linestyle=':', lw=1.5, label='$\\lambda = 4.5$')
ax.axvline(0.15, color='gray', linestyle='--', lw=1.5, label='$\\lambda = 0.15$')
ax.set_xlabel('Penalty weight $\\lambda$ (log scale)', fontsize=12)
ax.set_ylabel('Final constraint violation $\\max(0, 0.5 - x_1)$', fontsize=12)
ax.set_title('Penalty Method: Final Constraint Violation vs $\\lambda$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')

plt.suptitle('Q5: Penalty Method Sensitivity Analysis (200 iterations)', fontsize=12)
plt.tight_layout()
plt.savefig('figures/penalty_sensitivity.pdf', bbox_inches='tight')
plt.close()
print("Saved: penalty_sensitivity.pdf")

print("All third-batch figures generated.")
