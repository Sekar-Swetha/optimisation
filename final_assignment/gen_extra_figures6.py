"""
Generate sixth batch of additional figures:
  1. fw_gap.pdf  -- Frank-Wolfe duality gap g_k vs iteration for interior and boundary cases
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
# Frank-Wolfe gap analysis
# ============================================================
bounds = [(0.5, 5), (-5, 10)]
D = np.sqrt((5 - 0.5)**2 + (10 - (-5))**2)  # ~15.7

def fw_step(x, grad_fn, bounds, beta):
    g = grad_fn(x)
    # LP subproblem: argmin g^T z over box
    z = np.array([bounds[i][0] if g[i] > 0 else bounds[i][1] for i in range(len(g))])
    x_new = beta * x + (1 - beta) * z
    # FW gap: g^T (x - z)
    gap = np.dot(g, x - z)
    return x_new, z, gap

# Case 1: Interior optimum f(x) = (x1-1)^2 + (x2-5)^2, x*=(1,5) inside X
def loss_int(x): return (x[0]-1)**2 + (x[1]-5)**2
def grad_int(x): return np.array([2*(x[0]-1), 2*(x[1]-5)])
f_star_int = 0.0
L_int = 2.0   # largest eigenvalue of Hessian = 2

# Case 2: Boundary optimum f(x) = x1^2 + x2^2, x*=(0.5,0) on boundary
def loss_bnd(x): return x[0]**2 + x[1]**2
def grad_bnd(x): return np.array([2*x[0], 2*x[1]])
f_star_bnd = 0.5**2 + 0**2  # constrained min at (0.5, 0) = 0.25
L_bnd = 2.0

# Theoretical bound: f(x_k) - f* <= 2*L*D^2 / (k+2)
k_arr = np.arange(1, 141)
theory_int = 2 * L_int * D**2 / (k_arr + 2)
theory_bnd = 2 * L_bnd * D**2 / (k_arr + 2)

n_iters = 140

# Run interior case with two beta values
def run_fw(x0, loss_fn, grad_fn, bounds, beta, n_iters):
    x = x0.copy().astype(float)
    gaps, fvals = [], [loss_fn(x)]
    for _ in range(n_iters):
        x, z, gap = fw_step(x, grad_fn, bounds, beta)
        gaps.append(gap)
        fvals.append(loss_fn(x))
    return np.array(gaps), np.array(fvals)

x0_int = np.array([1.0, 1.0])   # interior start
x0_bnd = np.array([3.0, 3.0])   # boundary start

gap_int_90, fval_int_90  = run_fw(x0_int, loss_int, grad_int, bounds, 0.90,  n_iters)
gap_int_985,fval_int_985 = run_fw(x0_int, loss_int, grad_int, bounds, 0.985, n_iters)
gap_bnd_93, fval_bnd_93  = run_fw(x0_bnd, loss_bnd, grad_bnd, bounds, 0.93,  n_iters)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Top-left: FW gap for interior case
ax = axes[0, 0]
ax.loglog(k_arr, np.maximum(gap_int_90,  1e-10), 'b-',  lw=2, label='$\\beta=0.90$ (gap)')
ax.loglog(k_arr, np.maximum(gap_int_985, 1e-10), 'r--', lw=2, label='$\\beta=0.985$ (gap)')
ax.loglog(k_arr, theory_int, 'gray', lw=1.5, linestyle=':', label='$2LD^2/(k+2)$ bound')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('FW gap $g_k$ (log scale)', fontsize=11)
ax.set_title('FW Gap: Interior Optimum\n$f(x)=(x_1-1)^2+(x_2-5)^2$, $f^\\star=0$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')

# Top-right: Objective gap for interior case
ax = axes[0, 1]
ax.loglog(k_arr, np.maximum(fval_int_90[1:]  - f_star_int, 1e-10), 'b-',  lw=2, label='$\\beta=0.90$ (obj gap)')
ax.loglog(k_arr, np.maximum(fval_int_985[1:] - f_star_int, 1e-10), 'r--', lw=2, label='$\\beta=0.985$ (obj gap)')
ax.loglog(k_arr, theory_int, 'gray', lw=1.5, linestyle=':', label='$2LD^2/(k+2)$ bound')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)', fontsize=11)
ax.set_title('Objective Gap: Interior Optimum')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')

# Bottom-left: FW gap for boundary case
ax = axes[1, 0]
ax.loglog(k_arr, np.maximum(gap_bnd_93, 1e-10), 'g-', lw=2, label='$\\beta=0.93$ (gap)')
ax.loglog(k_arr, theory_bnd, 'gray', lw=1.5, linestyle=':', label='$2LD^2/(k+2)$ bound')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('FW gap $g_k$ (log scale)', fontsize=11)
ax.set_title('FW Gap: Boundary Optimum\n$f(x)=x_1^2+x_2^2$, $f^\\star=0.25$ at $(0.5,0)$')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')

# Bottom-right: Objective gap for boundary case
ax = axes[1, 1]
ax.loglog(k_arr, np.maximum(fval_bnd_93[1:] - f_star_bnd, 1e-10), 'g-', lw=2, label='$\\beta=0.93$ (obj gap)')
ax.loglog(k_arr, theory_bnd, 'gray', lw=1.5, linestyle=':', label='$2LD^2/(k+2)$ theoretical')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)', fontsize=11)
ax.set_title('Objective Gap: Boundary Optimum')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')

plt.suptitle('Q6: Frank--Wolfe Duality Gap and Objective Gap vs.\ Iteration\n'
             '(Log-log plots; $O(1/k)$ slope verifies convergence rate)', fontsize=12)
plt.tight_layout()
plt.savefig('figures/fw_gap.pdf', bbox_inches='tight')
plt.close()
print("Saved: fw_gap.pdf")

print("All sixth-batch figures generated.")
