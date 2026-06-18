"""Pass 5 supplementary figures for CS7DS2 report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ================================================================
# BENCHMARK B definitions
# ================================================================
def loss_B(x):
    return (x[0] - 1)**2 + 5*(x[1] - 2)**2 + np.sin(x[0])

def grad_B(x):
    return np.array([2*(x[0] - 1) + np.cos(x[0]), 10*(x[1] - 2)])

# ================================================================
# FIGURE 1: Penalty method landscape for different lambda values
# Shows how the penalised objective F_lambda changes with lambda
# ================================================================
def penalty_loss(x, lam):
    return loss_B(x) + lam * max(0.0, 0.5 - x[0])

x1_vals = np.linspace(-0.5, 3.0, 400)
x2_vals = np.linspace(0.0, 5.5, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

lambdas = [0.0, 0.15, 1.8, 4.5]
fig, axes = plt.subplots(1, 4, figsize=(20, 6))

for idx, lam in enumerate(lambdas):
    ax = axes[idx]
    Z = np.vectorize(lambda a, b: penalty_loss(np.array([a, b]), lam))(X1, X2)
    cs = ax.contour(X1, X2, Z, levels=30, cmap='viridis', alpha=0.8)
    ax.axvline(x=0.5, color='red', lw=2, linestyle='--', alpha=0.8)
    ax.fill_betweenx([0, 5.5], -0.5, 0.5, alpha=0.1, color='red')
    ax.plot(0.5828, 2.0, 'r*', ms=14, zorder=5,
            label=f'Constrained min' if idx == 0 else '')
    ax.set_xlabel('$x_1$')
    if idx == 0:
        ax.set_ylabel('$x_2$')
    ax.set_title(f'$\\lambda = {lam}$\n{"(unconstrained)" if lam == 0 else ""}')
    ax.grid(True, alpha=0.2)

plt.suptitle('Q5: Penalised Objective $F_\\lambda(x) = f(x) + \\lambda\\max(0, 0.5-x_1)$ '
             'for Different $\\lambda$\n(red dashed = constraint boundary $x_1=0.5$)',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('figures/supp_penalty_landscape.pdf', bbox_inches='tight')
plt.close()
print('Generated supp_penalty_landscape.pdf')

# ================================================================
# FIGURE 2: Newton step stability analysis on Rosenbrock
# Shows why alpha=0.22 is needed (full Newton step overshoots)
# ================================================================
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

def newton_step_once(x, alpha, damping=1e-8):
    g = grad_C(x)
    H = hessian_C(x) + damping * np.eye(2)
    p = np.linalg.solve(H, g)
    return x - alpha * p

# Show first Newton step from x0 for different alpha values
alphas_test = [1.0, 0.5, 0.22, 0.1]
colors_test = ['red', 'orange', 'green', 'blue']

x1_grid = np.linspace(-1.5, 1.5, 400)
x2_grid = np.linspace(-0.5, 2.0, 400)
X1C, X2C = np.meshgrid(x1_grid, x2_grid)
ZC = np.vectorize(lambda a, b: loss_C(np.array([a, b])))(X1C, X2C)

fig, ax = plt.subplots(figsize=(9, 7))
ax.contour(X1C, X2C, ZC, levels=np.logspace(-1, 3.5, 25), cmap='viridis', alpha=0.6)
ax.plot(*x0_C, 'k*', ms=14, zorder=5, label='Start $x_0=(-1,1)$')
ax.plot(1.0, 1.0, 'g*', ms=14, zorder=5, label='Optimum $(1,1)$')

for alpha, color in zip(alphas_test, colors_test):
    x1 = newton_step_once(x0_C, alpha)
    f1 = loss_C(x1)
    ax.annotate('', xy=x1, xytext=x0_C,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax.plot(*x1, 'o', color=color, ms=8, zorder=4,
            label=f'$\\alpha={alpha}$: $x_1=({x1[0]:.2f},{x1[1]:.2f})$, $f={f1:.2f}$')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title("Q3: First Newton Step on Rosenbrock from $x_0=(-1,1)$\nfor Different Step Sizes $\\alpha$")
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.5, 2.0)
plt.tight_layout()
plt.savefig('figures/supp_newton_stepsize.pdf', bbox_inches='tight')
plt.close()
print('Generated supp_newton_stepsize.pdf')

# ================================================================
# FIGURE 3: Grid search curse of dimensionality illustration
# Shows required grid points vs dimension for fixed resolution
# ================================================================
resolutions = [0.1, 0.05, 0.01]
dims = np.arange(1, 11)
domain_size = 4.0  # [-2, 2] per dimension

fig, ax = plt.subplots(figsize=(8, 5))
for res in resolutions:
    n_per_dim = int(domain_size / res) + 1
    total_evals = n_per_dim**dims
    mask = total_evals < 1e18  # practical limit
    ax.semilogy(dims[mask], total_evals[mask], '-o', ms=5,
                label=f'Resolution $\\Delta x = {res}$ ({n_per_dim} pts/dim)')

ax.axhline(y=3025, color='gray', lw=1.5, linestyle=':', label='Q4 grid (55$^2$=3025 pts, 2D)')
ax.set_xlabel('Problem dimension $d$')
ax.set_ylabel('Total function evaluations (log scale)')
ax.set_title('Curse of Dimensionality: Grid Search Evaluation Count\n'
             'Total evaluations $= (\\text{domain}/\\Delta x)^d$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(dims)
plt.tight_layout()
plt.savefig('figures/supp_grid_curse.pdf', bbox_inches='tight')
plt.close()
print('Generated supp_grid_curse.pdf')

print('\nAll pass 5 figures done.')
