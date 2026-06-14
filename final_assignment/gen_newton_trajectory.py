"""
Generate Newton trajectory analysis figure for Rosenbrock.
Shows non-monotone f values, update magnitudes, and trajectory on contour.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def loss_C(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def grad_C(x): return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
def hessian_C(x): return np.array([[2+1200*x[0]**2-400*x[1],-400*x[0]],[-400*x[0],200]])

x0 = np.array([-1.0, 1.0])
alpha_newton = 0.22
alpha_gd = 0.001
n_newton = 20
n_gd = 80

# Run Newton
x = x0.copy()
traj_newton = [x.copy()]
f_newton = [loss_C(x)]
kappa_newton = []
for _ in range(n_newton):
    g = grad_C(x)
    H = hessian_C(x) + 1e-8*np.eye(2)
    evals = np.linalg.eigvalsh(H)
    kappa_newton.append(abs(evals[1])/abs(evals[0]))
    p = np.linalg.solve(H, g)
    x = x - alpha_newton * p
    traj_newton.append(x.copy())
    f_newton.append(loss_C(x))
traj_newton = np.array(traj_newton)
f_newton = np.array(f_newton)
kappa_newton = np.array(kappa_newton)

# Run GD
x = x0.copy()
traj_gd = [x.copy()]
f_gd = [loss_C(x)]
for _ in range(n_gd):
    g = grad_C(x)
    x = x - alpha_gd * g
    traj_gd.append(x.copy())
    f_gd.append(loss_C(x))
traj_gd = np.array(traj_gd)
f_gd = np.array(f_gd)

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: convergence curves
ax = axes[0]
ax.semilogy(f_newton, 'g-o', ms=4, lw=2, label=f'Newton (α={alpha_newton}, 20 iters)')
ax.semilogy(np.arange(0, n_gd+1, 4), f_gd[::4], 'b--s', ms=3, lw=1.5, label=f'GD (α={alpha_gd}, 80 iters)')
ax.axhline(0.0, color='gold', lw=1.5, ls=':', label='f*=0')
# Mark the non-monotone first step
ax.annotate('f increases\n4.0→6.18', xy=(1, f_newton[1]), xytext=(4, 8),
            fontsize=8, color='red',
            arrowprops=dict(arrowstyle='->', color='red'))
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective f(x)')
ax.set_title('Q3: Newton vs GD on Rosenbrock')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: condition number along Newton trajectory
ax = axes[1]
ax.plot(range(n_newton), kappa_newton, 'r-o', ms=4, lw=2)
ax.axhline(2504, color='gray', ls='--', lw=1.5, label='κ at optimum ≈2504')
ax.set_xlabel('Newton iteration')
ax.set_ylabel('Condition number κ')
ax.set_title('Hessian condition number along Newton trajectory')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Panel 3: contour + trajectories
ax = axes[2]
x1g = np.linspace(-1.5, 1.5, 200)
x2g = np.linspace(-0.5, 1.5, 200)
X1, X2 = np.meshgrid(x1g, x2g)
Z = (1-X1)**2 + 100*(X2-X1**2)**2
ax.contour(X1, X2, np.log10(Z+1e-8), levels=25, cmap='viridis', alpha=0.6)
ax.plot(traj_newton[:,0], traj_newton[:,1], 'g-o', ms=4, lw=2, label='Newton')
ax.plot(traj_gd[:,0], traj_gd[:,1], 'b--', lw=1.2, alpha=0.7, label='GD')
ax.plot(*x0, 'ks', ms=8, label='Start (-1,1)')
ax.plot(1, 1, 'r*', ms=12, label='Optimum (1,1)')
# Mark the second iterate (f=6.18) as the non-monotone step
ax.plot(*traj_newton[1], 'ro', ms=8, mfc='none', mew=2, label='Step 1 (f↑6.18)')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Trajectories on Rosenbrock (log contours)')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig('figures/q3_newton_analysis.pdf', bbox_inches='tight')
plt.close()
print(f'Newton final f: {f_newton[-1]:.4f} at {traj_newton[-1]}')
print(f'GD final f: {f_gd[-1]:.4f} at {traj_gd[-1]}')
print(f'Newton kappa range: {kappa_newton.min():.1f} to {kappa_newton.max():.1f}')
print('Saved q3_newton_analysis.pdf')
