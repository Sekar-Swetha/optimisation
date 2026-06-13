"""Generate augmented Lagrangian dual convergence figure for Q5."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)

# Problem: min (mu/2)||x||^2  s.t. Ax = b
# A = I_2, b = [1,1]^T, mu = 2
# x* = [1,1]^T, y* = -mu*x* = [-2,-2]^T
# lambda_min(A^TA) = 1
# Rate c(rho) = mu/(mu + rho * lambda_min) = 2/(2 + rho)

mu = 2.0
A = np.eye(2)
b = np.array([1.0, 1.0])

x_star = np.array([1.0, 1.0])
y_star = -mu * x_star   # = [-2, -2]

def run_auglag(rho, n_iters=40, y_init=None):
    """Run augmented Lagrangian method and return ||y_k - y*||."""
    if y_init is None:
        y = np.zeros(2)
    else:
        y = y_init.copy()
    norms = [np.linalg.norm(y - y_star)]
    for _ in range(n_iters):
        # x_{k+1} = (mu*I + rho*A^TA)^{-1} A^T (rho*b - y)
        M = mu * np.eye(2) + rho * A.T @ A
        rhs = A.T @ (rho * b - y)
        x_new = np.linalg.solve(M, rhs)
        # y_{k+1} = y + rho*(A x_{k+1} - b)
        y = y + rho * (A @ x_new - b)
        norms.append(np.linalg.norm(y - y_star))
    return np.array(norms)

rhos = [0.5, 1.0, 2.0, 5.0, 10.0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
n_iters = 35
y0 = np.array([3.0, -1.0])  # non-zero start

# Theoretical rates
def theory_rate(rho):
    return mu / (mu + rho)

plt.rcParams.update({'font.size': 11})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: dual convergence ||y_k - y*|| on semilogy
iters = np.arange(n_iters + 1)
for rho, c in zip(rhos, colors):
    norms = run_auglag(rho, n_iters=n_iters, y_init=y0)
    cr = theory_rate(rho)
    theory = norms[0] * cr**iters
    ax1.semilogy(iters, norms, color=c, lw=2, label=fr'$\rho={rho}$ (empirical, $c={cr:.3f}$)')
    ax1.semilogy(iters, theory, color=c, lw=1, ls='--', alpha=0.5)

ax1.set_xlabel('Iteration $k$')
ax1.set_ylabel(r'$\|y_k - y^\star\|$ (log scale)')
ax1.set_title(r'Augmented Lagrangian: dual variable convergence')
ax1.legend(fontsize=8.5, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, n_iters)

# Right: convergence rate c(rho) = mu/(mu + rho) vs rho
rho_range = np.linspace(0.01, 15, 300)
c_range = mu / (mu + rho_range)

ax2.plot(rho_range, c_range, 'b-', lw=2, label=r'$c(\rho) = \frac{\mu}{\mu + \rho\,\lambda_{\min}(A^TA)}$')
for rho, c in zip(rhos, colors):
    ax2.scatter([rho], [theory_rate(rho)], color=c, s=80, zorder=5)
    ax2.annotate(fr'$\rho={rho}$', xy=(rho, theory_rate(rho)),
                 xytext=(rho + 0.4, theory_rate(rho) + 0.03),
                 fontsize=8.5, color=c)

ax2.axhline(0, color='gray', lw=0.8, ls=':')
ax2.axhline(1, color='gray', lw=0.8, ls=':')
ax2.set_xlabel(r'Penalty parameter $\rho$')
ax2.set_ylabel(r'Linear convergence rate $c(\rho)$')
ax2.set_title(r'Rate $c(\rho) = \mu/(\mu + \rho)$ for $\mu=2$, $\lambda_{\min}=1$')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 15)
ax2.set_ylim(-0.05, 1.1)
ax2.text(7, 0.6, r'Smaller $c$ $\Rightarrow$ faster convergence', fontsize=9, color='navy',
         ha='center')

fig.suptitle(r'Augmented Lagrangian convergence: $\min \frac{\mu}{2}\|x\|^2$ s.t. $Ax=b$ ($\mu=2$, $A=I_2$, $\lambda_{\min}=1$)',
             fontsize=11)
fig.tight_layout()
fig.savefig('figures/q5_auglag_convergence.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved q5_auglag_convergence.pdf')
print(f'Rates: ' + ', '.join(f'rho={r}: c={theory_rate(r):.4f}' for r in rhos))
