"""Generate two new figures for the routine report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                     'axes.labelsize': 11, 'legend.fontsize': 10,
                     'figure.dpi': 150})

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: GD vs Nesterov convergence rate bounds for strongly convex
#           (1-1/kappa)^k  vs  (1-1/sqrt(kappa))^k  for several kappa values
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

k = np.arange(0, 201)
kappas = [10, 100, 1000]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

ax = axes[0]
ax.set_title(r'GD convergence rate: $(1-1/\kappa)^k$')
for kappa, c in zip(kappas, colors):
    rate = (1 - 1/kappa)**k
    ax.semilogy(k, rate, color=c, lw=2, label=fr'$\kappa={kappa}$')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'$(1-1/\kappa)^k$  (normalised error bound)')
ax.legend()
ax.set_xlim(0, 200)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title(r'Nesterov strongly convex rate: $(1-1/\sqrt{\kappa})^k$')
for kappa, c in zip(kappas, colors):
    rate = (1 - 1/np.sqrt(kappa))**k
    ax.semilogy(k, rate, color=c, lw=2, label=fr'$\kappa={kappa}$')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel(r'$(1-1/\sqrt{\kappa})^k$  (normalised error bound)')
ax.legend()
ax.set_xlim(0, 200)
ax.grid(True, alpha=0.3)

# Add iteration counts to reach 1e-6
for ax_i, rate_fn in [(axes[0], lambda k: (1-1/1000)**k),
                       (axes[1], lambda k: (1-1/np.sqrt(1000))**k)]:
    iters = np.where(rate_fn(k) < 1e-6)[0]
    if len(iters):
        ax_i.axhline(1e-6, color='gray', lw=0.8, ls=':', alpha=0.7)
        ax_i.text(0.97, 1.2e-6, r'$10^{-6}$', ha='right',
                  transform=ax_i.get_yaxis_transform(), fontsize=9, color='gray')

fig.suptitle(r'Convergence rate comparison: GD vs Nesterov (strongly convex, $\kappa \in \{10,100,1000\}$)',
             fontsize=12)
fig.tight_layout()
fig.savefig('figures/q2_rate_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved q2_rate_comparison.pdf')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Forward vs Central difference gradient error vs delta
#           on Benchmark B at x=(0, 3)
# ─────────────────────────────────────────────────────────────────────────────
def f_B(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])

def grad_B_exact(x):
    return np.array([2*(x[0]-1) + np.cos(x[0]), 10*(x[1]-2)])

x0 = np.array([0.0, 3.0])
g_true = grad_B_exact(x0)

deltas = np.logspace(-14, 0, 300)
err_fwd = np.zeros(len(deltas))
err_ctr = np.zeros(len(deltas))

for j, delta in enumerate(deltas):
    g_fwd = np.array([(f_B(x0 + delta*np.eye(2)[i]) - f_B(x0)) / delta
                      for i in range(2)])
    g_ctr = np.array([(f_B(x0 + delta*np.eye(2)[i]) - f_B(x0 - delta*np.eye(2)[i])) / (2*delta)
                      for i in range(2)])
    err_fwd[j] = np.linalg.norm(g_fwd - g_true)
    err_ctr[j] = np.linalg.norm(g_ctr - g_true)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(deltas, err_fwd, color='#1f77b4', lw=2, label='Forward diff $O(\\delta)$ truncation')
ax.loglog(deltas, err_ctr, color='#d62728', lw=2, label='Central diff $O(\\delta^2)$ truncation')

# Theoretical reference lines
d_ref = deltas[50:220]
eps_mach = 2.2e-16
ax.loglog(d_ref, 0.5 * d_ref, '--', color='#1f77b4', alpha=0.6, lw=1.2, label=r'$O(\delta)$ slope')
ax.loglog(d_ref, 0.1 * d_ref**2, '--', color='#d62728', alpha=0.6, lw=1.2, label=r'$O(\delta^2)$ slope')
ax.loglog(d_ref, eps_mach/d_ref, ':', color='gray', lw=1.2, label=r'$O(\varepsilon_{\rm mach}/\delta)$ rounding')

# Mark optimal delta for each
delta_opt_fwd = np.sqrt(eps_mach)      # ~1e-8
delta_opt_ctr = eps_mach**(1/3)        # ~1e-5.3

ax.axvline(delta_opt_fwd, color='#1f77b4', ls=':', lw=1.5, alpha=0.8)
ax.axvline(delta_opt_ctr, color='#d62728', ls=':', lw=1.5, alpha=0.8)
ax.text(delta_opt_fwd*1.5, 0.015, r'$\delta_{\rm opt}^{\rm fwd}\approx10^{-8}$',
        color='#1f77b4', fontsize=9)
ax.text(delta_opt_ctr*1.5, 0.015, r'$\delta_{\rm opt}^{\rm ctr}\approx10^{-5.3}$',
        color='#d62728', fontsize=9)

# Mark delta=0.05 and delta=0.8
for dv, lbl in [(0.05, '$\\delta=0.05$'), (0.8, '$\\delta=0.8$')]:
    ax.axvline(dv, color='green', ls='--', lw=1, alpha=0.7)
    ax.text(dv*1.1, 1e-13, lbl, color='green', fontsize=9)

ax.set_xlabel(r'Step size $\delta$')
ax.set_ylabel(r'Gradient approximation error $\|\hat{g} - \nabla f\|$')
ax.set_title('Forward vs Central Difference: gradient error on Benchmark B at $x=(0,3)$')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1e-14, 2)
ax.set_ylim(1e-14, 5)

fig.tight_layout()
fig.savefig('figures/q4_fd_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved q4_fd_comparison.pdf')

print('Done.')
