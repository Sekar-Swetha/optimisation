"""Generate SVRG vs SGD vs GD comparison figure for Benchmark A."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ─── Benchmark A setup ───────────────────────────────────────────────────────
m, n = 1000, 2
theta_star = np.array([3.0, 4.0])
X = np.random.randn(m, n)
eps = np.random.randn(m)
y = X @ theta_star + eps

# Full gradient and loss
def full_grad(theta):
    return (X.T @ (X @ theta - y)) / m

def loss(theta):
    r = X @ theta - y
    return 0.5 * np.dot(r, r) / m

def stoch_grad(theta, idx):
    xi = X[idx]
    ri = X[idx] @ theta - y[idx]
    return xi.T @ (xi * ri[:, None] if xi.ndim == 2 else xi * ri)

# Find f_star (OLS)
theta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
f_star = loss(theta_ols)

theta0 = np.zeros(2)
n_epochs = 60

# ─── Full-batch GD ────────────────────────────────────────────────────────────
alpha_gd = 0.08
theta = theta0.copy()
gd_hist = [loss(theta)]
for _ in range(n_epochs * (m // m)):  # 1 iteration per epoch
    g = full_grad(theta)
    theta = theta - alpha_gd * g
    gd_hist.append(loss(theta))

# ─── Mini-batch SGD (b=40) ───────────────────────────────────────────────────
alpha_sgd = 0.06
b = 40
theta = theta0.copy()
sgd_hist = [loss(theta)]
for epoch in range(n_epochs):
    perm = np.random.permutation(m)
    for j in range(0, m, b):
        idx = perm[j:j+b]
        xi = X[idx]
        ri = xi @ theta - y[idx]
        g_hat = xi.T @ ri / len(idx)
        theta = theta - alpha_sgd * g_hat
    sgd_hist.append(loss(theta))

# ─── SVRG ─────────────────────────────────────────────────────────────────────
alpha_svrg = 0.06
inner_m = m  # inner loop size = n (full dataset)
theta = theta0.copy()
x_tilde = theta0.copy()
svrg_hist = [loss(theta)]

for epoch in range(n_epochs):
    # Full snapshot gradient
    g_tilde = full_grad(x_tilde)
    theta_inner = x_tilde.copy()
    for j in range(inner_m):
        i = np.random.randint(m)
        # xi as 1D vector
        xi = X[i]
        ri_new = xi @ theta_inner - y[i]
        ri_old = xi @ x_tilde - y[i]
        g_hat = xi * ri_new - xi * ri_old + g_tilde
        theta_inner = theta_inner - alpha_svrg * g_hat
    # Update snapshot
    x_tilde = theta_inner.copy()
    theta = theta_inner.copy()
    svrg_hist.append(loss(theta))

# ─── Plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 11})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: objective values over epochs
epochs_gd = np.arange(len(gd_hist))
epochs_sgd = np.arange(len(sgd_hist))
epochs_svrg = np.arange(len(svrg_hist))

ax1.semilogy(epochs_gd, [f - f_star + 1e-10 for f in gd_hist],
             'b-', lw=2, label='Full GD ($\\alpha=0.08$, 1 epoch/iter)')
ax1.semilogy(epochs_sgd, [f - f_star + 1e-10 for f in sgd_hist],
             'r-', lw=2, label='Mini-batch SGD ($b=40$, $\\alpha=0.06$)')
ax1.semilogy(epochs_svrg, [f - f_star + 1e-10 for f in svrg_hist],
             'g-', lw=2, label='SVRG ($m=n=1000$, $\\alpha=0.06$)')

noise_floor = 0.3 * alpha_sgd**2 * 1.25  # rough estimate
ax1.axhline(1e-10, ls=':', color='gray', alpha=0.5)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('$f(\\theta_k) - f^\\star$ (log scale)')
ax1.set_title('Convergence on Benchmark A: GD vs SGD vs SVRG')
ax1.legend(fontsize=9)
ax1.set_xlim(0, n_epochs)
ax1.grid(True, alpha=0.3)

# Right: log-linear to show rates — add eps to avoid log(0)
eps_plot = 1e-14
final_vals = {
    'GD': max(gd_hist[-1] - f_star, eps_plot),
    'SGD': max(sgd_hist[-1] - f_star, eps_plot),
    'SVRG': max(svrg_hist[-1] - f_star, eps_plot)
}
colors_bar = ['#1f77b4', '#d62728', '#2ca02c']
bars = ax2.bar(['Full GD', 'SGD (b=40)', 'SVRG'],
               [final_vals['GD'], final_vals['SGD'], final_vals['SVRG']],
               color=colors_bar, alpha=0.8)
ax2.set_yscale('log')
ax2.set_ylabel('Final $f(\\theta) - f^\\star$')
ax2.set_title(f'Final suboptimality after {n_epochs} epochs on Benchmark A')
ax2.grid(True, alpha=0.3, axis='y')

# Annotate bars
for bar, val in zip(bars, [final_vals['GD'], final_vals['SGD'], final_vals['SVRG']]):
    ax2.text(bar.get_x() + bar.get_width()/2, val * 1.5,
             f'{val:.2e}', ha='center', va='bottom', fontsize=9)

# Add noise floor annotation
ax2.text(0.5, 0.85, 'SGD noise floor\n(cannot converge below\nfor fixed step size)',
         transform=ax2.transAxes, ha='center', fontsize=8.5,
         color='#d62728', alpha=0.9)

fig.tight_layout()
fig.savefig('figures/q2_svrg_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print(f'Saved q2_svrg_comparison.pdf')
print(f'Final: GD={final_vals["GD"]:.4e}, SGD={final_vals["SGD"]:.4e}, SVRG={final_vals["SVRG"]:.4e}')
