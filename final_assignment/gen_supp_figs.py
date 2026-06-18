"""Generate supplementary figures for the routine report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# =================================================================
# BENCHMARK DEFINITIONS (copied from main code)
# =================================================================
m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
eps_noise = np.random.randn(m)
y_data = X_data @ theta_star + eps_noise

def loss_B(x):
    return (x[0] - 1)**2 + 5*(x[1] - 2)**2 + np.sin(x[0])

def grad_B(x):
    return np.array([2*(x[0] - 1) + np.cos(x[0]), 10*(x[1] - 2)])

x0_B = np.array([-1.0, 4.0])

# =================================================================
# FIGURE 1: Convergence rate comparison (log-log) on Benchmark B
# =================================================================
x = x0_B.copy().astype(float)
gd_hist = [loss_B(x)]
for _ in range(200):
    x = x - 0.06 * grad_B(x)
    gd_hist.append(loss_B(x))
gd_hist = np.array(gd_hist)

x = x0_B.copy().astype(float)
z = np.zeros(2)
nes_hist = [loss_B(x)]
for k in range(1, 201):
    beta_k = min((k - 1) / (k + 2), 0.92)
    lk = x + beta_k * z
    g = grad_B(lk)
    z = beta_k * z - 0.035 * g
    x = x + z
    nes_hist.append(loss_B(x))
nes_hist = np.array(nes_hist)

f_star_B = 0.7244
iters = np.arange(1, 201)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(gd_hist, 'b-', lw=2, label='GD ($\\alpha=0.06$)')
ax.plot(nes_hist, 'r-', lw=2, label='Nesterov ($\\alpha=0.035$, $\\beta_{\\max}=0.92$)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective value')
ax.set_title('Convergence Comparison -- Benchmark B')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
gap_gd = gd_hist[1:] - f_star_B
gap_nes = nes_hist[1:] - f_star_B
mask_gd = gap_gd > 1e-12
mask_nes = gap_nes > 1e-12
if mask_gd.any():
    ax.loglog(iters[mask_gd], gap_gd[mask_gd], 'b-', lw=2, label='GD gap $f(x_k)-f^*$')
if mask_nes.any():
    ax.loglog(iters[mask_nes], gap_nes[mask_nes], 'r-', lw=2, label='Nesterov gap $f(x_k)-f^*$')
k_ref = np.array([5.0, 200.0])
ax.loglog(k_ref, 5.0 / k_ref, 'b--', lw=1, alpha=0.6, label='$O(1/k)$ reference')
ax.loglog(k_ref, 50.0 / k_ref**2, 'r--', lw=1, alpha=0.6, label='$O(1/k^2)$ reference')
ax.set_xlabel('Iteration $k$ (log scale)')
ax.set_ylabel('$f(x_k) - f^*$ (log scale)')
ax.set_title('Log-Log: Convergence Gap vs Iteration')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('figures/supp_convergence_rates.pdf', bbox_inches='tight')
plt.close()
print('Generated supp_convergence_rates.pdf')

# =================================================================
# FIGURE 2: Finite difference gradient error vs delta on Benchmark B
# =================================================================
x_test = np.array([0.5, 3.0])
g_exact = grad_B(x_test)

deltas = np.logspace(-10, 1, 200)
errors = []
for delta in deltas:
    g_fd = np.array([
        (loss_B(x_test + delta * np.array([1.0, 0.0])) - loss_B(x_test)) / delta,
        (loss_B(x_test + delta * np.array([0.0, 1.0])) - loss_B(x_test)) / delta
    ])
    errors.append(np.linalg.norm(g_fd - g_exact))
errors = np.array(errors)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(deltas, errors, 'b-', lw=2, label='FD gradient error $\\|g_{FD} - g_{exact}\\|$')
ax.axvline(x=0.05, color='green', lw=1.5, linestyle='--', alpha=0.8, label='$\\delta=0.05$ (Q4 ``good\'\')')
ax.axvline(x=0.8, color='red', lw=1.5, linestyle='--', alpha=0.8, label='$\\delta=0.8$ (Q4 ``poor\'\')')
ax.axvline(x=1e-8, color='purple', lw=1.5, linestyle=':', alpha=0.8,
           label='Optimal $\\delta\\approx\\sqrt{\\varepsilon_{mach}}$')
# Reference lines
d_ref1 = np.logspace(-2, 0.5, 50)
ax.loglog(d_ref1, 0.6 * d_ref1, 'g--', lw=1, alpha=0.5, label='$O(\\delta)$ truncation')
d_ref2 = np.logspace(-10, -7, 30)
ax.loglog(d_ref2, 5e-16 / d_ref2, 'm--', lw=1, alpha=0.5, label='$O(\\varepsilon_{mach}/\\delta)$ round-off')
ax.set_xlabel('Perturbation size $\\delta$')
ax.set_ylabel('Gradient approximation error')
ax.set_title('Q4: FD Gradient Error vs $\\delta$ at $x=(0.5,\\,3.0)$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('figures/supp_fd_error.pdf', bbox_inches='tight')
plt.close()
print('Generated supp_fd_error.pdf')

# =================================================================
# FIGURE 3: Hessian eigenvalues / condition numbers for each benchmark
# =================================================================
np.random.seed(42)
m = 1000
X_data = np.random.randn(m, 2)
H_A = X_data.T @ X_data / m
eigs_A = np.linalg.eigvalsh(H_A)

x_star_B = np.array([0.5828, 2.0])
H_B_mat = np.array([[2.0 - np.sin(x_star_B[0]), 0.0], [0.0, 10.0]])
eigs_B = np.linalg.eigvalsh(H_B_mat)

x_star_C = np.array([1.0, 1.0])
h11 = 2.0 + 1200.0 * x_star_C[0]**2 - 400.0 * x_star_C[1]
h12 = -400.0 * x_star_C[0]
H_C_mat = np.array([[h11, h12], [h12, 200.0]])
eigs_C = np.linalg.eigvalsh(H_C_mat)

benchmarks = [
    ('A: Linear Regression', eigs_A, eigs_A[1] / eigs_A[0]),
    ('B: Toy Neural Network', eigs_B, eigs_B[1] / eigs_B[0]),
    ('C: Rosenbrock at $(1,1)$', eigs_C, eigs_C[1] / eigs_C[0])
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (name, eigs, kappa) in enumerate(benchmarks):
    ax = axes[idx]
    bars = ax.bar([1, 2], eigs, color=['steelblue', 'tomato'], width=0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'$\\lambda_{{\\min}}$', f'$\\lambda_{{\\max}}$'])
    ax.set_title(f'Bench. {name}\n$\\kappa = {kappa:.1f}$', fontsize=10)
    ax.set_ylabel('Eigenvalue')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(eigs):
        ax.text(i + 1, v * 1.03, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Hessian Eigenvalues and Condition Numbers at Optimal Points\n'
             'Higher $\\kappa = \\lambda_{\\max}/\\lambda_{\\min}$ implies slower GD convergence',
             fontsize=12, y=1.04)
plt.tight_layout()
plt.savefig('figures/supp_condition_numbers.pdf', bbox_inches='tight')
plt.close()
print('Generated supp_condition_numbers.pdf')

print('\nAll supplementary figures generated successfully.')
