import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def loss_B(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])

def grad_B(x):
    return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

x0 = np.array([-1.0, 4.0])
exact_grad = grad_B(x0)
print(f"Exact gradient at x0=(-1,4): {exact_grad}")
print(f"|grad| = {np.linalg.norm(exact_grad):.6f}")
print(f"f(x0) = {loss_B(x0):.6f}")
print(f"f''_11 = 2 - sin(-1) = {2 - np.sin(-1):.6f}")
print(f"f''_22 = 10")
print(f"L = max eigenvalue of H = 10")
print(f"Theoretical delta* (unit scale): {np.sqrt(2*np.finfo(float).eps/10):.2e}")
print(f"Theoretical delta* (scaled by |f|): {np.sqrt(4*np.finfo(float).eps*loss_B(x0)/10):.2e}")

deltas = np.logspace(-10, 0, 200)
errors = []
for d in deltas:
    fd_grad = np.zeros(2)
    f0 = loss_B(x0)
    for i in range(2):
        ei = np.zeros(2); ei[i] = 1.0
        fd_grad[i] = (loss_B(x0 + d*ei) - f0) / d
    errors.append(np.linalg.norm(fd_grad - exact_grad))

errors = np.array(errors)
min_idx = np.argmin(errors)
print(f"\nEmpirical optimal delta = {deltas[min_idx]:.2e}")
print(f"Minimum FD error = {errors[min_idx]:.2e}")

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(deltas, errors, 'b-', lw=2)
ax.axvline(deltas[min_idx], color='green', linestyle='--', lw=1.5, label=f'Empirical opt $\\delta\\approx${deltas[min_idx]:.0e}')
for dd, label, color in [(0.05, '$\\delta=0.05$ (good)', 'orange'), (0.8, '$\\delta=0.8$ (poor)', 'red'),
                          (1e-8, '$\\delta=10^{-8}$\n(theory opt.)', 'purple')]:
    ax.axvline(dd, color=color, linestyle=':', lw=1.5, label=label)
ax.set_xlabel('Perturbation $\\delta$')
ax.set_ylabel('FD gradient error $\\|\\hat{g} - \\nabla f\\|$')
ax.set_title('Forward FD gradient error vs $\\delta$ at $x_0=(-1,4)$, Benchmark B')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('figures/q4_fd_error.pdf', bbox_inches='tight')
plt.close()
print('Saved q4_fd_error.pdf')
