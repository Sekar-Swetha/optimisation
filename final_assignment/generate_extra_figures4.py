"""
Additional figures for the report - Pass 14
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
# Benchmark definitions
# ============================================================
def loss_C(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_C(x):
    dx1 = -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2)
    dx2 = 200*(x[1]-x[0]**2)
    return np.array([dx1, dx2])

def hess_C(x):
    h11 = 2 - 400*(x[1]-x[0]**2) + 800*x[0]**2
    h12 = -400*x[0]
    h22 = 200.0
    return np.array([[h11, h12], [h12, h22]])

def newton(x0, alpha, n_iters, damping=1e-8):
    x = x0.copy().astype(float)
    f_hist = [loss_C(x)]
    for _ in range(n_iters):
        g = grad_C(x)
        H = hess_C(x) + damping * np.eye(2)
        try:
            p = np.linalg.solve(H, g)
        except:
            p = g
        x = x - alpha * p
        f_hist.append(loss_C(x))
    return np.array(f_hist)

# ============================================================
# Figure 1: Newton step size sensitivity on Rosenbrock
# ============================================================
x0 = np.array([-1.0, 1.0])
alphas = [0.05, 0.10, 0.15, 0.22, 0.30, 0.40, 0.50, 0.70, 1.00]
colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for alpha, color in zip(alphas, colors):
    f_hist = newton(x0, alpha, 20)
    ax.semilogy(range(21), f_hist, '-', color=color, lw=1.8, alpha=0.85,
                label=f'$\\alpha={alpha}$')
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$f(x_k)$ (log scale)')
ax.set_title('Newton Convergence on Rosenbrock\nfor Different Step Sizes $\\alpha$')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

ax = axes[1]
final_vals = []
for alpha in alphas:
    f_hist = newton(x0, alpha, 20)
    final_vals.append(f_hist[-1])

ax.semilogy(alphas, final_vals, 'bo-', lw=2, markersize=8)
ax.axvline(x=0.22, color='r', ls='--', lw=2, label='Used: $\\alpha=0.22$')
ax.axhline(y=0.686, color='r', ls=':', lw=1.5, alpha=0.7, label='$f=0.686$ at $\\alpha=0.22$')
ax.set_xlabel('Step size $\\alpha$')
ax.set_ylabel(r'$f(x_{20})$ after 20 iterations (log scale)')
ax.set_title('Final Objective vs.\ Newton Step Size\n(20 iterations, Rosenbrock)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(alphas)
ax.set_xticklabels([str(a) for a in alphas], rotation=45)

plt.suptitle("Newton's Method Step Size Sensitivity on Benchmark C", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q3_newton_alpha_sensitivity.pdf', bbox_inches='tight')
plt.close()
print("Saved q3_newton_alpha_sensitivity.pdf")

# ============================================================
# Figure 2: Comparison of adaptive methods' effective step size evolution
# ============================================================
def loss_B(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def grad_B(x):
    return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

x0_B = np.array([-1.0, 4.0])
n_iters = 120
alpha0_adagrad = 1.2
alpha0_rms = 0.14
beta_rms = 0.9
eps = 1e-5

# Adagrad
x = x0_B.copy().astype(float)
G = np.zeros(2)
adagrad_eff = []
adagrad_f = []
for _ in range(n_iters):
    g = grad_B(x)
    G += g**2
    eff_alpha = alpha0_adagrad / (np.sqrt(G) + eps)
    adagrad_eff.append(np.mean(eff_alpha))
    x = x - eff_alpha * g
    adagrad_f.append(loss_B(x))

# RMSprop
x = x0_B.copy().astype(float)
v = np.zeros(2)
rms_eff = []
rms_f = []
for _ in range(n_iters):
    g = grad_B(x)
    v = beta_rms * v + (1-beta_rms) * g**2
    eff_alpha = alpha0_rms / (np.sqrt(v) + eps)
    rms_eff.append(np.mean(eff_alpha))
    x = x - eff_alpha * g
    rms_f.append(loss_B(x))

# Adam
alpha_adam = 0.08; beta1 = 0.80; beta2 = 0.999
x = x0_B.copy().astype(float)
m_vec = np.zeros(2); v_vec = np.zeros(2)
adam_eff = []
adam_f = []
for t in range(1, n_iters+1):
    g = grad_B(x)
    m_vec = beta1*m_vec + (1-beta1)*g
    v_vec = beta2*v_vec + (1-beta2)*g**2
    m_hat = m_vec / (1-beta1**t)
    v_hat = v_vec / (1-beta2**t)
    step = alpha_adam / (np.sqrt(v_hat) + 1e-8)
    adam_eff.append(np.mean(step))
    x = x - step * m_hat
    adam_f.append(loss_B(x))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
iters = np.arange(1, n_iters+1)

ax = axes[0]
ax.semilogy(iters, adagrad_eff, 'b-', lw=2, label='Adagrad ($\\alpha_0=1.2$)')
ax.semilogy(iters, rms_eff, 'g-', lw=2, label='RMSprop ($\\alpha_0=0.14$)')
ax.semilogy(iters, adam_eff, 'r-', lw=2, label='Adam ($\\alpha=0.08$)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean effective step size (log scale)')
ax.set_title('Effective Step Size Evolution on Benchmark B')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
f_star_B = 0.7244
ax.semilogy(iters, np.array(adagrad_f) - f_star_B + 1e-8, 'b-', lw=2, label='Adagrad')
ax.semilogy(iters, np.array(rms_f) - f_star_B + 1e-8, 'g-', lw=2, label='RMSprop')
ax.semilogy(iters, np.array(adam_f) - f_star_B + 1e-8, 'r-', lw=2, label='Adam')
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$f(x_k) - f^{\star}$ (log scale)')
ax.set_title('Convergence: Adagrad vs RMSprop vs Adam\n(Benchmark B)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('Adaptive Methods: Effective Step Size vs Convergence (Benchmark B)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q1_adaptive_step_comparison.pdf', bbox_inches='tight')
plt.close()
print("Saved q1_adaptive_step_comparison.pdf")

# ============================================================
# Figure 3: Gradient clipping effect on Heavy Ball (Benchmark C)
# ============================================================
def heavy_ball_clipped(x0, alpha, beta, n_iters, clip_norm=None):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    f_hist = [loss_C(x)]
    for _ in range(n_iters):
        g = grad_C(x)
        if clip_norm is not None:
            gnorm = np.linalg.norm(g)
            if gnorm > clip_norm:
                g = g * clip_norm / gnorm
        z = beta * z + alpha * g
        x = x - z
        f_hist.append(loss_C(x))
    return np.array(f_hist)

x0 = np.array([-1.0, 1.0])
alpha_hb = 0.0008; beta_hb = 0.86

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
clip_norms = [None, 200, 100, 50, 20]
labels = ['No clipping', 'clip=200', 'clip=100', 'clip=50', 'clip=20']
colors_clip = ['k', 'b', 'g', 'orange', 'r']
for clip, label, color in zip(clip_norms, labels, colors_clip):
    f_hist = heavy_ball_clipped(x0, alpha_hb, beta_hb, 300, clip)
    ax.semilogy(range(301), f_hist, '-', color=color, lw=1.8, label=label)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$f(x_k)$ (log scale)')
ax.set_title('Gradient Clipping Effect on Heavy Ball\n(Benchmark C)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Show gradient norms over time (no clipping)
x = x0.copy().astype(float)
z = np.zeros_like(x)
gnorms = []
for _ in range(300):
    g = grad_C(x)
    gnorms.append(np.linalg.norm(g))
    z = beta_hb * z + alpha_hb * g
    x = x - z

ax = axes[1]
ax.semilogy(range(300), gnorms, 'k-', lw=2)
ax.axhline(y=200, color='b', ls='--', alpha=0.7, label='clip=200')
ax.axhline(y=100, color='g', ls='--', alpha=0.7, label='clip=100')
ax.axhline(y=50, color='orange', ls='--', alpha=0.7, label='clip=50')
ax.axhline(y=20, color='r', ls='--', alpha=0.7, label='clip=20')
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\|\nabla f(x_k)\|$ (log scale)')
ax.set_title('Gradient Norm Evolution\n(Heavy Ball, no clipping)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Gradient Clipping: Effect on Convergence and Gradient Norm (Benchmark C)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q2_gradient_clipping.pdf', bbox_inches='tight')
plt.close()
print("Saved q2_gradient_clipping.pdf")

print("All figures generated successfully.")
