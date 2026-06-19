#!/usr/bin/env python3
"""Pass 9 figures: Adam bias correction effect, RMSprop EMA analysis,
Nesterov momentum schedule comparison."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(42)

# Benchmarks
def bench_A_loss(theta, X, y):
    return 0.5 * np.mean((X @ theta - y)**2)

def bench_A_grad(theta, X, y):
    return X.T @ (X @ theta - y) / len(y)

def bench_B_loss(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])

def bench_B_grad(x):
    return np.array([2*(x[0]-1) + np.cos(x[0]), 10*(x[1]-2)])

def bench_C_loss(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
y_data = X_data @ theta_star + np.random.randn(m)
H_mat = X_data.T @ X_data / m
b_vec = X_data.T @ y_data / m
f_star_A = 0.5 * np.mean(y_data**2) - 0.5 * b_vec @ np.linalg.solve(H_mat, b_vec)

# ─── Figure 1: Adam bias correction effect ────────────────────────────────────
def adam_run(loss_fn, grad_fn, x0, alpha, beta1, beta2, eps, n_iters, bias_correct=True):
    x = x0.copy().astype(float)
    m_v = np.zeros_like(x); v_v = np.zeros_like(x)
    losses = [loss_fn(x)]
    for t in range(1, n_iters+1):
        g = grad_fn(x)
        m_v = beta1 * m_v + (1 - beta1) * g
        v_v = beta2 * v_v + (1 - beta2) * g**2
        if bias_correct:
            m_hat = m_v / (1 - beta1**t)
            v_hat = v_v / (1 - beta2**t)
        else:
            m_hat = m_v
            v_hat = v_v
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        losses.append(loss_fn(x))
    return np.array(losses)

x0_B = np.array([-1.0, 4.0])
n_iters = 150
alpha_adam = 0.08

losses_adam_bc = adam_run(bench_B_loss, bench_B_grad, x0_B, alpha_adam, 0.80, 0.999, 1e-8, n_iters, True)
losses_adam_no = adam_run(bench_B_loss, bench_B_grad, x0_B, alpha_adam, 0.80, 0.999, 1e-8, n_iters, False)

# Adam with different beta1 values
beta1_vals = [0.5, 0.7, 0.9, 0.99]
colors_b1 = ['#d62728', '#ff7f0e', '#2196F3', '#9C27B0']
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
f_star_B = 0.7244
ax.semilogy(losses_adam_bc - f_star_B, color='#2196F3', linewidth=2.5,
            label='Adam (bias-corrected, $\\beta_1=0.80$)')
ax.semilogy(losses_adam_no - f_star_B, color='#FF5722', linewidth=2.5, linestyle='--',
            label='Adam (NO bias correction, $\\beta_1=0.80$)')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('$f(x_k) - f^\\star$ (log)', fontsize=12)
ax.set_title('Effect of Bias Correction in Adam\n(Benchmark B, $\\alpha=0.08$, $\\beta_2=0.999$)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
for b1, col in zip(beta1_vals, colors_b1):
    losses_b1 = adam_run(bench_B_loss, bench_B_grad, x0_B, alpha_adam, b1, 0.999, 1e-8, n_iters, True)
    ax2.semilogy(losses_b1 - f_star_B, color=col, linewidth=2, label=f'$\\beta_1 = {b1}$')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('$f(x_k) - f^\\star$ (log)', fontsize=12)
ax2.set_title('Adam: Effect of $\\beta_1$ (First Moment Decay)\n(Benchmark B)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/adam_bias_correction.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: adam_bias_correction.pdf")

# ─── Figure 2: Nesterov vs Heavy Ball momentum schedule ──────────────────────
def heavy_ball_run(loss_fn, grad_fn, x0, alpha, beta, n_iters):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    losses = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        z = beta * z + alpha * g
        x = x - z
        losses.append(loss_fn(x))
    return np.array(losses)

def nesterov_run(loss_fn, grad_fn, x0, alpha, beta_max, n_iters):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    losses = [loss_fn(x)]
    for k in range(1, n_iters+1):
        beta_k = min((k-1)/(k+2), beta_max)
        la = x + beta_k * z
        g = grad_fn(la)
        z = beta_k * z - alpha * g
        x = x + z
        losses.append(loss_fn(x))
    return np.array(losses)

# On Rosenbrock
x0_C = np.array([-1.0, 1.0])
n_iters = 150

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Left: Rosenbrock – compare momentum strategies
ax = axes[0]
# GD
def gd_run(loss_fn, grad_fn, x0, alpha, n):
    x = x0.copy().astype(float)
    losses = [loss_fn(x)]
    for _ in range(n):
        x -= alpha * grad_fn(x)
        losses.append(loss_fn(x))
    return np.array(losses)

losses_gd_C = gd_run(bench_C_loss, lambda x: np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)]),
                      x0_C, 0.0012, n_iters)
losses_hb_C = heavy_ball_run(bench_C_loss, lambda x: np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)]),
                              x0_C, 0.0008, 0.86, n_iters)
losses_nes_C = nesterov_run(bench_C_loss, lambda x: np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)]),
                             x0_C, 0.0007, 0.90, n_iters)

ax.semilogy(losses_gd_C, '-', color='#2196F3', linewidth=2, label='GD ($\\alpha=0.0012$)')
ax.semilogy(losses_hb_C, '--', color='#FF5722', linewidth=2, label='Heavy Ball ($\\alpha=0.0008$, $\\beta=0.86$)')
ax.semilogy(losses_nes_C, '-', color='#4CAF50', linewidth=2.5, label='Nesterov ($\\alpha=0.0007$, $\\beta_{max}=0.90$)')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('$f(x_k)$ (log)', fontsize=12)
ax.set_title('GD vs Heavy Ball vs Nesterov\nRosenbrock ($f^\\star=0$)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Middle: momentum schedule beta_k for Nesterov
ax2 = axes[1]
k_arr = np.arange(1, 151)
beta_max_vals = [0.7, 0.85, 0.90, 0.95, 0.99]
cols_beta = ['#d62728', '#ff7f0e', '#4CAF50', '#2196F3', '#9C27B0']
for bmax, col in zip(beta_max_vals, cols_beta):
    betas = np.minimum((k_arr - 1)/(k_arr + 2), bmax)
    ax2.plot(k_arr, betas, color=col, linewidth=2, label=f'$\\beta_{{max}}={bmax}$')
ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1)
ax2.set_xlabel('Iteration $k$', fontsize=12)
ax2.set_ylabel('Momentum $\\beta_k = \\min\\left(\\frac{k-1}{k+2}, \\beta_{max}\\right)$', fontsize=12)
ax2.set_title("Nesterov's Adaptive Momentum Schedule\n$\\beta_k = \\min((k-1)/(k+2),\\, \\beta_{max})$", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

# Right: effect of beta_max on Benchmark B
ax3 = axes[2]
x0_B2 = np.array([-1.0, 4.0])
for bmax, col in zip(beta_max_vals, cols_beta):
    losses_n = nesterov_run(bench_B_loss, bench_B_grad, x0_B2, 0.035, bmax, n_iters)
    ax3.semilogy(losses_n - f_star_B, color=col, linewidth=2, label=f'$\\beta_{{max}}={bmax}$')
ax3.set_xlabel('Iteration', fontsize=12)
ax3.set_ylabel('$f(x_k) - f^\\star$ (log)', fontsize=12)
ax3.set_title('Effect of $\\beta_{max}$ on Nesterov Convergence\n(Benchmark B, $\\alpha=0.035$)', fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/momentum_schedule_analysis.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: momentum_schedule_analysis.pdf")

# ─── Figure 3: RMSprop EMA window analysis ────────────────────────────────────
def rmsprop_run(loss_fn, grad_fn, x0, alpha0, beta, eps, n_iters):
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    losses = [loss_fn(x)]
    for _ in range(n_iters):
        g = grad_fn(x)
        v = beta * v + (1 - beta) * g**2
        eff_a = alpha0 / (np.sqrt(v) + eps)
        x -= eff_a * g
        losses.append(loss_fn(x))
    return np.array(losses)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
beta_vals = [0.5, 0.7, 0.9, 0.95, 0.99]
colors_r = ['#d62728', '#ff7f0e', '#4CAF50', '#2196F3', '#9C27B0']
x0_B2 = np.array([-1.0, 4.0])
for b, col in zip(beta_vals, colors_r):
    eff_window = 1/(1-b)
    losses_r = rmsprop_run(bench_B_loss, bench_B_grad, x0_B2, 0.14, b, 1e-5, n_iters)
    ax.semilogy(losses_r - f_star_B + 1e-12, color=col, linewidth=2,
                label=f'$\\beta={b}$ (eff.window $\\approx {eff_window:.0f}$)')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('$f(x_k) - f^\\star$ (log)', fontsize=12)
ax.set_title('RMSprop: Effect of EMA Decay $\\beta$\n(Benchmark B, $\\alpha_0=0.14$)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: effective window visualization
ax2 = axes[1]
t_arr = np.arange(1, 51)
for b, col in zip([0.5, 0.9, 0.99], ['#d62728', '#4CAF50', '#9C27B0']):
    weights = (1 - b) * b**(t_arr - 1)
    ax2.plot(t_arr, weights, 'o-', color=col, linewidth=2, markersize=4,
             label=f'$\\beta={b}$, eff.window $\\approx {1/(1-b):.0f}$')
ax2.set_xlabel('Steps ago $j$', fontsize=12)
ax2.set_ylabel('EMA weight $(1-\\beta)\\beta^{j}$', fontsize=12)
ax2.set_title('RMSprop EMA Effective Window\nWeight $(1-\\beta)\\beta^j$ for gradient $j$ steps ago', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 50)

plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/rmsprop_ema_analysis.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: rmsprop_ema_analysis.pdf")

print("All pass-9 figures generated.")
