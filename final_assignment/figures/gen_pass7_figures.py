#!/usr/bin/env python3
"""Pass 7 figures: convergence rate hierarchy, condition number sensitivity, Frank-Wolfe rate verification."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# ─── Figure 1: Convergence Rate Hierarchy ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

k = np.linspace(1, 100, 500)
eps0 = 1.0  # initial gap

# Left: convex case – showing order differences
ax = axes[0]
# GD: O(1/k) for L-smooth convex
ax.loglog(k, eps0 / k, '-', color='#2196F3', linewidth=2.5, label=r'GD: $O(1/k)$ ($L$-smooth convex)')
# Nesterov: O(1/k^2)
ax.loglog(k, 4 * eps0 / k**2, '-', color='#4CAF50', linewidth=2.5, label=r'Nesterov: $O(1/k^2)$ (optimal, convex)')
# Heavy ball / str conv GD: linear O((1-mu/L)^k)
kappa_str = 10.0
r_gd = 1 - 1/kappa_str
r_hb = ((np.sqrt(kappa_str) - 1)/(np.sqrt(kappa_str) + 1))**2
ax.semilogy(k, eps0 * r_gd**k, '--', color='#FF5722', linewidth=2, label=rf'GD str. convex: $O(\rho^k)$, $\rho={r_gd:.2f}$ ($\kappa={kappa_str:.0f}$)')
ax.semilogy(k, eps0 * r_hb**k, '--', color='#9C27B0', linewidth=2, label=rf'H.B./Nesterov str.conv: $\rho={r_hb:.3f}$ ($\kappa={kappa_str:.0f}$)')
# Newton: O(C^(2^k)) - quadratic convergence
C = 0.5
# After a burn-in of ~5 iterations, quadratic kicks in
k_newton = np.arange(1, 12)
newton_gaps = eps0 * C**(2**k_newton - 1)
ax.semilogy(k_newton, newton_gaps, 'o-', color='#FF9800', linewidth=2.5, markersize=6, label=r"Newton: $O(C^{2^k})$ (quadratic, local)")
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('Optimality gap $f(x_k) - f^\\star$', fontsize=12)
ax.set_title('Convergence Rate Hierarchy\n(log--log scale)', fontsize=12)
ax.legend(fontsize=8.5, loc='lower left')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(1, 100); ax.set_ylim(1e-15, 2)

# Right: effect of condition number on GD vs Nesterov
ax2 = axes[1]
kappas = [2, 5, 10, 50, 100, 1000]
colors_k = plt.cm.plasma(np.linspace(0.1, 0.9, len(kappas)))
k_vals = np.arange(0, 151)

for kap, col in zip(kappas, colors_k):
    r_gd_k = 1 - 2/(kap + 1)
    r_nes_k = 1 - 2/np.sqrt(kap)
    ax2.semilogy(k_vals, r_gd_k**k_vals, '-', color=col, linewidth=1.8, alpha=0.7,
                  label=rf'GD $\kappa={kap}$: $\rho={r_gd_k:.3f}$')
    ax2.semilogy(k_vals, (r_nes_k)**k_vals, '--', color=col, linewidth=1.8,
                  label=rf'Nes $\kappa={kap}$: $\rho={r_nes_k:.3f}$')

ax2.set_xlabel('Iteration $k$', fontsize=12)
ax2.set_ylabel('$(1-\\mu/L)^k$ or $(1-2/\\sqrt{\\kappa})^k$', fontsize=12)
ax2.set_title('GD (solid) vs Nesterov (dashed)\nLinear convergence rates vs condition number $\\kappa$', fontsize=11)
ax2.set_xlim(0, 150); ax2.set_ylim(1e-8, 1)

# Custom legend for kappas
from matplotlib.lines import Line2D
kap_legend = [Line2D([0],[0], color=colors_k[i], linewidth=2, label=f'$\\kappa={kappas[i]}$')
               for i in range(len(kappas))]
style_legend = [Line2D([0],[0], color='gray', linestyle='-', label='GD'),
                Line2D([0],[0], color='gray', linestyle='--', label='Nesterov')]
ax2.legend(handles=kap_legend + style_legend, fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/convergence_rate_hierarchy.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: convergence_rate_hierarchy.pdf")

# ─── Figure 2: Condition number sensitivity across benchmarks ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

def quadratic_loss(x, H, b):
    return 0.5 * x @ H @ x - b @ x

def quadratic_grad(x, H, b):
    return H @ x - b

def run_gd(H, b, x0, alpha, n):
    x = x0.copy()
    losses = [quadratic_loss(x, H, b)]
    for _ in range(n):
        x -= alpha * quadratic_grad(x, H, b)
        losses.append(quadratic_loss(x, H, b))
    return np.array(losses)

def run_nesterov_quad(H, b, x0, alpha, n):
    x = x0.copy().astype(float)
    z = np.zeros_like(x)
    losses = [quadratic_loss(x, H, b)]
    for k in range(1, n+1):
        beta_k = (k-1)/(k+2)
        la = x + beta_k * z
        g = quadratic_grad(la, H, b)
        z = beta_k * z - alpha * g
        x = x + z
        losses.append(quadratic_loss(x, H, b))
    return np.array(losses)

# Quadratic with different kappas
kappas_test = [2.0, 10.0, 100.0]
for idx, kap in enumerate(kappas_test):
    ax = axes[idx]
    mu_k = 1.0
    L_k = kap
    H_k = np.diag([mu_k, L_k])
    b_k = np.array([mu_k * 0.5, L_k * 1.5])  # x* = (0.5, 1.5)
    f_star_k = quadratic_loss(np.linalg.solve(H_k, b_k), H_k, b_k)
    x0_k = np.zeros(2)

    alpha_gd = 1.0 / L_k
    alpha_nes = 1.0 / L_k

    n_iters = 200
    losses_gd = run_gd(H_k, b_k, x0_k, alpha_gd, n_iters)
    losses_nes = run_nesterov_quad(H_k, b_k, x0_k, alpha_nes, n_iters)

    # Heavy ball optimal
    beta_hb = ((np.sqrt(kap) - 1)/(np.sqrt(kap) + 1))**2
    alpha_hb = 4.0 / (np.sqrt(L_k) + np.sqrt(mu_k))**2
    x_hb = x0_k.copy().astype(float)
    z_hb = np.zeros(2)
    losses_hb = [quadratic_loss(x_hb, H_k, b_k)]
    for _ in range(n_iters):
        g = quadratic_grad(x_hb, H_k, b_k)
        z_hb = beta_hb * z_hb + alpha_hb * g
        x_hb = x_hb - z_hb
        losses_hb.append(quadratic_loss(x_hb, H_k, b_k))
    losses_hb = np.array(losses_hb)

    it = np.arange(n_iters + 1)
    gap_gd = losses_gd - f_star_k
    gap_nes = losses_nes - f_star_k
    gap_hb = losses_hb - f_star_k

    ax.semilogy(it, np.clip(gap_gd, 1e-15, None), '-', color='#2196F3', linewidth=2, label='GD')
    ax.semilogy(it, np.clip(gap_nes, 1e-15, None), '-', color='#4CAF50', linewidth=2, label='Nesterov')
    ax.semilogy(it, np.clip(gap_hb, 1e-15, None), '--', color='#9C27B0', linewidth=2, label='Heavy Ball (opt.)')

    # Theoretical rates overlay
    r_gd_th = (1 - mu_k/L_k)
    r_nes_th = (1 - 1/np.sqrt(kap))
    r_hb_th = beta_hb
    gap0 = gap_gd[0]
    ax.semilogy(it, gap0 * r_gd_th**it, ':', color='#2196F3', linewidth=1.2, alpha=0.7, label=rf'Theory GD $\rho={r_gd_th:.3f}$')
    ax.semilogy(it, gap0 * r_nes_th**it, ':', color='#4CAF50', linewidth=1.2, alpha=0.7, label=rf'Theory Nes $\rho={r_nes_th:.3f}$')
    ax.semilogy(it, gap0 * r_hb_th**it, ':', color='#9C27B0', linewidth=1.2, alpha=0.7, label=rf'Theory HB $\rho={r_hb_th:.3f}$')

    ax.set_xlabel('Iteration $k$', fontsize=11)
    ax.set_ylabel('$f(x_k) - f^\\star$', fontsize=11)
    ax.set_title(f'$\\kappa = L/\\mu = {kap:.0f}$\n($\\mu={mu_k}, L={L_k}$)', fontsize=11)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_iters)

plt.suptitle('Convergence on Quadratic: Empirical vs Theoretical Rates (GD, Nesterov, Heavy Ball)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/condition_number_sensitivity.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: condition_number_sensitivity.pdf")

print("All pass-7 figures generated.")
