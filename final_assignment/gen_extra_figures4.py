"""Generate pass-10 extra figures:
1. q2_nesterov_vs_adam_trajectory.pdf  -- x1 and x2 components over iterations
   to show Nesterov valley traversal vs Adam coordinate normalisation
2. q1_heavy_ball_stability.pdf         -- stability region in (alpha, beta) space
   for Heavy Ball on a strongly convex quadratic
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ─── Benchmark C: Rosenbrock ─────────────────────────────────────────────────
def f_C(x): return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
def g_C(x): return np.array([
    -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
     200*(x[1] - x[0]**2)
])

x0_C = np.array([-1.0, 1.0])
NITERS = 150

def nesterov(x0, alpha, beta_max, niters, grad_fn, loss_fn):
    x = x0.copy()
    z = np.zeros_like(x)
    xh = [x.copy()]
    fh = [loss_fn(x)]
    for k in range(1, niters+1):
        beta_k = min((k-1)/(k+2), beta_max)
        lh = x + beta_k * z
        g = grad_fn(lh)
        z = beta_k * z - alpha * g
        x = x + z
        xh.append(x.copy())
        fh.append(loss_fn(x))
    return np.array(xh), np.array(fh)

def adam(x0, alpha, b1, b2, eps, niters, grad_fn, loss_fn):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    xh = [x.copy()]
    fh = [loss_fn(x)]
    for t in range(1, niters+1):
        g = grad_fn(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        x = x - alpha * mh / (np.sqrt(vh) + eps)
        xh.append(x.copy())
        fh.append(loss_fn(x))
    return np.array(xh), np.array(fh)

def gd(x0, alpha, niters, grad_fn, loss_fn):
    x = x0.copy()
    xh = [x.copy()]
    fh = [loss_fn(x)]
    for _ in range(niters):
        x = x - alpha * grad_fn(x)
        xh.append(x.copy())
        fh.append(loss_fn(x))
    return np.array(xh), np.array(fh)

xh_gd,  fh_gd  = gd(x0_C, 0.0012, NITERS, g_C, f_C)
xh_nes, fh_nes = nesterov(x0_C, 0.0007, 0.90, NITERS, g_C, f_C)
xh_adam,fh_adam = adam(x0_C, 0.006, 0.80, 0.999, 1e-8, NITERS, g_C, f_C)

iters = np.arange(NITERS + 1)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Top-left: x1 component vs iteration
ax = axes[0, 0]
ax.plot(iters, xh_gd[:,0],  'k-',  lw=1.2, label='GD', alpha=0.7)
ax.plot(iters, xh_nes[:,0], 'r-',  lw=2.0, label='Nesterov')
ax.plot(iters, xh_adam[:,0],'c-',  lw=2.0, label='Adam')
ax.axhline(1.0, color='gray', ls='--', lw=1.0, label='$x_1^\\star = 1$')
ax.set_xlabel('Iteration')
ax.set_ylabel('$x_1(k)$')
ax.set_title('$x_1$ component vs iteration (Rosenbrock)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Top-right: x2 component vs iteration
ax = axes[0, 1]
ax.plot(iters, xh_gd[:,1],  'k-',  lw=1.2, label='GD', alpha=0.7)
ax.plot(iters, xh_nes[:,1], 'r-',  lw=2.0, label='Nesterov')
ax.plot(iters, xh_adam[:,1],'c-',  lw=2.0, label='Adam')
ax.axhline(1.0, color='gray', ls='--', lw=1.0, label='$x_2^\\star = 1$')
ax.set_xlabel('Iteration')
ax.set_ylabel('$x_2(k)$')
ax.set_title('$x_2$ component vs iteration (Rosenbrock)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom-left: f(x_k) - f* on log scale
ax = axes[1, 0]
fstar = 0.0
ax.semilogy(iters, np.maximum(fh_gd - fstar,  1e-12), 'k-',  lw=1.2, label='GD')
ax.semilogy(iters, np.maximum(fh_nes - fstar, 1e-12), 'r-',  lw=2.0, label='Nesterov')
ax.semilogy(iters, np.maximum(fh_adam - fstar,1e-12), 'c-',  lw=2.0, label='Adam')
ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)')
ax.set_title('Convergence on Rosenbrock')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom-right: 2D trajectory projection
ax = axes[1, 1]
# Rosenbrock contour
x1g = np.linspace(-1.2, 1.4, 300)
x2g = np.linspace(-0.5, 1.5, 300)
X1, X2 = np.meshgrid(x1g, x2g)
Z = (1 - X1)**2 + 100*(X2 - X1**2)**2
levels = np.logspace(-1, 4, 25)
ax.contour(X1, X2, Z, levels=levels, colors='gray', linewidths=0.5, alpha=0.6)
# trajectories (plot every 3rd point to avoid clutter)
s = 3
ax.plot(xh_gd[::s, 0],   xh_gd[::s, 1],   'k.',   ms=3, label='GD', alpha=0.6)
ax.plot(xh_nes[::s, 0],  xh_nes[::s, 1],  'r-o',  ms=3, lw=1.5, label='Nesterov', alpha=0.8)
ax.plot(xh_adam[::s, 0], xh_adam[::s, 1], 'c-^',  ms=3, lw=1.5, label='Adam', alpha=0.8)
ax.plot(1, 1, 'g*', ms=12, label='Optimum $(1,1)$')
ax.plot(-1, 1, 'ks', ms=8, label='Start $(-1,1)$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Rosenbrock trajectory (2D)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xlim(-1.3, 1.4)
ax.set_ylim(-0.6, 1.6)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'q2_nesterov_vs_adam_trajectory.pdf'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved q2_nesterov_vs_adam_trajectory.pdf")

# ─── Figure 2: Heavy Ball stability region ───────────────────────────────────
# For a strongly convex quadratic with L and mu, the stability condition is:
# alpha(1+beta) < 2/L  AND  0 <= beta < 1
# The optimal convergence rate for fixed alpha,beta is (sqrt(kappa)-1)/(sqrt(kappa)+1)
# at optimal alpha* = 4/(sqrt(L)+sqrt(mu))^2, beta* = ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2

# Use Benchmark A: L ~ 1.033, mu ~ 0.925
L_A = 1.033
mu_A = 0.925
kappa_A = L_A / mu_A

alphas = np.linspace(0, 2.5/L_A, 300)
betas = np.linspace(0, 0.999, 300)
A, B = np.meshgrid(alphas, betas)

# Stability condition: alpha(1+beta) < 2/L
stable = A * (1 + B) < 2.0 / L_A

# Convergence rate (for strongly convex quadratic, approximation):
# max eigenvalue of iteration matrix
# For Heavy Ball on quadratic, spectral radius ~ max(|r1|, |r2|) where
# r = (beta + 1 - alpha*mu ± sqrt((beta+1-alpha*mu)^2 - 4*beta)) / 2  (complex roots)
# Use simplified: rate ~ (1 - alpha*mu) * (1 + beta) when real, else |r| = sqrt(beta)
# More precisely use: spectral radius of [[beta - alpha*L, alpha*L], [1, 0]] ...
# ... actually let's just show the stability region and the optimal point.

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
# Show stability region
ax.contourf(A, B, stable.astype(float), levels=[0.5, 1.5], colors=['lightblue'], alpha=0.5)
ax.contour(A, B, (A*(1+B) - 2.0/L_A), levels=[0.0], colors=['blue'], linewidths=2)
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)

# Mark optimal Heavy Ball parameters
alpha_opt = 4.0 / (np.sqrt(L_A) + np.sqrt(mu_A))**2
beta_opt  = ((np.sqrt(kappa_A) - 1) / (np.sqrt(kappa_A) + 1))**2
ax.plot(alpha_opt, beta_opt, 'r*', ms=14, zorder=5, label=f'Optimal ($\\alpha^*={alpha_opt:.2f}$, $\\beta^*={beta_opt:.2f}$)')

# Mark experimental settings for Benchmark A: alpha=0.045, beta=0.88
ax.plot(0.045, 0.88, 'g^', ms=10, zorder=5, label='Experiment ($\\alpha=0.045$, $\\beta=0.88$)')

# GD line: beta=0
ax.axhline(1/L_A, color='gray', ls=':', lw=1)

ax.text(0.5/L_A, 0.3, 'STABLE\nregion', ha='center', va='center',
        fontsize=11, color='navy', alpha=0.7)
ax.text(1.8/L_A, 0.7, 'UNSTABLE', ha='center', va='center',
        fontsize=11, color='red', alpha=0.7)
ax.set_xlabel('Step size $\\alpha$')
ax.set_ylabel('Momentum $\\beta$')
ax.set_title(f'Heavy Ball Stability Region\n(Benchmark A, $L={L_A:.3f}$, $\\mu={mu_A:.3f}$, $\\kappa={kappa_A:.3f}$)')
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(0, 2.5/L_A)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
# Use Benchmark C: L ~ 1002 at start (lambda_max of Hessian at x0=(-1,1))
# H_C(-1,1): h11 = 2 + 1200*1 - 400*1 = 802, h12=400, h22=200 -> lambda_max ~ 1002
L_C_approx = 1002  # correct lambda_max of H at x0=(-1,1)
mu_C_approx = 0.4  # small eigenvalue near optimum
kappa_C = L_C_approx / mu_C_approx  # ~2505

alphas_C = np.linspace(0, 2.5/L_C_approx, 300)
betas_C = np.linspace(0, 0.999, 300)
AC, BC = np.meshgrid(alphas_C, betas_C)
stable_C = AC * (1 + BC) < 2.0 / L_C_approx

ax2.contourf(AC * L_C_approx, BC, stable_C.astype(float), levels=[0.5, 1.5],
             colors=['lightcoral'], alpha=0.5)
ax2.contour(AC * L_C_approx, BC, (AC*(1+BC) - 2.0/L_C_approx), levels=[0.0],
            colors=['red'], linewidths=2)

# Mark experimental settings: alpha=0.0008, beta=0.86  -> alpha*L = 0.0008*1002 = 0.802
ax2.plot(0.0008 * L_C_approx, 0.86, 'g^', ms=10, zorder=5,
         label=f'Experiment ($\\alpha L={0.0008*L_C_approx:.2f}$, $\\beta=0.86$)')

ax2.text(0.6, 0.3, 'STABLE', ha='center', va='center', fontsize=11, color='darkred', alpha=0.7)
ax2.text(1.6, 0.7, 'UNSTABLE', ha='center', va='center', fontsize=11, color='red', alpha=0.7)
ax2.set_xlabel('Normalised step size $\\alpha L$')
ax2.set_ylabel('Momentum $\\beta$')
ax2.set_title(f'Heavy Ball Stability Region\n(Bench. C, $L\\approx{L_C_approx}$ at start, $\\kappa\\approx{kappa_C:.0f}$)')
ax2.legend(fontsize=9, loc='upper left')
ax2.set_xlim(0, 2.5)
ax2.set_ylim(0, 1.0)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'q1_heavy_ball_stability.pdf'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved q1_heavy_ball_stability.pdf")
