"""Generate pass-8 extra figures:
1. all_methods_benchmark_B.pdf  -- convergence of all Q1+Q2 methods on Benchmark B
2. q3_newton_decrement.pdf      -- Newton decrement lambda^2/2 vs iteration on Benchmarks A/B/C
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ─── Benchmark B ────────────────────────────────────────────────────────────
def f_B(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])

def g_B(x):
    return np.array([2*(x[0]-1) + np.cos(x[0]), 10*(x[1]-2)])

def H_B(x):
    return np.diag([2 - np.sin(x[0]), 10.0])

x0_B = np.array([-1.0, 4.0])
f_star_B = 0.7244
NITERS = 150

# ── GD baseline ─────────────────────────────────────────────────────────────
def gd(x0, alpha, niters, grad_fn, loss_fn):
    x = x0.copy()
    fh = [loss_fn(x)]
    for _ in range(niters):
        x = x - alpha * grad_fn(x)
        fh.append(loss_fn(x))
    return np.array(fh)

# ── Polyak ───────────────────────────────────────────────────────────────────
def polyak(x0, fstar, eps, niters, grad_fn, loss_fn):
    x = x0.copy()
    fh = [loss_fn(x)]
    for _ in range(niters):
        g = grad_fn(x)
        alpha_k = (loss_fn(x) - fstar) / (np.dot(g,g) + eps)
        x = x - alpha_k * g
        fh.append(loss_fn(x))
    return np.array(fh)

# ── Adagrad ──────────────────────────────────────────────────────────────────
def adagrad(x0, alpha0, eps, niters, grad_fn, loss_fn):
    x = x0.copy()
    G = np.zeros_like(x)
    fh = [loss_fn(x)]
    for _ in range(niters):
        g = grad_fn(x)
        G += g**2
        x = x - alpha0 / (np.sqrt(G) + eps) * g
        fh.append(loss_fn(x))
    return np.array(fh)

# ── RMSprop ──────────────────────────────────────────────────────────────────
def rmsprop(x0, alpha0, beta, eps, niters, grad_fn, loss_fn):
    x = x0.copy()
    v = np.zeros_like(x)
    fh = [loss_fn(x)]
    for _ in range(niters):
        g = grad_fn(x)
        v = beta * v + (1 - beta) * g**2
        x = x - alpha0 / (np.sqrt(v) + eps) * g
        fh.append(loss_fn(x))
    return np.array(fh)

# ── Heavy Ball ────────────────────────────────────────────────────────────────
def heavy_ball(x0, alpha, beta, niters, grad_fn, loss_fn):
    x = x0.copy()
    z = np.zeros_like(x)
    fh = [loss_fn(x)]
    for _ in range(niters):
        g = grad_fn(x)
        z = beta * z + alpha * g
        x = x - z
        fh.append(loss_fn(x))
    return np.array(fh)

# ── Nesterov ─────────────────────────────────────────────────────────────────
def nesterov(x0, alpha, beta_max, niters, grad_fn, loss_fn):
    x = x0.copy()
    z = np.zeros_like(x)
    fh = [loss_fn(x)]
    for k in range(1, niters+1):
        beta_k = min((k-1)/(k+2), beta_max)
        lh = x + beta_k * z
        g = grad_fn(lh)
        z = beta_k * z - alpha * g
        x = x + z
        fh.append(loss_fn(x))
    return np.array(fh)

# ── Adam ──────────────────────────────────────────────────────────────────────
def adam(x0, alpha, b1, b2, eps, niters, grad_fn, loss_fn):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    fh = [loss_fn(x)]
    for t in range(1, niters+1):
        g = grad_fn(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        x = x - alpha * mh / (np.sqrt(vh) + eps)
        fh.append(loss_fn(x))
    return np.array(fh)

# Compute all curves on Benchmark B
fh_gd     = gd(x0_B, 0.06, NITERS, g_B, f_B)
fh_polyak = polyak(x0_B, 0.0, 1e-4, NITERS, g_B, f_B)   # misspecified f*=0
fh_agrad  = adagrad(x0_B, 1.2, 1e-5, NITERS, g_B, f_B)
fh_rms    = rmsprop(x0_B, 0.14, 0.9, 1e-5, NITERS, g_B, f_B)
fh_hb     = heavy_ball(x0_B, 0.035, 0.90, NITERS, g_B, f_B)
fh_nes    = nesterov(x0_B, 0.035, 0.92, NITERS, g_B, f_B)
fh_adam   = adam(x0_B, 0.08, 0.80, 0.999, 1e-8, NITERS, g_B, f_B)

# Clip to avoid log(<=0)
def safe_log(fh):
    return np.maximum(fh - f_star_B, 1e-12)

iters = np.arange(NITERS + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogy(iters, safe_log(fh_gd),     'k-',  lw=1.5, label='GD (baseline)')
ax.semilogy(iters, safe_log(fh_agrad),  'b-',  lw=1.5, label='Adagrad')
ax.semilogy(iters, safe_log(fh_rms),    'g-',  lw=1.5, label='RMSprop')
ax.semilogy(iters, safe_log(fh_hb),     'm-',  lw=1.5, label='Heavy Ball')
ax.semilogy(iters, safe_log(fh_nes),    'r-',  lw=2.0, label='Nesterov')
ax.semilogy(iters, safe_log(fh_adam),   'c-',  lw=2.0, label='Adam')
ax.semilogy(iters, safe_log(fh_polyak), 'y--', lw=1.5, label='Polyak ($f^\\star=0$, wrong)')
ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x_k) - f^\\star$')
ax.set_title('Benchmark B: All Q1+Q2 Methods (log scale)')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim(0, NITERS)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
# Zoom: exclude diverging Polyak
ax2.semilogy(iters, safe_log(fh_gd),    'k-',  lw=1.5, label='GD (baseline)')
ax2.semilogy(iters, safe_log(fh_agrad), 'b-',  lw=1.5, label='Adagrad')
ax2.semilogy(iters, safe_log(fh_rms),   'g-',  lw=1.5, label='RMSprop')
ax2.semilogy(iters, safe_log(fh_hb),    'm-',  lw=1.5, label='Heavy Ball')
ax2.semilogy(iters, safe_log(fh_nes),   'r-',  lw=2.0, label='Nesterov')
ax2.semilogy(iters, safe_log(fh_adam),  'c-',  lw=2.0, label='Adam')
# O(1/k^2) and O(1/k) reference
k_ref = np.arange(1, NITERS + 1)
C1 = safe_log(fh_gd)[5] * 5
C2 = safe_log(fh_nes)[5] * 25
ax2.semilogy(k_ref, C1/k_ref,    'k--', lw=0.8, alpha=0.6, label='$O(1/k)$')
ax2.semilogy(k_ref, C2/k_ref**2, 'r--', lw=0.8, alpha=0.6, label='$O(1/k^2)$')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('$f(x_k) - f^\\star$')
ax2.set_title('Benchmark B: Correct-$f^\\star$ Methods with Rate Refs')
ax2.set_ylim(1e-8, 1e2)
ax2.legend(fontsize=8, loc='upper right')
ax2.set_xlim(0, NITERS)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'all_methods_benchmark_B.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved all_methods_benchmark_B.pdf")


# ─── Figure 2: Newton Decrement vs iteration ─────────────────────────────────
# Benchmark A: linear regression Hessian
np.random.seed(42)
m, d = 1000, 2
theta_star = np.array([3.0, 4.0])
X_A = np.random.randn(m, d)
eps_A = np.random.randn(m)
y_A = X_A @ theta_star + eps_A
H_A_mat = X_A.T @ X_A / m

def f_A(th): return 0.5/m * np.sum((X_A @ th - y_A)**2)
def g_A(th): return X_A.T @ (X_A @ th - y_A) / m
def H_A(th): return H_A_mat

# Benchmark C: Rosenbrock
def f_C(x): return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
def g_C(x): return np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
                               200*(x[1]-x[0]**2)])
def H_C(x):
    h11 = 2 + 1200*x[0]**2 - 400*x[1]
    h12 = -400*x[0]
    h22 = 200.0
    return np.array([[h11, h12],[h12, h22]])

def newton_with_decrement(x0, alpha, niters, grad_fn, hess_fn, loss_fn, lam=1e-8):
    x = x0.copy().astype(float)
    fh = [loss_fn(x)]
    dec_sq = []
    for _ in range(niters):
        g = grad_fn(x)
        H = hess_fn(x) + lam * np.eye(len(x))
        try:
            p = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            p = g
        # Newton decrement squared: g^T H^{-1} g = g^T p
        lam_sq = float(g @ p)
        dec_sq.append(max(lam_sq / 2, 1e-16))
        x = x - alpha * p
        fh.append(loss_fn(x))
    return np.array(fh), np.array(dec_sq)

n_newton = 20
x0_A = np.zeros(2)
x0_C = np.array([-1.0, 1.0])
x0_B_n = np.array([-1.0, 4.0])

fh_nA, dec_A = newton_with_decrement(x0_A, 1.0, n_newton, g_A, H_A, f_A)
fh_nB, dec_B = newton_with_decrement(x0_B_n, 0.85, n_newton, g_B, H_B, f_B)
fh_nC, dec_C = newton_with_decrement(x0_C, 0.22, n_newton, g_C, H_C, f_C)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, dec, label, fh, fname in zip(axes,
        [dec_A, dec_B, dec_C],
        ['A ($\\kappa\\approx1.12$)', 'B ($\\kappa\\approx6.9$)', 'C ($\\kappa\\approx2504$)'],
        [fh_nA, fh_nB, fh_nC],
        ['Benchmark A', 'Benchmark B', 'Benchmark C']):
    iters_d = np.arange(1, n_newton + 1)
    ax.semilogy(iters_d, dec, 'b-o', markersize=4, lw=1.5, label='$\\lambda^2/2$ (Newton decrement)')
    fstar_local = min(fh[-1], fh_nA[-1]) if fname == 'Benchmark A' else (0.7244 if 'B' in fname else 0.0)
    fgap = np.maximum(fh[1:] - fstar_local, 1e-16)
    ax.semilogy(iters_d, fgap, 'r--s', markersize=4, lw=1.5, label='$f(x_k)-f^\\star$')
    ax.axhline(0.5, color='gray', ls=':', lw=0.8, label='$\\varepsilon_{tol}=0.5$')
    ax.set_xlabel('Newton iteration')
    ax.set_title(f'Bench.\ {label}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_newton)

axes[0].set_ylabel('Value (log scale)')
plt.suptitle("Newton Decrement $\\lambda^2/2$ vs Suboptimality $f(x_k)-f^\\star$", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'q3_newton_decrement.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved q3_newton_decrement.pdf")
