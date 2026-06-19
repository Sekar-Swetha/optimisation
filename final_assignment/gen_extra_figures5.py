"""Generate pass-13 extra figures:
1. all_methods_convergence_A.pdf -- comprehensive convergence comparison on Benchmark A:
   GD, Polyak(wrong), Adagrad, RMSprop, Heavy Ball, Nesterov, Adam, Newton
   With theoretical O(1/k) and O(1/k^2) reference lines, clearly showing method hierarchy
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)
FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ─── Benchmark A setup ─────────────────────────────────────────────────────
m, d = 1000, 2
theta_star = np.array([3.0, 4.0])
X_A = np.random.randn(m, d)
eps_A = np.random.randn(m)
y_A = X_A @ theta_star + eps_A
H_A_mat = X_A.T @ X_A / m
f_star_A = 0.5/m * np.sum((X_A @ (np.linalg.solve(H_A_mat, X_A.T @ y_A / m)) - y_A)**2)

def f_A(th): return 0.5/m * np.sum((X_A @ th - y_A)**2)
def g_A(th): return X_A.T @ (X_A @ th - y_A) / m
def H_A(th): return H_A_mat

x0_A = np.zeros(2)
NITERS = 120

# ── Methods ──────────────────────────────────────────────────────────────────
def gd(x0, alpha, niters):
    x = x0.copy(); fh = [f_A(x)]
    for _ in range(niters):
        x = x - alpha * g_A(x); fh.append(f_A(x))
    return np.array(fh)

def polyak(x0, fstar, eps, niters):
    x = x0.copy(); fh = [f_A(x)]
    for _ in range(niters):
        g = g_A(x); alpha_k = (f_A(x) - fstar) / (g@g + eps)
        x = x - alpha_k * g; fh.append(f_A(x))
    return np.array(fh)

def adagrad(x0, alpha0, eps, niters):
    x = x0.copy(); G = np.zeros_like(x); fh = [f_A(x)]
    for _ in range(niters):
        g = g_A(x); G += g**2; x = x - alpha0/(np.sqrt(G)+eps)*g; fh.append(f_A(x))
    return np.array(fh)

def rmsprop(x0, alpha0, beta, eps, niters):
    x = x0.copy(); v = np.zeros_like(x); fh = [f_A(x)]
    for _ in range(niters):
        g = g_A(x); v = beta*v+(1-beta)*g**2; x = x - alpha0/(np.sqrt(v)+eps)*g; fh.append(f_A(x))
    return np.array(fh)

def heavy_ball(x0, alpha, beta, niters):
    x = x0.copy(); z = np.zeros_like(x); fh = [f_A(x)]
    for _ in range(niters):
        g = g_A(x); z = beta*z+alpha*g; x = x-z; fh.append(f_A(x))
    return np.array(fh)

def nesterov(x0, alpha, beta_max, niters):
    x = x0.copy(); z = np.zeros_like(x); fh = [f_A(x)]
    for k in range(1, niters+1):
        beta_k = min((k-1)/(k+2), beta_max)
        lh = x + beta_k*z; g = g_A(lh); z = beta_k*z - alpha*g; x = x+z; fh.append(f_A(x))
    return np.array(fh)

def adam(x0, alpha, b1, b2, eps, niters):
    x = x0.copy(); m_v = np.zeros_like(x); v = np.zeros_like(x); fh = [f_A(x)]
    for t in range(1, niters+1):
        g = g_A(x); m_v = b1*m_v+(1-b1)*g; v = b2*v+(1-b2)*g**2
        mh = m_v/(1-b1**t); vh = v/(1-b2**t); x = x-alpha*mh/(np.sqrt(vh)+eps); fh.append(f_A(x))
    return np.array(fh)

def newton_1d(x0, alpha, niters, lam=1e-8):
    x = x0.copy(); fh = [f_A(x)]
    for _ in range(niters):
        g = g_A(x); H = H_A(x) + lam*np.eye(len(x))
        p = np.linalg.solve(H, g); x = x - alpha*p; fh.append(f_A(x))
    return np.array(fh)

fh_gd    = gd(x0_A, 0.08, NITERS)
fh_polyak_wrong = polyak(x0_A, 0.0, 1e-4, NITERS)  # wrong f*=0
fh_polyak_right = polyak(x0_A, f_star_A*0.99, 1e-4, NITERS)  # nearly correct f*
fh_agrad = adagrad(x0_A, 1.8, 1e-5, NITERS)
fh_rms   = rmsprop(x0_A, 0.22, 0.9, 1e-5, NITERS)
fh_hb    = heavy_ball(x0_A, 0.045, 0.88, NITERS)
fh_nes   = nesterov(x0_A, 0.06, 0.90, NITERS)
fh_adam  = adam(x0_A, 0.12, 0.82, 0.999, 1e-8, NITERS)
fh_newton = newton_1d(x0_A, 1.0, min(20, NITERS))

f_star = f_star_A  # exact

iters = np.arange(NITERS + 1)
newton_iters = np.arange(len(fh_newton))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
ax.semilogy(iters, np.maximum(fh_gd - f_star, 1e-15),    'k-',  lw=1.5, label='GD')
ax.semilogy(iters, np.maximum(fh_agrad - f_star, 1e-15),  'b-',  lw=1.5, label='Adagrad')
ax.semilogy(iters, np.maximum(fh_rms - f_star, 1e-15),    'g-',  lw=1.5, label='RMSprop')
ax.semilogy(iters, np.maximum(fh_hb - f_star, 1e-15),     'm-',  lw=1.5, label='Heavy Ball')
ax.semilogy(iters, np.maximum(fh_nes - f_star, 1e-15),    'r-',  lw=2.0, label='Nesterov')
ax.semilogy(iters, np.maximum(fh_adam - f_star, 1e-15),   'c-',  lw=2.0, label='Adam')
ax.semilogy(newton_iters, np.maximum(fh_newton - f_star, 1e-15), 'r^-', lw=2.0, ms=6, label='Newton (20 it)')
# Reference lines
k_ref = np.arange(1, NITERS+1)
C1 = (fh_gd[5] - f_star) * 5
C2 = (fh_nes[5] - f_star) * 25
ax.semilogy(k_ref, np.maximum(C1/k_ref, 1e-15), 'k--', lw=0.8, alpha=0.5, label='$O(1/k)$')
ax.semilogy(k_ref, np.maximum(C2/k_ref**2, 1e-15), 'r--', lw=0.8, alpha=0.5, label='$O(1/k^2)$')
ax.set_xlabel('Iteration')
ax.set_ylabel('$f(x_k) - f^\\star$ (log scale)')
ax.set_title('Benchmark A: All Methods (excl.\ diverging Polyak)')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim(0, NITERS)
ax.grid(True, alpha=0.3)
ax.set_ylim(1e-15, None)

ax2 = axes[1]
# Show Polyak wrong vs right, and GD/Nesterov for context
ax2.semilogy(iters, np.maximum(fh_gd - f_star, 1e-15),            'k-', lw=1.5, label='GD baseline')
ax2.semilogy(iters, np.maximum(fh_polyak_wrong - f_star, 1e-15),  'r--',lw=2.0, label='Polyak ($f^\\star$=0, wrong)')
ax2.semilogy(iters, np.maximum(fh_polyak_right - f_star, 1e-15),  'g-', lw=2.0, label='Polyak ($f^\\star\\approx f^\\star_{\\rm true}$)')
ax2.semilogy(iters, np.maximum(fh_nes - f_star, 1e-15),           'b-', lw=2.0, label='Nesterov')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('$|f(x_k) - f^\\star|$ (log scale)')
ax2.set_title('Benchmark A: Polyak Sensitivity to $f^\\star$ Specification')
ax2.legend(fontsize=9)
ax2.set_xlim(0, NITERS)
ax2.grid(True, alpha=0.3)

plt.suptitle("Benchmark A ($\\kappa \\approx 1.117$): Comprehensive Method Comparison", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'all_methods_convergence_A.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved all_methods_convergence_A.pdf")
print(f"f_star_A = {f_star_A:.6f}")
