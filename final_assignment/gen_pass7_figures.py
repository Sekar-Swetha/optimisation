"""Pass 7 figures: PL inequality illustration, Adagrad regret proof illustration,
Newton trust region, SGD convergence analysis, convergence rate hierarchy."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(42)

# ── Functions ─────────────────────────────────────────────────────────────────
def rosen(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def rosen_grad(x):
    g0 = -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2)
    g1 = 200*(x[1]-x[0]**2)
    return np.array([g0,g1])
def bench_b(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def bench_b_grad(x):
    return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])

# ============================================================
# Figure 1: Polyak-Lojasiewicz Inequality Illustration
# ============================================================
# PL condition: (1/2)||grad f||^2 >= mu*(f - f*)
# For f(x) = x^2: grad = 2x, ||grad||^2 = 4x^2, f-f* = x^2
# So (1/2)*4x^2 = 2x^2 >= mu*x^2 => mu <= 2. PL constant = 2 = mu (strongly convex).
# For f(x) = sin^2(x): f* = 0, grad = 2sin(x)cos(x) = sin(2x)
# ||grad||^2 = sin^2(2x), f - f* = sin^2(x)
# PL: sin^2(2x)/2 >= mu*sin^2(x)
# 4sin^2(x)cos^2(x)/2 = 2cos^2(x)*sin^2(x) >= mu*sin^2(x)
# => 2cos^2(x) >= mu. Min is at x=pi/2: cos(pi/2)=0, so PL fails at x*=pi/2.
# Actually sin^2(x) has f* = 0, f' = sin(2x), at x=pi/2, f' = 0 but f = 1. FAILS PL.

# Show PL for strongly convex quadratic vs non-PL function
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Left: strongly convex f(x) = x^2
xs = np.linspace(-2, 2, 400)
f1 = xs**2
g1_sq = (2*xs)**2
f1_star = 0

mu_sc = 2.0  # PL constant

ax = axes[0]
ax.plot(xs, f1, 'b-', lw=2, label='$f(x) = x^2$')
ax.fill_between(xs, 0, (1/(2*mu_sc)) * g1_sq,
                alpha=0.2, color='green', label=r'$\frac{1}{2\mu}\|\nabla f\|^2$ (upper bound for PL)')
ax.plot(xs, (1/(2*2)) * g1_sq, 'g--', lw=1.5,
        label=r'$\frac{1}{2\mu}\|\nabla f\|^2$, $\mu=2$')
ax.set_xlabel('$x$'); ax.set_ylabel('Value')
ax.set_title('PL Condition: $f(x) = x^2$ (SC, $\mu=2$)\n'
             r'$\frac{1}{2}\|\nabla f(x)\|^2 \geq \mu(f(x)-f^\star)$ satisfied', fontsize=9)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 4.5); ax.set_xlim(-2, 2)
ax.text(0.05, 0.85, 'PL constant $\mu=2$\n(= strong convexity)', transform=ax.transAxes,
        fontsize=8.5, bbox=dict(facecolor='lightyellow', alpha=0.8))

# Middle: non-strongly convex but PL: f(x) = (x^2 + sin(x))^2 / 4 ... try f(x)=x^4
xs2 = np.linspace(-1.5, 1.5, 400)
f2 = xs2**4
g2_sq = (4*xs2**3)**2
# PL: (1/2)*16x^6 >= mu*x^4 => 8x^2 >= mu. At x->0, mu -> 0. NOT PL uniformly.
# Show it fails near x=0.
mu_try = 0.01  # very small
PL_lhs = 0.5 * g2_sq
PL_rhs = mu_try * f2

ax2 = axes[1]
ax2.plot(xs2, f2, 'b-', lw=2, label='$f(x) = x^4$')
ax2.plot(xs2, PL_lhs, 'g--', lw=1.5, label=r'$\frac{1}{2}\|\nabla f\|^2$')
ax2.plot(xs2, PL_rhs, 'r:', lw=1.5, label=rf'$\mu(f-f^\star)$, $\mu={mu_try}$')
ratio = np.where(f2 > 1e-8, PL_lhs / f2, np.nan)
ax2.set_xlabel('$x$'); ax2.set_ylabel('Value')
ax2.set_title('PL Failure: $f(x)=x^4$, degenerate min\n'
              r'$\frac{1}{2}\|\nabla f\|^2 / (f-f^\star) \to 0$ as $x\to 0$', fontsize=9)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.01, 1.5); ax2.set_xlim(-1.5, 1.5)
ax2.text(0.05, 0.85, 'PL fails: ratio\n$\\to 0$ at $x^\\star$\n(degenerate Hessian)',
         transform=ax2.transAxes, fontsize=8.5, bbox=dict(facecolor='lightyellow', alpha=0.8))

# Right: PL implies convergence for GD
# For mu-PL function: f(x_k) - f* <= (1 - 2*alpha*mu)^k * (f(x_0) - f*)
k_arr = np.arange(101)
f0_minus_fstar = 3.5  # Rosenbrock

# Different mu values (effective PL constant along trajectory)
for mu_eff, label, color, ls in [
    (1e-3, r'$\mu=10^{-3}$ (near GD on Rosen)', 'steelblue', '-'),
    (0.01, r'$\mu=0.01$', 'green', '--'),
    (0.1,  r'$\mu=0.1$',  'orange', '-.'),
    (1.0,  r'$\mu=1.0$ (well-conditioned)', 'red', ':'),
]:
    alpha_opt = 1 / (2 * mu_eff * 1000)  # 1/(2*mu_eff*L) with L=1000
    alpha_use = min(alpha_opt, 0.0012)   # cap at stability
    rate = max(0, 1 - 2 * alpha_use * mu_eff)
    conv = f0_minus_fstar * rate**k_arr
    axes[2].semilogy(k_arr, conv, color=color, linestyle=ls, lw=1.8, label=f'{label}\nrate={(rate):.4f}')

axes[2].set_xlabel('Iteration $k$'); axes[2].set_ylabel('$f(x_k) - f^\\star$')
axes[2].set_title('GD Convergence Under PL Condition\n$f(x_k)-f^\\star \\leq (1-2\\alpha\\mu)^k(f(x_0)-f^\\star)$', fontsize=9)
axes[2].legend(fontsize=7.5, loc='lower left'); axes[2].grid(True, which='both', alpha=0.3)
axes[2].set_xlim(0, 100)

plt.suptitle('Polyak--Łojasiewicz (PL) Inequality and GD Convergence', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'pl_inequality.pdf'), bbox_inches='tight')
plt.close()
print("Saved: pl_inequality.pdf")

# ============================================================
# Figure 2: Adagrad Regret Bound Illustration
# ============================================================
# Online convex opt: at each step t, receive loss f_t, update x_t+1
# Adagrad regret: R(T) = sum_{t=1}^T [f_t(x_t) - f_t(x*)]
# <= alpha_0 * sum_i sqrt(sum_{t=1}^T g_{t,i}^2)
# Simulate online linear regression

m_samples = 500
d = 5
T = 200  # online rounds

# Ground truth
theta_star = np.random.randn(d)
X_all = np.random.randn(m_samples, d)
y_all = X_all @ theta_star + 0.1 * np.random.randn(m_samples)

def online_gd(alpha, T):
    theta = np.zeros(d)
    regrets = []
    for t in range(T):
        idx = t % m_samples
        xt = X_all[idx]
        yt = y_all[idx]
        loss_theta = 0.5 * (xt @ theta - yt)**2
        loss_star  = 0.5 * (xt @ theta_star - yt)**2
        regrets.append(loss_theta - loss_star)
        g = xt * (xt @ theta - yt)
        theta -= alpha * g
    return np.cumsum(regrets)

def online_adagrad(alpha0, eps, T):
    theta = np.zeros(d)
    G = np.zeros(d)
    regrets = []
    for t in range(T):
        idx = t % m_samples
        xt = X_all[idx]
        yt = y_all[idx]
        loss_theta = 0.5 * (xt @ theta - yt)**2
        loss_star  = 0.5 * (xt @ theta_star - yt)**2
        regrets.append(loss_theta - loss_star)
        g = xt * (xt @ theta - yt)
        G += g**2
        theta -= (alpha0 / (np.sqrt(G) + eps)) * g
    return np.cumsum(regrets)

# Run
r_gd_fast  = online_gd(0.5, T)
r_gd_slow  = online_gd(0.05, T)
r_adagrad  = online_adagrad(2.0, 1e-5, T)

# Theoretical O(sqrt(T)) and O(T) references
ts = np.arange(1, T+1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(ts, np.maximum(r_gd_fast, 0),  'b-',  lw=2, label='GD $\\alpha=0.5$ (large)')
ax.plot(ts, np.maximum(r_gd_slow, 0),  'b--', lw=2, label='GD $\\alpha=0.05$ (small)')
ax.plot(ts, np.maximum(r_adagrad, 0),  'r-',  lw=2, label='Adagrad $\\alpha_0=2.0$')
# O(T) and O(sqrt(T)) references
c_T   = r_gd_fast[-1] / T
c_sqT = r_adagrad[-1] / np.sqrt(T)
ax.plot(ts, c_T * ts,        'k--', lw=1, alpha=0.7, label=r'$O(T)$ ref')
ax.plot(ts, c_sqT * np.sqrt(ts), 'k:',  lw=1, alpha=0.7, label=r'$O(\sqrt{T})$ ref')
ax.set_xlabel('Online round $t$'); ax.set_ylabel('Cumulative regret $R(T)$')
ax.set_title('Online Regret: GD vs Adagrad ($d=5$ linear regression)', fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Right: per-round regret (instantaneous)
ax2 = axes[1]
window = 10
from numpy.lib.stride_tricks import sliding_window_view

def smooth(arr, w):
    return np.convolve(arr, np.ones(w)/w, mode='valid')

rnd_gd_fast = np.diff(np.concatenate([[0], r_gd_fast]))
rnd_gd_slow = np.diff(np.concatenate([[0], r_gd_slow]))
rnd_adagrad = np.diff(np.concatenate([[0], r_adagrad]))

s = 5
ax2.plot(ts[s-1:], smooth(rnd_gd_fast, s),  'b-',  lw=1.5, label='GD large')
ax2.plot(ts[s-1:], smooth(rnd_gd_slow, s),  'b--', lw=1.5, label='GD small')
ax2.plot(ts[s-1:], smooth(rnd_adagrad, s),  'r-',  lw=1.5, label='Adagrad')
ax2.axhline(0, color='k', lw=1, linestyle=':', alpha=0.6)
ax2.set_xlabel('Online round $t$')
ax2.set_ylabel('Per-round regret $f_t(x_t) - f_t(x^\\star)$ (smoothed)')
ax2.set_title('Per-Round Regret (5-step moving average)', fontsize=10)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
ax2.text(0.98, 0.95,
         'Adagrad: negative regret possible\n(better than $x^\\star$ on some rounds)',
         transform=ax2.transAxes, ha='right', va='top', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Adagrad Regret Bound: $R(T) \\leq \\alpha_0 \\sum_i \\sqrt{\\sum_t g_{t,i}^2} = O(\\sqrt{T})$', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q1_adagrad_regret.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q1_adagrad_regret.pdf")

# ============================================================
# Figure 3: Newton Trust Region (why damping is needed on Rosen)
# ============================================================
# Show how ||x_{k+1} - x_k||_H (the Newton decrement) relates to
# the radius where the quadratic model is accurate.

x_pts = [np.array([-0.8, 0.6]), np.array([0.0, 0.0]), np.array([0.7, 0.45])]
labels_pts = ['$x$ near start', '$x$ at valley', '$x$ near optimum']

def rosen_hess(x):
    h11 = 2 - 400*(x[1] - x[0]**2) + 1600*x[0]**2
    h12 = -400*x[0]
    h22 = 200.0
    return np.array([[h11, h12], [h12, h22]])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (xc, lbl) in enumerate(zip(x_pts, labels_pts)):
    H = rosen_hess(xc)
    g = rosen_grad(xc)
    f0 = rosen(xc)

    try:
        p = np.linalg.solve(H + 1e-8*np.eye(2), g)
    except:
        p = g

    # Newton decrement: lambda^2 = g^T H^{-1} g = g^T p
    lam2 = g @ p
    newton_dec = np.sqrt(max(lam2, 0))

    # Grid around xc
    rng = 0.6
    xs_l = np.linspace(xc[0]-rng, xc[0]+rng, 200)
    ys_l = np.linspace(xc[1]-rng, xc[1]+rng, 200)
    Xs, Ys = np.meshgrid(xs_l, ys_l)

    # True function
    F_true = (1-Xs)**2 + 100*(Ys-Xs**2)**2
    # Quadratic approx
    DX = Xs - xc[0]; DY = Ys - xc[1]
    d_vec = np.stack([DX.ravel(), DY.ravel()]).T
    quad = f0 + d_vec @ g + 0.5 * np.sum((d_vec @ H) * d_vec, axis=1)
    F_quad = quad.reshape(Xs.shape)

    # Error = |True - Quad| / |True - f0|
    denom = np.abs(F_true - f0) + 1e-10
    error = np.abs(F_true - F_quad) / denom

    ax = axes[idx]
    im = ax.contourf(xs_l, ys_l, error, levels=[0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                     cmap='RdYlGn_r', alpha=0.8)
    plt.colorbar(im, ax=ax, label='Relative model error', shrink=0.85)
    ax.contour(xs_l, ys_l, (1-Xs)**2 + 100*(Ys-Xs**2)**2,
               levels=10, colors='gray', linewidths=0.5, alpha=0.4)

    # Mark current point and Newton step
    ax.plot(xc[0], xc[1], 'wo', ms=9, zorder=5)
    x_new = xc - p
    ax.annotate('', xy=(x_new[0], x_new[1]), xytext=(xc[0], xc[1]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.plot(x_new[0], x_new[1], 'b^', ms=8, zorder=5, label='Newton step $x - H^{-1}g$')

    # Show "trust radius" where error < 10%
    # approximate as ellipse: d^T H d <= r^2, r where error becomes 10%
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    kap = np.linalg.cond(H)
    ax.set_title(f'{lbl}\n$f={f0:.3f}$, $\\kappa(H)={kap:.0f}$, $\\lambda_N={newton_dec:.3f}$',
                 fontsize=9)
    ax.legend(fontsize=8, loc='lower right')

plt.suptitle('Newton Step Accuracy: Where Quadratic Model Approximates Rosenbrock\n'
             '(green = accurate, red = inaccurate; blue arrow = Newton step)', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q3_newton_trust.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q3_newton_trust.pdf")

# ============================================================
# Figure 4: Convergence Rate Hierarchy (all methods, all benchmarks)
# ============================================================
# Summary bar chart showing final values after equal compute budget
# All methods run for 150 gradient evaluations (to normalise by compute)

def gd(gfn, lfn, x0, a, n):
    x = x0.copy().astype(float)
    h = [lfn(x)]
    for _ in range(n):
        x -= a * gfn(x)
        h.append(lfn(x))
    return h

def polyak(gfn, lfn, x0, fs, eps, n):
    x = x0.copy().astype(float)
    h = [lfn(x)]
    for _ in range(n):
        g = gfn(x); f = lfn(x)
        a = (f - fs) / (g@g + eps)
        x -= a*g; h.append(lfn(x))
    return h

def adagrad(gfn, lfn, x0, a0, eps, n):
    x = x0.copy().astype(float)
    G = np.zeros_like(x); h = [lfn(x)]
    for _ in range(n):
        g = gfn(x); G += g**2
        x -= a0/(np.sqrt(G)+eps)*g; h.append(lfn(x))
    return h

def rmsprop(gfn, lfn, x0, a0, b, eps, n):
    x = x0.copy().astype(float)
    v = np.zeros_like(x); h = [lfn(x)]
    for _ in range(n):
        g = gfn(x); v = b*v+(1-b)*g**2
        x -= a0/(np.sqrt(v)+eps)*g; h.append(lfn(x))
    return h

def heavy_ball(gfn, lfn, x0, a, b, n):
    x = x0.copy().astype(float)
    z = np.zeros_like(x); h = [lfn(x)]
    for _ in range(n):
        g = gfn(x); z = b*z+a*g; x -= z; h.append(lfn(x))
    return h

def nesterov(gfn, lfn, x0, a, bmax, n):
    x = x0.copy().astype(float)
    z = np.zeros_like(x); h = [lfn(x)]
    for k in range(1,n+1):
        bk = min((k-1)/(k+2), bmax)
        la = x + bk*z; g = gfn(la)
        z = bk*z - a*g; x += z; h.append(lfn(x))
    return h

def adam(gfn, lfn, x0, a, b1, b2, eps, n):
    x = x0.copy().astype(float)
    m = np.zeros_like(x); v = np.zeros_like(x); h = [lfn(x)]
    for t in range(1,n+1):
        g = gfn(x)
        m = b1*m+(1-b1)*g; v = b2*v+(1-b2)*g**2
        mh = m/(1-b1**t); vh = v/(1-b2**t)
        x -= a*mh/(np.sqrt(vh)+eps); h.append(lfn(x))
    return h

def newton_method(gfn, hfn, lfn, x0, a, n, reg=1e-8):
    x = x0.copy().astype(float); h = [lfn(x)]
    for _ in range(n):
        g = gfn(x); H = hfn(x)+reg*np.eye(2)
        try: p = np.linalg.solve(H, g)
        except: p = g
        x -= a*p; h.append(lfn(x))
    return h

# Benchmark B setup
def bench_b_hess(x):
    return np.array([[2-np.sin(x[0]), 0],[0, 10]])

x0_B = np.array([-1., 4.]); fstar_B = 0.7244
x0_C = np.array([-1., 1.]); fstar_C = 0.0

# Run all for 120 iters
N = 120
res_B = {
    'GD':          gd(bench_b_grad, bench_b, x0_B, 0.06, N)[-1],
    'Polyak':      polyak(bench_b_grad, bench_b, x0_B, 0, 1e-4, N)[-1],
    'Adagrad':     adagrad(bench_b_grad, bench_b, x0_B, 1.2, 1e-5, N)[-1],
    'RMSprop':     rmsprop(bench_b_grad, bench_b, x0_B, 0.14, 0.9, 1e-5, N)[-1],
    'Heavy Ball':  heavy_ball(bench_b_grad, bench_b, x0_B, 0.035, 0.90, N)[-1],
    'Nesterov':    nesterov(bench_b_grad, bench_b, x0_B, 0.035, 0.92, N)[-1],
    'Adam':        adam(bench_b_grad, bench_b, x0_B, 0.08, 0.82, 0.999, 1e-8, N)[-1],
    'Newton':      newton_method(bench_b_grad, bench_b_hess, bench_b, x0_B, 0.85, 20, 1e-8)[-1],
}

res_C = {
    'GD':          gd(rosen_grad, rosen, x0_C, 0.0012, N)[-1],
    'Polyak':      polyak(rosen_grad, rosen, x0_C, 0, 1e-3, N)[-1],
    'Adagrad':     adagrad(rosen_grad, rosen, x0_C, 0.45, 1e-5, N)[-1],
    'RMSprop':     rmsprop(rosen_grad, rosen, x0_C, 0.0035, 0.9, 1e-5, N)[-1],
    'Heavy Ball':  heavy_ball(rosen_grad, rosen, x0_C, 0.0008, 0.86, N)[-1],
    'Nesterov':    nesterov(rosen_grad, rosen, x0_C, 0.0007, 0.90, N)[-1],
    'Adam':        adam(rosen_grad, rosen, x0_C, 0.006, 0.80, 0.999, 1e-8, N)[-1],
    'Newton':      newton_method(rosen_grad, rosen_hess, rosen, x0_C, 0.22, 20, 1e-8)[-1],
}

methods = list(res_B.keys())
vals_B = [res_B[m] for m in methods]
vals_C = [res_C[m] for m in methods]

# Sort by Benchmark C performance
order = np.argsort(vals_C)[::-1]
methods_sorted = [methods[i] for i in order]
vals_B_sorted = [vals_B[i] for i in order]
vals_C_sorted = [vals_C[i] for i in order]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors_bar = plt.cm.viridis(np.linspace(0.1, 0.9, len(methods)))

x_pos = np.arange(len(methods))

ax = axes[0]
bars = ax.bar(x_pos, vals_B_sorted, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.axhline(fstar_B, color='r', linestyle='--', lw=1.5, label=f'$f^\\star = {fstar_B}$')
ax.set_xticks(x_pos); ax.set_xticklabels(methods_sorted, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Final $f(x_{120})$')
ax.set_title('Benchmark B: Final Objective (120 iters)', fontsize=10)
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars, vals_B_sorted):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=7.5)

ax2 = axes[1]
vals_C_plot = np.maximum(vals_C_sorted, 1e-3)
bars2 = ax2.bar(x_pos, vals_C_plot, color=colors_bar, edgecolor='black', linewidth=0.5)
ax2.axhline(fstar_C, color='r', linestyle='--', lw=1.5, label=f'$f^\\star = {fstar_C}$')
ax2.set_yscale('log')
ax2.set_xticks(x_pos); ax2.set_xticklabels(methods_sorted, rotation=35, ha='right', fontsize=9)
ax2.set_ylabel('Final $f(x_{120})$ (log scale)')
ax2.set_title('Benchmark C: Final Objective (120 iters, log scale)', fontsize=10)
ax2.legend(fontsize=9); ax2.grid(True, axis='y', alpha=0.3, which='both')
for bar, val in zip(bars2, vals_C_sorted):
    ax2.text(bar.get_x()+bar.get_width()/2, max(val*1.1, 1e-3*1.1), f'{val:.4f}',
             ha='center', va='bottom', fontsize=7)

plt.suptitle('All Methods Final Performance: Benchmarks B and C (120 iterations)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'all_methods_ranked.pdf'), bbox_inches='tight')
plt.close()
print("Saved: all_methods_ranked.pdf")

# ============================================================
# Figure 5: SGD Learning Rate Schedule Comparison
# ============================================================
# Compare constant vs decaying vs cosine annealing LR for mini-batch SGD

np.random.seed(42)
m, d = 1000, 2
theta_star = np.array([3.0, 4.0])
X = np.random.randn(m, d)
y = X @ theta_star + np.random.randn(m)

def full_grad(theta):
    return X.T @ (X @ theta - y) / m
def loss(theta):
    return 0.5 * np.mean((X @ theta - y)**2)

fstar_A = 0.5  # approximate noise floor

def sgd_schedule(x0, schedule_fn, batch, epochs):
    theta = x0.copy()
    hist = [loss(theta)]
    n_batches = m // batch
    step = 0
    for ep in range(epochs):
        idx = np.random.permutation(m)
        for b in range(n_batches):
            batch_idx = idx[b*batch:(b+1)*batch]
            Xb = X[batch_idx]; yb = y[batch_idx]
            g = Xb.T @ (Xb @ theta - yb) / batch
            alpha_k = schedule_fn(step)
            theta -= alpha_k * g
            step += 1
        hist.append(loss(theta))
    return np.array(hist)

total_steps = 50 * (m // 20)  # 50 epochs, batch=20

# Schedule functions
def constant(t): return 0.06
def step_decay(t):
    return 0.06 * (0.5 ** (t // (total_steps // 5)))
def cosine(t):
    return 0.001 + 0.5*(0.06 - 0.001) * (1 + np.cos(np.pi * t / total_steps))
def poly_decay(t):
    return 0.06 / (1 + 0.01 * t)

x0 = np.zeros(2)
h_const    = sgd_schedule(x0, constant,    20, 50)
h_step     = sgd_schedule(x0, step_decay,  20, 50)
h_cosine   = sgd_schedule(x0, cosine,      20, 50)
h_poly     = sgd_schedule(x0, poly_decay,  20, 50)

eps_arr = np.arange(51)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.semilogy(eps_arr, h_const,  'b-',  lw=2, label='Constant $\\alpha=0.06$')
ax.semilogy(eps_arr, h_step,   'g--', lw=2, label='Step decay (halve every 10 epochs)')
ax.semilogy(eps_arr, h_cosine, 'r-.', lw=2, label='Cosine annealing')
ax.semilogy(eps_arr, h_poly,   'm:',  lw=2, label='Polynomial decay $\\alpha_0/(1+ct)$')
ax.axhline(fstar_A, color='k', linestyle=':', lw=1.2, label='Noise floor $J^\\star \\approx 0.5$')
ax.set_xlabel('Epoch'); ax.set_ylabel('$J(\\theta)$')
ax.set_title('Mini-Batch SGD: Learning Rate Schedule Comparison\n(Benchmark A, batch=20, 50 epochs)', fontsize=10)
ax.legend(fontsize=8.5); ax.grid(True, which='both', alpha=0.3)

# Right: schedule values over time
t_arr = np.arange(total_steps)
ax2 = axes[1]
ax2.plot(t_arr, [constant(t) for t in t_arr],   'b-',  lw=2, label='Constant')
ax2.plot(t_arr, [step_decay(t) for t in t_arr],  'g--', lw=2, label='Step decay')
ax2.plot(t_arr, [cosine(t) for t in t_arr],      'r-.', lw=2, label='Cosine')
ax2.plot(t_arr, [poly_decay(t) for t in t_arr],  'm:',  lw=2, label='Polynomial')
ax2.set_xlabel('SGD step $t$'); ax2.set_ylabel('Learning rate $\\alpha_t$')
ax2.set_title('Learning Rate Schedule Values', fontsize=10)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.suptitle('Learning Rate Schedules for Mini-Batch SGD', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q2_lr_schedules.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q2_lr_schedules.pdf")

print("All Pass 7 figures generated successfully.")
