"""Pass 10 figures: condition number effect on GD, mini-batch variance,
centered vs forward FD accuracy, convergence rate illustration for all theorems."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(42)

# ============================================================
# Figure 1: Condition Number Effect on Convergence
# ============================================================
# For a quadratic f(x) = x^T H x / 2 with H = diag(1, kappa),
# show how the condition number affects GD convergence

kappas = [1, 5, 10, 50, 100, 500, 2508]
N = 500

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for kappa in kappas:
    mu = 1.0; L = kappa
    alpha = 2.0 / (mu + L)  # optimal step for GD
    rate = (kappa - 1) / (kappa + 1)
    conv = rate**(2*np.arange(N+1))
    ax.semilogy(np.arange(N+1), conv, linewidth=1.8,
                label=f'$\\kappa={kappa}$, rate$={(rate):.4f}$')

ax.set_xlabel('Iteration $k$')
ax.set_ylabel('$(f(x_k) - f^\\star) / (f(x_0) - f^\\star)$')
ax.set_title('GD Convergence Rate vs Condition Number\n(optimal $\\alpha^\\star = 2/(\\mu+L)$)', fontsize=10)
ax.legend(fontsize=7.5, ncol=2); ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(0, N)

ax2 = axes[1]
kappa_range = np.logspace(0, 4, 300)
gd_iters = (kappa_range - 1) / (kappa_range + 1)  # per-iteration contraction
hb_iters = (np.sqrt(kappa_range) - 1) / (np.sqrt(kappa_range) + 1)
# Iterations to reach 1e-6 accuracy:
eps = 1e-6
iters_gd = np.ceil(np.log(eps) / np.log(gd_iters**2 + 1e-300))
iters_hb = np.ceil(np.log(eps) / np.log(hb_iters**2 + 1e-300))
iters_nes = np.ceil(np.sqrt(kappa_range) * np.log(1.0/eps) / 2)  # O(sqrt(kappa) log 1/eps)

ax2.loglog(kappa_range, iters_gd,  'b-',  lw=2, label='GD: $O(\\kappa \\log 1/\\varepsilon)$')
ax2.loglog(kappa_range, iters_hb,  'g--', lw=2, label='Heavy Ball: $O(\\sqrt{\\kappa} \\log 1/\\varepsilon)$')
ax2.loglog(kappa_range, iters_nes, 'r-.',  lw=2, label='Nesterov: $O(\\sqrt{\\kappa} \\log 1/\\varepsilon)$')
ax2.axvline(2508, color='purple', linestyle=':', lw=1.5, label='Rosenbrock $\\kappa=2508$')
ax2.set_xlabel('Condition number $\\kappa = L/\\mu$')
ax2.set_ylabel('Iterations to $\\varepsilon = 10^{-6}$ accuracy')
ax2.set_title('Iterations Required vs Condition Number\n(to reach $\\varepsilon = 10^{-6}$ accuracy)', fontsize=10)
ax2.legend(fontsize=9); ax2.grid(True, which='both', alpha=0.3)

plt.suptitle('Effect of Condition Number on First-Order Method Complexity', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'condition_number_effect.pdf'), bbox_inches='tight')
plt.close()
print("Saved: condition_number_effect.pdf")

# ============================================================
# Figure 2: Newton Convergence: log(error) vs iteration
# ============================================================
# Show that Newton on quadratic f(x) = (x-1)^2 + 10(y-2)^2 converges in 1 step
# and on Rosenbrock converges slowly due to damping

def rosen(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def rosen_grad(x):
    return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
def rosen_hess(x):
    h11 = 2-400*(x[1]-x[0]**2)+1600*x[0]**2
    h12 = -400*x[0]; h22 = 200.0
    return np.array([[h11,h12],[h12,h22]])

def bench_b(x): return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def bench_b_grad(x): return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])
def bench_b_hess(x): return np.array([[2-np.sin(x[0]),0],[0,10]])

def newton_hist(gfn, hfn, lfn, x0, alpha, n, reg=1e-8):
    x = x0.copy().astype(float); h = [lfn(x)]
    for _ in range(n):
        g = gfn(x); H = hfn(x)+reg*np.eye(2)
        try: p = np.linalg.solve(H,g)
        except: p = g
        x -= alpha*p; h.append(lfn(x))
    return np.array(h)

def gd_hist(gfn, lfn, x0, a, n):
    x = x0.copy().astype(float); h = [lfn(x)]
    for _ in range(n):
        x -= a*gfn(x); h.append(lfn(x))
    return np.array(h)

# Bench A (quadratic): Newton in 1 step
def make_quadratic_from_data():
    np.random.seed(42)
    m, d = 1000, 2; t_star = np.array([3.,4.])
    X = np.random.randn(m,d); y = X@t_star + np.random.randn(m)
    def loss(t): return 0.5*np.mean((X@t-y)**2)
    def grad(t): return X.T@(X@t-y)/m
    def hess(t): return X.T@X/m
    return loss, grad, hess, np.array([0.,0.])

lA, gA, hA, x0A = make_quadratic_from_data()
fstar_A = lA(np.linalg.solve(hA(x0A), gA(x0A)*0 + hA(x0A)@x0A - gA(x0A)))
# Actually just run Newton 1 step to get f*
h_newton_A = newton_hist(gA, hA, lA, x0A, 1.0, 20)
h_gd_A = gd_hist(gA, lA, x0A, 0.08, 80)

fstar_B = 0.7244; x0B = np.array([-1.,4.])
h_newton_B = newton_hist(bench_b_grad, bench_b_hess, bench_b, x0B, 0.85, 20)
h_gd_B = gd_hist(bench_b_grad, bench_b, x0B, 0.06, 80)

fstar_C = 0.0; x0C = np.array([-1.,1.])
h_newton_C = newton_hist(rosen_grad, rosen_hess, rosen, x0C, 0.22, 20)
h_gd_C = gd_hist(rosen_grad, rosen, x0C, 0.001, 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, hn, hg, fstar, title, ylim in [
    (axes[0], h_newton_A, h_gd_A, h_newton_A[-1], 'Benchmark A (Quadratic)', None),
    (axes[1], h_newton_B, h_gd_B, fstar_B, 'Benchmark B (Toy NN)', None),
    (axes[2], h_newton_C, h_gd_C, fstar_C, 'Benchmark C (Rosenbrock)', None),
]:
    sub_n = np.maximum(hn - fstar, 1e-16)
    sub_g = np.maximum(hg - fstar, 1e-16)
    ax.semilogy(np.arange(len(sub_n)), sub_n, 'r-o', lw=2, ms=5, label=f'Newton ({len(hn)-1} iters)')
    ax.semilogy(np.arange(len(sub_g)), sub_g, 'b-', lw=2, label=f'GD ({len(hg)-1} iters)')
    ax.set_xlabel('Iteration $k$')
    ax.set_ylabel('$f(x_k) - f^\\star$')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

# Annotate Bench A: Newton hits machine precision at k=1
axes[0].annotate('1 iteration\nto machine\nprecision', (1, sub_n[1]),
                 textcoords='offset points', xytext=(30, 10), fontsize=8,
                 arrowprops=dict(arrowstyle='->', color='red'))

plt.suptitle("Newton's Method vs Gradient Descent: Convergence on All 3 Benchmarks", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q3_newton_vs_gd_all.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q3_newton_vs_gd_all.pdf")

# ============================================================
# Figure 3: GD Oscillation Pattern for Ill-Conditioned Problem
# ============================================================
# Show the zig-zag oscillation pattern of GD on an ellipse
# f(x1,x2) = 0.5*(x1^2 + kappa*x2^2)

kappa_illcond = 100
x0 = np.array([1.0, 1.0])

def quad_illcond_grad(x):
    return np.array([x[0], kappa_illcond * x[1]])
def quad_illcond_loss(x):
    return 0.5*(x[0]**2 + kappa_illcond*x[1]**2)

L_ill = kappa_illcond  # max eigenvalue
mu_ill = 1.0

alpha_gd  = 2.0/(mu_ill + L_ill)  # optimal
alpha_too_big = 1.8/L_ill  # slightly aggressive

def trajectory(gfn, x0, a, n):
    x = x0.copy().astype(float); xs = [x.copy()]
    for _ in range(n):
        x -= a * gfn(x); xs.append(x.copy())
    return np.array(xs)

traj_opt = trajectory(quad_illcond_grad, x0, alpha_gd, 60)
traj_big = trajectory(quad_illcond_grad, x0, alpha_too_big, 60)

# Also Heavy Ball with optimal params
beta_hb = ((np.sqrt(kappa_illcond)-1)/(np.sqrt(kappa_illcond)+1))**2
alpha_hb = (1-np.sqrt(beta_hb))**2 / mu_ill

def hb_trajectory(gfn, x0, a, b, n):
    x = x0.copy().astype(float); z = np.zeros_like(x); xs = [x.copy()]
    for _ in range(n):
        z = b*z + a*gfn(x); x -= z; xs.append(x.copy())
    return np.array(xs)

traj_hb = hb_trajectory(quad_illcond_grad, x0, alpha_hb, beta_hb, 60)

# Grid for contours
xx = np.linspace(-1.2, 1.2, 300)
yy = np.linspace(-1.2, 1.2, 300)
XX, YY = np.meshgrid(xx, yy)
ZZ = 0.5*(XX**2 + kappa_illcond*YY**2)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

ax = axes[0]
ax.contourf(XX, YY, ZZ, levels=20, cmap='YlOrRd', alpha=0.3)
ax.contour(XX, YY, ZZ, levels=20, colors='gray', linewidths=0.4, alpha=0.6)
ax.plot(traj_opt[:,0], traj_opt[:,1], 'b-o', ms=3, lw=1.5,
        label=f'GD optimal $\\alpha={alpha_gd:.4f}$ ($= 2/(\\mu+L)$)')
ax.plot(traj_hb[:,0], traj_hb[:,1], 'g-s', ms=3, lw=1.5,
        label=f'Heavy Ball $\\alpha={alpha_hb:.5f}$, $\\beta={beta_hb:.3f}$')
ax.plot(0, 0, 'r*', ms=12, label='Optimum $(0,0)$')
ax.plot(x0[0], x0[1], 'ks', ms=8)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title(f'GD vs Heavy Ball on $f = (x_1^2 + {kappa_illcond}x_2^2)/2$\n'
             f'($\\kappa = {kappa_illcond}$, 60 iters)', fontsize=10)
ax.legend(fontsize=8.5); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')

ax2 = axes[1]
f_opt = np.array([quad_illcond_loss(x) for x in traj_opt])
f_hb  = np.array([quad_illcond_loss(x) for x in traj_hb])
k_arr = np.arange(len(traj_opt))
ax2.semilogy(k_arr, f_opt, 'b-', lw=2, label='GD (optimal $\\alpha$)')
ax2.semilogy(k_arr, f_hb,  'g-', lw=2, label='Heavy Ball (optimal $\\alpha, \\beta$)')
# Theory
rate_gd = (kappa_illcond-1)/(kappa_illcond+1)
rate_hb = (np.sqrt(kappa_illcond)-1)/(np.sqrt(kappa_illcond)+1)
ax2.semilogy(k_arr, f_opt[0] * rate_gd**(2*k_arr), 'b--', lw=1, alpha=0.7,
             label=f'GD theory: $(({kappa_illcond}-1)/({kappa_illcond}+1))^{{2k}}$')
ax2.semilogy(k_arr, f_hb[0] * rate_hb**(2*k_arr), 'g--', lw=1, alpha=0.7,
             label=f'HB theory: $((\\sqrt{{{kappa_illcond}}}-1)/(\\sqrt{{{kappa_illcond}}}+1))^{{2k}}$')
ax2.set_xlabel('Iteration $k$')
ax2.set_ylabel('$f(x_k)$')
ax2.set_title(f'Convergence: GD vs HB ($\\kappa = {kappa_illcond}$)\n'
              f'HB converges $\\sqrt{{\\kappa}}$-times faster', fontsize=10)
ax2.legend(fontsize=8); ax2.grid(True, which='both', alpha=0.3)

plt.suptitle(f'Ill-Conditioning ($\\kappa={kappa_illcond}$): Zig-Zag GD vs Momentum HB', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'gd_vs_hb_oscillation.pdf'), bbox_inches='tight')
plt.close()
print("Saved: gd_vs_hb_oscillation.pdf")

# ============================================================
# Figure 4: Variance of SGD gradient estimate vs batch size
# ============================================================
np.random.seed(42)
m = 1000; d = 2
theta_star = np.array([3.,4.])
X = np.random.randn(m,d); y = X@theta_star + np.random.randn(m)
full_grad = X.T@(X@theta_star-y)/m

# Compute per-sample gradients at theta_star
grads = np.array([X[i:i+1].T @ (X[i:i+1]@theta_star - y[i:i+1]) for i in range(m)]).squeeze()
# grads shape: (m, d)
sigma_sq = np.mean(np.sum((grads - full_grad)**2, axis=1))

batch_sizes = np.arange(1, 201)
# Theoretical variance (without replacement)
var_theory = (m - batch_sizes) / (batch_sizes * (m-1)) * sigma_sq
# With replacement
var_wr = sigma_sq / batch_sizes

# Empirical: compute actual variance for a few batch sizes
batch_test = [1, 5, 10, 20, 50, 100, 200]
var_empirical = []
for b in batch_test:
    samp_vars = []
    for _ in range(1000):
        idx = np.random.choice(m, b, replace=False)
        gb = np.mean(grads[idx], axis=0)
        samp_vars.append(np.sum((gb - full_grad)**2))
    var_empirical.append(np.mean(samp_vars))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(batch_sizes, var_theory, 'b-', lw=2, label='Without replacement: $(m-b)/(b(m-1))\\sigma^2$')
ax.plot(batch_sizes, var_wr, 'r--', lw=2, label='With replacement: $\\sigma^2/b$')
ax.scatter(batch_test, var_empirical, color='g', s=80, zorder=5, label='Empirical (1000 resamples)')
ax.axvline(m, color='k', linestyle=':', lw=1, label=f'Full batch ($b=m={m}$)')
ax.set_xlabel('Batch size $b$')
ax.set_ylabel('Gradient variance $\\mathbb{E}[\\|\\hat{g} - \\nabla J\\|^2]$')
ax.set_title('Mini-Batch SGD Gradient Variance vs Batch Size\n'
             '(Benchmark A, $m=1000$, $d=2$)', fontsize=10)
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 210)

ax2 = axes[1]
# Log-log version for range insight
ax2.loglog(batch_sizes, var_theory, 'b-', lw=2, label='Without replacement')
ax2.loglog(batch_sizes, var_wr, 'r--', lw=2, label='With replacement')
ax2.scatter(batch_test, var_empirical, color='g', s=80, zorder=5, label='Empirical')
ax2.loglog(batch_sizes, var_wr[0]/batch_sizes, 'k:', lw=1, alpha=0.7, label='$O(1/b)$ reference')
ax2.set_xlabel('Batch size $b$')
ax2.set_ylabel('Gradient variance (log scale)')
ax2.set_title('Gradient Variance vs Batch Size (log--log)\n'
              'W/R variance $\\propto 1/b$; W/O/R variance $\\to 0$ at $b=m$', fontsize=10)
ax2.legend(fontsize=8.5); ax2.grid(True, which='both', alpha=0.3)

plt.suptitle('SGD Gradient Variance: Finite Population Correction $(m-b)/(b(m-1))$', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'q2_sgd_variance.pdf'), bbox_inches='tight')
plt.close()
print("Saved: q2_sgd_variance.pdf")

print("All Pass 10 figures generated successfully.")
