"""
Generate fifth batch of additional figures:
  1. convergence_kappa.pdf   -- iterations to epsilon precision vs condition number for GD, Nesterov, HB
  2. method_budget_C.pdf     -- objective vs total function evaluations (not iterations) on Rosenbrock
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
# FIGURE 1: Iterations to reach epsilon precision vs condition number
# Theoretical rates for GD, Nesterov (optimal 1st order), Heavy Ball (quadratic),
# and Newton (for comparison - log scale on x axis)
# ============================================================
kappas = np.logspace(0, 4, 300)   # kappa from 1 to 10000
eps = 1e-6                          # target precision: f(x_k) - f* < eps
D = 1.0                             # ||x0 - x*|| = 1 (normalised)

# GD: O(kappa * log(1/eps)) iterations
# f_k - f* <= (1 - 1/kappa)^k * D^2 * L/2 <= eps when k >= kappa * log(D^2*L/(2*eps))
# use kappa * log(1/eps) as approximate bound
k_gd = kappas * np.log(1/eps)

# Nesterov: O(sqrt(kappa) * log(1/eps))
k_nes = np.sqrt(kappas) * np.log(1/eps)

# Heavy Ball (optimal for quadratics): O(sqrt(kappa) * log(1/eps)) same as Nesterov but different const
k_hb = np.sqrt(kappas) * np.log(1/eps)   # same rate, shown for completeness

# Newton: O(log(log(1/eps))) approximately - essentially constant in kappa
# after O(log(kappa)) steps to reach neighbourhood
k_newton = np.log(np.log(1/eps) + 1) + np.log(kappas)   # log(kappa) steps to enter local conv region

# Polyak step (known f*): same as GD on strongly convex (same factor of kappa in rate)
# but the rate is exactly (1 - mu/L)^k = (1 - 1/kappa)^k
k_polyak = kappas * np.log(1/eps)

# Mark our three benchmark condition numbers
kappa_A = 1.12
kappa_B = 6.9
kappa_C = 2508

fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(kappas, k_gd,     'k--', lw=2, label='GD: $O(\\kappa \\log(1/\\varepsilon))$')
ax.loglog(kappas, k_nes,    'b-',  lw=2.5, label='Nesterov: $O(\\sqrt{\\kappa}\\log(1/\\varepsilon))$')
ax.loglog(kappas, k_newton, 'g-',  lw=2, label="Newton: $O(\\log(\\kappa) + \\log\\log(1/\\varepsilon))$")

# Shade the gap between GD and Nesterov
ax.fill_between(kappas, k_nes, k_gd, alpha=0.08, color='blue', label='Savings from Nesterov over GD')

# Mark the benchmark condition numbers
for kap, name, col in [(kappa_A, 'Bench. A\n($\\kappa\\approx1.12$)', 'orange'),
                        (kappa_B, 'Bench. B\n($\\kappa\\approx6.9$)', 'purple'),
                        (kappa_C, 'Bench. C\n($\\kappa\\approx2508$)', 'red')]:
    ax.axvline(kap, color=col, linestyle=':', lw=1.5, alpha=0.8)
    ax.text(kap * 1.15, 3, name, color=col, fontsize=8, va='bottom', rotation=0)

# Annotate the gap at kappa=2508
gd_2508 = 2508 * np.log(1/eps)
nes_2508 = np.sqrt(2508) * np.log(1/eps)
ax.annotate(f'At $\\kappa=2508$:\nGD: {gd_2508:.0f} iters\nNesterov: {nes_2508:.0f} iters\n'
            f'Speedup: {gd_2508/nes_2508:.1f}$\\times$',
            xy=(2508, nes_2508), xytext=(400, 2e6),
            fontsize=9, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.2))

ax.set_xlabel('Condition number $\\kappa = L/\\mu$', fontsize=12)
ax.set_ylabel('Iterations to $\\varepsilon = 10^{-6}$ precision', fontsize=12)
ax.set_title('Theoretical Iterations vs.\ Condition Number\n'
             '(First-order methods on $L$-smooth $\\mu$-strongly convex $f$)')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim([1, 1e4])
ax.set_ylim([1, 1e8])

plt.tight_layout()
plt.savefig('figures/convergence_kappa.pdf', bbox_inches='tight')
plt.close()
print("Saved: convergence_kappa.pdf")

# ============================================================
# FIGURE 2: Total function evaluations (budget) vs final objective on Rosenbrock
# Compare methods on equal function-evaluation budget, not iteration budget
# ============================================================
def loss_C(x): return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
def grad_C(x):
    return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2),
                     200*(x[1]-x[0]**2)])
def hessian_C(x):
    h11 = 2 + 1200*x[0]**2 - 400*x[1]
    h12 = -400*x[0]
    return np.array([[h11, h12], [h12, 200]])

x0_C = np.array([-1.0, 1.0])
n_iters = 300

# GD: 1 gradient eval per step = 1 FE (ignoring loss computation)
def gd_budget(alpha, n_iters):
    x = x0_C.copy().astype(float)
    fe, fh = [0], [loss_C(x)]
    for k in range(1, n_iters+1):
        x = x - alpha * grad_C(x)
        fe.append(k)       # 1 FE per iteration
        fh.append(loss_C(x))
    return np.array(fe), np.array(fh)

# Nesterov: 1 gradient eval per step = 1 FE
def nesterov_budget(alpha, beta_max, n_iters):
    x = x0_C.copy().astype(float)
    z = np.zeros_like(x)
    fe, fh = [0], [loss_C(x)]
    for k in range(1, n_iters+1):
        beta_k = min((k-1)/(k+2), beta_max)
        lookahead = x + beta_k * z
        g = grad_C(lookahead)
        z = beta_k * z - alpha * g
        x = x + z
        fe.append(k)
        fh.append(loss_C(x))
    return np.array(fe), np.array(fh)

# Newton: 1 Hessian + 1 gradient per step = equivalent to ~3 FE (Hessian needs d^2 FDs)
# Here we account as 3 FE per Newton step (gradient + Hessian via 2d FDs)
def newton_budget(alpha, n_iters, damping=1e-8):
    x = x0_C.copy().astype(float)
    fe, fh = [0], [loss_C(x)]
    total_fe = 0
    for _ in range(n_iters):
        g = grad_C(x)
        H = hessian_C(x) + damping * np.eye(2)
        # Clip step to avoid explosion
        try:
            p = np.linalg.solve(H, g)
        except:
            p = g
        if np.linalg.norm(alpha * p) > 10:
            p = p / np.linalg.norm(p) * 10 / alpha
        x = x - alpha * p
        total_fe += 3   # 1 grad + 2 extra for Hessian columns (d=2, central diff)
        fe.append(total_fe)
        fh.append(loss_C(x))
    return np.array(fe), np.array(fh)

# NRS: 2 function evaluations per step (no gradient)
def nrs_budget(alpha, delta, n_iters):
    x = x0_C.copy().astype(float)
    rng = np.random.RandomState(0)
    fe, fh = [0], [loss_C(x)]
    for k in range(1, n_iters+1):
        u = rng.randn(2); u /= np.linalg.norm(u)
        g_hat = (loss_C(x + delta*u) - loss_C(x)) / delta * u
        x = x - alpha * g_hat
        fe.append(2*k)    # 2 FE per step
        fh.append(loss_C(x))
    return np.array(fe), np.array(fh)

# FD GD (good): d+1 = 3 FE per step
def fd_gd_budget(alpha, delta, n_iters):
    x = x0_C.copy().astype(float)
    fe, fh = [0], [loss_C(x)]
    for k in range(1, n_iters+1):
        f0 = loss_C(x)
        g = np.zeros(2)
        for i in range(2):
            ei = np.zeros(2); ei[i] = 1
            g[i] = (loss_C(x + delta*ei) - f0) / delta
        x = x - alpha * g
        fe.append(3*k)   # d+1 = 3 FE per step
        fh.append(loss_C(x))
    return np.array(fe), np.array(fh)

fe_gd, fh_gd     = gd_budget(0.0012, n_iters)
fe_nes, fh_nes   = nesterov_budget(0.0007, 0.90, n_iters)
fe_nwt, fh_nwt   = newton_budget(0.22, 40)
fe_nrs, fh_nrs   = nrs_budget(0.005, 0.08, n_iters)
fe_fd,  fh_fd    = fd_gd_budget(0.0008, 0.05, n_iters)

# Clip for plotting
clip = 1e5
for fh in [fh_gd, fh_nes, fh_nwt, fh_nrs, fh_fd]:
    fh[:] = np.clip(fh, 1e-10, clip)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(fe_gd,  fh_gd,  'k--',  lw=1.8, label='GD ($\\alpha=0.0012$, 1 FE/iter)')
ax.semilogy(fe_nes, fh_nes, 'b-',   lw=2.0, label='Nesterov ($\\alpha=0.0007$, 1 FE/iter)')
ax.semilogy(fe_nwt, fh_nwt, 'g-o',  lw=2.0, ms=5, label="Newton ($\\alpha=0.22$, 3 FE/iter)")
ax.semilogy(fe_fd,  fh_fd,  'orange', lw=1.8, linestyle='-.', label='FD GD ($\\delta=0.05$, 3 FE/iter)')
ax.semilogy(fe_nrs, fh_nrs, 'r--',  lw=1.5, label='NRS ($\\alpha=0.005$, 2 FE/iter)')

ax.set_xlabel('Total function evaluations', fontsize=12)
ax.set_ylabel('$f(x_k)$ (log scale)', fontsize=12)
ax.set_title('Method Comparison on Rosenbrock: Equal Function-Evaluation Budget\n'
             '($d=2$, starting from $(-1,1)$, each method uses its natural FE/iter cost)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 400])
ax.set_ylim([0.1, 200])

plt.tight_layout()
plt.savefig('figures/method_budget_C.pdf', bbox_inches='tight')
plt.close()
print("Saved: method_budget_C.pdf")

print("All fifth-batch figures generated.")
