"""Pass 17: Degenerate Newton comparison, condition number sensitivity, penalty convergence theory."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
np.random.seed(42)

# ============================================================
# Figure 1: q3_degenerate_newton_comparison.pdf
# Compare Newton on g(x)=x^p for p in {2,3,4,6}
# ============================================================

def newton_gxp(p, x0, n_iter):
    """Newton's method on g(x) = x^p.
    g'(x) = p*x^(p-1), g''(x) = p*(p-1)*x^(p-2)
    Newton update: x_{k+1} = x_k - g'(x_k)/g''(x_k)
                            = x_k - x_k/(p-1)
                            = x_k * (1 - 1/(p-1))
                            = x_k * (p-2)/(p-1)
    """
    xs = [x0]
    x = x0
    for _ in range(n_iter):
        if abs(x) < 1e-300:
            x = 0.0
            xs.append(x)
            continue
        # g'(x) = p * x^(p-1), g''(x) = p*(p-1)*x^(p-2)
        gprime = p * x**(p - 1)
        gdoubleprime = p * (p - 1) * x**(p - 2)
        if abs(gdoubleprime) < 1e-300:
            x = 0.0
        else:
            x = x - gprime / gdoubleprime
        xs.append(x)
    return np.array(xs)

p_vals = [2, 3, 4, 6]
colors = {2: 'blue', 3: 'green', 4: 'orange', 6: 'red'}
x0 = 0.5
n_iter = 20

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
ax2 = axes[1]

for p in p_vals:
    xs = newton_gxp(p, x0, n_iter)
    iters = np.arange(len(xs))
    abs_xs = np.abs(xs)
    # Replace zeros with very small number for log plot
    abs_xs_plot = np.where(abs_xs > 1e-300, abs_xs, 1e-300)
    label = f'$p={p}$'
    ax1.semilogy(iters, abs_xs_plot, '-o', color=colors[p], label=label, markersize=4, lw=1.8)

ax1.set_xlabel('Iteration $k$', fontsize=11)
ax1.set_ylabel('$|x_k|$', fontsize=11)
ax1.set_title("Newton's Method on $g(x)=x^p$: Iterates", fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, n_iter)

# Right panel: log(|e_{k+1}|) / log(|e_k|) ratio
# x*=0, so error_k = |x_k - x*| = |x_k|
for p in p_vals:
    xs = newton_gxp(p, x0, n_iter)
    errors = np.abs(xs)
    ratios = []
    for k in range(1, len(errors) - 1):
        ek = errors[k]
        ek1 = errors[k + 1]
        if ek > 1e-290 and ek1 > 1e-290:
            log_ek = np.log(ek)
            log_ek1 = np.log(ek1)
            if abs(log_ek) > 1e-10:
                ratios.append((k, log_ek1 / log_ek))
    if ratios:
        ks, rs = zip(*ratios)
        ax2.plot(ks, rs, '-o', color=colors[p], label=f'$p={p}$', markersize=4, lw=1.8)

# Reference lines for rho values
rho_refs = {3: 0.5, 4: 2/3, 6: 4/5}
linestyles = {3: '--', 4: '-.', 6: ':'}
for p, rho in rho_refs.items():
    ax2.axhline(rho, color=colors[p], linestyle=linestyles[p], lw=1.2, alpha=0.7,
                label=f'$\\rho={rho:.2f}$ ($p={p}$)')

ax2.set_xlabel('Iteration $k$', fontsize=11)
ax2.set_ylabel(r'$\log|e_{k+1}| / \log|e_k|$', fontsize=11)
ax2.set_title('Linear Rate Constant $\\rho = (p-2)/(p-1)$', fontsize=11)
ax2.legend(fontsize=9, ncol=2)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, n_iter - 1)
ax2.set_ylim(-0.1, 1.1)

fig.suptitle("Degenerate Newton's Method: $g(x)=x^p$, $x_0=0.5$", fontsize=13, fontweight='bold')
fig.tight_layout()
fname1 = os.path.join(OUTDIR, 'q3_degenerate_newton_comparison.pdf')
fig.savefig(fname1, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved: q3_degenerate_newton_comparison.pdf")

# ============================================================
# Figure 2: q1_condition_number_comparison.pdf
# Effect of condition number on GD, Heavy Ball, Adagrad, RMSprop
# ============================================================

kappas = [1.5, 5, 20, 100, 500, 2508]
n_iters_q1 = 120
x0_q1 = np.array([2.0, 2.0])


def quadratic_f(x, kappa):
    return 0.5 * (x[0]**2 + kappa * x[1]**2)

def quadratic_grad(x, kappa):
    return np.array([x[0], kappa * x[1]])


def run_gd(kappa, n_iters):
    """Gradient descent with alpha = 1/(1+kappa)."""
    alpha = 1.0 / (1.0 + kappa)
    x = x0_q1.copy()
    for _ in range(n_iters):
        g = quadratic_grad(x, kappa)
        x = x - alpha * g
    return quadratic_f(x, kappa)


def run_hb(kappa, n_iters):
    """Heavy Ball with optimal parameters for quadratic.
    Optimal: alpha = 4/(sqrt(kappa)+1)^2, beta = ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2
    """
    sk = np.sqrt(kappa)
    alpha = 4.0 / (sk + 1.0)**2
    beta = ((sk - 1.0) / (sk + 1.0))**2
    x = x0_q1.copy()
    x_prev = x0_q1.copy()
    for _ in range(n_iters):
        g = quadratic_grad(x, kappa)
        x_new = x - alpha * g + beta * (x - x_prev)
        x_prev = x
        x = x_new
    return quadratic_f(x, kappa)


def run_adagrad(kappa, n_iters, alpha0=0.8):
    """Adagrad."""
    x = x0_q1.copy()
    G = np.zeros(2)
    eps = 1e-8
    for _ in range(n_iters):
        g = quadratic_grad(x, kappa)
        G = G + g**2
        x = x - alpha0 * g / (np.sqrt(G) + eps)
    return quadratic_f(x, kappa)


def run_rmsprop(kappa, n_iters, alpha0=0.15, beta=0.9):
    """RMSprop."""
    x = x0_q1.copy()
    v = np.zeros(2)
    eps = 1e-8
    for _ in range(n_iters):
        g = quadratic_grad(x, kappa)
        v = beta * v + (1 - beta) * g**2
        x = x - alpha0 * g / (np.sqrt(v) + eps)
    return quadratic_f(x, kappa)


final_gd = []
final_hb = []
final_adagrad = []
final_rmsprop = []

for kappa in kappas:
    final_gd.append(run_gd(kappa, n_iters_q1))
    final_hb.append(run_hb(kappa, n_iters_q1))
    final_adagrad.append(run_adagrad(kappa, n_iters_q1))
    final_rmsprop.append(run_rmsprop(kappa, n_iters_q1))

fig2, ax = plt.subplots(1, 1, figsize=(9, 6))

kappas_arr = np.array(kappas)
ax.loglog(kappas_arr, np.abs(final_gd), '-o', color='blue', lw=2, markersize=7, label='GD ($\\alpha=1/(1+\\kappa)$)')
ax.loglog(kappas_arr, np.abs(final_hb), '-s', color='green', lw=2, markersize=7, label='Heavy Ball (optimal params)')
ax.loglog(kappas_arr, np.abs(final_adagrad), '-^', color='orange', lw=2, markersize=7, label='Adagrad ($\\alpha_0=0.8$)')
ax.loglog(kappas_arr, np.abs(final_rmsprop), '-D', color='red', lw=2, markersize=7, label='RMSprop ($\\alpha_0=0.15, \\beta=0.9$)')

ax.set_xlabel('Condition Number $\\kappa$', fontsize=12)
ax.set_ylabel('Final Suboptimality $f(x^{(T)}) - f^*$', fontsize=12)
ax.set_title('$\\kappa$-Sensitivity: Final Suboptimality after 120 Iterations\n'
             r'$f(x) = \frac{1}{2}(x_1^2 + \kappa x_2^2)$, $x_0=[2,2]$', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')
ax.set_xticks(kappas)
ax.set_xticklabels([str(k) for k in kappas], fontsize=9)

fig2.tight_layout()
fname2 = os.path.join(OUTDIR, 'q1_condition_number_comparison.pdf')
fig2.savefig(fname2, bbox_inches='tight', dpi=150)
plt.close(fig2)
print(f"Saved: q1_condition_number_comparison.pdf")

# ============================================================
# Figure 3: q5_penalty_convergence_theory.pdf
# O(1/lambda) penalty convergence
# ============================================================

# Constrained problem: min f(x) + lambda * max(0, 0.5 - x1)^2
# f(x) = (x1-1)^2 + 5*(x2-2)^2 + sin(x1)
# The true constraint is x1 >= 0.5 (inequality constraint h(x)=0.5-x1 <= 0)
# Penalised: min (x1-1)^2 + 5*(x2-2)^2 + sin(x1) + lambda*max(0, 0.5-x1)^2

x1_exact = 0.582   # given exact constrained minimum x1 component
f_star = 0.7244    # given optimal f value


def penalty_f(x, lam):
    x1, x2 = x
    f = (x1 - 1)**2 + 5 * (x2 - 2)**2 + np.sin(x1)
    penalty = lam * max(0.0, 0.5 - x1)**2
    return f + penalty


def penalty_grad(x, lam):
    x1, x2 = x
    df_dx1 = 2 * (x1 - 1) + np.cos(x1)
    df_dx2 = 10 * (x2 - 2)
    if x1 < 0.5:
        dp_dx1 = -2 * lam * (0.5 - x1)
    else:
        dp_dx1 = 0.0
    dp_dx2 = 0.0
    return np.array([df_dx1 + dp_dx1, df_dx2 + dp_dx2])


def run_penalty_gd(lam, n_gd=200, alpha_gd=0.05):
    """200 GD steps to minimise penalised objective."""
    x = np.array([1.5, 2.0])  # starting point
    for _ in range(n_gd):
        g = penalty_grad(x, lam)
        x = x - alpha_gd * g
    return x


lambdas = np.logspace(np.log10(0.1), np.log10(100), 30)

x1_stars = []
f_vals = []

for lam in lambdas:
    x_opt = run_penalty_gd(lam)
    x1_stars.append(x_opt[0])
    # compute f without penalty at the found point
    x1, x2 = x_opt
    fval = (x1 - 1)**2 + 5 * (x2 - 2)**2 + np.sin(x1)
    f_vals.append(fval)

x1_stars = np.array(x1_stars)
f_vals = np.array(f_vals)

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))

ax_left = axes3[0]
ax_right = axes3[1]

# Left panel: |x1*(lambda) - x1*(exact)| vs lambda on log-log
errors_x1 = np.abs(x1_stars - x1_exact)
# Avoid zeros
errors_x1_plot = np.where(errors_x1 > 1e-15, errors_x1, 1e-15)

ax_left.loglog(lambdas, errors_x1_plot, 'b-o', lw=2, markersize=5, label='$|x_1^*(\\lambda) - x_1^*|$')

# O(1/lambda) reference line
ref_lambda = lambdas
ref_c = errors_x1_plot[5] * lambdas[5]   # calibrate at index 5
ax_left.loglog(ref_lambda, ref_c / ref_lambda, 'k--', lw=1.5, label='$O(1/\\lambda)$ reference')

ax_left.set_xlabel('Penalty parameter $\\lambda$', fontsize=11)
ax_left.set_ylabel('$|x_1^*(\\lambda) - x_1^*|$', fontsize=11)
ax_left.set_title('Penalty Error: $O(1/\\lambda)$ Convergence\n'
                  r'$x_1^* \approx 0.582$', fontsize=11)
ax_left.legend(fontsize=10)
ax_left.grid(True, alpha=0.3, which='both')

# Right panel: f(x*(lambda)) approaching f*
ax_right.semilogx(lambdas, f_vals, 'r-o', lw=2, markersize=5, label='$f(x^*(\\lambda))$')
ax_right.axhline(f_star, color='k', linestyle='--', lw=1.5, label=f'$f^* = {f_star}$')

ax_right.set_xlabel('Penalty parameter $\\lambda$', fontsize=11)
ax_right.set_ylabel('$f(x^*(\\lambda))$', fontsize=11)
ax_right.set_title('Objective Approaching $f^*$ as $\\lambda \\to \\infty$\n'
                   r'$f(x) = (x_1-1)^2 + 5(x_2-2)^2 + \sin(x_1)$', fontsize=11)
ax_right.legend(fontsize=10)
ax_right.grid(True, alpha=0.3, which='both')

fig3.suptitle('Penalty Method Convergence: $O(1/\\lambda)$ Theory',
              fontsize=13, fontweight='bold')
fig3.tight_layout()
fname3 = os.path.join(OUTDIR, 'q5_penalty_convergence_theory.pdf')
fig3.savefig(fname3, bbox_inches='tight', dpi=150)
plt.close(fig3)
print(f"Saved: q5_penalty_convergence_theory.pdf")
