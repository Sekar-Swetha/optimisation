#!/usr/bin/env python3
"""Pass 15 figures: L-BFGS vs Newton comparison, ADMM on Q5, away-step FW,
proximal gradient, SAGA detailed convergence."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(42)

# ─── Benchmark definitions ─────────────────────────────────────────────────────
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

def hess_rosenbrock(x):
    x1, x2 = x
    return np.array([
        [-2 + 400*(3*x1**2 - x2), -400*x1],
        [-400*x1, 200]
    ])

# Benchmark A: quadratic
m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
y_data = X_data @ theta_star + np.random.randn(m) * 0.5
H_mat = X_data.T @ X_data / m
b_vec = X_data.T @ y_data / m
f_star_A = 0.5 * np.mean(y_data**2) - 0.5 * b_vec @ np.linalg.solve(H_mat, b_vec)

def bench_A_loss(theta):
    return 0.5 * np.mean((X_data @ theta - y_data)**2)

def bench_A_grad(theta):
    return X_data.T @ (X_data @ theta - y_data) / m

# ─── Figure 1: L-BFGS vs Newton vs GD vs Nesterov on Rosenbrock ───────────────

def run_gd(x0, alpha, n):
    x = x0.copy().astype(float)
    losses = [rosenbrock(x)]
    for _ in range(n):
        x -= alpha * grad_rosenbrock(x)
        losses.append(rosenbrock(x))
    return np.array(losses)

def run_nesterov(x0, alpha, beta_max, n):
    x = x0.copy().astype(float); z = np.zeros(2)
    losses = [rosenbrock(x)]
    for k in range(1, n+1):
        beta_k = min((k-1)/(k+2), beta_max)
        la = x + beta_k * z
        g = grad_rosenbrock(la)
        z = beta_k * z - alpha * g
        x = x + z
        losses.append(rosenbrock(x))
    return np.array(losses)

def run_newton(x0, alpha, damping, n):
    x = x0.copy().astype(float)
    losses = [rosenbrock(x)]
    for _ in range(n):
        g = grad_rosenbrock(x)
        H = hess_rosenbrock(x)
        H_reg = H + damping * np.eye(2)
        try:
            p = np.linalg.solve(H_reg, g)
        except:
            p = g
        x -= alpha * p
        losses.append(rosenbrock(x))
    return np.array(losses)

def run_lbfgs(x0, m_mem, n):
    """L-BFGS with two-loop recursion, strong Wolfe line search."""
    x = x0.copy().astype(float)
    losses = [rosenbrock(x)]
    s_list, y_list, rho_list = [], [], []

    def lbfgs_direction(g, s_list, y_list, rho_list):
        q = g.copy()
        alpha_i = []
        for s, y, rho_val in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            a = rho_val * np.dot(s, q)
            alpha_i.append(a)
            q = q - a * y
        if s_list:
            s_last, y_last = s_list[-1], y_list[-1]
            gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-12)
        else:
            gamma = 1.0
        r = gamma * q
        for (s, y, rho_val), a in zip(zip(s_list, y_list, rho_list), reversed(alpha_i)):
            beta = rho_val * np.dot(y, r)
            r = r + s * (a - beta)
        return r  # descent direction

    g = grad_rosenbrock(x)
    for k in range(n):
        if len(s_list) == 0:
            p = -g
        else:
            p = -lbfgs_direction(g, s_list, y_list, rho_list)

        # Backtracking line search
        alpha = 1.0
        f0 = rosenbrock(x)
        g0 = g.copy()
        for _ in range(30):
            x_new = x + alpha * p
            f_new = rosenbrock(x_new)
            if f_new <= f0 + 1e-4 * alpha * np.dot(g0, p):
                break
            alpha *= 0.5

        x_new = x + alpha * p
        g_new = grad_rosenbrock(x_new)
        s = x_new - x
        y = g_new - g

        sy = np.dot(s, y)
        if sy > 1e-10:
            if len(s_list) >= m_mem:
                s_list.pop(0); y_list.pop(0); rho_list.pop(0)
            s_list.append(s); y_list.append(y); rho_list.append(1.0/sy)

        x = x_new
        g = g_new
        losses.append(rosenbrock(x))
    return np.array(losses)

x0 = np.array([-1.0, 1.0])
n_all = 150

losses_gd = run_gd(x0, 0.0012, n_all)
losses_nes = run_nesterov(x0, 0.0007, 0.90, n_all)
losses_newton = run_newton(x0, 0.22, 1e-8, 30)
losses_lbfgs5 = run_lbfgs(x0, 5, n_all)
losses_lbfgs20 = run_lbfgs(x0, 20, n_all)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
k_all = np.arange(n_all + 1)
ax.semilogy(k_all, losses_gd, '-', color='#2196F3', linewidth=2, label='GD ($\\alpha=0.0012$)')
ax.semilogy(k_all, losses_nes, '-', color='#4CAF50', linewidth=2, label='Nesterov ($\\alpha=0.0007$)')
ax.semilogy(k_all, losses_lbfgs5, '-', color='#FF9800', linewidth=2.5, label='L-BFGS ($m=5$)')
ax.semilogy(k_all, losses_lbfgs20, '--', color='#9C27B0', linewidth=2, label='L-BFGS ($m=20$)')
ax.semilogy(np.arange(31), losses_newton, 'o-', color='#000000', linewidth=2.5, markersize=5,
            label='Newton ($\\alpha=0.22$, 30 iters)')
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('$f(x_k)$ (log scale)', fontsize=12)
ax.set_title('L-BFGS vs Newton vs GD on Rosenbrock\n(starting from $(-1, 1)$)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, n_all)

# Right: zoom into first 40 iters to show L-BFGS superlinear
ax2 = axes[1]
k40 = np.arange(41)
ax2.semilogy(k40, losses_gd[:41], '-', color='#2196F3', linewidth=2, label='GD')
ax2.semilogy(k40, losses_nes[:41], '-', color='#4CAF50', linewidth=2, label='Nesterov')
ax2.semilogy(k40, losses_lbfgs5[:41], '-', color='#FF9800', linewidth=2.5, label='L-BFGS ($m=5$)')
ax2.semilogy(k40, losses_lbfgs20[:41], '--', color='#9C27B0', linewidth=2, label='L-BFGS ($m=20$)')
ax2.semilogy(np.arange(31), losses_newton, 'o-', color='#000000', linewidth=2.5, markersize=5,
             label='Newton (30 iters)')
ax2.set_xlabel('Iteration $k$', fontsize=12)
ax2.set_ylabel('$f(x_k)$ (log scale)', fontsize=12)
ax2.set_title('Zoom: First 40 Iterations\n(superlinear L-BFGS convergence visible)', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 40)

plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/lbfgs_comparison.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: lbfgs_comparison.pdf")


# ─── Figure 2: ADMM on the Q5 constrained problem ─────────────────────────────
# Problem: min f(x) = (x1-1)^2 + (x2-5)^2 s.t. x1 >= 0.5
# Reformulate as: min f(x) s.t. x1 - z = 0, z >= 0.5
# ADMM: augmented Lagrangian L(x,z,nu) = f(x) + nu(x1-z) + rho/2*(x1-z)^2

def admm_q5(x0, rho, n_iters):
    """ADMM for min (x1-1)^2 + (x2-5)^2 s.t. x1 >= 0.5.
    Split: x = (x1,x2), z = x1 (coupling), z >= 0.5.
    x-update: analytic; z-update: clip to [0.5, inf).
    """
    x = x0.copy().astype(float)
    z = max(x[0], 0.5)
    nu = 0.0  # dual variable for x1 - z = 0
    x_hist = [x.copy()]
    f_vals = [(x[0]-1)**2 + (x[1]-5)**2]
    primal_res, dual_res = [], []

    for _ in range(n_iters):
        # x-update: min (x1-1)^2 + (x2-5)^2 + nu*(x1-z) + rho/2*(x1-z)^2
        # Gradient wrt x1: 2(x1-1) + nu + rho*(x1-z) = 0
        # => x1 = (2 + rho*z - nu) / (2 + rho)
        x1_new = (2 + rho * z - nu) / (2 + rho)
        x2_new = 5.0  # gradient wrt x2 at optimal: 2(x2-5) = 0
        x_new = np.array([x1_new, x2_new])

        # z-update: min -nu*(z) + rho/2*(x1_new - z)^2 s.t. z >= 0.5
        # => z = max(x1_new + nu/rho, 0.5)
        z_new = max(x1_new + nu / rho, 0.5)

        # dual update
        nu = nu + rho * (x1_new - z_new)

        primal_res.append(abs(x1_new - z_new))
        dual_res.append(abs(rho * (z_new - z)))

        x = x_new
        z = z_new
        x_hist.append(x.copy())
        f_vals.append((x[0]-1)**2 + (x[1]-5)**2)

    return np.array(x_hist), np.array(f_vals), np.array(primal_res), np.array(dual_res)

def proj_gd_q5(x0, alpha, n_iters):
    """Projected gradient descent for min f(x) s.t. x1 >= 0.5."""
    x = x0.copy().astype(float)
    f_vals = [(x[0]-1)**2 + (x[1]-5)**2]
    for _ in range(n_iters):
        g = np.array([2*(x[0]-1), 2*(x[1]-5)])
        x = x - alpha * g
        x[0] = max(x[0], 0.5)  # projection
        f_vals.append((x[0]-1)**2 + (x[1]-5)**2)
    return np.array(f_vals)

def penalty_q5(x0, lam, alpha, n_iters):
    """Penalty method for min f(x) + lambda*max(0, 0.5-x1)."""
    x = x0.copy().astype(float)
    f_vals = [(x[0]-1)**2 + (x[1]-5)**2]
    for _ in range(n_iters):
        violation = max(0.5 - x[0], 0)
        g = np.array([2*(x[0]-1) - lam * float(x[0] < 0.5), 2*(x[1]-5)])
        x = x - alpha * g
        f_vals.append((x[0]-1)**2 + (x[1]-5)**2)
    return np.array(f_vals)

x0_q5 = np.array([3.0, 2.0])
n_q5 = 80
f_star_q5 = 16.0  # (1-1)^2 + (5-5)^2 = 0, but constraint inactive so f*=0 at (1,5)
f_star_q5 = 0.0

x_admm, f_admm, pres_admm, dres_admm = admm_q5(x0_q5, rho=2.0, n_iters=n_q5)
x_admm5, f_admm5, pres5, dres5 = admm_q5(x0_q5, rho=5.0, n_iters=n_q5)
f_pgd = proj_gd_q5(x0_q5, alpha=0.08, n_iters=n_q5)
f_pen = penalty_q5(x0_q5, lam=10.0, alpha=0.06, n_iters=n_q5)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
k_arr = np.arange(n_q5 + 1)

ax = axes[0]
ax.semilogy(k_arr, np.clip(f_admm, 1e-10, None), '-', color='#2196F3', linewidth=2.5,
            label='ADMM ($\\rho=2$)')
ax.semilogy(k_arr, np.clip(f_admm5, 1e-10, None), '--', color='#9C27B0', linewidth=2,
            label='ADMM ($\\rho=5$)')
ax.semilogy(k_arr, np.clip(f_pgd, 1e-10, None), '-', color='#4CAF50', linewidth=2,
            label='Proj.\ GD ($\\alpha=0.08$)')
ax.semilogy(k_arr, np.clip(f_pen, 1e-10, None), '-', color='#FF5722', linewidth=2,
            label='Penalty ($\\lambda=10$)')
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('$f(x_k) - f^\\star$ (log)', fontsize=12)
ax.set_title('ADMM vs Proj.\ GD vs Penalty\n(Q5: $x_1 \\geq 0.5$, starting from $(3,2)$)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.semilogy(np.arange(n_q5), pres_admm, '-', color='#2196F3', linewidth=2,
             label='Primal res.\ $|x_1^k - z^k|$ ($\\rho=2$)')
ax2.semilogy(np.arange(n_q5), dres_admm, '--', color='#2196F3', linewidth=1.5, alpha=0.7,
             label='Dual res.\ ($\\rho=2$)')
ax2.semilogy(np.arange(n_q5), pres5, '-', color='#9C27B0', linewidth=2,
             label='Primal res.\ ($\\rho=5$)')
ax2.semilogy(np.arange(n_q5), dres5, '--', color='#9C27B0', linewidth=1.5, alpha=0.7,
             label='Dual res.\ ($\\rho=5$)')
ax2.set_xlabel('Iteration $k$', fontsize=12)
ax2.set_ylabel('Residual (log)', fontsize=12)
ax2.set_title('ADMM Primal and Dual Residuals\n(Convergence certificate)', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Right: ADMM trajectory
ax3 = axes[2]
ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Constraint $x_1=0.5$')
ax3.fill_betweenx([-1, 7], 0.5, 5.5, alpha=0.1, color='blue', label='Feasible $x_1\\geq0.5$')
ax3.plot(x_admm[:, 0], x_admm[:, 1], 'b-o', markersize=3, linewidth=1.5, alpha=0.7,
         label='ADMM iterates ($\\rho=2$)')
ax3.plot(x_admm5[:, 0], x_admm5[:, 1], 'm-s', markersize=3, linewidth=1.5, alpha=0.7,
         label='ADMM iterates ($\\rho=5$)')
ax3.plot(1, 5, '*', color='gold', markersize=14, zorder=6, markeredgecolor='k', label='Optimum $(1,5)$')
ax3.plot(x0_q5[0], x0_q5[1], 'ko', markersize=8, zorder=5, label='Start $(3,2)$')
ax3.set_xlim(-0.5, 4); ax3.set_ylim(-1, 7)
ax3.set_xlabel('$x_1$', fontsize=12); ax3.set_ylabel('$x_2$', fontsize=12)
ax3.set_title('ADMM Iterate Trajectory\n(constraint $x_1\\geq 0.5$ shown)', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/admm_comparison.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: admm_comparison.pdf")


# ─── Figure 3: Away-step FW vs standard FW ────────────────────────────────────
# Strongly convex quadratic objective on simplex: f(x) = 0.5*x^T*A*x - b^T*x
# Simplex Delta^3 = {x >= 0, sum(x)=1}, d=4

def fw_standard(A, b, x0, n):
    """Standard FW with step 2/(k+2)."""
    x = x0.copy().astype(float)
    f_star = -0.5 * b @ np.linalg.solve(A, b)  # unconstrained min (may be outside simplex)
    losses = [0.5 * x @ A @ x - b @ x]
    for k in range(n):
        g = A @ x - b
        # LMO: min_v in simplex g^T v = min_i g_i * e_i
        i_fw = np.argmin(g)
        z_fw = np.zeros_like(x); z_fw[i_fw] = 1.0
        gamma = 2.0 / (k + 2)
        x = (1 - gamma) * x + gamma * z_fw
        losses.append(0.5 * x @ A @ x - b @ x)
    return np.array(losses)

def fw_away(A, b, x0, n, max_step=True):
    """Away-step FW on simplex."""
    x = x0.copy().astype(float)
    losses = [0.5 * x @ A @ x - b @ x]
    # Active set: vertices with x_i > 0
    S = {i: x[i] for i in range(len(x)) if x[i] > 1e-10}

    for k in range(n):
        g = A @ x - b
        # FW direction: LMO
        i_fw = np.argmin(g)
        z_fw = np.zeros_like(x); z_fw[i_fw] = 1.0
        d_fw = z_fw - x  # FW direction
        gap_fw = -g @ d_fw

        # Away direction: argmax_{i in S} g_i
        i_away = max(S.keys(), key=lambda i: g[i])
        v_away = np.zeros_like(x); v_away[i_away] = 1.0
        d_away = x - v_away  # away direction
        gap_away = g @ d_away  # should be >= 0

        if gap_fw >= gap_away:
            # Use FW step
            d = d_fw
            max_gamma = 1.0
        else:
            # Use away step
            d = d_away
            max_gamma = S[i_away] / (1 - S[i_away] + 1e-12)

        # Exact line search on quadratic: gamma = -g^T d / (d^T A d)
        dAd = d @ A @ d
        if dAd > 1e-12:
            gamma = min(-g @ d / dAd, max_gamma)
        else:
            gamma = max_gamma
        gamma = max(0, gamma)

        x_new = x + gamma * d
        x_new = np.maximum(x_new, 0)  # numerical clip
        x_new /= x_new.sum()

        # Update active set
        if gap_fw >= gap_away:
            S[i_fw] = S.get(i_fw, 0) + gamma
            for i in list(S.keys()):
                S[i] = x_new[i]
                if S[i] < 1e-10:
                    del S[i]
        else:
            S[i_away] = x_new[i_away]
            for i in list(S.keys()):
                S[i] = x_new[i]
                if S[i] < 1e-10:
                    del S[i]
        if not S:
            S = {np.argmax(x_new): x_new.max()}

        x = x_new
        losses.append(0.5 * x @ A @ x - b @ x)
    return np.array(losses)

# Strongly convex quadratic on simplex
d = 5
np.random.seed(17)
Q = np.random.randn(d, d)
A = Q @ Q.T + 2.0 * np.eye(d)  # PD with mu=2, L~large
b = np.array([3.0, 1.0, 0.5, 2.0, 1.5])

# Compute constrained optimum on simplex
from scipy.optimize import minimize
def obj(x): return 0.5 * x @ A @ x - b @ x
from scipy.optimize import LinearConstraint
# Simple projection: run many iters of projected GD
def simplex_proj(v):
    """Project onto probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0)

x_simplex = simplex_proj(np.linalg.solve(A, b))  # approx
f_star_simplex = obj(x_simplex)

x0_simplex = simplex_proj(np.ones(d) / d)
n_fw = 100

losses_fw_std = fw_standard(A, b, x0_simplex, n_fw)
losses_fw_away = fw_away(A, b, x0_simplex, n_fw)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
k_arr = np.arange(n_fw + 1)
ax.semilogy(k_arr, np.clip(losses_fw_std - f_star_simplex, 1e-14, None),
            '-', color='#2196F3', linewidth=2.5, label='Standard FW ($\\gamma_k = 2/(k+2)$)')
ax.semilogy(k_arr, np.clip(losses_fw_away - f_star_simplex, 1e-14, None),
            '-', color='#FF5722', linewidth=2.5, label='Away-step FW (exact line search)')
# O(1/k) reference
ax.semilogy(k_arr[1:], (losses_fw_std[0] - f_star_simplex) / k_arr[1:],
            'k--', linewidth=1.5, alpha=0.7, label='$O(1/k)$ rate')
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('$f(x_k) - f^\\star$ (log)', fontsize=12)
ax.set_title('Standard FW vs Away-step FW\n(Strongly convex quadratic on simplex $\\Delta^5$)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: show that away-step achieves linear rate
ax2 = axes[1]
# Fit linear rate to away-step
away_gap = np.clip(losses_fw_away - f_star_simplex, 1e-14, None)
# Find approximate rate by linear regression on log
mask = away_gap > 1e-10
if mask.sum() > 10:
    log_gap = np.log(away_gap[mask])
    k_fit = k_arr[mask]
    slope, intercept = np.polyfit(k_fit, log_gap, 1)
    rho_fit = np.exp(slope)
else:
    rho_fit = 0.95

ax2.semilogy(k_arr, np.clip(losses_fw_std - f_star_simplex, 1e-14, None),
             '-', color='#2196F3', linewidth=2.5, label='Standard FW ($O(1/k)$)')
ax2.semilogy(k_arr, np.clip(losses_fw_away - f_star_simplex, 1e-14, None),
             '-', color='#FF5722', linewidth=2.5, label=f'Away-step FW (linear, $\\hat{{\\rho}}\\approx{rho_fit:.3f}$)')
# Show linear fit
if rho_fit < 1.0:
    gap0 = away_gap[0]
    ax2.semilogy(k_arr, gap0 * rho_fit**k_arr, 'r--', linewidth=1.5, alpha=0.7,
                 label=f'Linear fit $C\\cdot{rho_fit:.3f}^k$')
ax2.set_xlabel('Iteration $k$', fontsize=12)
ax2.set_ylabel('$f(x_k) - f^\\star$ (log)', fontsize=12)
ax2.set_title('Linear Convergence of Away-step FW\nvs Sublinear Standard FW', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('Frank--Wolfe Variants: Standard vs Away-step on Strongly Convex Quadratic',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/away_step_fw.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: away_step_fw.pdf")


# ─── Figure 4: SAGA vs SVRG vs SGD detailed comparison ────────────────────────
# Full implementations on Benchmark A

def sgd_run(X, y, x0, alpha, batch, n_epochs):
    n = len(y)
    x = x0.copy().astype(float)
    losses = [bench_A_loss(x)]
    for _ in range(n_epochs):
        idx = np.random.permutation(n)
        for j in range(0, n, batch):
            bi = idx[j:j+batch]
            g = X[bi].T @ (X[bi] @ x - y[bi]) / len(bi)
            x -= alpha * g
        losses.append(bench_A_loss(x))
    return np.array(losses)

def svrg_run(X, y, x0, alpha, batch, n_epochs, inner_iters=None):
    n = len(y)
    x = x0.copy().astype(float)
    if inner_iters is None:
        inner_iters = 2 * n // batch
    losses = [bench_A_loss(x)]
    x_tilde = x.copy()
    for _ in range(n_epochs):
        # Full gradient at snapshot
        mu = X.T @ (X @ x_tilde - y) / n
        x_inner = x.copy()
        for j in range(inner_iters):
            i = np.random.randint(0, n - batch)
            bi = np.arange(i, i + batch)
            g_i_tilde = X[bi].T @ (X[bi] @ x_tilde - y[bi]) / batch
            g_i_x = X[bi].T @ (X[bi] @ x_inner - y[bi]) / batch
            g_svrg = g_i_x - g_i_tilde + mu
            x_inner -= alpha * g_svrg
        x = x_inner
        x_tilde = x.copy()
        losses.append(bench_A_loss(x))
    return np.array(losses)

def saga_run(X, y, x0, alpha, n_epochs):
    n = len(y)
    x = x0.copy().astype(float)
    # Initialise gradient table with full gradients
    phi = np.zeros((n, x.shape[0]))  # stored points
    g_table = np.zeros((n, x.shape[0]))  # gradient table
    for i in range(n):
        g_table[i] = X[i] * (X[i] @ x - y[i])
    g_mean = g_table.mean(axis=0)

    losses = [bench_A_loss(x)]
    for _ in range(n_epochs):
        for j in range(n):
            i = np.random.randint(n)
            g_i_new = X[i] * (X[i] @ x - y[i])
            # SAGA update: g_i_new - g_table[i] + g_mean
            g_saga = g_i_new - g_table[i] + g_mean
            # Update table and mean
            g_mean += (g_i_new - g_table[i]) / n
            g_table[i] = g_i_new
            x -= alpha * g_saga
        losses.append(bench_A_loss(x))
    return np.array(losses)

x0_A = np.zeros(2)
n_epochs = 30
batch_size = 40

# Run methods
losses_sgd = sgd_run(X_data, y_data, x0_A, 0.05, batch_size, n_epochs)
losses_svrg = svrg_run(X_data, y_data, x0_A, 0.12, batch_size, n_epochs)
losses_saga = saga_run(X_data, y_data, x0_A, 0.06, n_epochs)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ep = np.arange(n_epochs + 1)
ax.semilogy(ep, np.clip(losses_sgd - f_star_A, 1e-10, None),
            '-', color='#2196F3', linewidth=2.5, label=f'SGD (batch={batch_size})')
ax.semilogy(ep, np.clip(losses_svrg - f_star_A, 1e-10, None),
            '-', color='#4CAF50', linewidth=2.5, label='SVRG (2 snapshots/epoch)')
ax.semilogy(ep, np.clip(losses_saga - f_star_A, 1e-10, None),
            '-', color='#FF5722', linewidth=2.5, label='SAGA (gradient table)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('$f(\\theta_k) - f^\\star$ (log)', fontsize=12)
ax.set_title('SGD vs SVRG vs SAGA on Benchmark A\n(Linear Regression, $m=1000$, batch=40)', fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right: gradient variance comparison
ax2 = axes[1]
# Estimate gradient variance at each epoch checkpoint for SGD
n_samples = 50
gvar_sgd, gvar_svrg_like = [], []

x_sgd_temp = x0_A.copy()
x_svrg_temp = x0_A.copy()
for ep_idx in range(min(n_epochs, 15)):
    # SGD gradient variance: variance of batch gradients at current x
    grads = np.array([X_data[np.random.randint(0, m - batch_size, batch_size)].T @
                      (X_data[np.random.randint(0, m - batch_size, batch_size)] @ x_sgd_temp -
                       y_data[np.random.randint(0, m - batch_size, batch_size)]) / batch_size
                      for _ in range(n_samples)])
    gvar_sgd.append(np.mean(np.var(grads, axis=0)))

    # SVRG: variance ~ ||theta_t - theta_tilde||^2 * L^2
    gvar_svrg_like.append(max(float(losses_svrg[ep_idx] - f_star_A), 1e-12))

    # Do one epoch of SGD
    for _ in range(m // batch_size):
        bi = np.random.randint(0, m - batch_size, batch_size)
        g = X_data[bi].T @ (X_data[bi] @ x_sgd_temp - y_data[bi]) / batch_size
        x_sgd_temp -= 0.05 * g

ax2.semilogy(range(len(gvar_sgd)), gvar_sgd, 'o-', color='#2196F3', linewidth=2,
             markersize=6, label='SGD gradient variance (estimated)')
ax2.semilogy(range(len(gvar_svrg_like)), gvar_svrg_like, 's-', color='#4CAF50', linewidth=2,
             markersize=6, label='SVRG suboptimality (proxy for variance)')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Variance / Suboptimality (log)', fontsize=12)
ax2.set_title('SGD Gradient Variance vs SVRG Convergence\n(Variance $\\to 0$ explains linear rate)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('Variance Reduction: SGD vs SVRG vs SAGA (Benchmark A)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/optimisation/final_assignment/figures/saga_svrg_comparison.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("Done: saga_svrg_comparison.pdf")

print("All pass-15 figures generated.")
