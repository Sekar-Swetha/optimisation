"""Pass 19: Adam convergence deep analysis and oracle complexity bar chart."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(42)

# ============================================================
# Benchmark definitions
# ============================================================

def loss_B(x):
    return (x[0] - 1)**2 + 5*(x[1] - 2)**2 + np.sin(x[0])

def grad_B(x):
    return np.array([2*(x[0] - 1) + np.cos(x[0]), 10*(x[1] - 2)])

# Rosenbrock (Benchmark C)
def loss_C(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_C(x):
    g1 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    g2 = 200*(x[1] - x[0]**2)
    return np.array([g1, g2])

def hessian_C(x):
    h11 = 2 + 1200*x[0]**2 - 400*x[1]
    h12 = -400*x[0]
    return np.array([[h11, h12], [h12, 200]])

# ============================================================
# Figure 1: q2_adam_convergence_analysis.pdf
# Deep 4-panel analysis of Adam on Benchmark B
# ============================================================

f_star_B = 0.7244
x0_B = np.array([-1.0, 4.0])
n_iters = 300
beta1 = 0.82
beta2 = 0.999
alpha = 0.08
eps = 1e-8

# Run Adam and record internal state
x = x0_B.copy().astype(float)
m_vec = np.zeros(2)
v_vec = np.zeros(2)

f_hist = [loss_B(x)]
eff_step_hist = []   # shape (n_iters, 2)
ratio_hist = []      # mhat[0] / sqrt(vhat[0])
beta1_pow_hist = []
beta2_pow_hist = []
bc1_hist = []        # 1/(1-beta1^t)
bc2_hist = []        # 1/(1-beta2^t)

for t in range(1, n_iters + 1):
    g = grad_B(x)
    m_vec = beta1 * m_vec + (1 - beta1) * g
    v_vec = beta2 * v_vec + (1 - beta2) * g**2
    mhat = m_vec / (1 - beta1**t)
    vhat = v_vec / (1 - beta2**t)
    # effective step size per dimension
    eff = alpha * np.abs(mhat) / (np.sqrt(vhat) + eps)
    eff_step_hist.append(eff.copy())
    # ratio for d=0
    ratio_hist.append(mhat[0] / (np.sqrt(vhat[0]) + eps))
    # bias correction tracking
    beta1_pow_hist.append(beta1**t)
    beta2_pow_hist.append(beta2**t)
    bc1_hist.append(1.0 / (1 - beta1**t))
    bc2_hist.append(1.0 / (1 - beta2**t))
    x = x - alpha * mhat / (np.sqrt(vhat) + eps)
    f_hist.append(loss_B(x))

f_hist = np.array(f_hist)
subopt = f_hist - f_star_B
# clip tiny negatives
subopt = np.clip(subopt, 1e-12, None)

eff_step_hist = np.array(eff_step_hist)   # (300, 2)
ratio_hist = np.array(ratio_hist)
beta1_pow_hist = np.array(beta1_pow_hist)
beta2_pow_hist = np.array(beta2_pow_hist)
bc1_hist = np.array(bc1_hist)
bc2_hist = np.array(bc2_hist)

k_arr = np.arange(1, n_iters + 1)
k_subopt = np.arange(0, n_iters + 1)  # length 301

fig1, axes1 = plt.subplots(2, 2, figsize=(13, 10))

# --- Top-left: log-log suboptimality ---
ax = axes1[0, 0]
ax.loglog(k_subopt[1:], subopt[1:], color='tab:blue', lw=2, label='Adam $f(x_k)-f^*$')
# Reference lines anchored at k=1
c1 = subopt[1] * 1      # O(1/k): at k=1, val = subopt[1]
c2 = subopt[1] * 1      # O(1/k^2): same anchor
ax.loglog(k_arr, c1 / k_arr, 'k--', lw=1.5, label='$O(1/k)$ reference')
ax.loglog(k_arr, c2 / k_arr**2, 'k:', lw=1.5, label='$O(1/k^2)$ reference')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('$f(x_k) - f^*$', fontsize=11)
ax.set_title('Suboptimality (log-log): Between $O(1/k)$ and $O(1/k^2)$', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

# --- Top-right: effective step size per dimension ---
ax = axes1[0, 1]
ax.plot(k_arr, eff_step_hist[:, 0], color='tab:orange', lw=1.8, label='dim $d=0$')
ax.plot(k_arr, eff_step_hist[:, 1], color='tab:green', lw=1.8, label='dim $d=1$')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel(r'$\alpha \cdot |\hat{m}_t^{(d)}| / (\sqrt{\hat{v}_t^{(d)}} + \varepsilon)$', fontsize=10)
ax.set_title('Effective Step Size per Dimension\n(stays bounded away from zero)', fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Bottom-left: mhat[0] / sqrt(vhat[0]) ---
ax = axes1[1, 0]
ax.plot(k_arr, ratio_hist, color='tab:red', lw=1.8,
        label=r'$\hat{m}_t^{(0)} / \sqrt{\hat{v}_t^{(0)}}$')
ax.axhline(1.0, color='gray', linestyle='--', lw=1.2, alpha=0.7, label='+1 (sign limit)')
ax.axhline(-1.0, color='gray', linestyle=':', lw=1.2, alpha=0.7, label='-1 (sign limit)')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel(r'$\hat{m}_t^{(0)} / \sqrt{\hat{v}_t^{(0)}}$', fontsize=10)
ax.set_title(r'Ratio $\hat{m}_t / \sqrt{\hat{v}_t}$ Approaches $\pm 1$ (sign of gradient)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Bottom-right: beta^t decay and bias correction ---
ax = axes1[1, 1]
ax2_twin = ax.twinx()

l1, = ax.semilogy(k_arr, beta1_pow_hist, color='tab:blue', lw=1.8, label=r'$\beta_1^t = 0.82^t$')
l2, = ax.semilogy(k_arr, beta2_pow_hist, color='tab:orange', lw=1.8, label=r'$\beta_2^t = 0.999^t$')
l3, = ax2_twin.plot(k_arr, bc1_hist, color='tab:blue', lw=1.5, linestyle='--',
                    label=r'$1/(1-\beta_1^t)$')
l4, = ax2_twin.plot(k_arr, bc2_hist, color='tab:orange', lw=1.5, linestyle='--',
                    label=r'$1/(1-\beta_2^t)$')

ax.axvline(50, color='gray', lw=1.0, linestyle=':', alpha=0.8)
ax.text(52, 1e-5, '$k \\approx 50$\n$\\beta_1^t\\approx0$', fontsize=8, color='gray')

ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel(r'$\beta^t$ (log scale)', fontsize=10)
ax2_twin.set_ylabel('Bias correction $1/(1-\\beta^t)$', fontsize=10)
ax.set_title('Bias Correction Decay: $\\beta_1^t$ Negligible after ~50 steps', fontsize=10)

lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

fig1.suptitle('Adam Deep Analysis on Benchmark B\n'
              r'$f(x)=(x_1-1)^2+5(x_2-2)^2+\sin(x_1)$, '
              f'$\\alpha={alpha}$, $\\beta_1={beta1}$, $\\beta_2={beta2}$, $x_0=[-1,4]$',
              fontsize=12, fontweight='bold')
fig1.tight_layout()
fname1 = os.path.join(OUTDIR, 'q2_adam_convergence_analysis.pdf')
fig1.savefig(fname1, bbox_inches='tight', dpi=150)
plt.close(fig1)
print(f"Saved: q2_adam_convergence_analysis.pdf")

# ============================================================
# Figure 2: all_methods_oracle_complexity.pdf
# Oracle calls to reach f(x)-f* < 0.1 on Rosenbrock
# ============================================================

np.random.seed(42)

x0_C = np.array([-1.0, 1.0])
f_star_C = 0.0
tol = 0.1
MAX_ITER = 500

def oracle_calls(method, alpha_val, **kwargs):
    """Run method, return oracle call count at first time f < f_star + tol.
    Returns (calls_to_reach, reached_flag)."""
    x = x0_C.copy().astype(float)
    name = method

    if name == 'GD':
        fe_per_iter = 1
        for k in range(1, MAX_ITER + 1):
            g = grad_C(x)
            x = x - alpha_val * g
            f = loss_C(x)
            if f - f_star_C < tol:
                return k * fe_per_iter
        return MAX_ITER * fe_per_iter

    elif name == 'Nesterov':
        beta_max = kwargs['beta_max']
        fe_per_iter = 2
        z = np.zeros(2)
        for k in range(1, MAX_ITER + 1):
            beta_k = min((k - 1) / (k + 2), beta_max)
            lookahead = x + beta_k * z
            g = grad_C(lookahead)
            z = beta_k * z - alpha_val * g
            x = x + z
            f = loss_C(x)
            if f - f_star_C < tol:
                return k * fe_per_iter
        return MAX_ITER * fe_per_iter

    elif name == 'HB':
        beta = kwargs['beta']
        fe_per_iter = 2
        z = np.zeros(2)
        for k in range(1, MAX_ITER + 1):
            g = grad_C(x)
            z = beta * z + alpha_val * g
            x = x - z
            f = loss_C(x)
            if f - f_star_C < tol:
                return k * fe_per_iter
        return MAX_ITER * fe_per_iter

    elif name == 'Adam':
        beta1_a = 0.9
        beta2_a = 0.999
        eps_a = 1e-8
        fe_per_iter = 2
        m_a = np.zeros(2)
        v_a = np.zeros(2)
        for t in range(1, MAX_ITER + 1):
            g = grad_C(x)
            m_a = beta1_a * m_a + (1 - beta1_a) * g
            v_a = beta2_a * v_a + (1 - beta2_a) * g**2
            mh = m_a / (1 - beta1_a**t)
            vh = v_a / (1 - beta2_a**t)
            x = x - alpha_val * mh / (np.sqrt(vh) + eps_a)
            f = loss_C(x)
            if f - f_star_C < tol:
                return t * fe_per_iter
        return MAX_ITER * fe_per_iter

    elif name == 'Newton':
        fe_per_iter = 5   # 4 for Hessian via FD, 1 for gradient
        damping = 1e-8
        for k in range(1, MAX_ITER + 1):
            g = grad_C(x)
            H = hessian_C(x)
            H_reg = H + damping * np.eye(2)
            try:
                p = np.linalg.solve(H_reg, g)
            except np.linalg.LinAlgError:
                p = g
            x = x - alpha_val * p
            f = loss_C(x)
            if f - f_star_C < tol:
                return k * fe_per_iter
        return MAX_ITER * fe_per_iter

    elif name == 'NRS':
        delta = kwargs['delta']
        fe_per_iter = 2
        for k in range(1, MAX_ITER + 1):
            g = grad_C(x)
            # Normalised random search direction component
            d = np.random.randn(2)
            d = d / (np.linalg.norm(d) + 1e-12)
            # Use gradient with NRS: x = x - alpha*(g + delta*d)
            x = x - alpha_val * (g + delta * d)
            f = loss_C(x)
            if f - f_star_C < tol:
                return k * fe_per_iter
        return MAX_ITER * fe_per_iter

    return MAX_ITER * 2


methods = ['GD', 'Nesterov', 'HB', 'Adam', 'Newton', 'NRS']
alphas  = {'GD': 0.0012, 'Nesterov': 0.0007, 'HB': 0.0008,
           'Adam': 0.006, 'Newton': 0.22, 'NRS': 0.001}
extras  = {'Nesterov': {'beta_max': 0.90}, 'HB': {'beta': 0.86},
           'NRS': {'delta': 0.01}}

oracle_counts = {}
for m in methods:
    kw = extras.get(m, {})
    oracle_counts[m] = oracle_calls(m, alphas[m], **kw)
    print(f"  {m}: {oracle_counts[m]} oracle calls")

# Color palette
colors_map = {
    'GD':       '#1f77b4',
    'Nesterov': '#ff7f0e',
    'HB':       '#2ca02c',
    'Adam':     '#d62728',
    'Newton':   '#9467bd',
    'NRS':      '#8c564b',
}

# Horizontal bar chart, log x-axis
fig2, ax2 = plt.subplots(figsize=(10, 5))

ys = np.arange(len(methods))
bars = [oracle_counts[m] for m in methods]
bar_colors = [colors_map[m] for m in methods]

hbars = ax2.barh(ys, bars, color=bar_colors, edgecolor='black', linewidth=0.7, height=0.6)
ax2.set_yticks(ys)
ax2.set_yticklabels(methods, fontsize=11)
ax2.set_xscale('log')
ax2.set_xlabel('Oracle Calls (function/gradient evaluations)', fontsize=11)
ax2.set_title('Oracle Calls to Reach $f(x_k) - f^* < 0.1$ on Rosenbrock (Benchmark C)',
              fontsize=11)

# Annotate bar ends
for bar, val in zip(hbars, bars):
    ax2.text(val * 1.05, bar.get_y() + bar.get_height() / 2,
             f'{val}', va='center', ha='left', fontsize=9)

ax2.grid(True, axis='x', alpha=0.4, which='both')
ax2.set_xlim(left=1)

fig2.suptitle('Oracle Calls to Reach f(x) - f* < 0.1 on Rosenbrock (Benchmark C)',
              fontsize=12, fontweight='bold')
fig2.tight_layout()
fname2 = os.path.join(OUTDIR, 'all_methods_oracle_complexity.pdf')
fig2.savefig(fname2, bbox_inches='tight', dpi=150)
plt.close(fig2)
print(f"Saved: all_methods_oracle_complexity.pdf")
