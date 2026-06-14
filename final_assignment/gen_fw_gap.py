"""Generate Frank-Wolfe gap evolution figure."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

bounds = [(0.5, 5.0), (-5.0, 10.0)]

def lmo(g):
    """Box LMO: coordinate-wise argmin of g^T z over box."""
    return np.array([bounds[i][0] if g[i] > 0 else bounds[i][1] for i in range(2)])

# Interior optimum: f(x) = (x1-1)^2 + (x2-5)^2, minimum at (1,5) in X
def f_int(x): return (x[0]-1)**2 + (x[1]-5)**2
def g_int(x): return np.array([2*(x[0]-1), 2*(x[1]-5)])

# Boundary optimum: f(x) = x1^2 + x2^2, constrained min at (0.5, 0)
def f_bnd(x): return x[0]**2 + x[1]**2
def g_bnd(x): return np.array([2*x[0], 2*x[1]])

def run_fw(gf, ff, x0, beta, n):
    x = x0.copy().astype(float)
    gaps, fvals = [], [ff(x)]
    for _ in range(n):
        g = gf(x)
        z = lmo(g)
        gap = float(np.dot(g, x - z))
        gaps.append(gap)
        x = beta * x + (1 - beta) * z
        fvals.append(ff(x))
    return np.array(fvals), np.array(gaps)

x0_int = np.array([1.0, 1.0])
x0_bnd = np.array([3.0, 3.0])

f90, gap90 = run_fw(g_int, f_int, x0_int, 0.90, 180)
f985, gap985 = run_fw(g_int, f_int, x0_int, 0.985, 180)
fbnd, gapbnd = run_fw(g_bnd, f_bnd, x0_bnd, 0.93, 140)

# Theoretical O(1/k) bound: 2*L*D^2/(k+2)
D = np.sqrt(4.5**2 + 15**2)  # diameter
L_int = 2.0  # Hessian of f_int = 2I
L_bnd = 2.0
k_int = np.arange(1, 181)
k_bnd = np.arange(1, 141)
theory_int = 2 * L_int * D**2 / (k_int + 2)
theory_bnd = 2 * L_bnd * D**2 / (k_bnd + 2)

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: Interior - FW gap vs iteration
ax = axes[0]
ax.semilogy(k_int, gap90, 'b-', lw=2, label='Gap $\\mathcal{G}(x_k)$, $\\beta=0.90$')
ax.semilogy(k_int, gap985, 'r-', lw=2, label='Gap $\\mathcal{G}(x_k)$, $\\beta=0.985$')
ax.semilogy(k_int, f90[1:], 'b--', lw=1.5, alpha=0.7, label='$f(x_k)-f^*$, $\\beta=0.90$')
ax.semilogy(k_int, f985[1:], 'r--', lw=1.5, alpha=0.7, label='$f(x_k)-f^*$, $\\beta=0.985$')
ax.semilogy(k_int, theory_int, 'k:', lw=2, label='Theory $O(1/k)$: $2LD^2/(k+2)$')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Value (log scale)')
ax.set_title('Interior opt.: FW gap vs suboptimality')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Panel 2: Interior - Gap tightness ratio
ax = axes[1]
ratio90 = gap90 / np.maximum(f90[1:], 1e-10)
ratio985 = gap985 / np.maximum(f985[1:], 1e-10)
ax.semilogy(k_int, ratio90, 'b-', lw=2, label='$\\mathcal{G}/f$ ratio, $\\beta=0.90$')
ax.semilogy(k_int, ratio985, 'r-', lw=2, label='$\\mathcal{G}/f$ ratio, $\\beta=0.985$')
ax.axhline(1.0, color='k', ls='--', lw=1.5, label='Ratio = 1 (tight bound)')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('$\\mathcal{G}(x_k) / (f(x_k)-f^*)$')
ax.set_title('FW gap tightness: $\\mathcal{G}(x_k) \\geq f(x_k)-f^*$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Boundary case
ax = axes[2]
ax.semilogy(k_bnd, gapbnd, 'g-', lw=2, label='Gap $\\mathcal{G}(x_k)$, $\\beta=0.93$')
fstar_bnd = 0.25
ax.semilogy(k_bnd, np.maximum(fbnd[1:] - fstar_bnd, 1e-10), 'g--', lw=1.5, alpha=0.7,
            label='$f(x_k)-f^*$ ($f^*=0.25$)')
ax.semilogy(k_bnd, theory_bnd, 'k:', lw=2, label='Theory $2LD^2/(k+2)$')
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Value (log scale)')
ax.set_title('Boundary opt.: FW gap (loose at boundary)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('figures/q6_fw_gap.pdf', bbox_inches='tight')
plt.close()
print(f"Interior β=0.90:  final gap={gap90[-1]:.4f}, final f={f90[-1]:.4f}")
print(f"Interior β=0.985: final gap={gap985[-1]:.4f}, final f={f985[-1]:.4f}")
print(f"Boundary β=0.93:  final gap={gapbnd[-1]:.4f}, final f-f*={fbnd[-1]-0.25:.4f}")
print(f"Theory bound at k=180: {theory_int[-1]:.4f}")
print(f"Theory bound at k=140: {theory_bnd[-1]:.4f}")
print("Saved q6_fw_gap.pdf")
