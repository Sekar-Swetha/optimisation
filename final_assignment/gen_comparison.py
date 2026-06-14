import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
eps_noise = np.random.randn(m)
y_data = X_data @ theta_star + eps_noise

def loss_C(x): return (1-x[0])**2+100*(x[1]-x[0]**2)**2
def grad_C(x): return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
def hessian_C(x): return np.array([[2+1200*x[0]**2-400*x[1],-400*x[0]],[-400*x[0],200]])

x0_C = np.array([-1.0,1.0])

def gd(gf,lf,x0,a,n):
    x=x0.copy().astype(float); fh=[lf(x)]
    for _ in range(n): x=x-a*gf(x); fh.append(lf(x))
    return np.array(fh)

def polyak_step(gf,lf,x0,fs,eps,n):
    x=x0.copy().astype(float); fh=[lf(x)]
    for _ in range(n):
        g=gf(x); ak=(lf(x)-fs)/(np.dot(g,g)+eps); x=x-ak*g; fh.append(lf(x))
    return np.array(fh)

def hb(gf,lf,x0,a,b,n):
    x=x0.copy().astype(float); z=np.zeros_like(x); fh=[lf(x)]
    for _ in range(n): g=gf(x); z=b*z+a*g; x=x-z; fh.append(lf(x))
    return np.array(fh)

def nesterov(gf,lf,x0,a,bm,n):
    x=x0.copy().astype(float); z=np.zeros_like(x); fh=[lf(x)]
    for k in range(1,n+1):
        bk=min((k-1)/(k+2),bm); la=x+bk*z; g=gf(la); z=bk*z-a*g; x=x+z; fh.append(lf(x))
    return np.array(fh)

def adam(gf,lf,x0,a,b1,b2,eps,n):
    x=x0.copy().astype(float); mv=np.zeros_like(x); vv=np.zeros_like(x); fh=[lf(x)]
    for t in range(1,n+1):
        g=gf(x); mv=b1*mv+(1-b1)*g; vv=b2*vv+(1-b2)*g**2
        mh=mv/(1-b1**t); vh=vv/(1-b2**t); x=x-a*mh/(np.sqrt(vh)+eps); fh.append(lf(x))
    return np.array(fh)

def newton_m(gf,hf,lf,x0,a,n):
    x=x0.copy().astype(float); fh=[lf(x)]
    for _ in range(n):
        g=gf(x); H=hf(x)+1e-8*np.eye(2); x=x-a*np.linalg.solve(H,g); fh.append(lf(x))
    return np.array(fh)

n_max = 160
f_gd = gd(grad_C, loss_C, x0_C, 0.0012, n_max)
f_poly = polyak_step(grad_C, loss_C, x0_C, 0, 1e-3, 120)
f_hb = hb(grad_C, loss_C, x0_C, 0.0008, 0.86, 120)
f_nes = nesterov(grad_C, loss_C, x0_C, 0.0007, 0.90, 150)
f_adam = adam(grad_C, loss_C, x0_C, 0.006, 0.80, 0.999, 1e-8, 150)
f_newt = newton_m(grad_C, hessian_C, loss_C, x0_C, 0.22, 20)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.semilogy(f_gd, 'k--', lw=1.5, label='GD (Q1, 160 it.)')
ax.semilogy(range(121), f_poly[:121], 'b-', lw=2, label='Polyak (Q1, 120 it.)')
ax.semilogy(range(121), f_hb[:121], 'c-', lw=2, label='Heavy Ball (Q1, 120 it.)')
ax.semilogy(range(151), f_nes[:151], 'r-', lw=2.5, label='Nesterov (Q2, 150 it.)')
ax.semilogy(range(151), f_adam[:151], 'm-', lw=2, label='Adam (Q2, 150 it.)')
ax.semilogy(range(21), f_newt[:21], 'g-o', lw=2.5, ms=5, label="Newton (Q3, 20 it.)")
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective value (log scale)')
ax.set_title('Cross-Question: All Methods on Benchmark C (Rosenbrock)')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 160)

ax = axes[1]
methods = ['GD\n(Q1)', 'Polyak\n(Q1)', 'Heavy\nBall(Q1)', 'Nesterov\n(Q2)', 'Adam\n(Q2)', "Newton\n(Q3)"]
final_vals = [float(f_gd[160]), float(f_poly[120]), float(f_hb[120]),
              float(f_nes[150]), float(f_adam[150]), float(f_newt[20])]
colors = ['#666666', '#1f77b4', '#17becf', '#d62728', '#9467bd', '#2ca02c']
bars = ax.bar(methods, final_vals, color=colors, edgecolor='black', linewidth=0.8)
ax.axhline(y=0, color='gold', linestyle='--', lw=2, label='Global min f*=0')
for bar, val in zip(bars, final_vals):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.04,
            '{:.3f}'.format(val), ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_ylabel('Final objective value f(x_k)')
ax.set_title('Benchmark C: Final Values by Method and Question')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('figures/all_methods_benchmark_C.pdf', bbox_inches='tight')
plt.close()
print('Saved all_methods_benchmark_C.pdf')
