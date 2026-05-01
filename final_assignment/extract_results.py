"""Extract numerical results for the report."""
import numpy as np
np.random.seed(42)

# Reproduce data
m = 1000
X_data = np.random.randn(m, 2)
theta_star = np.array([3.0, 4.0])
eps_noise = np.random.randn(m)
y_data = X_data @ theta_star + eps_noise

def loss_A(theta):
    r = X_data @ theta - y_data
    return 0.5 * np.mean(r**2)
def grad_A(theta):
    return X_data.T @ (X_data @ theta - y_data) / m
def loss_B(x):
    return (x[0]-1)**2 + 5*(x[1]-2)**2 + np.sin(x[0])
def grad_B(x):
    return np.array([2*(x[0]-1)+np.cos(x[0]), 10*(x[1]-2)])
def loss_C(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def grad_C(x):
    return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])

theta0_A = np.array([0.0, 0.0])
x0_B = np.array([-1.0, 4.0])
x0_C = np.array([-1.0, 1.0])

# Re-run all methods and extract final values
def gd(g, f, x0, a, n):
    x = x0.copy().astype(float)
    for _ in range(n): x = x - a*g(x)
    return f(x), x

def polyak(g, f, x0, fs, eps, n):
    x = x0.copy().astype(float)
    for _ in range(n):
        gr = g(x); ak = (f(x)-fs)/(np.dot(gr,gr)+eps); x = x - ak*gr
    return f(x), x

def adag(g, f, x0, a0, eps, n):
    x = x0.copy().astype(float); G = np.zeros_like(x)
    for _ in range(n):
        gr = g(x); G += gr**2; x = x - a0/(np.sqrt(G)+eps)*gr
    return f(x), x

def rms(g, f, x0, a0, b, eps, n):
    x = x0.copy().astype(float); v = np.zeros_like(x)
    for _ in range(n):
        gr = g(x); v = b*v+(1-b)*gr**2; x = x - a0/(np.sqrt(v)+eps)*gr
    return f(x), x

def hb(g, f, x0, a, b, n):
    x = x0.copy().astype(float); z = np.zeros_like(x)
    for _ in range(n):
        gr = g(x); z = b*z+a*gr; x = x - z
    return f(x), x

def nest(g, f, x0, a, bm, n):
    x = x0.copy().astype(float); z = np.zeros_like(x)
    for k in range(1,n+1):
        bk = min((k-1)/(k+2), bm); la = x+bk*z; gr = g(la); z = bk*z-a*gr; x = x+z
    return f(x), x

def adam(g, f, x0, a, b1, b2, eps, n):
    x = x0.copy().astype(float); mv = np.zeros_like(x); vv = np.zeros_like(x)
    for t in range(1,n+1):
        gr = g(x); mv = b1*mv+(1-b1)*gr; vv = b2*vv+(1-b2)*gr**2
        mh = mv/(1-b1**t); vh = vv/(1-b2**t); x = x - a*mh/(np.sqrt(vh)+eps)
    return f(x), x

print("="*70)
print("Q1 FINAL VALUES (120 iterations)")
print("="*70)
print(f"{'Method':<15} {'Bench A':>12} {'Bench B':>12} {'Bench C':>12}")
print("-"*55)
for name, fn in [
    ("GD", lambda g,f,x0,b: gd(g,f,x0,{'A':0.08,'B':0.06,'C':0.0012}[b],120)),
    ("Polyak", lambda g,f,x0,b: polyak(g,f,x0,0,{'A':1e-4,'B':1e-4,'C':1e-3}[b],120)),
    ("Adagrad", lambda g,f,x0,b: adag(g,f,x0,{'A':1.8,'B':1.2,'C':0.45}[b],1e-5,120)),
    ("RMSprop", lambda g,f,x0,b: rms(g,f,x0,{'A':0.22,'B':0.14,'C':0.0035}[b],0.9,1e-5,120)),
    ("Heavy Ball", lambda g,f,x0,b: hb(g,f,x0,{'A':0.045,'B':0.035,'C':0.0008}[b],{'A':0.88,'B':0.90,'C':0.86}[b],120)),
]:
    vA = fn(grad_A, loss_A, theta0_A, 'A')[0]
    vB = fn(grad_B, loss_B, x0_B, 'B')[0]
    vC = fn(grad_C, loss_C, x0_C, 'C')[0]
    print(f"{name:<15} {vA:>12.6f} {vB:>12.6f} {vC:>12.6f}")

print("\n" + "="*70)
print("Q2 FINAL VALUES (150 iterations)")
print("="*70)
print(f"{'Method':<15} {'Bench A':>12} {'Bench B':>12} {'Bench C':>12}")
print("-"*55)
for name, fn in [
    ("GD", lambda g,f,x0,b: gd(g,f,x0,{'A':0.08,'B':0.06,'C':0.0012}[b],150)),
    ("Nesterov", lambda g,f,x0,b: nest(g,f,x0,{'A':0.06,'B':0.035,'C':0.0007}[b],{'A':0.90,'B':0.92,'C':0.90}[b],150)),
    ("Adam", lambda g,f,x0,b: adam(g,f,x0,{'A':0.12,'B':0.08,'C':0.006}[b],{'A':0.82,'B':0.80,'C':0.80}[b],0.999,1e-8,150)),
]:
    vA = fn(grad_A, loss_A, theta0_A, 'A')[0]
    vB = fn(grad_B, loss_B, x0_B, 'B')[0]
    vC = fn(grad_C, loss_C, x0_C, 'C')[0]
    print(f"{name:<15} {vA:>12.6f} {vB:>12.6f} {vC:>12.6f}")

print("\n" + "="*70)
print("Q3 FINAL VALUES")
print("="*70)
H_A = X_data.T @ X_data / m
for bname, g, hf, f, x0, ga, na in [
    ('A', grad_A, lambda x: H_A, loss_A, theta0_A, 0.08, 1.0),
    ('B', grad_B, lambda x: np.array([[2-np.sin(x[0]),0],[0,10]]), loss_B, x0_B, 0.06, 0.85),
    ('C', grad_C, lambda x: np.array([[2+1200*x[0]**2-400*x[1],-400*x[0]],[-400*x[0],200]]), loss_C, x0_C, 0.001, 0.22),
]:
    fgd = gd(g, f, x0, ga, 80)[0]
    # Newton
    x = x0.copy().astype(float)
    for _ in range(20):
        gr = g(x); H = hf(x) + 1e-8*np.eye(2)
        x = x - na*np.linalg.solve(H, gr)
    fn_val = f(x)
    print(f"Bench {bname}: GD(80)={fgd:.6f}, Newton(20)={fn_val:.6f}, Newton final x={x}")

print("\n" + "="*70)
print("Q5 FINAL POSITIONS")
print("="*70)
x0_q5 = np.array([0.2, 4.0])
# Projected GD
x = x0_q5.copy().astype(float)
for _ in range(100):
    g = grad_B(x); x = x - 0.08*g; x[0] = max(0.5, x[0])
print(f"Projected GD: x={x}, f={loss_B(x):.6f}, violation={max(0, 0.5-x[0]):.6f}")

# Penalty lambda=4.5
x = x0_q5.copy().astype(float)
for _ in range(100):
    g = grad_B(x).copy()
    if x[0] < 0.5: g[0] -= 4.5
    x = x - 0.03*g
print(f"Penalty 4.5:  x={x}, f={loss_B(x):.6f}, violation={max(0, 0.5-x[0]):.6f}")

print("\n" + "="*70)
print("Q6 FINAL POSITIONS")
print("="*70)
def fw(gf, ff, x0, bounds, beta, n):
    x = x0.copy().astype(float)
    for _ in range(n):
        g = gf(x)
        z = np.array([bounds[i][0] if g[i]>0 else bounds[i][1] for i in range(len(g))])
        x = beta*x + (1-beta)*z
    return ff(x), x

bds = [(0.5,5.0),(-5.0,10.0)]
f1,x1 = fw(lambda x: np.array([2*(x[0]-1),2*(x[1]-5)]), lambda x: (x[0]-1)**2+(x[1]-5)**2, np.array([1.0,1.0]), bds, 0.90, 180)
f2,x2 = fw(lambda x: np.array([2*(x[0]-1),2*(x[1]-5)]), lambda x: (x[0]-1)**2+(x[1]-5)**2, np.array([1.0,1.0]), bds, 0.985, 180)
f3,x3 = fw(lambda x: np.array([2*x[0],2*x[1]]), lambda x: x[0]**2+x[1]**2, np.array([3.0,3.0]), bds, 0.93, 140)
print(f"FW interior b=0.90:  x={x1}, f={f1:.6f}")
print(f"FW interior b=0.985: x={x2}, f={f2:.6f}")
print(f"FW boundary b=0.93:  x={x3}, f={f3:.6f}")
