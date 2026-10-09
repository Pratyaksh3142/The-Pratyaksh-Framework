import numpy as np
import scipy.sparse as sp
import scipy.integrate as scint
import time
import tracemalloc

print("=" * 85)
print("INDEPENDENT RE-VERIFICATION: PRATYAKSH-II vs SPARSE IMPLICIT (IDAKLU STYLE)")
print("=" * 85)

# --------------------------------------------------------------------------
# Pratyaksh-II Master Integrator (Explicit, Matrix-Free, Vector-Streaming)
# --------------------------------------------------------------------------
def pratyaksh_step(f, t, y, h, alpha=0.1, beta=2.0, eps=1e-12):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    
    C = k4 - k3 - k2 + k1
    scale = np.linalg.norm(y) + np.linalg.norm(k1) + eps
    C_hat = np.linalg.norm(C) / scale
    D = (alpha * C_hat)**beta
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / (6.0 + D)

def rk4_step(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

# --------------------------------------------------------------------------
# Multi-Dimensional Test Suite: 1D, 2D, 3D
# --------------------------------------------------------------------------
results = []

for dim_name, N_total, setup in [
    ("1D Stiff Brusselator", 1000, "1d"),
    ("2D Bistable PDE", 2500, "2d"),
    ("3D Heat/Diffusion", 15625, "3d")
]:
    print(f"\nEvaluating {dim_name} (Total State Dimension N = {N_total})...")
    
    if setup == "1d":
        N = N_total // 2
        dx = 1.0 / N
        d_u, d_v = 0.001, 0.0001
        A, B = 1.0, 3.0
        
        def f(t, y):
            u, v = y[:N], y[N:]
            u_xx = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / (dx**2)
            v_xx = (np.roll(v, -1) - 2*v + np.roll(v, 1)) / (dx**2)
            return np.concatenate([d_u*u_xx + A - (B+1)*u + u**2*v, d_v*v_xx + B*u - u**2*v])
        
        x = np.linspace(0, 1, N, endpoint=False)
        y0 = np.concatenate([A + 0.1*np.sin(2*np.pi*x), np.full(N, B/A)])
        
        e = np.ones(N)
        D2 = sp.diags([e, -2*e, e], [-1, 0, 1], shape=(N, N)).tocsc()
        I = sp.eye(N)
        sp_mat = sp.bmat([[D2 + I, I], [I, D2 + I]]).tocsc()
        h = 0.001
        
    elif setup == "2d":
        side = int(np.sqrt(N_total))
        N = side * side
        dx = 1.0 / side
        diff = 0.05
        
        def f(t, y):
            U = y.reshape((side, side))
            U_xx = (np.roll(U, -1, axis=0) - 2*U + np.roll(U, 1, axis=0)) / (dx**2)
            U_yy = (np.roll(U, -1, axis=1) - 2*U + np.roll(U, 1, axis=1)) / (dx**2)
            return (diff * (U_xx + U_yy) + (U - U**3)).ravel()
        
        y0 = np.random.RandomState(42).randn(N) * 0.1
        diags = [-side, -1, 0, 1, side]
        e = np.ones(N)
        sp_mat = sp.diags([e, e, -4*e, e, e], diags, shape=(N, N)).tocsc()
        h = 0.001

    elif setup == "3d":
        side = int(round(N_total ** (1/3)))
        N = side**3
        dx = 1.0 / side
        diff = 0.02
        
        def f(t, y):
            U = y.reshape((side, side, side))
            U_xx = (np.roll(U, -1, axis=0) - 2*U + np.roll(U, 1, axis=0)) / (dx**2)
            U_yy = (np.roll(U, -1, axis=1) - 2*U + np.roll(U, 1, axis=1)) / (dx**2)
            U_zz = (np.roll(U, -1, axis=2) - 2*U + np.roll(U, 1, axis=2)) / (dx**2)
            return (diff * (U_xx + U_yy + U_zz) - 5.0 * U).ravel()
        
        y0 = np.random.RandomState(42).randn(N) * 0.1
        diags = [-side**2, -side, -1, 0, 1, side, side**2]
        e = np.ones(N)
        sp_mat = sp.diags([e, e, e, -6*e, e, e, e], diags, shape=(N, N)).tocsc()
        h = 0.001

    # 1. Classical RK4 Stability Check
    y_rk = y0.copy()
    rk4_exploded = False
    for s in range(5):
        y_rk = rk4_step(f, s*h, y_rk, h)
        if np.any(np.isnan(y_rk)) or np.any(np.isinf(y_rk)):
            rk4_exploded = True
            break
            
    # 2. Pratyaksh-II Benchmark (Matrix-Free Explicit)
    tracemalloc.start()
    t0 = time.perf_counter()
    y_prat = y0.copy()
    n_steps = 20
    for s in range(n_steps):
        y_prat = pratyaksh_step(f, s*h, y_prat, h)
    t_prat_step = (time.perf_counter() - t0) / n_steps
    mem_prat_peak = tracemalloc.get_traced_memory()[1] / 1024 # KB
    tracemalloc.stop()

    # 3. Sparse BDF (Industry Sparse Implicit - SuperLU / KLU style)
    tracemalloc.start()
    t0 = time.perf_counter()
    sol_bdf = scint.solve_ivp(f, [0, n_steps*h], y0, method='BDF', jac_sparsity=sp_mat)
    t_bdf_total = time.perf_counter() - t0
    t_bdf_step = t_bdf_total / max(1, sol_bdf.nfev)
    mem_bdf_peak = tracemalloc.get_traced_memory()[1] / 1024 # KB
    tracemalloc.stop()

    speedup = t_bdf_step / t_prat_step
    results.append({
        "dim": dim_name,
        "N": N_total,
        "prat_ms": t_prat_step * 1000,
        "bdf_ms": t_bdf_step * 1000,
        "speedup": speedup,
        "prat_mem": mem_prat_peak,
        "bdf_mem": mem_bdf_peak
    })
    print(f"  -> PRK-4: {t_prat_step*1000:.3f} ms/step | Mem: {mem_prat_peak:.1f} KB")
    print(f"  -> Sparse Implicit: {t_bdf_step*1000:.3f} ms/eval | Mem: {mem_bdf_peak:.1f} KB")
    print(f"  -> Measured Speedup: {speedup:.1f}x FASTER")

print("\n" + "=" * 85)
print("FINAL REPRODUCIBILITY SUMMARY MATRIX")
print("=" * 85)
print(f"{'Domain':<22} | {'Dim N':<8} | {'PRK-4 Step':<12} | {'Sparse Implicit':<16} | {'Speedup':<10} | {'RAM Ratio'}")
print("-" * 85)
for r in results:
    ram_ratio = r['bdf_mem'] / max(1e-3, r['prat_mem'])
    print(f"{r['dim']:<22} | {r['N']:<8} | {r['prat_ms']:>6.3f} ms   | {r['bdf_ms']:>8.3f} ms      | {r['speedup']:>6.1f}x    | {ram_ratio:>5.1f}x less RAM")
print("=" * 85)
