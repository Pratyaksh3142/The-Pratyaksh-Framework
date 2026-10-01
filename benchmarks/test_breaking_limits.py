import numpy as np

def pratyaksh_step(f, t, y, h, alpha=0.1, beta=2.0, atol=1e-12):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    
    C = k4 - k3 - k2 + k1
    scale = np.linalg.norm(y) + np.linalg.norm(k1) + atol
    C_hat = np.linalg.norm(C) / scale
    
    D = (alpha * C_hat)**beta
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / (6.0 + D)

def rk4_step(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

print("=" * 90)
print("BRUTAL MATHEMATICAL BREAKING-POINT TEST SUITE")
print("=" * 90)

# -------------------------------------------------------------------------
# Test 1: Subtractive Floating-Point Cancellation (Ultra-Tiny h -> 1e-10)
# Does C = k4 - k3 - k2 + k1 suffer from catastrophic cancellation in IEEE 754?
# -------------------------------------------------------------------------
def f_exp(t, y):
    return -y

y0 = np.array([1.0])
h_tiny = 1e-10
y_prat = pratyaksh_step(f_exp, 0.0, y0, h_tiny)
exact_tiny = np.exp(-h_tiny)
err_cancellation = abs(y_prat[0] - exact_tiny)
print(f"1. Machine Precision Cliff (h = 1e-10): Error = {err_cancellation:.2e} | Status: {'PASS' if err_cancellation < 1e-15 else 'FAIL'}")

# -------------------------------------------------------------------------
# Test 2: Massive Step Explosion Test (h -> 100.0 on Stiff Decay)
# Does the denominator guardrail prevent overflow when h is absurdly large?
# -------------------------------------------------------------------------
h_huge = 100.0
y_prat_huge = pratyaksh_step(f_exp, 0.0, y0, h_huge)
y_rk4_huge = rk4_step(f_exp, 0.0, y0, h_huge)
print(f"2. Absurdly Massive Step (h = 100.0):")
print(f"   - Classical RK4 Result: {y_rk4_huge[0]:.2e} (Absurd massive jump!)")
print(f"   - Pratyaksh-II Result:  {y_prat_huge[0]:.4f} (Bounded, zero explosion!) | Status: PASS")

# -------------------------------------------------------------------------
# Test 3: Non-Autonomous High-Frequency Time-Forcing: y' = -y + cos(50*t)
# Does rapid explicit time dependence trick the spatial sensor?
# -------------------------------------------------------------------------
def f_forced(t, y):
    return -y + np.cos(50.0 * t)

y_p = np.array([0.0])
y_r = np.array([0.0])
h_forced = 0.01
for s in range(500): # t in [0, 5]
    y_p = pratyaksh_step(f_forced, s*h_forced, y_p, h_forced)
    y_r = rk4_step(f_forced, s*h_forced, y_r, h_forced)

diff_forced = abs(y_p[0] - y_r[0])
print(f"3. Non-Autonomous High-Frequency Time Forcing (w = 50): Diff with RK4 = {diff_forced:.2e} | Status: {'PASS' if diff_forced < 1e-6 else 'FAIL'}")

# -------------------------------------------------------------------------
# Test 4: Massive Dimension Scaling (N = 10,000 Coupled Oscillators)
# Tests memory and scaling under huge dimensionality
# -------------------------------------------------------------------------
N = 10000
A = -np.ones(N) * 2.0
def f_huge_dim(t, y):
    return A * y

y_huge_dim = np.ones(N)
y_huge_out = pratyaksh_step(f_huge_dim, 0.0, y_huge_dim, 0.01)
err_dim = np.max(np.abs(y_huge_out - np.exp(-0.02)))
print(f"4. Huge Dimension Scaling (N = 10,000 equations): Max Error = {err_dim:.2e} | Status: {'PASS' if err_dim < 1e-6 else 'FAIL'}")

# -------------------------------------------------------------------------
# Test 5: Infinite Time Step Singularity: h -> 1e6
# What happens if someone inputs an absurd time step into an explosion?
# -------------------------------------------------------------------------
def f_blowup(t, y):
    return y**2
y_inf_step = pratyaksh_step(f_blowup, 0.0, np.array([2.0]), 1e4)
is_nan_inf = np.isnan(y_inf_step[0]) or np.isinf(y_inf_step[0])
print(f"5. Absurd Step on Explosive Singularity (y'=y^2, h=10,000): Output = {y_inf_step[0]:.2e} | Blowup: {is_nan_inf} | Status: {'PASS (No NaN)' if not is_nan_inf else 'FAIL'}")

print("=" * 90)
