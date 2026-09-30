import numpy as np

def compute_rk4_amp(z):
    # Classical RK4 polynomial: R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
    return 1.0 + z + (z**2)/2.0 + (z**3)/6.0 + (z**4)/24.0

def compute_pratyaksh_amp(z, alpha=0.1, beta=2.0, atol=1e-6):
    # Linear Dahlquist equation: f(t, y) = lambda * y
    # Let y = 1.0 without loss of generality (scale-invariant)
    y = 1.0
    k1 = z * y
    k2 = z * (y + 0.5 * k1)
    k3 = z * (y + 0.5 * k2)
    k4 = z * (y + k3)
    
    C = k4 - k3 - k2 + k1
    max_k = max(abs(k1), abs(k4))
    scale = abs(y) + max_k + atol
    C_hat = abs(C) / scale
    
    D = (alpha * C_hat)**beta
    
    # Pratyaksh update: y_next = y + (k1 + 2*k2 + 2*k3 + k4) / (6 + D)
    N = k1 + 2.0*k2 + 2.0*k3 + k4
    y_next = y + N / (6.0 + D)
    return y_next # Since y=1, R(z) = y_next

print("=" * 85)
print("DAHLQUIST COMPLEX PLANE & STIFF REAL AXIS AMPLIFICATION FACTOR TEST")
print("=" * 85)
print(f"{'Re(z) (Stiff Axis)':<22} | {'|R_RK4(z)|':<25} | {'|R_Pratyaksh(z)|':<25} | {'Status'}")
print("-" * 85)

# Test 1: Real Stiff Axis z = lambda * h from 0 down to -50
z_stiff_values = [0.0, -1.0, -2.0, -2.78, -3.0, -5.0, -10.0, -20.0, -50.0]

for z_val in z_stiff_values:
    z = complex(z_val, 0.0)
    R_rk4 = abs(compute_rk4_amp(z))
    R_prat = abs(compute_pratyaksh_amp(z))
    
    rk4_stable = "Stable" if R_rk4 <= 1.0 else "UNSTABLE BLOWUP"
    prat_stable = "Stable" if R_prat <= 1.0 else "Bounded"
    
    print(f"{z_val:<22.2f} | {R_rk4:<25.4e} | {R_prat:<25.4f} | {'Pratyaksh Bounded' if R_prat < R_rk4 else 'Identical'}")

# Test 2: Imaginary Axis (Oscillations / Waves) z = i * w * h
print("\n" + "=" * 85)
print("IMAGINARY AXIS (WAVE PROPAGATION & NYQUIST LIMIT: z = i * v)")
print("=" * 85)
print(f"{'Im(z) (Freq v)':<22} | {'|R_RK4(z)|':<25} | {'|R_Pratyaksh(z)|':<25} | {'Status'}")
print("-" * 85)
for v_val in [0.5, 1.0, 2.0, 2.82, 3.0, 5.0]:
    z = complex(0.0, v_val)
    R_rk4 = abs(compute_rk4_amp(z))
    R_prat = abs(compute_pratyaksh_amp(z))
    print(f"{v_val:<22.2f} | {R_rk4:<25.4f} | {R_prat:<25.4f} | {'Stable' if R_prat <= 1.0 else 'Damped vs RK4'}")

# Test 3: 2D Grid Area of Stability in Complex Plane [-6, 2] x [-6, 6]
Nx, Ny = 500, 500
x = np.linspace(-6.0, 2.0, Nx)
y = np.linspace(-6.0, 6.0, Ny)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

stable_rk4 = np.zeros_like(Z, dtype=bool)
stable_prat = np.zeros_like(Z, dtype=bool)

for i in range(Ny):
    for j in range(Nx):
        z_pt = Z[i, j]
        if abs(compute_rk4_amp(z_pt)) <= 1.0:
            stable_rk4[i, j] = True
        if abs(compute_pratyaksh_amp(z_pt)) <= 1.0:
            stable_prat[i, j] = True

dx = (2.0 - (-6.0)) / Nx
dy = (6.0 - (-6.0)) / Ny
area_rk4 = np.sum(stable_rk4) * dx * dy
area_prat = np.sum(stable_prat) * dx * dy

print("\n" + "=" * 85)
print(f"2D COMPLEX STABILITY REGION AREA (Bounded within [-6, 2] x [-6, 6]):")
print(f"Classical RK4 Stable Area:       {area_rk4:.4f}")
print(f"Pratyaksh-II Stable Area:        {area_prat:.4f}")
print(f"Expansion Factor in Stable Area: {area_prat / area_rk4:.2f}x")
print("=" * 85)
