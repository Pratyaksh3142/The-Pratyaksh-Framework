import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Amplification Factors of Industry Architectures
# ----------------------------------------------------------------------

# 1. Forward Euler: R(z) = 1 + z
def r_euler(z):
    return 1.0 + z

# 2. Classical RK4: R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
def r_rk4(z):
    return 1.0 + z + (z**2)/2.0 + (z**3)/6.0 + (z**4)/24.0

# 3. DOPRI5 (Dormand-Prince 5(4) - Standard in SciPy RK45 and MATLAB ode45):
# Order 5 polynomial: R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 + z^5/120
def r_dopri5(z):
    return 1.0 + z + (z**2)/2.0 + (z**3)/6.0 + (z**4)/24.0 + (z**5)/120.0

# 4. Backward Euler (Implicit L-Stable): R(z) = 1 / (1 - z)
def r_back_euler(z):
    return 1.0 / (1.0 - z)

# 5. Crank-Nicolson / Implicit Midpoint (A-Stable): R(z) = (1 + z/2) / (1 - z/2)
def r_crank_nicolson(z):
    return (1.0 + z/2.0) / (1.0 - z/2.0)

# 6. Radau IIA (3-stage Order 5 - Gold standard in SciPy Radau):
# R(z) = (1 + 2z/5 + z^2/20) / (1 - 3z/5 + 3z^2/20 - z^3/60)
def r_radau_iia(z):
    num = 1.0 + (2.0/5.0)*z + (1.0/20.0)*(z**2)
    den = 1.0 - (3.0/5.0)*z + (3.0/20.0)*(z**2) - (1.0/60.0)*(z**3)
    return num / den

# 7. Pratyaksh-II (Ours - Explicit Matrix-Free):
def r_pratyaksh(z, alpha=0.1, beta=2.0, atol=1e-6):
    y = 1.0
    k1 = z * y
    k2 = z * (y + 0.5 * k1)
    k3 = z * (y + 0.5 * k2)
    k4 = z * (y + k3)
    C = k4 - k3 - k2 + k1
    scale = abs(y) + atol
    C_hat = abs(C) / scale
    D = (alpha * C_hat)**beta
    N = k1 + 2.0*k2 + 2.0*k3 + k4
    return 1.0 + N / (6.0 + D)

# ----------------------------------------------------------------------
# Benchmarking along Stiff Real Axis
# ----------------------------------------------------------------------
z_stiff = np.array([0.0, -1.0, -2.0, -2.78, -5.0, -10.0, -20.0, -50.0, -100.0, -500.0, -1000.0])

print("=" * 115)
print(f"{'Re(z)':<10} | {'Forward Euler':<15} | {'Classical RK4':<15} | {'DOPRI5 (RK45)':<15} | {'SciPy Radau':<15} | {'Pratyaksh-II (Ours)':<20}")
print("=" * 115)

for z in z_stiff:
    e_val = abs(r_euler(z))
    rk4_val = abs(r_rk4(z))
    dopri_val = abs(r_dopri5(z))
    radau_val = abs(r_radau_iia(z))
    prat_val = abs(r_pratyaksh(z))
    print(f"{z:<10.1f} | {e_val:<15.2e} | {rk4_val:<15.2e} | {dopri_val:<15.2e} | {radau_val:<15.4f} | {prat_val:<20.4f}")

print("=" * 115)

# ----------------------------------------------------------------------
# Generate Publication-Grade Industry Showdown Plot
# ----------------------------------------------------------------------
z_dense = np.linspace(-100.0, 0.0, 1000)

amp_euler = np.abs([r_euler(z) for z in z_dense])
amp_rk4 = np.abs([r_rk4(z) for z in z_dense])
amp_dopri5 = np.abs([r_dopri5(z) for z in z_dense])
amp_radau = np.abs([r_radau_iia(z) for z in z_dense])
amp_prat = np.abs([r_pratyaksh(z) for z in z_dense])

plt.figure(figsize=(11, 6))

plt.semilogy(z_dense, amp_euler, label="Forward Euler (Order 1 Explicit) - Blows up", color="#d9534f", linestyle="--", linewidth=1.5)
plt.semilogy(z_dense, amp_rk4, label="Classical RK4 (Order 4 Explicit) - Blows up to 10^6", color="#f0ad4e", linestyle="--", linewidth=2.0)
plt.semilogy(z_dense, amp_dopri5, label="DOPRI5 / SciPy RK45 (Order 5 Explicit) - Blows up to 10^8", color="#e74c3c", linestyle=":", linewidth=2.0)
plt.semilogy(z_dense, amp_radau, label="SciPy Radau IIA (Order 5 Implicit) - Requires O(N^3) Jacobians", color="#3498db", linestyle="-.", linewidth=2.0)
plt.semilogy(z_dense, amp_prat, label="Pratyaksh-II (Ours - Order 4 Explicit) - Asymptotes to 1.0 (Zero Jacobian!)", color="#2ecc71", linewidth=3.0)

plt.axhline(1.0, color="black", linestyle="-", linewidth=1.0, label="Absolute Stability Threshold (|R| = 1.0)")
plt.title("Industry Architecture Showdown on the Stiff Dahlquist Axis (Re(z) in [-100, 0])", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Stiffness Parameter: Re(z) = lambda * h (Log-scale negative axis)", fontsize=11)
plt.ylabel("Amplification Factor |R(z)| (Log Scale)", fontsize=11)
plt.xlim(-100, 0)
plt.ylim(1e-2, 1e8)
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.legend(loc="upper left", fontsize=9, framealpha=0.95)

plt.tight_layout()
output_img = "/Users/pi3.142/.gemini/antigravity/scratch/The-Pratyaksh-Framework/benchmarks/dahlquist_industry_showdown.png"
plt.savefig(output_img, dpi=300)
print(f"\nSaved industry showdown plot to: {output_img}")
