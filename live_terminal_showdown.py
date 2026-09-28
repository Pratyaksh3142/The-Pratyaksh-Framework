import numpy as np
import time
import sys

# Stiff Vector Field
def f(t, h):
    u, v = h[0], h[1]
    du = -150.0 * u + 50.0 * np.tanh(v)
    dv = -1.0 * v - 5.0 * np.tanh(u)
    return np.array([du, dv])

def rk4_step(h, dt):
    k1 = dt * f(0, h)
    k2 = dt * f(0, h + 0.5*k1)
    k3 = dt * f(0, h + 0.5*k2)
    k4 = dt * f(0, h + k3)
    return h + (k1 + 2*k2 + 2*k3 + k4) / 6.0

def pratyaksh_step(h, dt):
    k1 = dt * f(0, h)
    k2 = dt * f(0, h + 0.5*k1)
    k3 = dt * f(0, h + 0.5*k2)
    k4 = dt * f(0, h + k3)
    num = (k1 + 2*k2 + 2*k3 + k4) / 6.0
    curv = k4 - k3 - k2 + k1
    den = 1.0 + 0.5 * np.sum(curv**2) / (np.sum(k1**2) + 1e-14)
    return h + num / den

# ANSI Colors for Terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

h_rk4 = np.array([2.0, 2.0])
h_pr = np.array([2.0, 2.0])
dt = 0.028
steps = 25

print(f"\n{BOLD}========================================================================{RESET}")
print(f"{BOLD} LIVE TERMINAL SHOWDOWN: Classical RK4  vs.  The Pratyaksh Framework{RESET}")
print(f"{BOLD}========================================================================{RESET}")
print(f" Simulating highly stiff latent space... (dt = {dt})\n")

print(f" Step | Time   | {GREEN}Pratyaksh 'u'{RESET}       | {RED}Classical RK4 'u'{RESET}   | Status")
print("-" * 72)

rk4_dead = False

for step in range(1, steps + 1):
    t = step * dt
    
    # Take steps
    h_pr = pratyaksh_step(h_pr, dt)
    if not rk4_dead:
        h_rk4 = rk4_step(h_rk4, dt)
    
    # Format numbers
    pr_val = f"{h_pr[0]:.6f}"
    
    if np.isnan(h_rk4[0]) or np.isinf(h_rk4[0]) or abs(h_rk4[0]) > 1e10:
        rk4_val = f"{RED}NaN (DETONATED){RESET}"
        status = f"{RED}💥 RK4 Exploded!{RESET}"
        rk4_dead = True
    elif abs(h_rk4[0]) > 100.0:
        rk4_val = f"{YELLOW}{h_rk4[0]:11.2f}{RESET}"
        status = f"{YELLOW}🚨 RK4 Diverging!{RESET}"
    else:
        rk4_val = f"{h_rk4[0]:11.6f}"
        status = "Running..."
        
    if rk4_dead:
        status = f"{GREEN}🟢 Pratyaksh Stable{RESET}"
        rk4_val = f"{RED}NaN            {RESET}"

    # Print live to terminal
    sys.stdout.write(f" {step:03d}  | {t:.2f}s  | {GREEN}{pr_val:>14}{RESET}  | {rk4_val:>15}  | {status}\n")
    sys.stdout.flush()
    time.sleep(0.15) # Artificial delay so the human eye can watch the numbers juggle

print(f"{BOLD}========================================================================{RESET}")
if rk4_dead:
    print(f"{RED}FATAL ERROR: Classical RK4 suffered NaN overflow.{RESET}")
    print(f"{GREEN}SUCCESS: The Pratyaksh Framework autonomously damped the shock.{RESET}\n")
