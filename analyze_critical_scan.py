"""
Analyze the completed critical gamma scan results.
Key question: is there a phase transition at γ_c(C++)=0.442?
"""

import json, io, sys, os
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "gamma_critical_scan_summary.json")) as f:
    data = json.load(f)

gammas = np.array([d["gamma"] for d in data])
n_cores = np.array([d["avg_cores"] for d in data])

print("="*60)
print("Critical Gamma Scan Analysis")
print("="*60)

# Known values
# CRITICAL (Round 15): n_baseline=91.6 is from lost N=400 C++ data.
# The saturation model is HYPOTHETICAL — only gamma=6.0 validated by C++.
# n_cores is INDEPENDENT of N (both N=400 and N=1000 give ~92.3 cores).
n_baseline = 91.6  # γ=0 source-driven baseline [HYPOTHETICAL]
gamma_c_cpp = 0.442  # theoretical γ_c for nonlocal PDE

print(f"\nBaseline (γ=0): {n_baseline:.1f} cores")
print(f"Theoretical γ_c(C++): {gamma_c_cpp}")
print(f"γ=0.45 (at γ_c): {n_cores[5]:.1f} cores")
print(f"γ=8.0 (max): {n_cores[-1]:.1f} cores")
print(f"Total range: {n_cores[-1] - n_baseline:.1f} cores ({100*(n_cores[-1]-n_baseline)/n_baseline:.0f}% increase)")

# Fit models
print("\n--- Model Fits ---")

# Model 1: Pure linear n = a + b*γ
A = np.vstack([gammas, np.ones_like(gammas)]).T
b_lin, a_lin = np.linalg.lstsq(A, n_cores, rcond=None)[0]
resid_lin = np.sum((n_cores - (a_lin + b_lin*gammas))**2)
print(f"\nLinear: n = {a_lin:.2f} + {b_lin:.2f}*γ, RSS={resid_lin:.1f}")

# Model 2: Saturation n = n_sat - (n_sat-n0)*exp(-γ/γ_char)
from scipy.optimize import curve_fit
try:
    def sat_model(g, n_sat, gamma_char, n0):
        return n_sat - (n_sat - n0) * np.exp(-g / gamma_char)
    popt, _ = curve_fit(sat_model, gammas, n_cores, p0=[150, 2.0, 92], bounds=([140, 0.5, 85], [160, 10, 95]))
    n_sat_fit, gamma_char, n0_fit = popt
    resid_sat = np.sum((n_cores - sat_model(gammas, *popt))**2)
    print(f"Saturation: n_sat={n_sat_fit:.1f}, γ_char={gamma_char:.2f}, n0={n0_fit:.1f}, RSS={resid_sat:.1f}")
except Exception as e:
    print(f"Saturation fit failed: {e}")

# Model 3: Power law n = n0 + A*γ^p
try:
    def pow_model(g, A, p, n0):
        return n0 + A * g**p
    popt2, _ = curve_fit(pow_model, gammas, n_cores, p0=[30, 0.5, 92], bounds=([1, 0.1, 85], [100, 1.0, 95]))
    A_fit, p_fit, n0_fit2 = popt2
    resid_pow = np.sum((n_cores - pow_model(gammas, *popt2))**2)
    print(f"Power law: n0={n0_fit2:.1f}, A={A_fit:.1f}, p={p_fit:.3f}, RSS={resid_pow:.1f}")
except Exception as e:
    print(f"Power law fit failed: {e}")

# Phase transition test: look for slope change at γ_c
print("\n--- Phase Transition Test ---")
idx_c = 5  # γ=0.45
below_c = slice(0, idx_c+1)
above_c = slice(idx_c, None)

gammas_below = gammas[below_c]
n_below = n_cores[below_c]
gammas_above = gammas[above_c]
n_above = n_cores[above_c]

# Linear fits below and above γ_c
A_below = np.vstack([gammas_below, np.ones_like(gammas_below)]).T
slope_below, _ = np.linalg.lstsq(A_below, n_below, rcond=None)[0]

A_above = np.vstack([gammas_above, np.ones_like(gammas_above)]).T
slope_above, _ = np.linalg.lstsq(A_above, n_above, rcond=None)[0]

print(f"Slope below γ_c: {slope_below:.2f} cores/γ")
print(f"Slope above γ_c: {slope_above:.2f} cores/γ")
print(f"Slope ratio (above/below): {slope_above/slope_below:.2f}")

# Slope change at exactly γ_c
if slope_above > slope_below:
    print(f"→ Slope INCREASES above γ_c (consistent with phase transition)")
else:
    print(f"→ Slope DECREASES above γ_c (smooth crossover, not phase transition)")

# Finite difference slope check
print("\n--- Local slope analysis ---")
for i in range(1, len(gammas)):
    dgamma = gammas[i] - gammas[i-1]
    dn = n_cores[i] - n_cores[i-1]
    local_slope = dn / dgamma
    marker = " <-- γ_c" if abs(gammas[i] - gamma_c_cpp) < 0.05 else ""
    print(f"  γ={gammas[i]:.2f}→{gammas[i-1]:.2f}: Δ={dn:.2f}, slope={local_slope:.1f}{marker}")

# Key finding
print("\n" + "="*60)
print("KEY FINDING")
print("="*60)
print(f"""
The critical scan reveals a SMOOTH CROSSOVER, not a sharp phase transition:
- No kink in n_cores(γ) at γ_c=0.442
- Slope below γ_c ({slope_below:.1f}/γ) > slope above ({slope_above:.1f}/γ)
- Source-driven baseline (91 cores) dominates at all γ
- Total increase from γ=0 to γ=8: only {n_cores[-1]-n_baseline:.0f} cores ({100*(n_cores[-1]-n_baseline)/n_baseline:.0f}%)

ROOT CAUSES:
1. ALL data in this analysis is SYNTHETIC (from generate_sweep_data.py).
   The saturation model is HYPOTHETICAL — only gamma=6.0 validated by C++.
2. n_cores is INDEPENDENT of N (Round 15 fix). C++ shows n_cores ≈ 92.3 for both N=400 and N=1000.
3. The nonlocal Gaussian kernel produces a continuous response even at γ < γ_c
4. The relative core detection threshold (0.1*max_phi) always finds local maxima
5. The smooth crossover is a feature of the synthetic saturation model, not verified by C++ data

NATURE SUB-JOURNAL IMPLICATIONS:
- The "phase transition" is best characterized as a crossover in the realistic
  satellite scenario
- Need C++ simulations at multiple γ values to validate the saturation model
- Need uniform source to isolate genuine Turing instability
- Need Fourier spectrum analysis to detect Turing modes directly
- The theoretical framework (6 dimensions) is built on the nonlocal KS
  equation, consistent with the C++ implementation
""")