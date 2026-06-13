"""
===============================================================================
Dimension 2: Linear Stability Analysis & Turing Instability
===============================================================================

Purpose: Derive the dispersion relation for the modified Keller-Segel PDE,
         find critical conditions for pattern formation, and predict
         observable quantities (core spacing, growth rate).

Dependency: dim1_theory_report.json (for physical parameters)
Outputs:    dim2_stability_report.json
===============================================================================
"""

import json, sys, io
import numpy as np
from scipy.optimize import minimize_scalar

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
print("Dimension 2: Linear Stability Analysis")
print("=" * 70)

# =====================================================================
# Part A: Modified KS Equation & Uniform Steady State
# =====================================================================

print("\n" + "=" * 70)
print("Part A: Governing Equation & Steady State")
print("=" * 70)

print("""
The C++ simulation evolves:

  d(phi)/dt = D*lap(phi) - gamma*div(phi*grad(phi)) - beta*phi + S

with D=1 (normalized), source S = rho(x,y,z).

In the Fourier domain, the chemotaxis term phi*grad(phi) acts as a
nonlocal interaction. To leading order, the linearized nonlocal
chemotaxis maps to:

  F[phi*grad(phi)] ~ -phi_0 * k^2 / (1 + sigma^2*k^2) * phi_k

giving the dispersion relation:

  lambda(k) = -D*k^2 + gamma*phi_0 * k^2/(1 + sigma^2*k^2) - beta     ... (1)

where:
  k = |k| is the wavenumber
  phi_0 = S/beta is the uniform steady state
  sigma is the Gaussian kernel width (sigma_nd = 1.0 in simulation)
""")

# =====================================================================
# Part B: Parameter Space & Dispersion Relation
# =====================================================================

print("=" * 70)
print("Part B: Dispersion Relation Analysis")
print("=" * 70)

# Simulation parameters
D = 1.0
sigma = 1.0  # dimensionless kernel width

# Function to compute dispersion relation
def dispersion(k2, gamma, beta, phi0, sigma2=sigma**2):
    """lambda(k^2) for given parameters"""
    k2 = np.asarray(k2)
    return -D * k2 + gamma * phi0 * k2 / (1.0 + sigma2 * k2) - beta

def find_max_growth(gamma, beta, phi0, sigma2=sigma**2):
    """Find k^2_max and lambda_max"""
    def neg_lambda(k2):
        if k2 <= 0:
            return 1e10
        return -dispersion(k2, gamma, beta, phi0, sigma2)
    
    result = minimize_scalar(neg_lambda, bounds=(1e-6, 100), method='bounded')
    k2_max = result.x
    lambda_max = -result.fun
    return k2_max, lambda_max

# =====================================================================
# Part C: Parameter Space Analysis
# =====================================================================

print("=" * 70)
print("Part C: Stability Analysis for Key Parameter Values")
print("=" * 70)

# Analyze for key parameter combinations
# S is roughly 1 in the simulation (normalized)
# phi_0 = S/beta

parameter_sets = [
    {"label": "C++ default", "gamma": 6.0, "beta": 0.6, "S": 1.0},
    {"label": "Config file", "gamma": 8.0, "beta": 0.5, "S": 1.0},
    {"label": "Weak gamma", "gamma": 2.0, "beta": 0.6, "S": 1.0},
    {"label": "Strong gamma", "gamma": 12.0, "beta": 0.6, "S": 1.0},
    {"label": "Low beta", "gamma": 6.0, "beta": 0.2, "S": 1.0},
    {"label": "High beta", "gamma": 6.0, "beta": 1.5, "S": 1.0},
]

results = []

for ps in parameter_sets:
    gamma = ps["gamma"]
    beta = ps["beta"]
    S = ps["S"]
    phi0 = S / beta
    
    k2_max, lambda_max = find_max_growth(gamma, beta, phi0)
    k_max = np.sqrt(max(k2_max, 0))
    
    # Predicted core spacing (wavelength = 2*pi/k_max)
    lambda_wavelength = 2 * np.pi / max(k_max, 1e-10)
    
    # Critical condition for Turing instability
    # gamma*phi0 > 1 is the simplified condition
    # More precisely: (sqrt(gamma*phi0) - 1)^2 > beta
    turing_condition = gamma * phi0
    is_unstable = lambda_max > 0
    
    dx = 20.0 / 40  # grid spacing (dimensionless)
    spacing_grid = lambda_wavelength / dx if k_max > 0 else float('inf')
    
    result = {
        "label": ps["label"],
        "gamma": gamma, "beta": beta, "S": S, "phi0": phi0,
        "gamma_phi0": turing_condition,
        "k2_max": float(k2_max), "k_max": float(k_max),
        "lambda_max": float(lambda_max),
        "predicted_wavelength": float(lambda_wavelength),
        "predicted_core_spacing_grid_cells": float(spacing_grid),
        "is_unstable": bool(is_unstable),
    }
    results.append(result)
    
    status = "UNSTABLE (cores form)" if is_unstable else "STABLE (uniform)"
    print(f"\n{ps['label']}: gamma={gamma}, beta={beta}, phi0={phi0:.4f}")
    print(f"  gamma*phi0 = {turing_condition:.4f}")
    print(f"  k_max = {k_max:.4f}, lambda_max = {lambda_max:.4f}")
    print(f"  Predicted wavelength = {lambda_wavelength:.2f} (dimensionless)")
    print(f"  Predicted core spacing = {spacing_grid:.1f} grid cells")
    print(f"  Status: {status}")

# =====================================================================
# Part D: Critical Line in (gamma, beta) Space
# =====================================================================

print("\n" + "=" * 70)
print("Part D: Critical Line gamma_c(beta)")
print("=" * 70)

# Find gamma_c for each beta such that lambda_max = 0
# From our derivation: lambda_max = (sqrt(gamma*phi0) - 1)^2 - beta = 0
# → sqrt(gamma*S/beta) = 1 + sqrt(beta)
# → gamma*S/beta = (1 + sqrt(beta))^2
# → gamma_c = beta * (1 + sqrt(beta))^2 / S

betas = np.logspace(-1, np.log10(2.0), 50)
gamma_c_theory = betas * (1 + np.sqrt(betas))**2  # S=1

print("\nAnalytical critical line (S=1):")
print(f"  gamma_c(beta) = beta * (1 + sqrt(beta))^2")
print(f"\n  At beta=0.6: gamma_c = {0.6 * (1 + np.sqrt(0.6))**2:.4f}")
print(f"  At beta=0.5: gamma_c = {0.5 * (1 + np.sqrt(0.5))**2:.4f}")

critical_points = []
for beta_test in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]:
    gamma_c = beta_test * (1 + np.sqrt(beta_test))**2
    critical_points.append({"beta": beta_test, "gamma_c": float(gamma_c)})
    print(f"  beta={beta_test:.1f}: gamma_c={gamma_c:.4f}")

# =====================================================================
# Part E: Comparison with Simulation
# =====================================================================

print("\n" + "=" * 70)
print("Part E: Theory vs Simulation Predictions")
print("=" * 70)

# From the C++ test run: gamma=6, beta=0.6 → avg_cores=191
# The simulation grid is 40x40x40, domain [-10, 10] in each direction
# Total volume = 20^3 = 8000 (dimensionless)
# If core spacing ~ 8.5 grid cells, volume per core ~ (8.5)^3 ≈ 614
# Predicted cores ≈ 40^3 / (8.5)^3 ≈ 64000 / 614 ≈ 104

# But the simulation has R_max constraint, so not all cores form links
# And cores merge during detection

dx = 20.0 / 40  # = 0.5
grid_volume = 40**3  # = 64000 grid cells

for r in results:
    if r["is_unstable"]:
        spacing = r["predicted_core_spacing_grid_cells"]
        if spacing > 0 and spacing < 100:
            predicted_cores = grid_volume / (spacing**3)
            r["predicted_n_cores"] = float(predicted_cores)
            print(f"\n{r['label']}:")
            print(f"  Core spacing: {spacing:.1f} grid cells")
            print(f"  Volume per core: {spacing**3:.0f} cells^3")
            print(f"  Predicted n_cores: {predicted_cores:.0f}")

# =====================================================================
# Part F: Growth Rate vs k (for plotting)
# =====================================================================

print("\n" + "=" * 70)
print("Part F: Full Dispersion Curves")
print("=" * 70)

# Compute dispersion curves for key parameter sets
k2_range = np.linspace(0, 20, 200)
dispersion_curves = {}

for ps in parameter_sets:
    gamma = ps["gamma"]
    beta = ps["beta"]
    S = ps["S"]
    phi0 = S / beta
    lambdas = dispersion(k2_range, gamma, beta, phi0)
    
    key = f"gamma{gamma}_beta{beta}"
    dispersion_curves[key] = {
        "label": ps["label"],
        "gamma": gamma, "beta": beta,
        "k2": k2_range.tolist(),
        "lambda": lambdas.tolist(),
        "lambda_max": float(np.max(lambdas)),
    }
    
    lam_max = np.max(lambdas)
    k2_at_max = k2_range[np.argmax(lambdas)]
    print(f"  {ps['label']}: max lambda={lam_max:.4f} at k^2={k2_at_max:.3f}")

# =====================================================================
# Save Results
# =====================================================================

output = {
    "theory_version": "2.0",
    "governing_equation": "d(phi)/dt = D*lap(phi) - gamma*div(phi*grad(phi)) - beta*phi + S",
    "dispersion_relation": "lambda(k) = -D*k^2 + gamma*phi0*k^2/(1+sigma^2*k^2) - beta",
    "turing_instability_condition": "gamma*phi0 > (1+sqrt(beta))^2",
    "parameter_analysis": results,
    "critical_line": {
        "formula": "gamma_c(beta) = beta * (1 + sqrt(beta))^2 / S",
        "points": critical_points,
    },
    "dispersion_curves": dispersion_curves,
    "key_predictions": {
        "for_gamma6_beta06": {
            "gamma_phi0": 10.0,
            "k_max": next(r["k_max"] for r in results if r["label"] == "C++ default"),
            "lambda_max": next(r["lambda_max"] for r in results if r["label"] == "C++ default"),
            "predicted_wavelength": next(r["predicted_wavelength"] for r in results if r["label"] == "C++ default"),
        }
    }
}

with open("dim2_stability_report.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print("Dimension 2 COMPLETE. Report: dim2_stability_report.json")
print(f"{'='*70}")

print("""
=== Dimension 2 Key Conclusions ===

1. The modified KS equation exhibits Turing instability when:
   gamma*phi0 > (1 + sqrt(beta))^2

2. For C++ default params (gamma=6, beta=0.6):
   - gamma*phi0 = 10.0 > 3.15 (threshold) -> UNSTABLE -> cores form!
   - Most unstable wavenumber k_max = predicted from theory
   - Growth rate lambda_max = predicted from theory

3. The critical line gamma_c(beta) provides a falsifiable prediction
   that will be tested against parameter sweep simulations.

4. Core spacing prediction: d_c = 2*pi/k_max (to be verified)

5. At beta=0.6, the critical gamma is gamma_c ~ 3.15
   -> Simulation at gamma=6 is well into the ordered phase
""")