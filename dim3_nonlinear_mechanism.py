"""
===============================================================================
Dimension 3: Nonlinear Mechanism Analysis
              Constraint-Driven Saturation in the Nonlocal KS Equation

===============================================================================
SPECIFICATION (v4.0 — Round 24 Restructured)
===============================================================================

PURPOSE:
    Analyze the nonlinear saturation mechanism that produces stable finite-
    amplitude cores in the nonlocal KS PDE. The nonlocal operator N[phi] is
    LINEAR; the nonlinearity comes from the phi >= 0 constraint (clipping).

INPUT:
    From Phase 2 (dim2_linear_stability):
      - k2_disc = 16.0         (discrete Laplacian at Nyquist)
      - C0_Nyquist = 37.38     (|C(k_Nyquist)|)
      - gamma_c(beta) = (16+beta)/37.38  (exact critical line, C++ validated)
      - lambda_max(gamma,beta) = -16 + gamma*37.38 - beta

OUTPUT:
    - dim3_nonlinear_mechanism_report.json
    - Constraint-driven saturation mechanism description
    - Algebraic identity: g_eff = gamma_c*C0 = k2_disc + beta (constant)
    - Consistency relation: A_eq = epsilon (NOT an independent prediction)
    - Core radius formula (theoretical, epsilon << 1 limit)
    - Pattern selection analysis (theoretical)

VERIFICATION:
    Mathematical self-consistency only. The amplitude equation A_eq = epsilon
    is an ALGEBRAIC IDENTITY derived from the definition g_eff = gamma_c*C0,
    NOT an independent prediction validated by C++ data.

    This distinguishes the current analysis from standard weakly nonlinear
    theory where g is computed from mode coupling and A_eq is a genuine
    prediction. Here, g_eff is DEFINED to be gamma_c*C0, which makes
    A_eq = sqrt(mu/g_eff) = sqrt(gamma_c*C0*epsilon^2 / (gamma_c*C0)) = epsilon
    a tautology.

LIMITATIONS (CRITICAL):
    1. Perturbation theory requires epsilon << 1.
       C++ operating point: gamma=6.0, gamma_c=0.444, epsilon=3.54.
       The quantitative predictions (R_core, scaling exponents) are NOT
       valid at this operating point.

    2. The amplitude equation is an ALGEBRAIC IDENTITY (C2, Round 17).
       It provides a consistency check, not an independent prediction.
       If C++ data shows A_eq != epsilon, the constraint-driven saturation
       hypothesis needs revision. If A_eq ~ epsilon, the mechanism is
       qualitatively consistent.

    3. The constraint-driven saturation mechanism is a HYPOTHESIS.
       The actual C++ saturation may involve additional effects (mode
       competition, source distribution, boundary conditions) not
       captured by the single-mode Landau description.

DEPENDENCY: dim2_linear_stability (parameters only, no JSON dependency)
STATUS:    Theoretical analysis with identified limitations
===============================================================================
"""

import json, sys, io, os
import numpy as np
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
print("Dimension 3: Nonlinear Mechanism — Constraint-Driven Saturation (v4.0)")
print("=" * 70)

# =====================================================================
# Part A: Input Parameters (from Phase 2)
# =====================================================================

print("\n" + "=" * 70)
print("Part A: Input Parameters (from Phase 2 — Linear Stability)")
print("=" * 70)

D, sigma_val, S0 = 1.0, 1.0, 1.0
k2_disc = 16.0
C0_Nyquist = 37.38

def gamma_c_nl(beta):
    return (k2_disc + beta) / C0_Nyquist

def lambda_max_nl(gamma, beta):
    return -k2_disc + gamma * C0_Nyquist - beta

print(f"  k2_disc          = {k2_disc:.1f}")
print(f"  C0_Nyquist       = {C0_Nyquist:.4f}")
print(f"  gamma_c(beta)    = ({k2_disc:.1f} + beta) / {C0_Nyquist:.4f}")
print(f"  lambda_max(gamma) = -{k2_disc:.1f} + gamma*{C0_Nyquist:.4f} - beta")

# Operating point
gamma_0, beta_0 = 6.0, 0.6
gamma_c_def = gamma_c_nl(beta_0)
epsilon_def = np.sqrt(max((gamma_0 - gamma_c_def) / gamma_c_def, 0))

print(f"\n  C++ operating point:")
print(f"    gamma = {gamma_0}, beta = {beta_0}")
print(f"    gamma_c = {gamma_c_def:.4f}")
print(f"    epsilon = sqrt((gamma - gamma_c)/gamma_c) = {epsilon_def:.4f}")
print(f"    epsilon/epsilon_c = {epsilon_def:.2f}x (>> 1, perturbation theory invalid)")

# =====================================================================
# Part B: Constraint-Driven Saturation Mechanism
# =====================================================================

print("\n" + "=" * 70)
print("Part B: Constraint-Driven Saturation Mechanism")
print("=" * 70)

print("""
FUNDAMENTAL OBSERVATION: The nonlocal KS equation is LINEAR in phi:

    d(phi)/dt = D * lap(phi) - gamma * N[phi] - beta * phi + rho(r)    ... (1)

where N[phi] = sum_j (phi_j - phi_i) * G(r_ij)/r_ij is a LINEAR operator.
There is no intrinsic cubic nonlinearity.

The nonlinearity comes from the NON-NEGATIVITY CONSTRAINT:

    phi(r) >= 0  for all r                                              ... (2)

In the C++ simulation, enforced by clipping at each time step:
    phi_new = max(0, phi_new)                                          ... (3)

This is a LINEAR COMPLEMENTARITY PROBLEM.

SATURATION MECHANISM (qualitative):
    Consider a single Fourier mode near onset:
        phi(x) = phi_0 + A * cos(k_c * x)                              ... (4)
    where phi_0 = S0/beta is the uniform steady state.

    Without constraint: A grows exponentially (lambda_max > 0).
    With constraint phi >= 0: A is bounded by A <= phi_0
    (the cosine cannot dip below -phi_0 without violating phi >= 0).

    The constraint creates an effective nonlinear damping through the
    projection operator Pi_{>=0}.
""")

# =====================================================================
# Part C: Algebraic Identity Analysis (CRITICAL)
# =====================================================================

print("=" * 70)
print("Part C: Algebraic Identity Analysis")
print("=" * 70)

print("""
CRITICAL (Round 17, C2): The amplitude equation A_eq = epsilon is an
ALGEBRAIC IDENTITY, not an independent prediction.

PROOF:
    The effective Landau description is:
        dA/dt = mu * A - g_eff * A^3                                   ... (5)

    where:
        mu = lambda_max = gamma*C0_Nyquist - k2_disc - beta
           = gamma_c*C0_Nyquist * epsilon^2                            ... (6)

        g_eff is DEFINED as:
        g_eff = gamma_c * C0_Nyquist = k2_disc + beta                  ... (7)

    Then the steady-state amplitude is:
        A_eq = sqrt(mu / g_eff)
             = sqrt(gamma_c*C0_Nyquist * epsilon^2 / (gamma_c*C0_Nyquist))
             = sqrt(epsilon^2)
             = epsilon                                                  ... (8)

    This is a TAUTOLOGY. The result A_eq = epsilon follows directly from
    the DEFINITION g_eff = gamma_c*C0_Nyquist, not from any physical
    calculation of mode coupling coefficients.

    In standard weakly nonlinear theory (e.g., Ginzburg-Landau), g is
    COMPUTED from the cubic coupling between Fourier modes:
        g = f(gamma, k_c, sigma, ...)  [computed, not defined]
    and A_eq = sqrt(mu/g) is a genuine PREDICTION.

    Here, g_eff is DEFINED, not computed. The amplitude equation is a
    CONSISTENCY CHECK, not a prediction.

CONSISTENCY INTERPRETATION:
    If C++ simulation data shows A_eq != epsilon, the constraint-driven
    saturation hypothesis is inconsistent with the data.
    If A_eq ~ epsilon, the mechanism is qualitatively consistent but
    this does NOT constitute independent validation.

    The constraint-driven saturation mechanism provides a valuable
    CONCEPTUAL FRAMEWORK for understanding why cores saturate
    (phi >= 0 prevents unbounded growth), but the specific quantitative
    relationship A_eq = epsilon has no predictive power.
""")

# =====================================================================
# Part D: Effective Landau Coefficients (with identity annotation)
# =====================================================================

print("=" * 70)
print("Part D: Effective Landau Coefficients")
print("=" * 70)

# The effective Landau free energy:
# F_eff(A) = -mu*A^2/2 + g_eff*A^4/4
# g_eff = gamma_c*C0_Nyquist = k2_disc + beta (DEFINED, not computed)
# -> A_eq = sqrt(mu/g_eff) = epsilon (ALGEBRAIC IDENTITY)

param_sets = [
    {"label": "C++ default (epsilon=3.54)",    "gamma": 6.0,  "beta": 0.6},
    {"label": "Near onset (epsilon=0.1)",      "gamma": None, "beta": 0.6, "epsilon": 0.1},
    {"label": "Near onset (epsilon=0.05)",     "gamma": None, "beta": 0.6, "epsilon": 0.05},
    {"label": "Strong drive",                   "gamma": 12.0, "beta": 0.6},
    {"label": "Low beta",                       "gamma": 6.0,  "beta": 0.2},
    {"label": "High beta",                      "gamma": 6.0,  "beta": 2.0},
]

amplitude_results = []

for ps in param_sets:
    if ps["gamma"] is not None:
        gamma_val = ps["gamma"]
        beta_val = ps["beta"]
        gc = gamma_c_nl(beta_val)
        eps_val = np.sqrt(max((gamma_val - gc) / gc, 0))
    else:
        beta_val = ps["beta"]
        gc = gamma_c_nl(beta_val)
        eps_val = ps["epsilon"]
        gamma_val = gc * (1 + eps_val**2)

    mu_val = lambda_max_nl(gamma_val, beta_val)
    phi0_val = S0 / beta_val

    # g_eff = gamma_c*C0 = k2_disc + beta (DEFINED constant)
    g_eff_const = gc * C0_Nyquist  # = k2_disc + beta

    if mu_val > 0:
        A_steady = np.sqrt(mu_val / g_eff_const)  # = eps_val (identity)
    else:
        A_steady = 0.0

    # ALGEBRAIC IDENTITY CHECK
    identity_check = abs(A_steady - eps_val) < 1e-10 if mu_val > 0 else True

    # Core radius (theoretical, epsilon << 1 limit)
    if abs(eps_val) > 1e-6:
        R_core = np.pi * np.sqrt(D / (eps_val**2))
    else:
        R_core = float('inf')

    # Validity flag
    valid_perturbation = eps_val < 0.5

    r = {
        "label": ps["label"],
        "gamma": float(gamma_val), "beta": float(beta_val),
        "gamma_c": float(gc), "phi0": float(phi0_val),
        "epsilon": float(eps_val),
        "mu": float(mu_val),
        "g_eff_defined": float(g_eff_const),
        "A_steady": float(A_steady),
        "algebraic_identity_verified": bool(identity_check),
        "perturbation_theory_valid": bool(valid_perturbation),
        "R_core_dimensionless": float(R_core),
        "R_core_grid_cells": float(R_core / 0.5),
        "saturation_mechanism": "phi >= 0 clipping (constraint-driven)",
        "note": "g_eff is DEFINED as gamma_c*C0, not computed from mode coupling. "
                "A_eq = epsilon is an ALGEBRAIC IDENTITY (C2). "
                "Quantitative predictions valid only for epsilon << 1.",
    }
    amplitude_results.append(r)

    validity_mark = " [VALID epsilon<<1]" if valid_perturbation else " [INVALID epsilon>>1]"
    print(f"\n{ps['label']}:{validity_mark}")
    print(f"  gamma={gamma_val:.4f}, beta={beta_val}, gamma_c={gc:.4f}, epsilon={eps_val:.4f}")
    print(f"  mu = lambda_max = {mu_val:.4f}")
    print(f"  phi_0 = S0/beta = {phi0_val:.4f} (upper bound from constraint)")
    print(f"  g_eff = gamma_c*C0 = {g_eff_const:.4f} (DEFINED, = k2_disc + beta = {k2_disc + beta_val:.1f})")
    print(f"  A_steady = sqrt(mu/g_eff) = {A_steady:.4f} = epsilon = {eps_val:.4f} (IDENTITY: {identity_check})")
    if valid_perturbation:
        print(f"  R_core = pi*sqrt(D/epsilon^2) = {R_core:.2f} ({R_core/0.5:.1f} grid cells)")
    else:
        print(f"  R_core = pi*sqrt(D/epsilon^2) = {R_core:.2f} [THEORETICAL ONLY, epsilon>>1 invalid]")

# =====================================================================
# Part E: Core Radius Formula (Theoretical)
# =====================================================================

print("\n" + "=" * 70)
print("Part E: Core Radius Formula (Theoretical, epsilon << 1 limit)")
print("=" * 70)

print("""
From the effective Landau amplitude equation (near onset, epsilon << 1):

    R_core = pi * sqrt(xi / mu) = pi * sqrt(D / epsilon^2)
           = pi * sqrt(D * gamma_c / (gamma - gamma_c))

VALIDITY:
    This formula is derived from the amplitude equation which assumes
    epsilon << 1. At the C++ operating point (epsilon = 3.54), this
    formula is NOT quantitatively reliable.

    For epsilon = 3.54: R_core = pi / 3.54 ~ 0.89 dimensionless ~ 1.8 grid cells
    This predicts very small cores (~2 grid cells), which may be
    resolution-limited rather than physically meaningful.

    C++ validation needed: measure core radius from simulation data
    and compare with the theoretical formula at various epsilon values.
""")

# Compute core radius vs gamma for various beta (theoretical only)
gammas_scan = np.linspace(0.5, 20.0, 100)
betas_fixed = [0.2, 0.4, 0.6, 0.8, 1.0]
core_radius_curves = {}

for beta_fix in betas_fixed:
    gc_fix = gamma_c_nl(beta_fix)
    epsilons = np.sqrt(np.maximum(gammas_scan - gc_fix, 0) / gc_fix)
    R_cores = np.where(epsilons > 1e-6, np.pi * np.sqrt(D) / epsilons, np.inf)
    core_radius_curves[f"beta{beta_fix}"] = {
        "gamma_c": float(gc_fix),
        "gammas": gammas_scan.tolist(),
        "R_cores": R_cores.tolist(),
        "validity": "epsilon << 1 only",
    }
    print(f"\n  beta={beta_fix:.1f}, gamma_c={gc_fix:.4f}:")
    for g_target in [1.0, 2.0, 4.0, 8.0]:
        idx = np.argmin(np.abs(gammas_scan - g_target))
        eps = epsilons[idx]
        R = R_cores[idx]
        valid = eps < 0.5
        status = " [THEORETICAL]" if valid else " [INVALID epsilon>>1]"
        if np.isfinite(R):
            print(f"    gamma={g_target:.1f}: epsilon={eps:.3f}, R_core={R:.2f}{status}")

# =====================================================================
# Part F: Pattern Selection (Theoretical)
# =====================================================================

print("\n" + "=" * 70)
print("Part F: Pattern Selection — BCC vs HCP (Theoretical)")
print("=" * 70)

print("""
In 3D isotropic systems, the preferred pattern is determined by the
cross-coupling between modes with |k_j| = k_c.

For the nonlocal KS with constraint-driven nonlinearity, the cross-coupling
coefficient h is determined by the angular structure of the nonlocal kernel:

    h(theta) ~ C(k_c * cos(theta))

For the 26-neighbor stencil, the kernel is nearly isotropic, so h/g ~ 1
for all angles. This favors BODY-CENTERED CUBIC (BCC) patterns, which
maximize the number of modes that can coexist at the critical wavenumber.

The BCC lattice has reciprocal lattice vectors at angles of 60 and 90 deg,
both of which are well-supported by the 26-neighbor stencil.

NOTE: This is a theoretical analysis. C++ pattern data is needed to confirm
whether the actual core arrangement follows BCC ordering.
""")

# Cross-coupling for characteristic angles
theta_vals = [0, np.pi/3, np.pi/2, 2*np.pi/3]
k_c = np.pi / 0.5

print(f"Cross-coupling ratios at k_c = {k_c:.4f} (Nyquist):")
for theta in theta_vals:
    h_ratio = 1.0  # Nearly isotropic for 26-neighbor stencil
    print(f"  theta = {np.rad2deg(theta):6.0f} deg: h/g ~ {h_ratio:.2f}")

print("""
Since h/g ~ 1 for all angles > 0, the pattern selection is degenerate:
BCC and HCP are nearly degenerate. The phi >= 0 constraint breaks this
degeneracy slightly in favor of BCC, which has higher packing density.
""")

# =====================================================================
# Part G: Amplitude Dynamics (Numerical Demonstration)
# =====================================================================

print("\n" + "=" * 70)
print("Part G: Amplitude Dynamics (Numerical Demonstration)")
print("=" * 70)

mu_def = lambda_max_nl(gamma_0, beta_0)
g_eff_def = gamma_c_def * C0_Nyquist

def amplitude_ode(t, A, mu, g):
    return mu * A - g * A**3

initial_amplitudes = [0.001, 0.01, 0.1, 0.5, 1.0]
t_span = [0, 5]
t_eval = np.linspace(0, 5, 200)

print(f"\nSolving dA/dt = {mu_def:.4f}*A - {g_eff_def:.4f}*A^3:")
print(f"  g_eff = gamma_c*C0 = {g_eff_def:.4f} (DEFINED constant)")
print(f"  Steady-state A = sqrt(mu/g_eff) = {epsilon_def:.4f} = epsilon (IDENTITY)")
for A0 in initial_amplitudes:
    sol = solve_ivp(lambda t, y: amplitude_ode(t, y, mu_def, g_eff_def),
                    t_span, [A0], t_eval=t_eval, method='RK45', rtol=1e-8)
    A_final = sol.y[0, -1]
    print(f"  A(0)={A0:.3f} -> A(inf)={A_final:.4f} (theory: {epsilon_def:.4f})")

# =====================================================================
# Part H: Eckhaus Stability (Theoretical)
# =====================================================================

print("\n" + "=" * 70)
print("Part H: Eckhaus Stability Analysis (Theoretical)")
print("=" * 70)

Q_eckhaus = epsilon_def / np.sqrt(3)
print(f"""
The amplitude equation admits plane wave solutions:
    A(X) = A_0 * exp(i*Q*X)  with |A_0|^2 = (mu - xi*Q^2)/g_eff

Eckhaus instability: these solutions are stable only when
    |Q| < Q_Eckhaus = sqrt(mu/(3*xi)) = epsilon/sqrt(3)

For default parameters (epsilon = {epsilon_def:.4f}):
    Q_Eckhaus = {Q_eckhaus:.4f}
    Maximum spatial modulation wavelength: 2*pi/Q_Eckhaus = {2*np.pi/Q_eckhaus:.1f}

NOTE: At epsilon = 3.54, the Eckhaus boundary is in the strongly nonlinear
regime where the amplitude equation is not quantitatively valid.
""")

# =====================================================================
# Save Results
# =====================================================================

output = {
    "theory_version": "4.0",
    "restructuring": (
        "v4.0 (Round 24): Restructured from 'amplitude equation prediction' to "
        "'nonlinear mechanism analysis'. The amplitude equation A_eq = epsilon "
        "is explicitly identified as an ALGEBRAIC IDENTITY (C2), not an "
        "independent prediction. g_eff is DEFINED as gamma_c*C0 = k2_disc+beta, "
        "not computed from mode coupling. Validity regions (epsilon << 1) are "
        "explicitly marked. The constraint-driven saturation mechanism is "
        "presented as a conceptual framework, not a validated theory."
    ),
    "input_parameters": {
        "source": "Phase 2 (dim2_linear_stability)",
        "k2_disc": float(k2_disc),
        "C0_Nyquist": float(C0_Nyquist),
        "gamma_c_formula": f"gamma_c(beta) = ({k2_disc} + beta) / {C0_Nyquist}",
    },
    "operating_point": {
        "gamma": gamma_0, "beta": beta_0,
        "gamma_c": float(gamma_c_def),
        "epsilon": float(epsilon_def),
        "perturbation_theory_valid": False,
        "reason": "epsilon = 3.54 >> 1",
    },
    "constraint_driven_saturation": {
        "mechanism": "phi >= 0 constraint (clipping: phi_new = max(0, phi_new))",
        "nonlinear_source": "Projection operator Pi_{>=0} on linear PDE",
        "effective_landau": "dA/dt = mu*A - g_eff*A^3",
        "mu": "lambda_max = gamma*C0_Nyquist - k2_disc - beta",
        "g_eff_defined": "gamma_c*C0_Nyquist = k2_disc + beta (DEFINED, not computed)",
        "algebraic_identity": "A_eq = sqrt(mu/g_eff) = epsilon (TAUTOLOGY)",
        "identity_proof": (
            "A_eq = sqrt(mu/g_eff) = sqrt(gamma_c*C0*epsilon^2 / (gamma_c*C0)) "
            "= sqrt(epsilon^2) = epsilon. "
            "This follows from the DEFINITION g_eff = gamma_c*C0, not from "
            "any physical calculation of mode coupling coefficients."
        ),
    },
    "parameter_analysis": amplitude_results,
    "core_radius_prediction": {
        "formula": "R_core = pi*sqrt(D/epsilon^2) = pi*sqrt(D*gamma_c/(gamma-gamma_c))",
        "scaling": "R_core ~ 1/sqrt(gamma - gamma_c)",
        "validity": "epsilon << 1 only",
        "default_value": float(amplitude_results[0]["R_core_dimensionless"]),
        "default_grid_cells": float(amplitude_results[0]["R_core_grid_cells"]),
        "note": "At epsilon=3.54, R_core ~ 0.89 is resolution-limited, not physically meaningful",
        "core_radius_curves": core_radius_curves,
    },
    "pattern_selection": {
        "preferred_pattern": "BCC (Body-Centered Cubic)",
        "reason": "Nearly isotropic 26-neighbor stencil -> h/g ~ 1 for all angles",
        "status": "Theoretical, not validated by C++ pattern data",
    },
    "eckhaus_stability": {
        "boundary": "|Q| < epsilon/sqrt(3)",
        "Q_eckhaus_default": float(Q_eckhaus),
        "validity": "epsilon << 1 only",
    },
    "limitations": [
        "Perturbation theory requires epsilon << 1; C++ operates at epsilon = 3.54",
        "Amplitude equation A_eq = epsilon is an ALGEBRAIC IDENTITY (C2)",
        "Constraint-driven saturation is a HYPOTHESIS, not a validated theory",
        "Core radius formula not validated by C++ data",
        "Pattern selection not confirmed by C++ spatial correlation analysis",
    ],
}

with open(os.path.join(SCRIPT_DIR, "dim3_nonlinear_mechanism_report.json"), 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print("Dimension 3 COMPLETE (v4.0 — Restructured: Nonlinear Mechanism Analysis)")
print(f"Report: dim3_nonlinear_mechanism_report.json")
print(f"{'='*70}")

print("""
=== Dimension 3 Key Conclusions (v4.0 — RESTRUCTURED) ===

1. CONSTRAINT-DRIVEN SATURATION: The nonlocal KS is linear; nonlinearity
   comes from phi >= 0 constraint (clipping). This is a conceptual
   framework, not a validated theory.

2. ALGEBRAIC IDENTITY (C2): A_eq = sqrt(mu/g_eff) = epsilon is a TAUTOLOGY.
   g_eff = gamma_c*C0 is DEFINED, not computed from mode coupling.
   Standard weakly nonlinear theory computes g; here g is defined.
   This is a CONSISTENCY CHECK, not an independent prediction.

3. VALIDITY: Perturbation theory requires epsilon << 1.
   C++ operating point: epsilon = 3.54 >> 1.
   Quantitative predictions (R_core, scaling) are NOT valid at this point.

4. CORE RADIUS: R_core = pi*sqrt(D/epsilon^2) is a theoretical formula
   valid only for epsilon << 1. Needs C++ validation.

5. PATTERN SELECTION: BCC favored theoretically; not confirmed by data.
""")