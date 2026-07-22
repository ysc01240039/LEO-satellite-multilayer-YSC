"""
===============================================================================
Dimension 6: Variational Structure & Lyapunov Functional

===============================================================================
SPECIFICATION (v2.1 — Round 24 Updated)
===============================================================================

PURPOSE:
    Derive the variational (gradient-flow) structure of the nonlocal KS
    equation and prove the existence of a Lyapunov functional (free energy)
    that decreases during core formation for AUTONOMOUS systems.

INPUT:
    From Phase 1 (dim1): Nonlocal KS PDE with constraint phi >= 0
    - D = 1.0, gamma, beta, sigma = 1.0
    - Nonlocal operator N[phi] with 26-neighbor stencil

OUTPUT:
    - dim6_variational_report.json
    - Free energy functional F[phi] with three contributions:
        F_diff: diffusion (ordering)
        F_nonlocal: nonlocal interaction (aggregation)
        F_decay: decay (dissipation)
    - Gradient flow proof: d(phi)/dt = -delta_F/delta_phi
    - H-theorem: dF/dt <= 0 for AUTONOMOUS systems (rho = const)
    - Non-autonomous analysis: oscillation when rho = rho(r,t)
    - Connection to constraint-driven saturation

VERIFICATION:
    - Mathematical self-consistency: functional derivative matches PDE RHS
    - H-theorem: dF/dt <= 0 for autonomous systems (mathematically proven)
    - C3 RESOLVED (Round 20): H-theorem only applies to autonomous systems;
      C++ simulation has time-varying source rho(r,t) causing oscillations
    - Oscillation: Fourier spectrum shows deterministic narrowband signal
      (dominant period = 16.2, peak/mean = 97.2)

LIMITATIONS:
    - H-theorem dF/dt <= 0 proven for AUTONOMOUS systems only
    - C++ system is NON-AUTONOMOUS (rho varies with time)
    - The oscillation represents cycling between quasi-stable core configurations
    - Free energy provides a QUALITATIVE framework, not quantitative predictions

DEPENDENCY: None (all parameters are hardcoded from dim1 constants)
STATUS:    Theoretical — mathematically correct for autonomous systems
===============================================================================
"""

import json, sys, io, os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
print("Dimension 6: Variational Structure & Lyapunov Functional")
print("=" * 70)

# =====================================================================
# Part A: Nonlocal KS Equation & Gradient Flow Structure
# =====================================================================

print("\n" + "=" * 70)
print("Part A: Nonlocal KS Equation as a Gradient Flow")
print("=" * 70)

print("""
The C++ simulation evolves the NONLOCAL KS equation:

    ∂φ/∂t = D·∇²φ - γ·N[φ] - β·φ + ρ(r)              ... (1)

where the nonlocal operator is:
    N[φ](r) = Σ_j [φ(r) - φ(r + dr_j)] · K(dr_j)     ... (2)

with K(dr) = G(|dr|)/|dr|, G(r) = exp(-r²/2σ²), and the sum
runs over the 26-neighbor stencil on the 3D grid.

CRITICAL OBSERVATION: N[φ] is a LINEAR operator in φ. Unlike the
LOCAL KS equation (∂φ/∂t = D∇²φ - γ∇·(φ∇φ) - βφ + ρ), the
nonlocal form has no cubic nonlinearity from the chemotaxis term.

The nonlinearity that produces pattern formation (cores) comes from
the NON-NEGATIVITY CONSTRAINT:
    φ(r) ≥ 0  for all r                                 ... (3)

In the C++ simulation, this is enforced by clipping at each step:
    φ_new = max(0, φ_new)                               ... (4)

The PDE (1) with constraint (3) is a LINEAR COMPLEMENTARITY PROBLEM.
""")

# =====================================================================
# Part B: Free Energy Functional (NONLOCAL)
# =====================================================================

print("=" * 70)
print("Part B: Free Energy Functional for the Nonlocal KS Equation")
print("=" * 70)

print("""
The nonlocal KS equation (1) can be written as a gradient flow:

    ∂φ/∂t = -δF_total[φ]/δφ                           ... (5)

where F_total[φ] is the total free energy functional. We construct F_total
such that its functional derivative reproduces the RHS of (1).

The nonlocal operator N[φ]_i = Σ_j (φ_j - φ_i)·K_{ij} = -Σ_j (φ_i - φ_j)·K_{ij}.
In the PDE, the nonlocal term is -γ·N[φ]_i = +γ·Σ_j (φ_i - φ_j)·K_{ij}.

For the gradient flow to hold, we need:
    -δF_nl/δφ_i = -γ·N[φ]_i = +γ·Σ_j (φ_i - φ_j)·K_{ij}
    → δF_nl/δφ_i = -γ·Σ_j (φ_i - φ_j)·K_{ij}

This is satisfied by:
    F_nl[φ] = -(γ/2) Σ_{i,j} [φ_i - φ_j]² · K_{ij}    ... (6)

where K_{ij} = G(|r_i - r_j|)/|r_i - r_j| for the 26-neighbor stencil.

NOTE: The negative sign means F_nl is MINIMIZED by non-uniform φ (aggregation).
This reflects the anti-diffusive nature of chemotaxis: it creates order (cores)
by lowering the free energy through the nonlocal interaction.

The COMPLETE free energy functional is:

    F[φ] = (D/2) Σ_i |(∇φ)_i|² · ΔV                      [diffusion, ≥0]
         + (β/2) Σ_i φ_i² · ΔV                             [decay, ≥0]
         - Σ_i ρ_i · φ_i · ΔV                              [source driving]
         - (γ/2) Σ_{i,j} [φ_i - φ_j]² · K_{ij} · ΔV²     [nonlocal, ≤0]

where ΔV = dx³ is the grid cell volume.

Key properties of F[φ]:
  1. The diffusion and decay terms are CONVEX (penalize non-uniformity)
  2. The nonlocal term is CONCAVE (rewards non-uniformity, promotes aggregation)
  3. F_total is NON-CONVEX → multiple local minima → multiple stable core patterns
  4. On the feasible set {φ ≥ 0}, F is bounded below (the φ≥0 constraint and
     finite domain prevent unlimited growth of the nonlocal term)
  5. The competition between the convex (diffusion) and concave (chemotaxis)
     terms determines the core pattern: large γ makes the nonlocal term dominate,
     producing many sharp cores; small γ makes diffusion dominate, producing
     smooth, uniform solutions.

The functional derivative is:
    δF/δφ_i = -D(∇²φ)_i + β·φ_i - ρ_i - γ Σ_j [φ_i - φ_j]·K_{ij}  ... (7)

Therefore:
    ∂φ/∂t = -δF/δφ = D∇²φ + γ Σ_j [φ_i - φ_j]·K_{ij} - βφ + ρ
           = D∇²φ - γ·N[φ] - βφ + ρ  ✓
""")

# =====================================================================
# Part C: Constrained Minimization & Core Formation
# =====================================================================

print("=" * 70)
print("Part C: Constrained Free Energy Minimization")
print("=" * 70)

print("""
The C++ simulation enforces φ ≥ 0 via clipping. This is equivalent to
the CONSTRAINED minimization problem:

    min_{φ ≥ 0} F[φ]                                    ... (8)

where F[φ] is the NON-CONVEX free energy functional from Part B.

The KKT optimality conditions for (8) are:

    δF/δφ_i - λ_i = 0                                   ... (9a)
    λ_i ≥ 0, φ_i ≥ 0, λ_i · φ_i = 0                    ... (9b)

where λ_i are Lagrange multipliers for the non-negativity constraint.

In regions where φ_i > 0 (INSIDE CORES):
    λ_i = 0 → δF/δφ_i = 0 → D∇²φ - γN[φ] - βφ + ρ = 0

In regions where φ_i = 0 (OUTSIDE CORES):
    λ_i = δF/δφ_i ≥ 0 → the unconstrained gradient would drive φ negative

The NON-CONVEXITY of F[φ] (from the negative nonlocal term) is the mathematical
origin of MULTIPLE STABLE CORE PATTERNS:
  - The diffusion+decay terms (convex) favor smooth, uniform φ
  - The nonlocal chemotaxis term (concave) favors sharp, localized φ
  - The φ ≥ 0 constraint prevents the solution from "overshooting" negative
  - The competition creates MULTIPLE LOCAL MINIMA → degenerate core patterns
  - Different initial conditions converge to different local minima
  - The number of stable cores grows with γ (stronger nonlocal driving)

This is fundamentally different from the LOCAL KS equation, where the
nonlinearity is intrinsic (cubic) and the free energy is quartic.
""")

# =====================================================================
# Part D: H-Theorem for the Constrained Gradient Flow
# =====================================================================

print("=" * 70)
print("Part D: H-Theorem for the Projected Gradient Flow")
print("=" * 70)

print("""
The C++ time stepping is the PROJECTED GRADIENT METHOD:

    φ^{n+1} = Π_{≥0}(φ^n - dt·δF/δφ)                   ... (11)

where Π_{≥0}(v) = max(0, v) is the projection onto the feasible set.

Theorem (H-Theorem for Projected Gradient Flow):
  For dt < 2/L where L is the Lipschitz constant of δF/δφ, the free
  energy decreases monotonically:
    F(φ^{n+1}) ≤ F(φ^n)                                 ... (12)

  Equality holds iff φ^{n+1} = φ^n (steady state).

Proof sketch:
  By the projection property:
    ‖φ^{n+1} - (φ^n - dt·δF/δφ)‖² ≤ ‖φ^n - (φ^n - dt·δF/δφ)‖²
    = dt²·‖δF/δφ‖²

  Expanding the LHS and rearranging:
    ⟨φ^{n+1} - φ^n, δF/δφ⟩ ≤ -‖φ^{n+1} - φ^n‖²/(2dt) ≤ 0  ... (13)

  By the quadratic upper bound (L-smoothness of F):
    F(φ^{n+1}) - F(φ^n) ≤ ⟨δF/δφ, φ^{n+1} - φ^n⟩ + (L/2)‖φ^{n+1} - φ^n‖²
                        ≤ -(1/dt - L/2)·‖φ^{n+1} - φ^n‖²  ... (14)

  For dt < 2/L: F(φ^{n+1}) - F(φ^n) ≤ 0.  ∎

Lipschitz constant estimate for the discrete system:
  L = ‖δ²F/δφ²‖ ≤ 2D·6/dx² + 2β + 2γ·Σ_j K_j
  For dx=0.5, D=1, β=0.6, γ=6:
    L ≤ 2·6/0.25 + 1.2 + 2·6·(6·1.88 + 12·1.10 + 8·0.79)
    = 48 + 1.2 + 12·(11.28 + 13.22 + 6.34)
    = 49.2 + 12·30.84 ≈ 419

  dt < 2/419 ≈ 0.0048. The C++ uses dt = 0.004 < 0.0048, so the
  projected gradient flow guarantees monotonic energy decrease at
  current parameters.

Consequences:
  1. The system converges to a steady state (no oscillations)
  2. All attractors are constrained critical points of F
  3. The long-time behavior is completely determined by F[φ] and φ ≥ 0
  4. Adding stochastic noise → constrained Gibbs distribution
""")

# =====================================================================
# Part D2: H-Theorem Limitation — Time-Varying Source Term
# =====================================================================

print("=" * 70)
print("Part D2: CRITICAL LIMITATION — Time-Varying Source Term ρ(r,t)")
print("=" * 70)

print("""
IMPORTANT CAVEAT: The H-theorem proof in Part D assumes the system is AUTONOMOUS
— i.e., the source term ρ(r) is time-independent. The C++ simulation, however,
has a TIME-VARYING source term ρ(r,t) due to:

  1. Satellite orbital motion: positions updated every step (dt=0.004)
     → ρ(r,t) changes at every time step as satellites move

  2. Payload (cap, task) updates: every 10 steps (dt=0.04)
     → ρ(r,t) = 1.0 + cap(t) + task(t) changes stochastically

  3. Sparse point-source distribution: only ~1.6% of grid cells have non-zero ρ
     → ρ(r,t) is highly inhomogeneous in space

For a NON-AUTONOMOUS system with time-varying ρ(r,t):

  ∂φ/∂t = -δF[φ; ρ(t)]/δφ

the free energy F[φ; ρ(t)] depends explicitly on time through ρ(t). The total
time derivative becomes:

  dF/dt = ∂F/∂t + ⟨δF/δφ, ∂φ/∂t⟩
        = ⟨∂ρ/∂t, -φ⟩ - ‖δF/δφ‖²        (for unconstrained gradient flow)

The first term ⟨∂ρ/∂t, -φ⟩ represents the power injected by the time-varying
source. This term can be POSITIVE, meaning the source can pump energy into the
system, preventing convergence to a steady state.

When the projection Π_{≥0} is active, the energy balance becomes even more
complex because the projection may introduce additional non-monotonicity.

C++ SIMULATION EVIDENCE (Round 19 H1 analysis):

  Fourier spectrum of core count time series (1001 samples, 2h physical time):
  - Dominant frequency: 0.0617 (period = 16.2 dimensionless time ≈ 16.2 s)
  - Peak/mean ratio: 97.2 → narrow-band DETERMINISTIC oscillation
  - Spectrum type: narrow-band periodic (NOT broadband noise)
  - Lag-1 autocorrelation: -0.172 (negative → oscillatory signature)
  - PACF(1) = -0.172, PACF(2) = 0.110 → non-AR(1), persistent cycle

  The oscillation is NOT decaying: CV = 22.5%, amplitude = 122% of mean.
  The dominant period (16.2) is much longer than the source update interval
  (0.04) and the phi response time (1/β ≈ 1.7), but much shorter than the
  diffusion time (L²/D = 100). This suggests it is a PDE-intrinsic relaxation
  oscillation from core competition/merging dynamics.

CONCLUSION: The H-theorem (dF/dt ≤ 0 → convergence to steady state) is ONLY
valid for the AUTONOMOUS system (constant ρ). For the actual C++ simulation
with time-varying ρ(r,t), the system does NOT converge to a steady state but
exhibits persistent, deterministic oscillations. The H-theorem provides the
correct gradient-flow structure for the instantaneous free energy, but the
time-dependence of ρ prevents the system from reaching a fixed point.

This does NOT invalidate the gradient-flow formulation itself — it only means
that the long-time behavior is governed by the interplay between the gradient
descent dynamics and the time-varying driving force ρ(r,t).
""")

# =====================================================================
# Part E: Nonlocal Parameters & Landau Phenomenology
# =====================================================================

D, sigma_val, S0 = 1.0, 1.0, 1.0
gamma_0, beta_0 = 6.0, 0.6

# Nonlocal parameters from the discrete stencil analysis
C0_continuum = 30.1556  # stencil C0 (continuum limit ΣK_j) — unused, kept for reference
C0_Nyquist = 37.38      # |C(k_Nyquist)| for the nonlocal operator at discrete Nyquist
k2_disc = 16.0          # discrete Laplacian at the most unstable mode
gamma_c_nl = (k2_disc + beta_0) / C0_Nyquist  # nonlocal critical gamma = (16+β)/37.38

print("=" * 70)
print("Part E: Nonlocal Dispersion & Effective Parameters")
print("=" * 70)

print(f"""
Nonlocal KS parameters (from discrete 26-neighbor stencil, dx=0.5):
  C0_Nyquist = |C(k_max)| = {C0_Nyquist:.4f}
  k²_disc(k_max) = {k2_disc:.1f}
  Maximum growth rate: λ_max = -k²_disc + γ·C0_Nyquist - β
  Critical gamma: γ_c = (k²_disc + β) / C0_Nyquist = {gamma_c_nl:.4f}

For default parameters (γ={gamma_0}, β={beta_0}):
  λ_max = -{k2_disc} + {gamma_0}·{C0_Nyquist:.4f} - {beta_0} = {-k2_disc + gamma_0*C0_Nyquist - beta_0:.2f}
  γ/γ_c = {gamma_0/gamma_c_nl:.1f} (deep in the ordered phase)

The nonlocal KS is LINEAR, so there is no intrinsic cubic saturation.
Pattern formation is driven by the φ ≥ 0 constraint. The "effective"
nonlinearity can be characterized by a Landau-like phenomenology where
the constraint acts as an infinite potential barrier at φ = 0.

Effective Landau free energy for the mode amplitude A:
  F_eff(A) = -μ·A²/2 + g_eff·A⁴/4

where μ = λ_max = γ·C0 - k²_disc - β is the linear growth rate,
and g_eff is an EFFECTIVE cubic coefficient from the constraint.

The constraint-induced saturation gives:
  A_eq = sqrt(μ / g_eff) ∝ sqrt(γ - γ_c)

consistent with the mean-field order parameter exponent β̃ = 1/2.
""")

# =====================================================================
# Part F: Effective Landau Free Energy
# =====================================================================

print("=" * 70)
print("Part F: Landau Free Energy Near Onset")
print("=" * 70)

# Use nonlocal dispersion to compute effective parameters
phi0 = S0 / beta_0
mu_nl = gamma_0 * C0_Nyquist - k2_disc - beta_0  # nonlocal growth rate at Nyquist
epsilon_nl = np.sqrt(max((gamma_0 - gamma_c_nl) / gamma_c_nl, 0))

# Effective g from constraint: the constraint φ ≥ 0 creates an effective
# cubic nonlinearity dA/dt = μA - g_eff·A³ with g_eff = γ_c·C0_Nyquist
# = k²_disc + β (constant), giving A_eq = sqrt(μ/g_eff) = ε
g_eff = gamma_c_nl * C0_Nyquist  # = k²_disc + beta_0 = constant

def free_energy_landau(A, mu, g):
    """Landau free energy: F = F0 - (mu/2)*A^2 + (g/4)*A^4"""
    return -mu * A**2 / 2 + g * A**4 / 4

A_values = np.linspace(-2, 2, 200)
F_landau = free_energy_landau(A_values, mu_nl, g_eff)

A_min_val = np.sqrt(max(mu_nl / g_eff, 0))
F_min_val = free_energy_landau(A_min_val, mu_nl, g_eff)

print(f"\nEffective Landau parameters (nonlocal):")
print(f"  μ = λ_max = {mu_nl:.4f}")
print(f"  g_eff = γ_c·C0 = {g_eff:.4f} (constant, = k²_disc + β = {k2_disc + beta_0:.1f})")
print(f"  ε = sqrt((γ-γ_c)/γ_c) = {epsilon_nl:.4f}")
print(f"  Amplitude at minimum: A = ±{A_min_val:.4f}")
print(f"  Free energy at minima: F = {F_min_val:.4f}")
print(f"  Free energy at A=0 (uniform): F = 0")
print(f"  Energy barrier: ΔF = {-F_min_val:.4f}")

# =====================================================================
# Part G: Chemical Potential & Thermodynamic Analogy
# =====================================================================

print("\n" + "=" * 70)
print("Part G: Chemical Potential & Thermodynamic Analogy")
print("=" * 70)

print(f"""
The variational derivative defines the chemical potential:

    μ_chem(r) = δF/δφ = -D∇²φ + βφ - ρ + γN[φ]           ... (15)

The constrained gradient flow is:
    ∂φ/∂t = Π_{{≥0}}(-μ_chem)                              ... (16)

which is a constrained continuity equation:
    ∂φ/∂t = -∇·J   with J = -∇μ_chem, subject to φ ≥ 0

Thermodynamic analogy:
  - φ(r)       ↔ particle density / communication intensity
  - μ_chem     ↔ chemical potential
  - F[φ]       ↔ Helmholtz free energy
  - φ ≥ 0      ↔ hard-core exclusion (infinite repulsion at φ=0)
  - Cores      ↔ phase-separated droplets (constrained by φ≥0)
  - D|∇φ|²     ↔ surface tension / gradient penalty
  - βφ²        ↔ external potential / harmonic trap
  - γN[φ]      ↔ nonlocal attraction (difference-smoothing)

This MFGT (mean-field gradient theory) analogy provides:
  1. A rigorous framework for equilibrium analysis
  2. A natural definition of "effective temperature" T_eff
  3. Connection to obstacle problems and free boundary theory
  4. The constraint φ ≥ 0 as the physical origin of core sharpness
""")

# =====================================================================
# Part H: Effective Temperature & Fluctuation-Dissipation
# =====================================================================

print("=" * 70)
print("Part H: Effective Temperature & Stochastic Extension")
print("=" * 70)

T_eff = D / (2 * beta_0)

print(f"""
Adding stochastic fluctuations (e.g., packet arrival randomness):

    ∂φ/∂t = Π_{{≥0}}(-δF/δφ + √(2·T_eff)·η(r,t))          ... (17)

where η is spatiotemporal white noise and T_eff is the effective
temperature determined by:
    T_eff = (packet_variance)*D / (2*beta) ≈ D/(2β) = {T_eff:.4f}

For our parameters (D=1, β=0.6):
    T_eff = {T_eff:.4f}

The fluctuation-dissipation theorem (FDT) holds for the unconstrained
system (linear regime). The constrained equilibrium distribution is:

    P[φ] ∝ exp(-F[φ]/T_eff) · 1_{{φ ≥ 0}}                  ... (18)

This means:
  1. The system obeys detailed balance at equilibrium
  2. The steady-state distribution is the truncated Gibbs distribution
  3. Core formation ≡ constrained free energy minimization
  4. The most probable state is the constrained F-minimizer
""")

# =====================================================================
# Part I: Multi-Stability & Pattern Catalog
# =====================================================================

print("\n" + "=" * 70)
print("Part I: Multi-Stability from Constraint Geometry")
print("=" * 70)

print("""
For γ < γ_c, F[φ] is CONVEX, and the unconstrained problem has a unique global
minimum. For γ > γ_c, F[φ] becomes NON-CONVEX due to the negative nonlocal term
-(γ/2)Σ(φ-φ')²K. The constraint φ ≥ 0 creates a non-convex feasible
set boundary, leading to multiple local minima in the constrained problem.

Pattern Enumeration (for cubic domain with periodic BCs):

  1. Homogeneous: φ = φ₀ = S/β everywhere (feasible, but may not be
     a constrained minimum if the unconstrained minimum has φ < 0 somewhere)

  2. Single core: φ > 0 in one connected region, φ = 0 elsewhere

  3. Multiple cores: φ > 0 in several disconnected regions
     The number of cores depends on:
     (a) Source distribution ρ(r) (satellite positions)
     (b) Nonlocal kernel width σ (controls core size)
     (c) γ/γ_c ratio (controls depth of ordering)

Stability criterion:
  For a core pattern to be a constrained local minimum:
    (a) δF/δφ = 0 inside each core (φ > 0)
    (b) δF/δφ ≥ 0 outside cores (φ = 0, constraint active)
    (c) The second variation δ²F is positive definite on the tangent cone

The number of stable core patterns grows with domain size and γ/γ_c.
This is analogous to the multiplicity of solutions in obstacle problems.
""")

# =====================================================================
# Part J: Comparison with Local KS
# =====================================================================

print("=" * 70)
print("Part J: Local vs Nonlocal KS — Why the Distinction Matters")
print("=" * 70)

gamma_c_local = beta_0 * (1 + np.sqrt(beta_0))**2

print(f"""
The LOCAL KS equation (∂φ/∂t = D∇²φ - γ∇·(φ∇φ) - βφ + ρ) has:

  F_local[φ] = ∫[(D/2)|∇φ|² - (γ/6)φ³ + (β/2)φ² - ρφ] dr

  - The -(γ/6)φ³ term is CUBIC and provides INTRINSIC nonlinear saturation
  - The free energy is NON-CONVEX (due to the negative cubic term)
  - Critical gamma: γ_c = β(1+√β)² = {gamma_c_local:.4f}

The NONLOCAL KS equation (C++ simulation) has:

  F_nonlocal[φ] = (D/2)|∇φ|² + (β/2)φ² - ρφ - (γ/2)∫∫[φ-φ']²K

  - The -(γ/2)∫∫[φ-φ']²K term favors heterogeneity (γ > 0 promotes pattern formation)
  - For γ < γ_c: Hessian is positive definite → F is CONVEX (uniform phase stable)
  - For γ > γ_c: Hessian has negative eigenvalue → F is NON-CONVEX (ordered phase)
  - Nonlinearity comes from the CONSTRAINT φ ≥ 0
  - Critical gamma: γ_c = (k²_disc + β)/C0 = {gamma_c_nl:.4f}

KEY DIFFERENCES:
  1. Local KS γ_c = {gamma_c_local:.2f} vs Nonlocal KS γ_c = {gamma_c_nl:.2f}
     → Nonlocal onset is ~{gamma_c_local/gamma_c_nl:.1f}x lower (more sensitive)

  2. Local KS saturation is intrinsic (cubic term); Nonlocal KS saturation
     is constraint-driven (φ ≥ 0 boundary)

  3. Local KS free energy is non-convex → multiple unconstrained minima
     Nonlocal KS free energy is convex for γ < γ_c, non-convex for γ > γ_c
     → constraint-driven multi-stability in the ordered phase
     multiple minima only from the constraint

  4. Both give β̃ = 1/2 (mean-field order parameter), but for different
     physical reasons:
     - Local: from cubic term in Landau expansion
     - Nonlocal: from constraint saturation at φ = 0
""")

# =====================================================================
# Part K: Core Interface Profile (Obstacle Problem)
# =====================================================================

print("=" * 70)
print("Part K: Core Interface Profile (Free Boundary Problem)")
print("=" * 70)

# Interface width from the constrained minimization
# At the core boundary, φ drops from φ_core to 0 over width w
# The interface profile minimizes F_1D subject to φ ≥ 0
# This is a classical obstacle problem

# Effective interface width from dimensional analysis
# Balance D(dφ/dx)² ~ βφ² at the interface
w_interface_dim = np.sqrt(D / beta_0)
w_interface_grid = w_interface_dim / 0.5

print(f"""
The interface between a core (φ > 0) and the background (φ = 0)
is a FREE BOUNDARY determined by the obstacle condition.

In the 1D approximation, the interface profile satisfies:
    D·d²φ/dx² + γ·N_eff[φ] - βφ + ρ = 0,  φ > 0
    φ = 0 at the free boundary x = x*
    dφ/dx = 0 at x = x* (smooth contact with obstacle)

Interface width (dimensional analysis):
    w ≈ √(D/β) = {w_interface_dim:.3f} dimensionless units
    = {w_interface_grid:.2f} grid cells (dx = 0.5)

This is SHARPER than the local KS interface width, which is:
    w_local ≈ √(D/|β - γφ₀|) → diverges near γ_c

In the nonlocal KS, the interface width is INDEPENDENT of γ,
determined solely by D and β. This is a testable prediction:
  - Measure interface width from C++ simulation at different γ
  - If width ≈ {w_interface_dim:.3f} ± 10%, the obstacle model is confirmed
""")

# =====================================================================
# Save Results
# =====================================================================

output = {
    "theory_version": "6.1",
    "correction": "Fixed free energy functional for NONLOCAL KS equation. "
                  "Previous version (6.0) incorrectly used local KS free energy "
                  "with -(γ/6)φ³ term. The nonlocal KS has a QUADRATIC free energy "
                  "with nonlinearity from the φ ≥ 0 constraint.",
    "dependencies": ["dim1_theory_report.json"],
    "gradient_flow": {
        "form": "∂φ/∂t = -δF/δφ, with φ ≥ 0 constraint",
        "consequence": "F monotonically decreases for dt < 2/L",
        "projection": "φ^{n+1} = max(0, φ^n - dt·δF/δφ)",
    },
    "free_energy_functional": {
        "expression": (
            "F[φ] = (D/2)∫|∇φ|²dr + (β/2)∫φ²dr - ∫ρφ dr "
            "+ (γ/2)∫∫[φ(r)-φ(r')]²K(r,r')drdr'"
        ),
        "terms": {
            "diffusion": "(D/2)|∇φ|²",
            "decay": "(β/2)φ²",
            "source": "-ρφ",
            "nonlocal_interaction": "-(γ/2)∫∫[φ(r)-φ(r')]²K(r,r')drdr'",
        },
        "properties": [
            "quadratic (all terms ≤ order 2 in φ)",
            "convex for γ < γ_c (positive definite Hessian), non-convex for γ > γ_c",
            "bounded below",
            "coercive",
        ],
        "key_insight": (
            "For γ < γ_c, F[φ] is convex with unique minimum (uniform phase). "
            "For γ > γ_c, the nonlocal term -(γ/2)Σ(φ-φ')²K makes F[φ] non-convex, "
            "creating multiple local minima that correspond to distinct core patterns. "
            "The constraint φ ≥ 0 further restricts the feasible set."
        ),
    },
    "nonlocal_parameters": {
        "C0_Nyquist": float(C0_Nyquist),
        "C0_continuum": float(C0_continuum),
        "k2_disc": float(k2_disc),
        "gamma_c_nonlocal": float(gamma_c_nl),
        "gamma_c_local": float(gamma_c_local),
        "lambda_max": float(mu_nl),
        "epsilon": float(epsilon_nl),
    },
    "landau_free_energy": {
        "form_near_onset": "F(A) = -(μ/2)A² + (g_eff/4)A⁴",
        "mu": float(mu_nl),
        "g_eff": float(g_eff),
        "g_eff_origin": "constraint saturation: A ≤ φ₀, g_eff = μ/φ₀²",
        "minima_at": f"A = ±{float(A_min_val):.4f}",
        "barrier_height": float(round(-F_min_val, 4)),
    },
    "chemical_potential": {
        "expression": "μ_chem = δF/δφ = -D∇²φ + βφ - ρ + γN[φ]",
        "constrained_flow": "∂φ/∂t = Π_{≥0}(-μ_chem)",
    },
    "thermodynamic_analogy": {
        "phi": "particle density / communication intensity",
        "mu_chem": "chemical potential",
        "F": "Helmholtz free energy",
        "phi_ge_0": "hard-core exclusion (infinite barrier at φ=0)",
        "cores": "constrained free energy minima (obstacle regions)",
        "D|∇φ|²": "surface tension / gradient penalty",
        "beta φ²": "harmonic trap",
        "gamma N[φ]": "nonlocal difference-smoothing",
    },
    "effective_temperature": {
        "T_eff": float(T_eff),
        "formula": "T_eff = D/(2*beta)",
        "FDT_holds": True,
        "equilibrium_distribution": "P[φ] ∝ exp(-F[φ]/T_eff) · 1_{φ ≥ 0}",
    },
    "interface_analysis": {
        "type": "free boundary (obstacle problem)",
        "width_formula": "w = sqrt(D/beta)",
        "width_dimensionless": float(w_interface_dim),
        "width_grid_cells": float(w_interface_grid),
        "gamma_independent": True,
        "testable_prediction": "Interface width should be constant across γ values",
    },
    "h_theorem": {
        "statement": "F(φ^{n+1}) ≤ F(φ^n) for dt < 2/L (AUTONOMOUS system only)",
        "condition": f"dt < 2/L ≈ {2/419:.4f} (C++ dt=0.004 satisfies this)",
        "consequence": "All attractors are constrained critical points (for constant ρ)",
        "proof_method": "Projection inequality + L-smoothness bound",
        "limitation": {
            "issue": "H-theorem assumes AUTONOMOUS system (time-independent ρ)",
            "c++_reality": "ρ(r,t) is time-varying due to satellite motion + payload updates",
            "observed_behavior": "Persistent deterministic oscillation (CV=22.5%, dominant period=16.2)",
            "dF/dt_nonautonomous": "dF/dt = ⟨∂ρ/∂t, -φ⟩ - ‖δF/δφ‖², first term can be positive",
            "conclusion": "H-theorem valid for instantaneous gradient structure but does NOT guarantee convergence for non-autonomous system"
        },
    },
    "local_vs_nonlocal_comparison": {
        "local_ks": {
            "free_energy_type": "non-convex (cubic term)",
            "nonlinearity_origin": "intrinsic (γ∇·(φ∇φ))",
            "gamma_c": float(gamma_c_local),
            "interface_width": "γ-dependent, diverges at γ_c",
        },
        "nonlocal_ks": {
            "free_energy_type": "convex for γ < γ_c, non-convex for γ > γ_c",
            "nonlinearity_origin": "constraint (φ ≥ 0) + nonlocal attraction",
            "gamma_c": float(gamma_c_nl),
            "interface_width": "γ-independent, w = sqrt(D/β)",
        },
    },
    "key_insight": (
        "The nonlocal KS free energy is convex for γ < γ_c (uniform phase) "
        "and non-convex for γ > γ_c (ordered phase). The non-convexity arises "
        "from the nonlocal attraction term -(γ/2)Σ(φ-φ')²K, which favors "
        "spatial heterogeneity. The constraint φ ≥ 0 further restricts the "
        "feasible set, creating an obstacle problem structure. This is "
        "fundamentally different from the local KS, where the cubic "
        "chemotaxis term provides intrinsic nonlinear saturation."
    ),
}

with open(os.path.join(SCRIPT_DIR, "dim6_variational_report.json"), 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print("Dimension 6 COMPLETE. Report: dim6_variational_report.json")
print(f"{'='*70}")

print("""
=== Dimension 6 Key Conclusions (v6.1 — CORRECTED) ===

1. CORRECTION: The nonlocal KS equation has a QUADRATIC free energy:
   F[φ] = (D/2)∫|∇φ|² + (β/2)∫φ² - ∫ρφ + (γ/2)∫∫[φ-φ']²K
   This replaces the previous INCORRECT form with -(γ/6)φ³ term,
   which was for the local KS, not the nonlocal KS.

2. The nonlocal operator N[φ] is LINEAR. The nonlinearity that produces
   core patterns comes from the φ ≥ 0 CONSTRAINT (clipping in C++).

3. H-Theorem analog: dF/dt ≤ 0 for the projected gradient flow,
   provided dt < 2/L ≈ 0.0048. The C++ dt=0.004 satisfies this condition.
   HOWEVER, this proof assumes an AUTONOMOUS system (time-independent ρ).
   The C++ simulation has time-varying ρ(r,t) (satellite motion + payload
   updates), and exhibits persistent deterministic oscillations (CV=22.5%,
   dominant period=16.2). The H-theorem provides the correct gradient-flow
   structure for the instantaneous free energy, but time-varying ρ prevents
   convergence to a steady state. See Part D2 for detailed analysis.

4. Core formation = CONSTRAINED free energy minimization (obstacle problem).
   For γ > γ_c, the free energy is non-convex, creating multiple local minima
   corresponding to distinct core patterns. The φ ≥ 0 constraint further restricts
   the feasible set.

5. Interface width w = sqrt(D/β) ≈ 1.29 is INDEPENDENT of γ (testable prediction).

6. The nonlocal KS gives β̃ = 1.0 (ε convention), matching β̃ = 1/2 when
   expressed in the conventional t = ε² convention. The constraint-driven
   saturation produces an effective cubic nonlinearity ∂_tA = μA - g_eff·A³
   with g_eff = γ_c·C0 = constant, yielding m = sqrt(μ/g_eff) = ε.

7. The nonlocal critical gamma γ_c = (16+β)/37.38 ≈ 0.444 is ~4.3x lower
   than the local KS γ_c = 1.89, making the nonlocal system more sensitive
   to pattern formation.
""")