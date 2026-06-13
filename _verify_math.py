"""
Complete math derivation verification for nonlocal PDE.
Checks every step from stencil to final formulas.
"""
import numpy as np

dx = 0.5
sigma = 1.0

# === Step 1: Build stencil ===
stencil = []
for sx in [-1, 0, 1]:
    for sy in [-1, 0, 1]:
        for sz in [-1, 0, 1]:
            if sx == 0 and sy == 0 and sz == 0:
                continue
            dr = np.sqrt(sx*sx + sy*sy + sz*sz) * dx
            w = np.exp(-dr*dr / (2*sigma*sigma)) / dr
            stencil.append((sx, sy, sz, w))

C0 = sum(w for (_,_,_,w) in stencil)
print(f"Step 1: C0 = {C0:.4f} (should be 30.1556)")

# === Step 2: k2_disc at Nyquist ===
k = np.pi / dx
cos_sum = np.cos(k*dx) + np.cos(0) + np.cos(0)
k2_disc_max = 2.0 * (3.0 - cos_sum) / (dx*dx)
print(f"Step 2: k2_disc_max = {k2_disc_max:.4f} (should be 16.0)")

# === Step 3: C(k_max) at Nyquist ===
C_k_max = 0.0
for (sx, sy, sz, w) in stencil:
    phase = k * sx * dx
    C_k_max += (np.cos(phase) - 1.0) * w
print(f"Step 3: C(k_max) = {C_k_max:.4f} (should be -37.38)")

# === Step 4: Analytical gamma_c formula ===
print(f"\nStep 4: Analytical gamma_c(beta) = (k2_disc_max + beta) / |C(k_max)|")
print(f"  gamma_c(0)     = ({k2_disc_max:.4f} + 0) / {abs(C_k_max):.4f} = {k2_disc_max/abs(C_k_max):.4f}")
print(f"  gamma_c(0.6)   = ({k2_disc_max:.4f} + 0.6) / {abs(C_k_max):.4f} = {(k2_disc_max+0.6)/abs(C_k_max):.4f}")
print(f"  gamma_c(2.0)   = ({k2_disc_max:.4f} + 2.0) / {abs(C_k_max):.4f} = {(k2_disc_max+2.0)/abs(C_k_max):.4f}")

# Compare with numerical
print(f"\n  Numerical: gamma_c(0.6) = 0.4441")
print(f"  Analytical: gamma_c(0.6) = {(k2_disc_max+0.6)/abs(C_k_max):.4f}")
print(f"  Match: {abs((k2_disc_max+0.6)/abs(C_k_max) - 0.4441) < 0.001}")

# === Step 5: Old formula error ===
print(f"\nStep 5: Old formula gamma_c(0.6) = beta/C0 = 0.6/{C0:.4f} = {0.6/C0:.4f}")
print(f"  Error: {100*(0.6/C0 - 0.4441)/0.4441:.1f}%")

# === Step 6: Verify why C(k_max) != -C0 ===
# Count neighbors that contribute at k = (pi/dx, 0, 0)
contributing = 0
non_contributing = 0
C_contrib = 0.0
C_non = 0.0
for (sx, sy, sz, w) in stencil:
    if sx != 0:  # only neighbors with non-zero x component contribute
        contributing += 1
        C_contrib += w
    else:
        non_contributing += 1
        C_non += w
print(f"\nStep 6: At k=(pi/dx, 0, 0):")
print(f"  Contributing neighbors (sx!=0): {contributing} with sum w = {C_contrib:.4f}")
print(f"  Non-contributing (sx=0): {non_contributing} with sum w = {C_non:.4f}")
print(f"  C(k_max) = -2*C_contrib = -2*{C_contrib:.4f} = {-2*C_contrib:.4f}")
print(f"  C0 = C_contrib + C_non = {C_contrib + C_non:.4f}")
print(f"  |C(k_max)|/C0 = {2*C_contrib/C0:.4f}")

# === Step 7: Saturation formula self-consistency ===
n_baseline = 91.58676435787842
n_grid_max = 123.09121441194068
gamma_char = 0.5728874345930506
print(f"\nStep 7: Saturation formula")
print(f"  n(0) = {n_baseline + (n_grid_max-n_baseline)*(1-np.exp(0)):.2f} (should = {n_baseline:.2f})")
print(f"  n(inf) = {n_baseline + (n_grid_max-n_baseline)*(1-np.exp(-100)):.2f} (should = {n_grid_max:.2f})")
print(f"  n(gamma_c) = {n_baseline + (n_grid_max-n_baseline)*(1-np.exp(-0.4441/gamma_char)):.2f}")
print(f"  n_baseline + (n_grid_max - n_baseline) = {n_baseline + n_grid_max - n_baseline:.2f} = n_grid_max: {abs(n_baseline + n_grid_max - n_baseline - n_grid_max) < 1e-10}")

print("\n=== ALL VERIFICATION COMPLETE ===")