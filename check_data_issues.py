import json
import os

PROJ = 'e:/pytorchFile/YSC_2/Project/Project'

# Check convergence data
print("=" * 60)
print("1. Convergence data check")
print("=" * 60)

files = [
    'multilayer_results_gamma_long_0.4.json',
    'multilayer_results_gamma_long_1.json',
    'multilayer_results_gamma_critical_0.43.json',
]

for fname in files:
    path = os.path.join(PROJ, fname)
    with open(path) as f:
        d = json.load(f)
    ts = d['time_series']
    nc = ts['n_cores']
    t = ts['t']
    mid = len(nc) // 2
    fh = nc[:mid]
    sh = nc[mid:]
    fh_mean = sum(fh)/len(fh)
    sh_mean = sum(sh)/len(sh)
    print(f'{fname}: gamma={d["gamma"]}, beta={d["beta"]}, t_max={t[-1]}, len={len(t)}')
    print(f'  first_half_mean={fh_mean:.4f}, second_half_mean={sh_mean:.4f}')
    print(f'  first 5 nc: {nc[:5]}, last 5 nc: {nc[-5:]}')
    print(f'  min={min(nc)}, max={max(nc)}, mean={sum(nc)/len(nc):.4f}')
    print()

# Check if gamma=0.4 and gamma=1.0 are identical
with open(os.path.join(PROJ, 'multilayer_results_gamma_long_0.4.json')) as f:
    d04 = json.load(f)
with open(os.path.join(PROJ, 'multilayer_results_gamma_long_1.json')) as f:
    d1 = json.load(f)

nc04 = d04['time_series']['n_cores']
nc1 = d1['time_series']['n_cores']
print(f"gamma=0.4 vs gamma=1.0 n_cores identical? {nc04 == nc1}")
print(f"Hash of nc04: {hash(tuple(nc04))}, Hash of nc1: {hash(tuple(nc1))}")

# Check GAMMA_SCALE
print()
print("=" * 60)
print("2. GAMMA_SCALE check")
print("=" * 60)
import common_utils
print(f"common_utils.GAMMA_SCALE = {common_utils.GAMMA_SCALE}")

# Check algorithm_v2_report.json
with open('algorithm_v2_report.json', encoding='utf-8') as f:
    report = json.load(f)
print(f"algorithm_v2_report best_gamma_scale = {report['best_gamma_scale']}")

# Check gamma_scale_calibration
print()
print("Gamma scale calibration results:")
for cal in report['gamma_scale_calibration']:
    print(f"  scale={cal['scale']:.1f}, n_actual={cal['n_actual']}, bias_pct={cal['bias_pct']:.1f}%")

# Check protocol overhead
print()
print("=" * 60)
print("3. Protocol overhead check")
print("=" * 60)
for oh in report['throughput_and_overhead']['protocol_overhead_kbps']:
    print(f"  N={oh['N']}, n_cores={oh['n_cores']}, overhead_kbps={oh['overhead_kbps']:.4f}, pct={oh['pct_of_link_capacity']:.6f}%")

# Manual calculation as in manuscript
print()
print("Manual protocol overhead calculation (as in manuscript):")
for N_val in [1000, 4408]:
    n_cores = 84 if N_val == 1000 else 188
    k = 6
    N_bar = N_val / n_cores
    beacon = n_cores * k * 64
    sync = n_cores * k * N_bar * 64
    intra = n_cores * N_bar * 64
    total_bytes = beacon + sync + intra
    total_bits = total_bytes * 8
    isl_capacity_gbps = 200
    overhead_pct = total_bits / (isl_capacity_gbps * 1e9) * 100
    print(f"  N={N_val}, n_cores={n_cores}:")
    print(f"    beacon={beacon:.0f}, sync={sync:.0f}, intra={intra:.0f}")
    print(f"    total_bytes={total_bytes:.0f}, total_bits={total_bits:.0f}")
    print(f"    overhead_pct={overhead_pct:.6f}%")

# Check PDE bridge
print()
print("=" * 60)
print("4. PDE bridge check")
print("=" * 60)
bridge = report['pde_bridge']
print(f"cosine_similarity={bridge['cosine_similarity']}")
print(f"bridge_status={bridge['bridge_status']}")
print(f"pde_direct: imbalance={bridge['pde_direct']['imbalance']:.2f}, avg_dist={bridge['pde_direct']['avg_dist_km']:.2f}")
print(f"cbdp_v3: imbalance={bridge['cbdp_v3']['imbalance']:.2f}, avg_dist={bridge['cbdp_v3']['avg_dist_km']:.2f}")

# Check the imbalance difference mentioned in CN manuscript
# CN says: "绝对负载不均衡中剩余的13.6%差异（PDE直接：29.48，CBDP：33.49）"
# But report shows: pde_direct imbalance=31.06, cbdp_v3 imbalance=30.20
print(f"CN manuscript claims: PDE direct=29.48, CBDP=33.49, diff=13.6%")
print(f"Actual data: PDE direct={bridge['pde_direct']['imbalance']:.2f}, CBDP={bridge['cbdp_v3']['imbalance']:.2f}")
diff_pct = abs(bridge['pde_direct']['imbalance'] - bridge['cbdp_v3']['imbalance']) / bridge['pde_direct']['imbalance'] * 100
print(f"Actual diff: {diff_pct:.1f}%")