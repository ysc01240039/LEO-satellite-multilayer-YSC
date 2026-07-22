#!/usr/bin/env python3
"""
===============================================================================
Real Orbit Data Validation

Validates all 6 benchmark algorithms (Greedy, RoundRobin, Nearest-3,
CBDP v2, CBDP v3, ShortestPath) on real satellite orbit data from
parquet files, and compares with Fibonacci sphere network results.

Output: real_orbit_report.json
===============================================================================
"""

import json
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import shared utilities from common module
from common_utils import (
    GAMMA_SCALE, n_baseline, n_grid_max, gamma_char, REF_N,
    NumpyEncoder,
    predict_cores, required_gamma_for_core_fraction,
    generate_network, latlon_to_cart,
    benchmark_greedy, benchmark_roundrobin, benchmark_nearest3, benchmark_shortest_path,
    benchmark_cbdp, benchmark_cbdp_v3,
)

# ================================================================
# Section 4: CBDP Algorithms — imported from common_utils
# (benchmark_cbdp, benchmark_cbdp_v3)
# ================================================================

# ================================================================
# Section 5: Load Real Orbit Data
# ================================================================

def load_real_orbit_data(orbit_dir, max_sats_per_layer=200):
    """
    Load satellite positions from parquet files.
    Returns positions array and layer metadata.
    """
    positions = []
    layer_ids = []
    sat_ids = []
    heights_list = []

    # Layer heights from metadata (L1=500, L2=800, L3=1100, L4=1400, L5=1700)
    layer_heights = {1: 500, 2: 800, 3: 1100, 4: 1400, 5: 1700}

    for layer in range(1, 6):
        layer_dir = os.path.join(orbit_dir, f"L{layer}")
        # Check if files are directly in orbit_dir or in subdirectories
        if os.path.isdir(layer_dir):
            search_dir = layer_dir
        else:
            search_dir = orbit_dir

        count = 0
        for sat_idx in range(1, max_sats_per_layer + 1):
            fname = f"L{layer}_{sat_idx:04d}.parquet"
            fpath = os.path.join(search_dir, fname)
            if not os.path.exists(fpath):
                # Try orbit_parquet directly
                fpath = os.path.join(orbit_dir, fname)
                if not os.path.exists(fpath):
                    continue

            try:
                df = pd.read_parquet(fpath)
                # Take position at first timestep (t=0, first row)
                row = df.iloc[0]
                pos = np.array([row['x'], row['y'], row['z']], dtype=float)
                positions.append(pos)
                layer_ids.append(layer)
                sat_ids.append(sat_idx)
                heights_list.append(layer_heights.get(layer, 500))
                count += 1
            except Exception as e:
                print(f"  Warning: Failed to read {fname}: {e}")

        print(f"  Layer L{layer}: loaded {count} satellites")

    positions = np.array(positions)
    print(f"\n  Total real orbit satellites loaded: {len(positions)}")
    return positions, layer_ids, sat_ids, heights_list

# ================================================================
# Section 7: Load Ground Stations
# ================================================================

def load_ground_stations(gs_json_path):
    """Load ground stations from JSON and compute positions and demands."""
    with open(gs_json_path, encoding='utf-8') as f:
        gs_data = json.load(f)

    gs_lat_lon = []
    gs_positions = []
    for s in gs_data['stations']:
        gs_lat_lon.append((s['latitude'], s['longitude']))
        gs_positions.append(latlon_to_cart(s['latitude'], s['longitude']))

    gs_positions = np.array(gs_positions)
    gs_demands_raw = np.array([s['weight'] for s in gs_data['stations']])
    gs_demands = 1 + 99 * (gs_demands_raw / gs_demands_raw.max())

    print(f"  Loaded {len(gs_positions)} ground stations")
    print(f"  Demand range: [{gs_demands.min():.0f}, {gs_demands.max():.0f}]")
    return gs_positions, gs_demands, gs_lat_lon, gs_data

# ================================================================
# Section 8: Run All Benchmarks
# ================================================================

def run_all_benchmarks(sat_pos, gs_pos, gs_demand, gamma_opt, label=""):
    """Run all 6 benchmark algorithms and return results dict."""
    print(f"\n  [{label}] Running benchmarks on N={len(sat_pos)} satellites...")

    results = {}
    t0 = pd.Timestamp.now()

    # 1. Greedy
    r = benchmark_greedy(sat_pos, gs_pos, gs_demand)
    results['greedy'] = {'imbalance': float(r['imbalance']), 'avg_dist_km': float(r['avg_dist_km']),
                         'n_used': int(r['n_used']), 'max_load': float(r['max_load'])}
    print(f"    Greedy:        n_used={r['n_used']}, dist={r['avg_dist_km']:.0f}km, imb={r['imbalance']:.3f}")

    # 2. RoundRobin
    r = benchmark_roundrobin(sat_pos, gs_pos, gs_demand)
    results['roundrobin'] = {'imbalance': float(r['imbalance']), 'avg_dist_km': float(r['avg_dist_km']),
                             'n_used': int(r['n_used']), 'max_load': float(r['max_load'])}
    print(f"    RoundRobin:    n_used={r['n_used']}, dist={r['avg_dist_km']:.0f}km, imb={r['imbalance']:.3f}")

    # 3. Nearest-3
    r = benchmark_nearest3(sat_pos, gs_pos, gs_demand)
    results['nearest3'] = {'imbalance': float(r['imbalance']), 'avg_dist_km': float(r['avg_dist_km']),
                           'n_used': int(r['n_used']), 'max_load': float(r['max_load'])}
    print(f"    Nearest-3:     n_used={r['n_used']}, dist={r['avg_dist_km']:.0f}km, imb={r['imbalance']:.3f}")

    # 4. ShortestPath
    r = benchmark_shortest_path(sat_pos, gs_pos, gs_demand)
    results['shortestpath'] = {'imbalance': float(r['imbalance']), 'avg_dist_km': float(r['avg_dist_km']),
                               'n_used': int(r['n_used']), 'max_load': float(r['max_load'])}
    print(f"    ShortestPath:  n_used={r['n_used']}, dist={r['avg_dist_km']:.0f}km, imb={r['imbalance']:.3f}")

    # 5. CBDP v2
    r = benchmark_cbdp(sat_pos, gs_pos, gs_demand, gamma=gamma_opt)
    results['cbdp_v2'] = {'imbalance': float(r['imbalance']), 'avg_dist_km': float(r['avg_dist_km']),
                          'n_used': int(r['n_used']), 'n_cores': int(r['n_cores']),
                          'max_load': float(r['max_load'])}
    print(f"    CBDP v2:       n_used={r['n_used']}, cores={r['n_cores']}, dist={r['avg_dist_km']:.0f}km, imb={r['imbalance']:.3f}")

    # 6. CBDP v3 (grid search over alpha and k_cores)
    alphas = [0.1, 0.2, 0.3, 0.5]
    k_cores_vals = [1, 2, 3, 5]
    best_v3_score = float('inf')
    best_v3_result = None
    best_v3_params = (0.3, 3)

    # Pre-compute core detection once
    r_v3_first = benchmark_cbdp_v3(sat_pos, gs_pos, gs_demand,
                                   gamma=gamma_opt, alpha=0.3, k_cores=3)
    core_cache = {
        'core_positions': r_v3_first.get('core_positions'),
        'n_cores_real': r_v3_first['n_cores'],
        'sat_core': r_v3_first.get('sat_core'),
    }

    r_nearest3 = benchmark_nearest3(sat_pos, gs_pos, gs_demand)
    for alpha_try in alphas:
        for k_try in k_cores_vals:
            r_v3_try = benchmark_cbdp_v3(sat_pos, gs_pos, gs_demand,
                                         gamma=gamma_opt, alpha=alpha_try, k_cores=k_try,
                                         core_cache=core_cache)
            dist_ratio = r_v3_try['avg_dist_km'] / r_nearest3['avg_dist_km']
            imb_ratio = r_v3_try['imbalance'] / max(r_nearest3['imbalance'], 0.01)
            score = 0.3 * dist_ratio + 0.7 * imb_ratio
            if score < best_v3_score:
                best_v3_score = score
                best_v3_result = r_v3_try
                best_v3_params = (alpha_try, k_try)

    r = best_v3_result
    results['cbdp_v3'] = {'imbalance': float(r['imbalance']), 'avg_dist_km': float(r['avg_dist_km']),
                          'n_used': int(r['n_used']), 'n_cores': int(r['n_cores']),
                          'max_load': float(r['max_load']),
                          'alpha': best_v3_params[0], 'k_cores': best_v3_params[1]}
    print(f"    CBDP v3:       n_used={r['n_used']}, cores={r['n_cores']}, dist={r['avg_dist_km']:.0f}km, "
          f"imb={r['imbalance']:.3f}, alpha={best_v3_params[0]}, k={best_v3_params[1]}")

    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    print(f"    [{label}] Completed in {elapsed:.1f}s")

    return results

# ================================================================
# Section 9: Main
# ================================================================

def main():
    print("=" * 70)
    print("Real Orbit Data Validation")
    print("=" * 70)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    orbit_dir = os.path.join(script_dir, "orbit_parquet")
    gs_json_path = os.path.join(script_dir, "ground_stations.json")
    output_path = os.path.join(script_dir, "real_orbit_report.json")

    # Set random seed for reproducibility (generate_network uses internal RandomState)
    # np.random.seed(42) — no longer needed; generate_network has internal seed=42

    # ---- Step 1: Load Real Orbit Data ----
    print("\n" + "=" * 70)
    print("Step 1: Loading Real Orbit Data (t=0)")
    print("=" * 70)
    real_positions, layer_ids, sat_ids, heights_list = load_real_orbit_data(orbit_dir, max_sats_per_layer=200)
    N_real = len(real_positions)
    print(f"  Total: {N_real} satellites from {len(set(layer_ids))} layers")
    for lyr in sorted(set(layer_ids)):
        count = sum(1 for lid in layer_ids if lid == lyr)
        avg_h = np.mean([h for h, lid in zip(heights_list, layer_ids) if lid == lyr])
        print(f"    Layer L{lyr}: {count} satellites, avg height ~{avg_h:.0f} km")

    # ---- Step 2: Load Ground Stations ----
    print("\n" + "=" * 70)
    print("Step 2: Loading Ground Stations")
    print("=" * 70)
    gs_positions, gs_demands, gs_lat_lon, gs_data = load_ground_stations(gs_json_path)

    # ---- Step 3: Compute Optimal Gamma ----
    target_frac = 0.25
    gamma_opt = max(required_gamma_for_core_fraction(target_frac), 0.1)
    n_pred = predict_cores(gamma_opt, N=N_real)
    print(f"\n  Target fraction: {target_frac}, γ_opt = {gamma_opt:.3f}")
    print(f"  Predicted cores: {n_pred:.0f} ({n_pred / N_real * 100:.1f}%)")

    # ---- Step 4: Run Benchmarks on Real Orbit Data ----
    print("\n" + "=" * 70)
    print("Step 4: Benchmarks on REAL ORBIT Data")
    print("=" * 70)
    real_results = run_all_benchmarks(real_positions, gs_positions, gs_demands, gamma_opt, label="REAL")

    # ---- Step 5: Generate Fibonacci Sphere Network for Comparison ----
    print("\n" + "=" * 70)
    print("Step 5: Fibonacci Sphere Comparison Network")
    print("=" * 70)

    # Use same N and layer heights as real data
    unique_layers = sorted(set(layer_ids))
    real_heights = []
    for lyr in unique_layers:
        layer_heights = [h for h, lid in zip(heights_list, layer_ids) if lid == lyr]
        real_heights.append(np.mean(layer_heights))
    print(f"  Using heights: {[f'{h:.0f}' for h in real_heights]} km")

    fibo_positions = generate_network(N_real, real_heights)
    print(f"  Generated {len(fibo_positions)} Fibonacci sphere positions")

    # ---- Step 6: Run Benchmarks on Fibonacci Data ----
    print("\n" + "=" * 70)
    print("Step 6: Benchmarks on FIBONACCI Sphere Data")
    print("=" * 70)
    fibo_results = run_all_benchmarks(fibo_positions, gs_positions, gs_demands, gamma_opt, label="FIBO")

    # ---- Step 7: Comparison Analysis ----
    print("\n" + "=" * 70)
    print("Step 7: Comparison Analysis")
    print("=" * 70)

    comparison = {}
    for algo in ['greedy', 'roundrobin', 'nearest3', 'shortestpath', 'cbdp_v2', 'cbdp_v3']:
        real = real_results[algo]
        fibo = fibo_results[algo]
        comp = {
            'real_orbit': real,
            'fibonacci': fibo,
            'delta': {}
        }
        for key in ['imbalance', 'avg_dist_km', 'n_used']:
            if key in real and key in fibo:
                delta = real[key] - fibo[key]
                pct = (delta / max(abs(fibo[key]), 1e-6)) * 100
                comp['delta'][key] = {'absolute': float(delta), 'percent': float(pct)}
        for key in ['n_cores']:
            if key in real and key in fibo:
                delta = real[key] - fibo[key]
                comp['delta'][key] = {'absolute': int(delta)}

        comparison[algo] = comp

        print(f"\n  {algo}:")
        print(f"    Real:      dist={real['avg_dist_km']:.0f}km, imb={real['imbalance']:.3f}, "
              f"n_used={real.get('n_used', 'N/A')}")
        print(f"    Fibonacci: dist={fibo['avg_dist_km']:.0f}km, imb={fibo['imbalance']:.3f}, "
              f"n_used={fibo.get('n_used', 'N/A')}")
        if 'n_cores' in real:
            print(f"    Cores:     real={real['n_cores']}, fibo={fibo['n_cores']}")

    # ---- Step 8: Summary Statistics ----
    print("\n" + "=" * 70)
    print("Step 8: Summary")
    print("=" * 70)

    # Key findings
    for algo in ['cbdp_v2', 'cbdp_v3']:
        real = real_results[algo]
        fibo = fibo_results[algo]
        dist_diff_pct = (real['avg_dist_km'] - fibo['avg_dist_km']) / max(fibo['avg_dist_km'], 1e-6) * 100
        imb_diff_pct = (real['imbalance'] - fibo['imbalance']) / max(fibo['imbalance'], 1e-6) * 100
        print(f"  {algo}: Real vs Fibonacci → dist Δ={dist_diff_pct:+.1f}%, imb Δ={imb_diff_pct:+.1f}%")

    # Nearest-3 as reference
    for label, results in [("Real", real_results), ("Fibonacci", fibo_results)]:
        n3 = results['nearest3']
        v3 = results['cbdp_v3']
        dist_ratio = v3['avg_dist_km'] / n3['avg_dist_km']
        imb_ratio = v3['imbalance'] / max(n3['imbalance'], 1e-6)
        print(f"  {label}: CBDP v3 vs Nearest-3 → dist_ratio={dist_ratio:.2f}x, imb_ratio={imb_ratio:.2f}x")

    # ---- Step 9: Build Output Report ----
    output = {
        "version": "1.0",
        "description": "Real orbit data validation — comparing real satellite positions with Fibonacci sphere",
        "parameters": {
            "seed": 42,
            "target_frac": target_frac,
            "gamma_opt": float(gamma_opt),
            "n_predicted_cores": float(n_pred),
            "N_real": N_real,
            "N_fibonacci": len(fibo_positions),
            "layer_heights_km": [float(h) for h in real_heights],
            "n_ground_stations": len(gs_positions),
            "GAMMA_SCALE": GAMMA_SCALE,
            "phase_diagram": {
                "n_baseline": n_baseline,
                "n_grid_max": n_grid_max,
                "gamma_char": gamma_char,
                "REF_N": REF_N,
            },
        },
        "real_orbit": {
            "N": N_real,
            "layers": {f"L{lyr}": sum(1 for lid in layer_ids if lid == lyr) for lyr in sorted(set(layer_ids))},
            "results": real_results,
        },
        "fibonacci": {
            "N": len(fibo_positions),
            "results": fibo_results,
        },
        "comparison": comparison,
        "key_findings": {
            "cbdp_v3_vs_nearest3_real": {
                "dist_ratio": float(real_results['cbdp_v3']['avg_dist_km'] / max(real_results['nearest3']['avg_dist_km'], 1e-6)),
                "imb_ratio": float(real_results['cbdp_v3']['imbalance'] / max(real_results['nearest3']['imbalance'], 1e-6)),
            },
            "cbdp_v3_vs_nearest3_fibonacci": {
                "dist_ratio": float(fibo_results['cbdp_v3']['avg_dist_km'] / max(fibo_results['nearest3']['avg_dist_km'], 1e-6)),
                "imb_ratio": float(fibo_results['cbdp_v3']['imbalance'] / max(fibo_results['nearest3']['imbalance'], 1e-6)),
            },
        },
        "ground_stations": {
            "stations": [{"latitude": lat, "longitude": lon, "weight": float(w)}
                         for (lat, lon), w in zip(gs_lat_lon, gs_demands)],
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\n{'=' * 70}")
    print(f"Validation complete. Report saved to: {output_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()