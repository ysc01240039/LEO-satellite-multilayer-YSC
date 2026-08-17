#!/usr/bin/env python3
"""
Multi-seed re-run of the algorithm_v2 Part E benchmark with the CURRENT code.

Purpose: replace algorithm_v2_report.json benchmark_results (which were produced
by an uncommitted intermediate code version and are not reproducible) with
honest multi-seed mean/std data from the current codebase.

Protocol: identical to algorithm_v2.py Part E
  - 5 constellations (Iridium 66, Globalstar 48, Medium 500, Large 1000, Gen1 4408)
  - population-weighted ground-station demands (config B, from _fig5_gs_head.json,
    i.e. the ground_stations.json used to generate the original report)
  - gamma_opt = required_gamma_for_core_fraction(0.25), floored at 0.1
  - CBDP v2 = benchmark_cbdp(gamma_opt)
  - CBDP v3 = grid search alpha in {0.1,0.2,0.3,0.5}, k_cores in {1,2,3,5},
    score = 0.3*dist_ratio + 0.7*imb_ratio vs Nearest-3 (same as Part E)
  - seeds: 42, 123, 456, 789, 2024

Output: algorithm_v2_rerun_report.json
"""

import json, os, time
import numpy as np

import common_utils
from common_utils import (
    predict_cores, required_gamma_for_core_fraction,
    generate_network, latlon_to_cart,
    benchmark_greedy, benchmark_roundrobin, benchmark_nearest3,
    benchmark_cbdp, benchmark_cbdp_v3,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Ground stations (same 20 sites as algorithm_v2.py) ---
GS_LAT_LON = [
    (39.9, 116.4), (31.2, 121.5), (40.7, -74.0), (51.5, -0.1), (35.7, 139.7),
    (48.9, 2.3), (37.8, -122.4), (55.8, 37.6), (19.4, -99.1), (-33.9, 151.2),
    (1.3, 103.8), (28.6, 77.2), (-23.6, -46.6), (55.0, -3.4), (52.5, 13.4),
    (37.6, 127.0), (-6.2, 106.8), (22.3, 114.2), (25.2, 55.3), (35.0, 33.0),
]

CONSTELLATIONS = [
    {"name": "Globalstar-scale", "N": 48,  "layers": 1, "height": 1414},
    {"name": "Iridium-scale",    "N": 66,  "layers": 1, "height": 780},
    {"name": "Medium-scale",     "N": 500, "layers": 3, "heights": [500, 900, 1300]},
    {"name": "Large-scale",      "N": 1000, "layers": 5, "heights": [500, 800, 1100, 1400, 1700]},
    {"name": "Starlink Gen1",    "N": 4408, "layers": 5, "heights": [340, 550, 1110, 1130, 1275]},
]

SEEDS = [42, 123, 456, 789, 2024]

ALPHAS = [0.1, 0.2, 0.3, 0.5]
K_CORES_VALS = [1, 2, 3, 5]


def load_demands():
    """Config B: population-weighted demands from the original ground_stations.json."""
    path = os.path.join(SCRIPT_DIR, '_fig5_gs_head.json')
    with open(path, encoding='utf-8') as f:
        gs_data = json.load(f)
    raw = np.array([s['weight'] for s in gs_data['stations']], dtype=float)
    return 1 + 99 * (raw / raw.max())  # normalize to [1, 100], same as algorithm_v2.py


def run_one(cfg, seed, gs_positions, gs_demands):
    """Run all algorithms for one constellation and one seed (Part E protocol)."""
    layers_info = cfg.get('heights', cfg.get('height', 500))
    sat_pos = generate_network(cfg['N'], layers_info, seed=seed)

    r_greedy = benchmark_greedy(sat_pos, gs_positions, gs_demands)
    r_rr = benchmark_roundrobin(sat_pos, gs_positions, gs_demands)
    r_nearest3 = benchmark_nearest3(sat_pos, gs_positions, gs_demands)

    target_frac = 0.25
    gamma_opt = max(required_gamma_for_core_fraction(target_frac), 0.1)

    r_cbdp = benchmark_cbdp(sat_pos, gs_positions, gs_demands, gamma=gamma_opt)

    # CBDP v3 grid search with cached core detection (same as Part E)
    r_v3_first = benchmark_cbdp_v3(sat_pos, gs_positions, gs_demands,
                                   gamma=gamma_opt, alpha=0.3, k_cores=3)
    core_cache = {
        'core_positions': r_v3_first.get('core_positions'),
        'n_cores_real': r_v3_first['n_cores'],
        'sat_core': r_v3_first.get('sat_core'),
    }
    best_score = float('inf')
    best_v3 = None
    best_params = (0.3, 3)
    for a in ALPHAS:
        for k in K_CORES_VALS:
            r_try = benchmark_cbdp_v3(sat_pos, gs_positions, gs_demands,
                                      gamma=gamma_opt, alpha=a, k_cores=k,
                                      core_cache=core_cache)
            dist_ratio = r_try['avg_dist_km'] / r_nearest3['avg_dist_km']
            imb_ratio = r_try['imbalance'] / max(r_nearest3['imbalance'], 0.01)
            score = 0.3 * dist_ratio + 0.7 * imb_ratio
            if score < best_score:
                best_score = score
                best_v3 = r_try
                best_params = (a, k)

    return {
        'constellation': cfg['name'],
        'N': cfg['N'],
        'seed': seed,
        'gamma_opt': gamma_opt,
        'n_cores_v2': r_cbdp['n_cores'],
        'n_cores_v3': best_v3['n_cores'],
        'greedy':     {'imbalance': r_greedy['imbalance'], 'avg_dist_km': r_greedy['avg_dist_km'], 'n_used': r_greedy['n_used']},
        'roundrobin': {'imbalance': r_rr['imbalance'],     'avg_dist_km': r_rr['avg_dist_km'],     'n_used': r_rr['n_used']},
        'nearest3':   {'imbalance': r_nearest3['imbalance'],'avg_dist_km': r_nearest3['avg_dist_km'],'n_used': r_nearest3['n_used']},
        'cbdp':       {'imbalance': r_cbdp['imbalance'],   'avg_dist_km': r_cbdp['avg_dist_km'],   'n_used': r_cbdp['n_used']},
        'cbdp_v3':    {'imbalance': best_v3['imbalance'],  'avg_dist_km': best_v3['avg_dist_km'],  'n_used': best_v3['n_used'],
                       'alpha': best_params[0], 'k_cores': best_params[1]},
    }


def aggregate(rows):
    """Mean/std across seeds for each constellation and algorithm."""
    agg = []
    for cfg in CONSTELLATIONS:
        sub = [r for r in rows if r['N'] == cfg['N']]
        entry = {'constellation': cfg['name'], 'N': cfg['N'], 'n_seeds': len(sub),
                 'gamma_opt': float(np.mean([r['gamma_opt'] for r in sub]))}
        for alg in ['greedy', 'roundrobin', 'nearest3', 'cbdp', 'cbdp_v3']:
            for metric in ['imbalance', 'avg_dist_km', 'n_used']:
                vals = np.array([r[alg][metric] for r in sub], dtype=float)
                entry.setdefault(alg, {})[metric + '_mean'] = float(vals.mean())
                entry[alg][metric + '_std'] = float(vals.std())
        entry['n_cores_v2_mean'] = float(np.mean([r['n_cores_v2'] for r in sub]))
        entry['n_cores_v2_std'] = float(np.std([r['n_cores_v2'] for r in sub]))
        entry['n_cores_v3_mean'] = float(np.mean([r['n_cores_v3'] for r in sub]))
        entry['n_cores_v3_std'] = float(np.std([r['n_cores_v3'] for r in sub]))
        # v3 parameter choices across seeds
        entry['cbdp_v3_params'] = [(r['cbdp_v3']['alpha'], r['cbdp_v3']['k_cores']) for r in sub]
        agg.append(entry)
    return agg


def main():
    gs_demands = load_demands()
    gs_positions = np.array([latlon_to_cart(lat, lon) for lat, lon in GS_LAT_LON])

    rows = []
    for cfg in CONSTELLATIONS:
        for seed in SEEDS:
            t0 = time.time()
            row = run_one(cfg, seed, gs_positions, gs_demands)
            dt = time.time() - t0
            rows.append(row)
            print(f"[done] N={cfg['N']:>5} seed={seed:>4}  "
                  f"cbdp_imb={row['cbdp']['imbalance']:.2f}  "
                  f"v3_imb={row['cbdp_v3']['imbalance']:.2f}  "
                  f"n_cores_v2={row['n_cores_v2']}  ({dt:.1f}s)", flush=True)

    agg = aggregate(rows)

    out = {
        'description': 'Multi-seed re-run of algorithm_v2 Part E benchmark with current code',
        'date': '2026-08-07',
        'seeds': SEEDS,
        'demands': 'population-weighted (config B, original ground_stations.json)',
        'protocol': 'identical to algorithm_v2.py Part E (gamma_opt=frac0.25, v3 grid search)',
        'per_seed_results': rows,
        'benchmark_results_mean_std': agg,
    }
    out_path = os.path.join(SCRIPT_DIR, 'algorithm_v2_rerun_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, cls=common_utils.NumpyEncoder)
    print(f"\n[saved] {out_path}")

    # Console summary
    print("\n=== Mean +/- std across seeds ===")
    for e in agg:
        print(f"N={e['N']:>5}: CBDPv2 {e['cbdp']['imbalance_mean']:.2f}+/-{e['cbdp']['imbalance_std']:.2f} | "
              f"CBDPv3 {e['cbdp_v3']['imbalance_mean']:.2f}+/-{e['cbdp_v3']['imbalance_std']:.2f} | "
              f"Greedy {e['greedy']['imbalance_mean']:.2f} | N3 {e['nearest3']['imbalance_mean']:.2f} | "
              f"RR {e['roundrobin']['imbalance_mean']:.3f}")


if __name__ == '__main__':
    main()
