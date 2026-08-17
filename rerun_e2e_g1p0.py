#!/usr/bin/env python3
"""
Multi-seed end-to-end scalability benchmark at gamma = 1.0.

Replaces the legacy access-layer-only rerun_benchmark_g1p0.py.
All methods are evaluated end-to-end over the ISL graph (Kruskal MST + 4-NN)
in a shared topology, shared flow model (all ordered GS pairs), and shared
metric convention (demand-weighted path length).

Methods:
    dijkstra      centralized shortest-path oracle (upper bound)
    greedy        least-loaded of 5 nearest access satellites + SP transit
    nearest3      3 nearest access satellites (equal split) + SP transit
    roundrobin    3-unit demand chunks in global rotation + SP transit
    cbdp          proposed (paper Algorithm 2): SNC hierarchy + portal relay
    pfnsar        potential-field network-state-aware routing (Wei et al. 2025 TCOM)
    lpih          hierarchical logic-path routing (Yan et al. 2024)

Output:
    algorithm_v2_e2e_g1p0_report.json
        per_seed_results: one row per (constellation, seed)
        benchmark_results_mean_std: mean / std per constellation
"""

import json, os, time
import numpy as np

from common_utils import (
    generate_network, latlon_to_cart, benchmark_e2e_all, NumpyEncoder,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

SEEDS = list(range(42, 52))  # 10 seeds (same set as fig5 ablation)
GAMMA = 1.0


def load_demands():
    path = os.path.join(SCRIPT_DIR, '_fig5_gs_head.json')
    with open(path, encoding='utf-8') as f:
        gs_data = json.load(f)
    raw = np.array([s['weight'] for s in gs_data['stations']], dtype=float)
    return 1 + 99 * (raw / raw.max())


def main():
    gs_positions = np.array([latlon_to_cart(a, b) for a, b in GS_LAT_LON])
    gs_demands = load_demands()

    per_seed = []
    t0_all = time.time()
    for cfg in CONSTELLATIONS:
        for seed in SEEDS:
            layers_info = cfg.get('heights', cfg.get('height', 500))
            sat_pos = generate_network(cfg['N'], layers_info, seed=seed)
            t0 = time.time()
            r = benchmark_e2e_all(sat_pos, gs_positions, gs_demands,
                                   gamma=GAMMA, seed=seed)
            dt = time.time() - t0
            methods = r['methods']
            row = {
                'constellation': cfg['name'],
                'N': cfg['N'],
                'seed': seed,
                'gamma': GAMMA,
                'graph_edges': r['graph_edges'],
                'graph_deg_avg': r['graph_deg_avg'],
                'n_cores_cbdp': r['n_cores_cbdp'],
                'n_domains_lpih': r['n_domains_lpih'],
                'pfnsar_beta': r['pfnsar_beta'],
                'elapsed_sec': round(dt, 3),
            }
            for name, v in methods.items():
                row[name] = {
                    'imbalance': float(v['imbalance']),
                    'carried_imbalance': float(v['carried_imbalance']),
                    'avg_dist_km': float(v['avg_dist_km']),
                    'n_used': int(v['n_used']),
                    'max_carried': float(v['max_carried']),
                    'mean_carried': float(v['mean_carried']),
                    'overhead_bytes_per_cycle': float(v['overhead_bytes_per_cycle']),
                    'route_ops': float(v['route_ops']),
                }
            per_seed.append(row)
            print(f"N={cfg['N']:>5} seed={seed:>5}: "
                  f"CBDP imb={row['cbdp']['imbalance']:>7.2f} "
                  f"dist={row['cbdp']['avg_dist_km']:>9.1f} "
                  f"n_used={row['cbdp']['n_used']:>4} "
                  f"cores={row['n_cores_cbdp']:>3} "
                  f"({dt:.1f}s)", flush=True)

    # aggregate mean / std per constellation
    bench = []
    for cfg in CONSTELLATIONS:
        rows = [r for r in per_seed if r['N'] == cfg['N']]
        agg = {'constellation': cfg['name'], 'N': cfg['N'], 'gamma': GAMMA,
               'n_seeds': len(rows)}
        metrics = ['imbalance', 'carried_imbalance', 'avg_dist_km', 'n_used',
                   'max_carried', 'mean_carried', 'overhead_bytes_per_cycle',
                   'route_ops']
        for name in ['dijkstra', 'greedy', 'nearest3', 'roundrobin',
                     'cbdp', 'pfnsar', 'lpih']:
            entry = {}
            for m in metrics:
                vals = np.array([r[name][m] for r in rows], dtype=float)
                entry[f"{m}_mean"] = float(vals.mean())
                entry[f"{m}_std"] = float(vals.std())
            agg[name] = entry
        agg['n_cores_cbdp_mean'] = float(np.mean([r['n_cores_cbdp'] for r in rows]))
        agg['n_cores_cbdp_std'] = float(np.std([r['n_cores_cbdp'] for r in rows]))
        agg['n_domains_lpih_mean'] = float(np.mean([r['n_domains_lpih'] for r in rows]))
        agg['n_domains_lpih_std'] = float(np.std([r['n_domains_lpih'] for r in rows]))
        bench.append(agg)

    out = {
        'description': 'E2E scalability benchmark at gamma=1.0, '
                       'all methods over shared ISL graph (Kruskal MST + 4-NN), '
                       'all-pairs GS flows, 10 seeds.',
        'date': time.strftime('%Y-%m-%d'),
        'seeds': SEEDS,
        'gamma': GAMMA,
        'demands': 'population-weighted (config B)',
        'per_seed_results': per_seed,
        'benchmark_results_mean_std': bench,
    }
    out_path = os.path.join(SCRIPT_DIR, 'algorithm_v2_e2e_g1p0_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nSaved: {out_path}  (total {time.time()-t0_all:.1f}s)")

    # Console summary
    print("\n=== Mean +/- std across seeds ===")
    for e in bench:
        print(f"N={e['N']:>5}: "
              f"Dijkstra {e['dijkstra']['imbalance_mean']:>7.2f}+-{e['dijkstra']['imbalance_std']:>5.2f} | "
              f"Greedy {e['greedy']['imbalance_mean']:>7.2f}+-{e['greedy']['imbalance_std']:>5.2f} | "
              f"N3 {e['nearest3']['imbalance_mean']:>7.2f}+-{e['nearest3']['imbalance_std']:>5.2f} | "
              f"CBDP {e['cbdp']['imbalance_mean']:>7.2f}+-{e['cbdp']['imbalance_std']:>5.2f} | "
              f"PFNSAR {e['pfnsar']['imbalance_mean']:>7.2f}+-{e['pfnsar']['imbalance_std']:>5.2f} | "
              f"LPIH {e['lpih']['imbalance_mean']:>7.2f}+-{e['lpih']['imbalance_std']:>5.2f}")


if __name__ == '__main__':
    main()
