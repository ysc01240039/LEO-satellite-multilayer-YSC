#!/usr/bin/env python3
"""
fig4 benchmark re-run at the paper's nominal operating point gamma=1.0.

Rationale: the first multi-seed rerun used gamma=0.444 (near-critical, from the
legacy 25% core-fraction heuristic). The rest of the evaluation (fig5 ablation,
ns-3 validation center, failure recovery) uses gamma=1.0. This run unifies the
scaling benchmark at gamma=1.0 so that all experimental figures share one
operating point.

Protocol: same constellations, demands (config B), and seeds as
rerun_benchmark_v2.py; algorithms: Greedy, Round-Robin, Nearest-3, CBDP
(v2 routing base = paper Algorithm 2: nearest SNC + intra-core top-5 spread).

Output: algorithm_v2_rerun_g1p0_report.json
"""

import json, os, time
import numpy as np

from common_utils import (
    generate_network, latlon_to_cart,
    benchmark_greedy, benchmark_roundrobin, benchmark_nearest3,
    benchmark_cbdp, NumpyEncoder,
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

SEEDS = list(range(42, 52))          # 10 seeds, same set as fig5 ablation
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
    t0 = time.time()
    for cfg in CONSTELLATIONS:
        for seed in SEEDS:
            layers_info = cfg.get('heights', cfg.get('height', 500))
            sat_pos = generate_network(cfg['N'], layers_info, seed=seed)
            r = {
                'constellation': cfg['name'], 'N': cfg['N'], 'seed': seed,
                'gamma': GAMMA,
                'greedy':     benchmark_greedy(sat_pos, gs_positions, gs_demands),
                'roundrobin': benchmark_roundrobin(sat_pos, gs_positions, gs_demands),
                'nearest3':   benchmark_nearest3(sat_pos, gs_positions, gs_demands),
                'cbdp':       benchmark_cbdp(sat_pos, gs_positions, gs_demands,
                                             gamma=GAMMA),
            }
            slim = {'constellation': cfg['name'], 'N': cfg['N'], 'seed': seed,
                    'gamma': GAMMA}
            for alg in ['greedy', 'roundrobin', 'nearest3', 'cbdp']:
                slim[alg] = {'imbalance': float(r[alg]['imbalance']),
                             'avg_dist_km': float(r[alg]['avg_dist_km']),
                             'n_used': int(r[alg]['n_used'])}
            slim['n_cores_cbdp'] = int(r['cbdp']['n_cores'])
            per_seed.append(slim)
            print(f"N={cfg['N']:>5} seed={seed:>5}: "
                  f"greedy={slim['greedy']['imbalance']:>7.2f} "
                  f"rr={slim['roundrobin']['imbalance']:>6.2f} "
                  f"n3={slim['nearest3']['imbalance']:>7.2f} "
                  f"cbdp={slim['cbdp']['imbalance']:>7.2f} "
                  f"cores={slim['n_cores_cbdp']:>4}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    # aggregate mean/std per constellation
    bench = []
    for cfg in CONSTELLATIONS:
        rows = [r for r in per_seed if r['N'] == cfg['N']]
        agg = {'constellation': cfg['name'], 'N': cfg['N'], 'gamma': GAMMA,
               'n_seeds': len(rows)}
        for alg in ['greedy', 'roundrobin', 'nearest3', 'cbdp']:
            imb = np.array([r[alg]['imbalance'] for r in rows])
            dst = np.array([r[alg]['avg_dist_km'] for r in rows])
            nu = np.array([r[alg]['n_used'] for r in rows])
            agg[alg] = {'imbalance_mean': float(imb.mean()),
                        'imbalance_std': float(imb.std()),
                        'avg_dist_km_mean': float(dst.mean()),
                        'avg_dist_km_std': float(dst.std()),
                        'n_used_mean': float(nu.mean()),
                        'n_used_std': float(nu.std())}
        nc = np.array([r['n_cores_cbdp'] for r in rows])
        agg['n_cores_cbdp_mean'] = float(nc.mean())
        agg['n_cores_cbdp_std'] = float(nc.std())
        bench.append(agg)

    out = {'description': 'fig4 scaling benchmark at gamma=1.0 (nominal operating point), '
                          'v2 routing base (paper Algorithm 2), 5 seeds, config B demands',
           'date': time.strftime('%Y-%m-%d'), 'seeds': SEEDS, 'gamma': GAMMA,
           'demands': 'population-weighted (config B)',
           'per_seed_results': per_seed, 'benchmark_results_mean_std': bench}
    with open(os.path.join(SCRIPT_DIR, 'algorithm_v2_rerun_g1p0_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print('\nSaved: algorithm_v2_rerun_g1p0_report.json')


if __name__ == '__main__':
    main()
