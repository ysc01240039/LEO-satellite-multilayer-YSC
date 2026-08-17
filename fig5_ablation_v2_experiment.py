#!/usr/bin/env python3
"""
===============================================================================
Fig5 Ablation Study v2-base — REAL experiments (supersedes v3-based ablation)
===============================================================================
Rationale: paper Algorithm 2 assigns each GS to its NEAREST SNC (v2 structure:
benchmark_cbdp), not the v3 alpha-split structure. To keep fig4 (v2) and fig5
consistent, the ablation is rebuilt on the v2 routing base.

Setup: N=1000, 5 layers [500,800,1100,1400,1700] km, gamma=1.0,
20 ground stations, 10 seeds (42..51), population-weighted demands (config B,
the configuration under which the benchmark report was generated).

Variants (v2 routing base = nearest SNC + intra-core top-k spreading):
  1. full_cbdp      : PDE cores (_detect_cores, gamma=1.0) + top-5 spread
  2. no_pde_snc     : k-means cores (same count, same seed) + top-5 spread
  3. no_hierarchy   : no cores at all; each GS -> single nearest satellite
  4. no_spread      : PDE cores + top-1 (no intra-core redundancy)
References: nearest3, greedy.

All numbers produced by real code runs. No hand-picked values.
===============================================================================
"""

import json
import os
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from common_utils import (
    latlon_to_cart, generate_network,
    benchmark_greedy, benchmark_nearest3, _detect_cores, NumpyEncoder,
)
from algorithm_v2 import gs_lat_lon_20

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GS_POS = np.array([latlon_to_cart(lat, lon) for lat, lon in gs_lat_lon_20])

_gs_head_path = os.path.join(SCRIPT_DIR, '_fig5_gs_head.json')
with open(_gs_head_path, encoding='utf-8') as f:
    _gs_data = json.load(f)
_w = np.array([s['weight'] for s in _gs_data['stations']])
DEMANDS = 1 + 99 * (_w / _w.max())   # population-weighted, same as algorithm_v2.py

N = 1000
HEIGHTS = [500, 800, 1100, 1400, 1700]
GAMMA = 1.0
SEEDS = list(range(42, 52))          # 10 runs


def cbdp_v2_route(sat_pos, gs_pos, gs_demand, core_cache, k_spread=5):
    """benchmark_cbdp logic with injectable core cache and spread parameter.

    Faithful replica of common_utils.benchmark_cbdp routing (nearest SNC per
    GS, demand split among the k_spread nearest member satellites), extended
    to accept an externally computed core cache (for the k-means ablation).
    """
    sat_pos = np.asarray(sat_pos)
    M = len(gs_pos)
    core_positions = core_cache['core_positions']
    sat_core = core_cache['sat_core']

    core_tree = cKDTree(core_positions)
    sat_tree = cKDTree(sat_pos)
    load = np.zeros(len(sat_pos))
    total_dist = 0.0

    for j in range(M):
        _, core_c = core_tree.query(gs_pos[j])
        core_sats = np.where(sat_core == core_c)[0]
        if len(core_sats) > 0:
            k = min(k_spread, len(core_sats))
            sat_dists = np.array([np.linalg.norm(gs_pos[j] - sat_pos[s])
                                  for s in core_sats])
            sort_idx = np.argsort(sat_dists)
            sorted_idx = core_sats[sort_idx]
            for idx in sorted_idx[:k]:
                load[idx] += gs_demand[j] / k
            total_dist += np.mean(sat_dists[sort_idx[:k]])
        else:
            d_nearest, nearest_idx = sat_tree.query(gs_pos[j])
            load[nearest_idx] += gs_demand[j]
            total_dist += d_nearest

    n_used = int(np.sum(load > 0))
    imb = ((load.max() - load[load > 0].min()) / max(load.mean(), 1e-6)
           if n_used > 0 else 0.0)
    return {'imbalance': float(imb), 'avg_dist_km': float(total_dist / M),
            'n_used': n_used}


def nearest1(sat_pos, gs_pos, gs_demand):
    """No hierarchy: each GS sends all demand to its single nearest satellite."""
    sat_tree = cKDTree(sat_pos)
    load = np.zeros(len(sat_pos))
    total_dist = 0.0
    for j in range(len(gs_pos)):
        d, s = sat_tree.query(gs_pos[j])
        load[s] += gs_demand[j]
        total_dist += d
    n_used = int(np.sum(load > 0))
    imb = ((load.max() - load[load > 0].min()) / max(load.mean(), 1e-6)
           if n_used > 0 else 0.0)
    return {'imbalance': float(imb), 'avg_dist_km': float(total_dist / len(gs_pos)),
            'n_used': n_used}


def kmeans_core_cache(sat_pos, n_cores, seed):
    km = KMeans(n_clusters=n_cores, n_init=10, random_state=seed)
    labels = km.fit_predict(sat_pos)
    return {'core_positions': km.cluster_centers_,
            'n_cores_real': int(n_cores), 'sat_core': labels}


def main():
    variants = ['full_cbdp', 'no_pde_snc', 'no_hierarchy', 'no_spread',
                'nearest3_ref', 'greedy_ref']
    per_seed = {v: [] for v in variants}
    n_cores_log = []

    for seed in SEEDS:
        sat = generate_network(N, HEIGHTS, seed=seed)
        pde = _detect_cores(sat, GAMMA, N)
        km = kmeans_core_cache(sat, pde['n_cores_real'], seed)
        n_cores_log.append(int(pde['n_cores_real']))

        r_full = cbdp_v2_route(sat, GS_POS, DEMANDS, pde, k_spread=5)
        r_nopde = cbdp_v2_route(sat, GS_POS, DEMANDS, km, k_spread=5)
        r_nohier = nearest1(sat, GS_POS, DEMANDS)
        r_nospread = cbdp_v2_route(sat, GS_POS, DEMANDS, pde, k_spread=1)
        r_n3 = benchmark_nearest3(sat, GS_POS, DEMANDS)
        r_greedy = benchmark_greedy(sat, GS_POS, DEMANDS)

        for name, r in [('full_cbdp', r_full), ('no_pde_snc', r_nopde),
                        ('no_hierarchy', r_nohier), ('no_spread', r_nospread),
                        ('nearest3_ref', r_n3), ('greedy_ref', r_greedy)]:
            per_seed[name].append({'imbalance': r['imbalance'],
                                   'avg_dist_km': r['avg_dist_km'],
                                   'n_used': r['n_used']})
        print(f"seed={seed}: full={r_full['imbalance']:.2f} "
              f"-PDE={r_nopde['imbalance']:.2f} -Hier={r_nohier['imbalance']:.2f} "
              f"-Spread={r_nospread['imbalance']:.2f} "
              f"N3={r_n3['imbalance']:.2f} Greedy={r_greedy['imbalance']:.2f} "
              f"cores={pde['n_cores_real']}")

    summary = {}
    for v in variants:
        imb = np.array([r['imbalance'] for r in per_seed[v]])
        dst = np.array([r['avg_dist_km'] for r in per_seed[v]])
        summary[v] = {'imb_mean': float(imb.mean()), 'imb_std': float(imb.std()),
                      'dist_mean': float(dst.mean()), 'dist_std': float(dst.std()),
                      'n_used_mean': float(np.mean([r['n_used'] for r in per_seed[v]])),
                      'imb_per_seed': imb.tolist(), 'dist_per_seed': dst.tolist()}

    full = summary['full_cbdp']['imb_mean']
    deg = {v: float(100.0 * (summary[v]['imb_mean'] - full) / full)
           for v in ['no_pde_snc', 'no_hierarchy', 'no_spread']}

    print('\nSUMMARY (config B, population-weighted demands, 10 seeds)')
    print(f"{'Variant':<16} {'Imbalance':>17} {'Dist(km)':>15} {'Degr.%':>8}")
    for v in variants:
        s = summary[v]
        d = f"{deg[v]:>7.1f}%" if v in deg else ('       0%' if v == 'full_cbdp' else '       -')
        print(f"{v:<16} {s['imb_mean']:>7.2f} +/- {s['imb_std']:<6.2f} "
              f"{s['dist_mean']:>7.0f} +/- {s['dist_std']:<5.0f} {d}")

    out = {'config': {'N': N, 'heights': HEIGHTS, 'gamma': GAMMA, 'seeds': SEEDS,
                      'demands': 'population-weighted (config B)',
                      'base': 'v2 routing (nearest SNC + intra-core top-k spread)'},
           'n_cores_pde_per_seed': n_cores_log,
           'summary': summary, 'degradation_pct_vs_full': deg,
           'per_seed': per_seed}
    with open(os.path.join(SCRIPT_DIR, 'fig5_ablation_v2_results.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print('\nSaved: fig5_ablation_v2_results.json')


if __name__ == '__main__':
    main()
