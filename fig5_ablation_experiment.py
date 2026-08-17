#!/usr/bin/env python3
"""
===============================================================================
*** DEPRECATED — SUPERSEDED BY fig5_ablation_v2_experiment.py (2026-08) ***
This v1 script uses the CBDP v3 mesh routing base (full baseline imbalance
46.62). The paper's ablation figure and text are generated from the v2 script,
which uses the CBDP v2 routing base (Algorithm 2, full baseline imbalance
39.44). The two bases yield different absolute numbers; do NOT run this script
to reproduce the paper's Fig. 5. The core qualitative conclusion (SNC hierarchy
and intra-core load spreading are the dominant components; PDE detection alone
gives no routing benefit) holds in both versions. This file is retained only
for historical/audit reference.
===============================================================================
Fig5 Ablation Study — REAL experiments (replaces hardcoded fig5 data)
===============================================================================
Setup: N=1000 (Large-scale, heights [500,800,1100,1400,1700]), gamma=1.0,
20 ground stations, 10 seeds (42..51). Two demand configurations:
  A) fallback hardcoded demands [13,10,...,15]  (current working tree state)
  B) population-weighted demands from ground_stations.json @git HEAD
     (the configuration under which algorithm_v2_report.json was generated;
     the file is deleted in the working tree, so it was recovered via git)

Variants (all use the existing routing benchmark framework):
  1. Full CBDP            = PDE cores (demand-weighted = "dynamic reconfig" ON)
                            + CBDP v3 mesh routing with grid search (alpha, k_cores)
  2. -PDE SNC             = k-means cores (demand-weighted, same k as PDE) + same routing
  3. -Dynamic Reconfig    = PDE cores with UNIFORM weights (no demand adaptation)
                            + same routing
  4. -SNC Mesh (flat)     = PDE demand-weighted cores, routing alpha=1.0
                            (core hierarchy unused -> direct nearest-satellite)
  5. -SNC Mesh (1-core)   = PDE demand-weighted cores, alpha=0, k_cores=1
                            (hierarchy kept, but no multi-core mesh splitting)
  6. -Nonlinear Damping   = NOT REALIZABLE in the Python routing benchmark
                            (kappa is a parameter of the C++ PDE time integrator
                            only; no routing-level equivalent exists)

Cross-checks:
  * Replication attempt of algorithm_v2_report.json N=1000 row
    (gamma=0.4440873874393292, seed=42) under both demand sets.
  * CBDP v2 (benchmark_cbdp) at gamma=1.0 for reference.
  * 24-hour dynamic reconfiguration test: re-detect cores per hour (demand-weighted,
    gamma scaled by demand ratio, as in algorithm_v2 Part K) vs frozen cores.

All numbers below are produced by real code runs. No hand-picked values.
===============================================================================
"""

import json
import os
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

import common_utils
from common_utils import (
    latlon_to_cart, generate_network,
    benchmark_greedy, benchmark_nearest3, benchmark_shortest_path,
    benchmark_cbdp, benchmark_cbdp_v3, _detect_cores, NumpyEncoder,
)
from algorithm_v2 import (
    gs_lat_lon_20, benchmark_cbdp_v3_demand_weighted,
    generate_time_varying_demand,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Benchmark configuration -------------------------------------------------
GS_POS = np.array([latlon_to_cart(lat, lon) for lat, lon in gs_lat_lon_20])
DEMANDS_FALLBACK = np.array([13, 10, 19, 14, 20, 5, 15, 5, 17, 12,
                             8, 16, 11, 6, 9, 10, 14, 13, 7, 15], dtype=float)

_gs_head_path = os.path.join(SCRIPT_DIR, '_fig5_gs_head.json')
with open(_gs_head_path, encoding='utf-8') as f:
    _gs_data = json.load(f)
_w = np.array([s['weight'] for s in _gs_data['stations']])
DEMANDS_POPWEIGHT = 1 + 99 * (_w / _w.max())   # same normalization as algorithm_v2.py

N = 1000
HEIGHTS = [500, 800, 1100, 1400, 1700]
GAMMA = 1.0
SEEDS = list(range(42, 52))          # 10 runs
ALPHAS = [0.1, 0.2, 0.3, 0.5]        # same grid as algorithm_v2.py
K_CORES_VALS = [1, 2, 3, 5]


def demand_weights(sat_pos, gs_pos, gs_dem):
    """Same demand-weight construction as benchmark_cbdp_v3_demand_weighted."""
    w = np.ones(len(sat_pos))
    tree = cKDTree(sat_pos)
    for j in range(len(gs_pos)):
        _, s = tree.query(gs_pos[j])
        w[s] += gs_dem[j] / max(gs_dem.mean(), 1e-6)
    return w


def kmeans_core_cache(sat_pos, n_cores, seed, sat_weight=None):
    """Replace PDE core detection with k-means clustering (ablation)."""
    km = KMeans(n_clusters=n_cores, n_init=10, random_state=seed)
    labels = km.fit_predict(sat_pos, sample_weight=sat_weight)
    return {'core_positions': km.cluster_centers_,
            'n_cores_real': int(n_cores),
            'sat_core': labels}


def v3_grid_search(sat_pos, gs_pos, gs_dem, ref, core_cache):
    """Same (alpha, k_cores) grid-search protocol as algorithm_v2.py Part E."""
    best, best_score, best_params = None, np.inf, None
    for a in ALPHAS:
        for k in K_CORES_VALS:
            r = benchmark_cbdp_v3(sat_pos, gs_pos, gs_dem, gamma=GAMMA,
                                  alpha=a, k_cores=k, core_cache=core_cache)
            score = (0.3 * r['avg_dist_km'] / ref['avg_dist_km']
                     + 0.7 * r['imbalance'] / max(ref['imbalance'], 0.01))
            if score < best_score:
                best_score, best, best_params = score, r, (a, k)
    return best, best_params


def slim(r):
    return {'imbalance': float(r['imbalance']),
            'avg_dist_km': float(r['avg_dist_km']),
            'n_used': int(r['n_used'])}


def replication_check(gs_dem, label):
    """Try to reproduce algorithm_v2_report.json N=1000 row exactly."""
    sat42 = generate_network(N, HEIGHTS, seed=42)
    g_report = 0.4440873874393292
    r_v2 = benchmark_cbdp(sat42, GS_POS, gs_dem, gamma=g_report)
    print(f"  [{label}] CBDP v2 @gamma=0.444, seed=42: "
          f"imbalance={r_v2['imbalance']:.4f} (report 35.9655), "
          f"dist={r_v2['avg_dist_km']:.2f} (report 1284.39), "
          f"cores={r_v2['n_cores']} (report 84)")
    return {'demand_set': label,
            'cbdp_v2': slim(r_v2) | {'n_cores': int(r_v2['n_cores'])},
            'report_values': {'imbalance': 35.96547454129859,
                              'avg_dist_km': 1284.3919430247784,
                              'n_cores': 84}}


def run_config(gs_dem, label):
    """Full ablation suite for one demand configuration."""
    print("\n" + "=" * 72)
    print(f"Config [{label}]: static ablation, N={N}, gamma={GAMMA}, "
          f"seeds {SEEDS[0]}..{SEEDS[-1]}")
    print("=" * 72)

    variants = ['full_cbdp', 'no_pde_snc', 'no_dynamic_reconfig',
                'no_snc_mesh_flat', 'no_snc_mesh_1core',
                'cbdp_v2_ref', 'nearest3_ref', 'greedy_ref', 'shortest_ref']
    per_seed = {v: [] for v in variants}
    extra = {'n_cores_pde_uniform': [], 'n_cores_pde_demandw': [],
             'params_full': [], 'params_no_pde': [], 'params_no_reconfig': []}

    for seed in SEEDS:
        sat = generate_network(N, HEIGHTS, seed=seed)
        ref_n3 = benchmark_nearest3(sat, GS_POS, gs_dem)
        r_greedy = benchmark_greedy(sat, GS_POS, gs_dem)
        r_short = benchmark_shortest_path(sat, GS_POS, gs_dem)
        r_v2 = benchmark_cbdp(sat, GS_POS, gs_dem, gamma=GAMMA)

        w = demand_weights(sat, GS_POS, gs_dem)

        pde_dw = _detect_cores(sat, GAMMA, N, sat_weight=w)   # demand-adaptive
        pde_u = _detect_cores(sat, GAMMA, N)                  # uniform
        km_dw = kmeans_core_cache(sat, pde_dw['n_cores_real'], seed, sat_weight=w)

        r_full, p_full = v3_grid_search(sat, GS_POS, gs_dem, ref_n3, pde_dw)
        r_nopde, p_nopde = v3_grid_search(sat, GS_POS, gs_dem, ref_n3, km_dw)
        r_norec, p_norec = v3_grid_search(sat, GS_POS, gs_dem, ref_n3, pde_u)
        r_flat = benchmark_cbdp_v3(sat, GS_POS, gs_dem, gamma=GAMMA,
                                   alpha=1.0, k_cores=3, core_cache=pde_dw)
        r_1core = benchmark_cbdp_v3(sat, GS_POS, gs_dem, gamma=GAMMA,
                                    alpha=0.0, k_cores=1, core_cache=pde_dw)

        per_seed['full_cbdp'].append(slim(r_full))
        per_seed['no_pde_snc'].append(slim(r_nopde))
        per_seed['no_dynamic_reconfig'].append(slim(r_norec))
        per_seed['no_snc_mesh_flat'].append(slim(r_flat))
        per_seed['no_snc_mesh_1core'].append(slim(r_1core))
        per_seed['cbdp_v2_ref'].append(slim(r_v2))
        per_seed['nearest3_ref'].append(slim(ref_n3))
        per_seed['greedy_ref'].append(slim(r_greedy))
        per_seed['shortest_ref'].append(slim(r_short))
        extra['n_cores_pde_uniform'].append(int(pde_u['n_cores_real']))
        extra['n_cores_pde_demandw'].append(int(pde_dw['n_cores_real']))
        extra['params_full'].append(p_full)
        extra['params_no_pde'].append(p_nopde)
        extra['params_no_reconfig'].append(p_norec)
        print(f"  seed={seed}: full={r_full['imbalance']:.3f} "
              f"-PDE={r_nopde['imbalance']:.3f} -Reconf={r_norec['imbalance']:.3f} "
              f"-Mesh(flat)={r_flat['imbalance']:.3f} -Mesh(1c)={r_1core['imbalance']:.3f} "
              f"cores(u/dw)={pde_u['n_cores_real']}/{pde_dw['n_cores_real']}")

    # ---- 24h dynamic reconfiguration (frozen vs re-detected cores) ----
    print(f"  [{label}] 24h dynamic reconfiguration (alpha=0.3, k_cores=3 fixed)")
    dyn = {'reconfig': [], 'frozen': []}
    for seed in SEEDS:
        sat = generate_network(N, HEIGHTS, seed=seed)
        frozen_cache = _detect_cores(sat, GAMMA, N)   # uniform, fixed for 24h
        imb_rec, imb_frz, dist_rec, dist_frz = [], [], [], []
        for t in range(24):
            dem_t = generate_time_varying_demand(GS_POS, gs_lat_lon_20, t, gs_dem)
            gamma_t = GAMMA * dem_t.sum() / gs_dem.sum()
            r_rec = benchmark_cbdp_v3_demand_weighted(sat, GS_POS, dem_t,
                                                      gamma=gamma_t, alpha=0.3, k_cores=3)
            r_frz = benchmark_cbdp_v3(sat, GS_POS, dem_t, gamma=GAMMA,
                                      alpha=0.3, k_cores=3, core_cache=frozen_cache)
            imb_rec.append(r_rec['imbalance']); dist_rec.append(r_rec['avg_dist_km'])
            imb_frz.append(r_frz['imbalance']); dist_frz.append(r_frz['avg_dist_km'])
        dyn['reconfig'].append({'imb_24h_mean': float(np.mean(imb_rec)),
                                'dist_24h_mean': float(np.mean(dist_rec))})
        dyn['frozen'].append({'imb_24h_mean': float(np.mean(imb_frz)),
                              'dist_24h_mean': float(np.mean(dist_frz))})
        print(f"    seed={seed}: reconfig 24h mean imb={np.mean(imb_rec):.3f} "
              f"frozen={np.mean(imb_frz):.3f}")

    # ---- Aggregate ----
    summary = {}
    for v in variants:
        imb = np.array([r['imbalance'] for r in per_seed[v]])
        dst = np.array([r['avg_dist_km'] for r in per_seed[v]])
        summary[v] = {'imb_mean': float(imb.mean()), 'imb_std': float(imb.std()),
                      'dist_mean': float(dst.mean()), 'dist_std': float(dst.std()),
                      'n_used_mean': float(np.mean([r['n_used'] for r in per_seed[v]])),
                      'imb_per_seed': imb.tolist(), 'dist_per_seed': dst.tolist()}

    imb_full = summary['full_cbdp']['imb_mean']
    deg = {v: float(100.0 * (summary[v]['imb_mean'] - imb_full) / imb_full)
           for v in ['no_pde_snc', 'no_dynamic_reconfig',
                     'no_snc_mesh_flat', 'no_snc_mesh_1core']}

    rec = np.array([d['imb_24h_mean'] for d in dyn['reconfig']])
    frz = np.array([d['imb_24h_mean'] for d in dyn['frozen']])
    dyn_summary = {
        'reconfig_imb_mean': float(rec.mean()), 'reconfig_imb_std': float(rec.std()),
        'frozen_imb_mean': float(frz.mean()), 'frozen_imb_std': float(frz.std()),
        'frozen_degradation_pct': float(100 * (frz.mean() - rec.mean()) / rec.mean()),
        'reconfig_dist_mean': float(np.mean([d['dist_24h_mean'] for d in dyn['reconfig']])),
        'frozen_dist_mean': float(np.mean([d['dist_24h_mean'] for d in dyn['frozen']])),
    }

    labels = {'full_cbdp': 'Full CBDP (PDE+reconfig+mesh)',
              'no_pde_snc': '-PDE SNC (k-means cores)',
              'no_dynamic_reconfig': '-Dynamic Reconfig (uniform)',
              'no_snc_mesh_flat': '-SNC Mesh (flat, alpha=1)',
              'no_snc_mesh_1core': '-SNC Mesh (1-core, a=0,k=1)',
              'cbdp_v2_ref': 'CBDP v2 (reference)',
              'nearest3_ref': 'Nearest-3 (reference)',
              'greedy_ref': 'Greedy (reference)',
              'shortest_ref': 'Shortest-Path (reference)'}
    print(f"\n  [{label}] SUMMARY (mean +/- std over {len(SEEDS)} seeds)")
    print(f"  {'Variant':<30} {'Imbalance':>17} {'Dist(km)':>17} {'Degr.%':>8}")
    print(f"  {'-'*74}")
    for v in variants:
        s = summary[v]
        d = (f"{deg[v]:>7.1f}%" if v in deg
             else ('       0%' if v == 'full_cbdp' else '       -'))
        print(f"  {labels[v]:<30} {s['imb_mean']:>7.2f} +/- {s['imb_std']:<6.2f} "
              f"{s['dist_mean']:>7.0f} +/- {s['dist_std']:<6.0f} {d}")
    print(f"  24h: reconfig {dyn_summary['reconfig_imb_mean']:.2f} vs frozen "
          f"{dyn_summary['frozen_imb_mean']:.2f} "
          f"(frozen degradation {dyn_summary['frozen_degradation_pct']:+.1f}%)")

    return {'demand_set': label, 'static_ablation': summary,
            'static_degradation_pct_vs_full': deg,
            'core_counts': extra,
            'dynamic_24h': {'per_seed': dyn, 'summary': dyn_summary}}


def main():
    out = {'config': {'N': N, 'heights': HEIGHTS, 'gamma': GAMMA,
                      'seeds': SEEDS, 'n_gs': len(GS_POS),
                      'demands_fallback': DEMANDS_FALLBACK.tolist(),
                      'demands_popweight': DEMANDS_POPWEIGHT.tolist(),
                      'alphas': ALPHAS, 'k_cores_vals': K_CORES_VALS,
                      'kmeans': 'sklearn KMeans(n_init=10, random_state=seed)'}}

    print("=" * 72)
    print("Step 0: Replication attempt of algorithm_v2_report.json N=1000 row")
    print("=" * 72)
    out['replication_check'] = [
        replication_check(DEMANDS_FALLBACK, 'fallback demands'),
        replication_check(DEMANDS_POPWEIGHT, 'population-weighted demands')]

    out['runs'] = [
        run_config(DEMANDS_FALLBACK, 'A: fallback demands'),
        run_config(DEMANDS_POPWEIGHT, 'B: population-weighted demands')]

    with open(os.path.join(SCRIPT_DIR, 'fig5_ablation_results.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print("\nSaved: fig5_ablation_results.json")


if __name__ == '__main__':
    main()
