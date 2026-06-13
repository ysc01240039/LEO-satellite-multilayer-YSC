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
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# Section 0: Constants & Parameters (from algorithm_v2.py)
# ================================================================

R_earth = 6371.0          # Earth radius (km)
GAMMA_SCALE = 13.6         # PDE gamma → algorithm gamma scale factor
n_baseline = 91.6          # baseline core count (N=400 reference)
n_grid_max = 123.1         # grid-saturated core count
gamma_char = 0.573         # characteristic gamma
C0 = 30.1556               # stencil C0
gamma_c_06 = 0.444         # critical gamma for β=0.6
REF_N = 400                # reference N for fraction-based model

# ================================================================
# Section 1: NumpyEncoder
# ================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ================================================================
# Section 2: Phase Diagram Prediction Functions
# ================================================================

def predict_cores(gamma, N=400):
    """Predict number of communication cores using the fitted PDE model."""
    n_baseline_frac = n_baseline / REF_N
    n_grid_max_frac = n_grid_max / REF_N
    n_frac = n_baseline_frac + (n_grid_max_frac - n_baseline_frac) * (1 - np.exp(-gamma / gamma_char))
    return min(n_frac * N, N)

def required_gamma_for_core_fraction(fraction):
    """Find γ needed to achieve target core fraction."""
    n_baseline_frac = n_baseline / REF_N
    n_grid_max_frac = n_grid_max / REF_N
    target_frac = min(fraction, n_grid_max_frac * 0.98)
    if target_frac <= n_baseline_frac:
        return 0.0
    if target_frac >= n_grid_max_frac * 0.98:
        return 20.0
    ratio = (n_grid_max_frac - target_frac) / (n_grid_max_frac - n_baseline_frac)
    if ratio <= 0:
        return 20.0
    gamma = -gamma_char * np.log(max(ratio, 1e-10))
    return gamma

# ================================================================
# Section 3: Core Detection Helpers (shared by CBDP v2/v3)
# ================================================================

def _calibrate_cores(phi, n_target, grid_res, dx, domain_extent, gamma, sat_pos):
    """Adaptive threshold calibration: binary search for ~n_target cores."""
    filter_size = max(2, int(8 - np.log10(max(n_target, 1) + 1) * 2))
    local_max = (phi == maximum_filter(phi, size=filter_size))
    phi_max = phi.max()

    lo, hi = 0.01 * phi_max, 0.95 * phi_max
    best_idx = None
    best_count = 0

    for _ in range(12):
        mid = (lo + hi) / 2
        core_mask = local_max & (phi > mid)
        core_idx = np.argwhere(core_mask)
        count = len(core_idx)

        if count < n_target * 0.4:
            hi = mid
        elif count > n_target * 2.5:
            lo = mid
        else:
            best_idx = core_idx
            best_count = count
            break

        if abs(count - n_target) < abs(best_count - n_target):
            best_idx = core_idx
            best_count = count

    if best_idx is None or best_count == 0:
        threshold = max(0.02, 0.5 * np.exp(-gamma / 5.0)) * phi_max
        core_mask = local_max & (phi > threshold)
        best_idx = np.argwhere(core_mask)
        best_count = len(best_idx)

    max_cores = max(n_target * 3, 10)
    if best_count > max_cores:
        core_vals = phi[best_idx[:, 0], best_idx[:, 1], best_idx[:, 2]]
        top_idx = np.argsort(core_vals)[-max_cores:]
        best_idx = best_idx[top_idx]
        best_count = max_cores

    if best_count == 0:
        return np.mean(sat_pos, axis=0).reshape(1, 3), 1

    core_positions = best_idx * dx - domain_extent + dx / 2
    return core_positions, best_count


def _detect_cores(sat_pos, gamma, N):
    """Detect cores using gamma-controlled density field."""
    n_target = int(predict_cores(gamma, N))
    n_target = max(n_target, 3)
    grid_res = max(6, min(120, int(np.sqrt(N * 5.0))))
    domain_extent = np.max(np.abs(sat_pos)) * 1.2
    dx = 2 * domain_extent / grid_res

    phi = np.zeros((grid_res, grid_res, grid_res))
    for i in range(N):
        x = int((sat_pos[i, 0] + domain_extent) / dx)
        y = int((sat_pos[i, 1] + domain_extent) / dx)
        z = int((sat_pos[i, 2] + domain_extent) / dx)
        if 0 <= x < grid_res and 0 <= y < grid_res and 0 <= z < grid_res:
            phi[x, y, z] += 1.0

    gamma_eff = gamma * GAMMA_SCALE
    sigma_smooth = max(0.5, 2.0 * np.exp(-gamma_eff / 5.0))
    phi = gaussian_filter(phi, sigma=sigma_smooth)

    core_positions, n_cores_real = _calibrate_cores(phi, n_target, grid_res, dx, domain_extent, gamma_eff, sat_pos)
    core_tree = cKDTree(core_positions)
    sat_core = np.full(N, -1)
    for i in range(N):
        dist, c_idx = core_tree.query(sat_pos[i])
        sat_core[i] = c_idx

    return {'core_positions': core_positions, 'n_cores_real': n_cores_real,
            'sat_core': sat_core, 'grid_res': grid_res}

# ================================================================
# Section 4: Benchmark Algorithms
# ================================================================

def benchmark_greedy(sat_pos, gs_pos, gs_demand):
    """Greedy: each GS assigned to nearest 3 least-loaded satellites."""
    N = len(sat_pos)
    M = len(gs_pos)
    sat_tree = cKDTree(sat_pos)
    load = np.zeros(N)
    total_dist = 0.0

    for j in range(M):
        dists, idxs = sat_tree.query(gs_pos[j], k=min(5, N))
        best = idxs[np.argmin(load[idxs])]
        load[best] += gs_demand[j]
        total_dist += np.linalg.norm(gs_pos[j] - sat_pos[best])

    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
    }


def benchmark_roundrobin(sat_pos, gs_pos, gs_demand):
    """Round-robin: equal distribution across all satellites."""
    N = len(sat_pos)
    M = len(gs_pos)
    load = np.zeros(N)
    total_dist = 0.0

    sat_idx = 0
    for j in range(M):
        chunks = max(1, int(np.ceil(gs_demand[j] / 3)))
        chunk_sats = []
        for _ in range(chunks):
            s = sat_idx % N
            chunk_sats.append(s)
            load[s] += gs_demand[j] / chunks
            sat_idx += 1
        unique_sats = set(chunk_sats)
        for s in unique_sats:
            weight = chunk_sats.count(s) / chunks
            total_dist += np.linalg.norm(gs_pos[j] - sat_pos[s]) * weight

    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load > 0].min()) / load.mean() if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
    }


def benchmark_nearest3(sat_pos, gs_pos, gs_demand):
    """Nearest-3 heuristic: each GS splits load equally to its 3 nearest satellites."""
    N = len(sat_pos)
    M = len(gs_pos)
    load = np.zeros(N)
    total_dist = 0.0

    for j in range(M):
        dists = np.array([np.linalg.norm(gs_pos[j] - sat_pos[i]) for i in range(N)])
        best_idxs = np.argsort(dists)[:3]
        for idx in best_idxs:
            load[idx] += gs_demand[j] / 3
        total_dist += np.mean(dists[best_idxs])

    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load > 0].min()) / load.mean() if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
    }


def benchmark_shortest_path(sat_pos, gs_pos, gs_demand):
    """Shortest Path: each GS routes to its single nearest satellite."""
    N = len(sat_pos)
    M = len(gs_pos)
    load = np.zeros(N)
    total_dist = 0.0

    for j in range(M):
        dists = np.array([np.linalg.norm(gs_pos[j] - sat_pos[i]) for i in range(N)])
        best = np.argmin(dists)
        load[best] += gs_demand[j]
        total_dist += dists[best]

    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
    }


def benchmark_cbdp_v2(sat_pos, gs_pos, gs_demand, gamma=6.0):
    """CBDP v2: Core-Based Distributed Protocol with gamma-controlled cores."""
    N = len(sat_pos)
    M = len(gs_pos)

    n_target = int(predict_cores(gamma, N))
    n_target = max(n_target, 3)

    grid_res = max(10, min(50, int(np.sqrt(N * 2.5))))
    domain_extent = np.max(np.abs(sat_pos)) * 1.2
    dx = 2 * domain_extent / grid_res

    phi = np.zeros((grid_res, grid_res, grid_res))
    for i in range(N):
        x = int((sat_pos[i, 0] + domain_extent) / dx)
        y = int((sat_pos[i, 1] + domain_extent) / dx)
        z = int((sat_pos[i, 2] + domain_extent) / dx)
        if 0 <= x < grid_res and 0 <= y < grid_res and 0 <= z < grid_res:
            phi[x, y, z] += 1.0

    gamma_eff = gamma * GAMMA_SCALE
    sigma_smooth = max(0.5, 2.0 * np.exp(-gamma_eff / 5.0))
    phi = gaussian_filter(phi, sigma=sigma_smooth)

    core_positions, n_cores_real = _calibrate_cores(phi, n_target, grid_res, dx, domain_extent, gamma_eff, sat_pos)

    core_tree = cKDTree(core_positions)
    sat_core = np.full(N, -1)
    for i in range(N):
        dist, c_idx = core_tree.query(sat_pos[i])
        sat_core[i] = c_idx

    load = np.zeros(N)
    total_latency = 0.0

    for j in range(M):
        _, core_c = core_tree.query(gs_pos[j])
        core_sats = np.where(sat_core == core_c)[0]
        n_core_sats = len(core_sats)

        if n_core_sats > 0:
            k = min(5, n_core_sats)
            sat_dists = np.array([np.linalg.norm(gs_pos[j] - sat_pos[s]) for s in core_sats])
            sorted_idx = core_sats[np.argsort(sat_dists)]
            for idx in sorted_idx[:k]:
                load[idx] += gs_demand[j] / k
            total_latency += np.mean(sat_dists[np.argsort(sat_dists)][:k])
        else:
            sat_tree_all = cKDTree(sat_pos)
            d_nearest, nearest_idx = sat_tree_all.query(gs_pos[j])
            load[nearest_idx] += gs_demand[j]
            total_latency += d_nearest

    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'n_cores': n_cores_real,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_latency / M,
        'max_load': load.max(),
    }


def benchmark_cbdp_v3(sat_pos, gs_pos, gs_demand, gamma=6.0, alpha=0.3, k_cores=3, core_cache=None):
    """
    CBDP v3: Improved Core-Based Distributed Protocol.
    alpha: fraction of load routed directly (0-1)
    k_cores: number of nearest cores per GS for core-routed portion
    core_cache: optional pre-computed core detection dict
    """
    N = len(sat_pos)
    M = len(gs_pos)

    if core_cache is not None:
        core_positions = core_cache['core_positions']
        n_cores_real = core_cache['n_cores_real']
        sat_core = core_cache['sat_core']
    else:
        n_target = int(predict_cores(gamma, N))
        n_target = max(n_target, 3)

        grid_res = max(12, min(60, int(np.sqrt(N * 2.5))))
        domain_extent = np.max(np.abs(sat_pos)) * 1.2
        dx = 2 * domain_extent / grid_res

        phi = np.zeros((grid_res, grid_res, grid_res))
        for i in range(N):
            x = int((sat_pos[i, 0] + domain_extent) / dx)
            y = int((sat_pos[i, 1] + domain_extent) / dx)
            z = int((sat_pos[i, 2] + domain_extent) / dx)
            if 0 <= x < grid_res and 0 <= y < grid_res and 0 <= z < grid_res:
                phi[x, y, z] += 1.0

        gamma_eff = gamma * GAMMA_SCALE
        sigma_smooth = max(0.5, 2.0 * np.exp(-gamma_eff / 5.0))
        phi = gaussian_filter(phi, sigma=sigma_smooth)

        core_positions, n_cores_real = _calibrate_cores(phi, n_target, grid_res, dx, domain_extent, gamma_eff, sat_pos)

        core_tree_temp = cKDTree(core_positions)
        sat_core = np.full(N, -1)
        for i in range(N):
            dist, c_idx = core_tree_temp.query(sat_pos[i])
            sat_core[i] = c_idx

    core_tree = cKDTree(core_positions)
    sat_tree = cKDTree(sat_pos)
    load = np.zeros(N)
    total_latency = 0.0

    for j in range(M):
        dist_direct, sat_direct = sat_tree.query(gs_pos[j])

        actual_k = min(k_cores, n_cores_real)
        core_dists, core_idxs = core_tree.query(gs_pos[j], k=actual_k)
        core_dists = np.atleast_1d(core_dists)
        core_idxs = np.atleast_1d(core_idxs)

        load[sat_direct] += gs_demand[j] * alpha
        total_latency += dist_direct * alpha

        for c_idx, c_dist in zip(core_idxs, core_dists):
            core_sats = np.where(sat_core == c_idx)[0]
            n_core_sats = len(core_sats)
            if n_core_sats > 0:
                sat_dists = np.array([np.linalg.norm(gs_pos[j] - sat_pos[s]) for s in core_sats])
                best_s = core_sats[np.argmin(sat_dists)]
                load[best_s] += gs_demand[j] * (1 - alpha) / len(core_idxs)
                total_latency += np.min(sat_dists) * (1 - alpha) / len(core_idxs)
            else:
                _, nearest = sat_tree.query(gs_pos[j])
                load[nearest] += gs_demand[j] * (1 - alpha) / len(core_idxs)
                total_latency += dist_direct * (1 - alpha) / len(core_idxs)

    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'n_cores': n_cores_real,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_latency / M,
        'max_load': load.max(),
        'core_positions': core_positions,
        'sat_core': sat_core,
    }

# ================================================================
# Section 5: Network Generation (Fibonacci Sphere)
# ================================================================

def generate_network(N, layers_info):
    """Generate satellite positions on orbital shells (Fibonacci sphere)."""
    if isinstance(layers_info, list):
        n_layers = len(layers_info)
        heights = layers_info
    else:
        n_layers = layers_info
        heights = [layers_info] * n_layers

    sats_per_layer = [N // n_layers] * n_layers
    sats_per_layer[-1] += N - sum(sats_per_layer)

    positions = []
    for l_idx, (h, n_s) in enumerate(zip(heights, sats_per_layer)):
        r = R_earth + h
        phi_golden = np.pi * (3 - np.sqrt(5))
        phase_offset = np.random.uniform(0, 2 * np.pi)
        for i in range(n_s):
            y = 1 - (i / max(n_s - 1, 1)) * 2
            radius_at_y = np.sqrt(1 - y * y)
            theta = phi_golden * i + phase_offset
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            positions.append([x * r, y * r, z * r])

    return np.array(positions)


def latlon_to_cart(lat, lon, r=R_earth):
    """Convert latitude/longitude to Cartesian coordinates."""
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    x = r * np.cos(lat_r) * np.cos(lon_r)
    y = r * np.cos(lat_r) * np.sin(lon_r)
    z = r * np.sin(lat_r)
    return np.array([x, y, z])

# ================================================================
# Section 6: Load Real Orbit Data
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
    r = benchmark_cbdp_v2(sat_pos, gs_pos, gs_demand, gamma=gamma_opt)
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

    # Set random seed for reproducibility
    np.random.seed(42)

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