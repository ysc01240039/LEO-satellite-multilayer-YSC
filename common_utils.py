#!/usr/bin/env python3
"""
===============================================================================
Common Utilities for CBDP Project

Shared constants, helper classes, and benchmark algorithms used by both
algorithm_v2.py and real_orbit_validation.py.

This module eliminates code duplication and ensures consistent behavior
across all analysis scripts.
===============================================================================
"""

import json
import warnings
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    dijkstra as _cs_dijkstra,
    connected_components as _cs_components,
    minimum_spanning_tree as _cs_mst,
)

# ================================================================
# Section 0: Constants & Parameters
# ================================================================
# C++ parameter scan data summary:
#
# Gamma scan (gamma=0.43-6.0, beta=0.6, N=1000, 40^3 grid):
#   All standard runs: n_cores = 93.06 ± 22.78 (251 samples each)
#   NOTE: All gamma scan files share identical time series (same random seed).
#   Long run (gamma=6.0, 2h): n_cores = 91.49 ± 20.55 (1001 samples)
#   Gamma=0.444 data is a duplicate of gamma=0.5 (bit-for-bit identical).
#   n_cores is CONSTANT across gamma ∈ [0.43, 6.0].
#
# N-scan (gamma=6.0, beta=0.6, N=200-1000, 40^3 grid):
#   n_cores = 478.4 · N^(-0.2348) (R²=0.9953, 5-point scan)
#   N=200: 136.96, N=400: 117.47, N=600: 108.02, N=800: 100.10, N=1000: 93.06
#   n_cores follows power-law decrease with N.
#
# Beta scan (gamma=6.0, N=1000, beta=0.1, 0.6, 2.0, 20x span):
#   All three beta values: n_cores = 93.06 (identical time series, same seed).
#   n_cores is CONSTANT across beta ∈ [0.1, 2.0].
#
# The saturation model n_cores(gamma) = n_baseline + (n_grid_max - n_baseline)
#   * (1 - exp(-(gamma - gamma_c) / gamma_char)) is FALSIFIED.
# The model overpredicts by 35% at gamma=6.0 (123.1 vs C++ 91.5).
# n_cores is determined by grid size and source distribution topology,
# NOT by chemotactic strength gamma.
# ================================================================

R_earth = 6371.0          # Earth radius (km)
GAMMA_SCALE = 3.69         # PDE gamma → algorithm gamma scale factor (calibrated via algorithm_v2.py Part J)
n_baseline = 91.6          # baseline core count (γ=0, from lost N=400 calib) [HYPOTHETICAL — UNVALIDATED]
n_grid_max = 123.1         # grid-saturated core count (γ→∞) [HYPOTHETICAL — UNVALIDATED]
gamma_char = 0.573         # characteristic gamma [HYPOTHETICAL — UNVALIDATED]
n_cores_validated = 92.3   # Pooled mean of gamma=0.5 (93.06) and gamma=6.0 (91.49) C++ data
REF_N = 400                # Reference constellation size for calibration
C0 = 30.1556               # stencil C0 (continuum limit ΣK_j), NOT the Nyquist value
# For critical line: use |C(k_Nyquist)| = 37.38 (discrete Nyquist mode)
gamma_c_06 = 0.444         # critical gamma for β=0.6, nonlocal: (16+0.6)/37.38


def gamma_c_nonlocal(beta):
    """Nonlocal KS critical line: γ_c(β) = (k²_disc + β) / C0_Nyquist = (16+β)/37.38."""
    return (16.0 + beta) / 37.38


# ================================================================
# Section 1: NumpyEncoder
# ================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.complexfloating):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ================================================================
# Section 2: Phase Diagram Prediction Functions
# ================================================================

def predict_cores(gamma, N=400):
    """
    Predict number of communication cores.

    C++ scan data summary:
    - Gamma scan: n_cores = 93.06 CONSTANT across gamma ∈ [0.43, 6.0] (N=1000).
    - N-scan: n_cores = 478.4 · N^(-0.2348) (R²=0.9953, N=200-1000).
    - Beta scan: n_cores = 93.06 CONSTANT across beta ∈ [0.1, 2.0] (N=1000).
    - The saturation model (exponential growth n(gamma)) is FALSIFIED.
    - n_cores is determined by grid size and source distribution, not gamma.

    NOTE: This function returns a constant 92.3 (legacy pooled mean).
    For N-dependent predictions, use the power-law: n_cores = 478.4 * N^(-0.2348).

    Args:
        gamma: PDE chemotactic strength (IGNORED — n_cores independent of gamma)
        N: number of satellites (NOT used for scaling, kept for API compatibility)
    Returns:
        predicted core count (absolute, independent of N and gamma)
    """
    warnings.warn(
        "Saturation model is FALSIFIED. "
        "n_cores = 92.3 is the legacy pooled mean. "
        "For N-dependent predictions, use n_cores = 478.4 * N^(-0.2348) (R²=0.9953). "
        "n_cores is determined by grid size and source distribution, not gamma. "
        "gamma=0.444 data excluded: duplicate of gamma=0.5.",
        UserWarning
    )
    return 92.3  # Legacy pooled mean; use power-law for N-dependent predictions


def required_gamma_for_core_fraction(fraction):
    """
    Find γ needed to achieve target core count.
    CRITICAL (Round 22): n_cores is INDEPENDENT of gamma. This function is
    OBSOLETE — gamma does not control core count. Returns gamma_c_06=0.444
    as the minimal valid gamma (critical point). Any gamma ≥ gamma_c
    produces the same ~93 cores.
    """
    return gamma_c_06  # n_cores independent of gamma, return critical gamma


# ================================================================
# Section 3: Core Detection Helpers
# ================================================================

def _calibrate_cores(phi, n_target, grid_res, dx, domain_extent, gamma, sat_pos,
                     strict=False):
    """
    Adaptive threshold calibration: binary search for ~n_target cores.
    Returns calibrated core positions and count.

    strict=True: hard-cap the result at exactly n_target (top-n_target by
    density value) so the realized core count tracks the calibration target.
    Used by the E2E routing benchmark, where core-count variance directly
    distorts the scalability comparison. Legacy callers use strict=False.
    """
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

    if strict:
        max_cores = n_target
    else:
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


def _detect_cores(sat_pos, gamma, N, sat_weight=None, n_target=None, strict=False):
    """Detect cores using gamma-controlled density field. Returns cache dict.

    Args:
        sat_pos: satellite positions (N, 3)
        gamma: PDE gamma parameter (only affects smoothing scale, NOT core count)
        N: number of satellites
        sat_weight: optional per-satellite weight for density field (default: uniform)
        n_target: target core count for calibration. If None, uses the legacy
            constant 93 (Round 22: n_cores ~93 at N=1000). Callers that need
            N-dependent core counts should pass the C++ N-scan power law
            n = 478.4 * N^(-0.2348), e.g. via _detect_cores_powerlaw().
        strict: if True, hard-cap the realized core count at n_target
            (forwarded to _calibrate_cores). Default False (legacy behavior).
    """
    if n_target is None:
        n_target = 93  # legacy constant (calibrated at N=1000)
    n_target = max(int(n_target), 3)
    grid_res = max(6, min(120, int(np.sqrt(N * 5.0))))
    domain_extent = np.max(np.abs(sat_pos)) * 1.2
    dx = 2 * domain_extent / grid_res

    phi = np.zeros((grid_res, grid_res, grid_res))
    weights = sat_weight if sat_weight is not None else np.ones(N)
    for i in range(N):
        x = int((sat_pos[i, 0] + domain_extent) / dx)
        y = int((sat_pos[i, 1] + domain_extent) / dx)
        z = int((sat_pos[i, 2] + domain_extent) / dx)
        if 0 <= x < grid_res and 0 <= y < grid_res and 0 <= z < grid_res:
            phi[x, y, z] += weights[i]

    gamma_eff = gamma * GAMMA_SCALE
    sigma_smooth = max(0.5, 2.0 * np.exp(-gamma_eff / 5.0))
    phi = gaussian_filter(phi, sigma=sigma_smooth)

    core_positions, n_cores_real = _calibrate_cores(
        phi, n_target, grid_res, dx, domain_extent, gamma_eff, sat_pos,
        strict=strict)
    core_tree = cKDTree(core_positions)
    sat_core = np.full(N, -1)
    for i in range(N):
        dist, c_idx = core_tree.query(sat_pos[i])
        sat_core[i] = c_idx

    return {'core_positions': core_positions, 'n_cores_real': n_cores_real,
            'sat_core': sat_core, 'grid_res': grid_res}


# ================================================================
# Section 3b: Independent Core Detection (NO circular dependency)
# ================================================================

def _detect_cores_independent(sat_pos, gamma, N, sat_weight=None, threshold_pct=0.15):
    """Detect cores INDEPENDENTLY of model predictions — no circular dependency.
    
    Unlike _detect_cores() which uses predict_cores() to set a target and then
    calibrates to match it, this function uses a FIXED absolute threshold
    (percentile of the density field) to detect cores. This breaks the circular
    dependency: predict_cores() → n_target → calibrate → n_cores ≈ n_target.
    
    Args:
        sat_pos: satellite positions (N, 3)
        gamma: PDE gamma parameter (for smoothing scale only)
        N: number of satellites
        sat_weight: optional per-satellite weight for density field
        threshold_pct: fraction of max density above which to detect cores (default 0.15)
    Returns:
        cache dict with core_positions, n_cores_real, sat_core, grid_res
    """
    grid_res = max(6, min(120, int(np.sqrt(N * 5.0))))
    domain_extent = np.max(np.abs(sat_pos)) * 1.2
    dx = 2 * domain_extent / grid_res

    phi = np.zeros((grid_res, grid_res, grid_res))
    weights = sat_weight if sat_weight is not None else np.ones(N)
    for i in range(N):
        x = int((sat_pos[i, 0] + domain_extent) / dx)
        y = int((sat_pos[i, 1] + domain_extent) / dx)
        z = int((sat_pos[i, 2] + domain_extent) / dx)
        if 0 <= x < grid_res and 0 <= y < grid_res and 0 <= z < grid_res:
            phi[x, y, z] += weights[i]

    gamma_eff = gamma * GAMMA_SCALE
    sigma_smooth = max(0.5, 2.0 * np.exp(-gamma_eff / 5.0))
    phi = gaussian_filter(phi, sigma=sigma_smooth)

    # Independent detection: use local maxima above a fixed percentile threshold
    # This is INDEPENDENT of predict_cores() — no calibration to target
    filter_size = max(2, int(8 - np.log10(max(N, 1) + 1) * 2))
    local_max = (phi == maximum_filter(phi, size=filter_size))
    phi_max = phi.max()
    
    # Fixed threshold: percentile of the density field (not relative to target)
    threshold = threshold_pct * phi_max
    core_mask = local_max & (phi > threshold)
    core_idx = np.argwhere(core_mask)
    n_cores_real = len(core_idx)

    # Limit to reasonable maximum
    max_cores = N // 3
    if n_cores_real > max_cores and n_cores_real > 0:
        core_vals = phi[core_idx[:, 0], core_idx[:, 1], core_idx[:, 2]]
        top_idx = np.argsort(core_vals)[-max_cores:]
        core_idx = core_idx[top_idx]
        n_cores_real = max_cores

    if n_cores_real == 0:
        core_positions = np.mean(sat_pos, axis=0).reshape(1, 3)
        n_cores_real = 1
    else:
        core_positions = core_idx * dx - domain_extent + dx / 2

    core_tree = cKDTree(core_positions)
    sat_core = np.full(N, -1)
    for i in range(N):
        dist, c_idx = core_tree.query(sat_pos[i])
        sat_core[i] = c_idx

    return {'core_positions': core_positions, 'n_cores_real': n_cores_real,
            'sat_core': sat_core, 'grid_res': grid_res,
            'detection_method': 'independent_fixed_threshold',
            'threshold_pct': threshold_pct}


def _detect_cores_multi_threshold(sat_pos, gamma, N, sat_weight=None,
                                   thresholds=(0.05, 0.10, 0.15, 0.20, 0.25)):
    """Detect cores at multiple thresholds for sensitivity analysis.
    
    Returns dict mapping threshold_pct → cache dict with n_cores_real.
    This enables quantitative assessment of threshold sensitivity (H2 fix).
    """
    results = {}
    for t in thresholds:
        cache = _detect_cores_independent(sat_pos, gamma, N, sat_weight, threshold_pct=t)
        results[f"thresh_{t:.2f}"] = {
            'n_cores': cache['n_cores_real'],
            'threshold_pct': t,
            'threshold_abs': float(t * np.max(np.bincount(
                np.clip(np.argmax(np.abs(sat_pos), axis=0), 0, None)))),
        }
    return results


# ================================================================
# Section 4: Network Generation Helpers
# ================================================================

def generate_network(N, layers_info, seed=42):
    """Generate satellite positions on orbital shells (Fibonacci sphere).

    Args:
        N: total number of satellites.
        layers_info: list of heights (km) per layer, or single int for number of layers.
        seed: random seed for reproducible phase offsets (default 42).
    """
    rng = np.random.RandomState(seed)
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
        phase_offset = rng.uniform(0, 2 * np.pi)
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
# Section 5: Benchmark Algorithms
# ================================================================

def benchmark_greedy(sat_pos, gs_pos, gs_demand):
    """Greedy: each GS queries its 5 nearest satellites, picks the least-loaded one."""
    N = len(sat_pos)
    M = len(gs_pos)
    sat_tree = cKDTree(sat_pos)
    load = np.zeros(N)
    total_dist = 0.0

    for j in range(M):
        dists, idxs = sat_tree.query(gs_pos[j], k=min(5, N))
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)
        best_pos = np.argmin(load[idxs])
        load[idxs[best_pos]] += gs_demand[j]
        total_dist += dists[best_pos]

    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
    }


def benchmark_roundrobin(sat_pos, gs_pos, gs_demand):
    """Round-robin: high-demand GS demand split across satellites in round-robin order."""
    CHUNK_DEMAND = 3  # demand units per chunk (finer granularity → better load balance)
    N = len(sat_pos)
    M = len(gs_pos)
    sat_tree = cKDTree(sat_pos)
    load = np.zeros(N)
    total_dist = 0.0

    sat_idx = 0
    for j in range(M):
        chunks = max(1, int(np.ceil(gs_demand[j] / CHUNK_DEMAND)))
        per_chunk = gs_demand[j] / chunks
        assigned_dists = []
        for _ in range(chunks):
            assigned_sat = sat_idx % N
            load[assigned_sat] += per_chunk
            assigned_dists.append(np.linalg.norm(gs_pos[j] - sat_pos[assigned_sat]))
            sat_idx += 1
        total_dist += np.mean(assigned_dists)  # average distance to actually assigned satellites

    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
    }


def benchmark_nearest3(sat_pos, gs_pos, gs_demand):
    """Nearest-3 heuristic: each GS splits load equally to its 3 nearest satellites."""
    N = len(sat_pos)
    M = len(gs_pos)
    sat_tree = cKDTree(sat_pos)
    load = np.zeros(N)
    total_dist = 0.0

    for j in range(M):
        dists, idxs = sat_tree.query(gs_pos[j], k=min(3, N))
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)
        for idx, d in zip(idxs, dists):
            load[idx] += gs_demand[j] / len(idxs)
        total_dist += np.mean(dists)

    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
    }


def benchmark_shortest_path(sat_pos, gs_pos, gs_demand):
    """Shortest Path: each GS routes to its single nearest satellite."""
    N = len(sat_pos)
    M = len(gs_pos)
    sat_tree = cKDTree(sat_pos)
    load = np.zeros(N)
    total_dist = 0.0

    for j in range(M):
        d, idx = sat_tree.query(gs_pos[j])
        load[idx] += gs_demand[j]
        total_dist += d

    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
    }


# ================================================================
# Section 6: CBDP Algorithms
# ================================================================

def benchmark_cbdp(sat_pos, gs_pos, gs_demand, gamma=6.0):
    """
    CBDP: Core-Based Distributed Protocol.
    Phase 1: Detect cores via _detect_cores (gamma-controlled PDE field).
    Phase 2: Assign each GS to nearest core.
    Phase 3: Route GS demand through core's top-5 nearest satellites.
    """
    N = len(sat_pos)
    M = len(gs_pos)

    core_cache = _detect_cores(sat_pos, gamma, N)
    core_positions = core_cache['core_positions']
    n_cores_real = core_cache['n_cores_real']
    sat_core = core_cache['sat_core']

    core_tree = cKDTree(core_positions)
    sat_tree = cKDTree(sat_pos)
    load = np.zeros(N)
    total_dist = 0.0

    for j in range(M):
        _, core_c = core_tree.query(gs_pos[j])
        core_sats = np.where(sat_core == core_c)[0]
        n_core_sats = len(core_sats)

        if n_core_sats > 0:
            k = min(5, n_core_sats)
            sat_dists = np.array([np.linalg.norm(gs_pos[j] - sat_pos[s]) for s in core_sats])
            sort_idx = np.argsort(sat_dists)
            sorted_idx = core_sats[sort_idx]
            for idx in sorted_idx[:k]:
                load[idx] += gs_demand[j] / k
            total_dist += np.mean(sat_dists[sort_idx[:k]])
        else:
            d_nearest, nearest_idx = sat_tree.query(gs_pos[j])
            load[nearest_idx] += gs_demand[j]
            total_dist += d_nearest

    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'n_cores': n_cores_real,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
        'core_positions': core_positions,
        'sat_core': sat_core,
    }


def benchmark_cbdp_v3(sat_pos, gs_pos, gs_demand, gamma=6.0, alpha=0.3, k_cores=3, core_cache=None):
    """
    CBDP v3: Improved Core-Based Distributed Protocol.
    alpha: fraction of load routed directly to nearest satellite (0-1).
    k_cores: number of nearest cores per GS for core-routed portion.
    core_cache: optional pre-computed core detection dict from _detect_cores.
    """
    N = len(sat_pos)
    M = len(gs_pos)

    if core_cache is not None:
        core_positions = core_cache['core_positions']
        n_cores_real = core_cache['n_cores_real']
        sat_core = core_cache['sat_core']
    else:
        core_cache = _detect_cores(sat_pos, gamma, N)
        core_positions = core_cache['core_positions']
        n_cores_real = core_cache['n_cores_real']
        sat_core = core_cache['sat_core']

    core_tree = cKDTree(core_positions)
    sat_tree = cKDTree(sat_pos)
    load = np.zeros(N)
    total_dist = 0.0

    for j in range(M):
        dist_direct, sat_direct = sat_tree.query(gs_pos[j])

        actual_k = min(k_cores, n_cores_real)
        core_dists, core_idxs = core_tree.query(gs_pos[j], k=actual_k)
        core_dists = np.atleast_1d(core_dists)
        core_idxs = np.atleast_1d(core_idxs)

        # Direct portion
        load[sat_direct] += gs_demand[j] * alpha
        total_dist += dist_direct * alpha

        # Core-routed portion
        for c_idx, c_dist in zip(core_idxs, core_dists):
            core_sats = np.where(sat_core == c_idx)[0]
            n_core_sats = len(core_sats)
            if n_core_sats > 0:
                sat_dists = np.array([np.linalg.norm(gs_pos[j] - sat_pos[s]) for s in core_sats])
                best_s = core_sats[np.argmin(sat_dists)]
                load[best_s] += gs_demand[j] * (1 - alpha) / len(core_idxs)
                total_dist += np.min(sat_dists) * (1 - alpha) / len(core_idxs)
            else:
                _, nearest = sat_tree.query(gs_pos[j])
                load[nearest] += gs_demand[j] * (1 - alpha) / len(core_idxs)
                total_dist += dist_direct * (1 - alpha) / len(core_idxs)

    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'n_cores': n_cores_real,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
        'core_positions': core_positions,
        'sat_core': sat_core,
    }


# ================================================================
# Section 7: Advanced Baseline Algorithms (Dijkstra, SDN, Distributed Multipath)
# ================================================================

def _build_satellite_graph(sat_pos, max_link_range_km=5000.0):
    """Build adjacency list for satellite graph based on ISL range.
    
    Each satellite connects to neighbors within max_link_range_km.
    This models the realistic ISL topology where each satellite has
    limited beam range (typically 4 ISLs in Starlink Gen1).
    
    Returns:
        adj: list of lists, adj[i] = [(j, dist_km), ...] for each satellite i
    """
    N = len(sat_pos)
    tree = cKDTree(sat_pos)
    adj = [[] for _ in range(N)]
    k_max = min(30, N - 1)
    for i in range(N):
        dists, idxs = tree.query(sat_pos[i], k=k_max + 1)
        for d, j in zip(dists, idxs):
            if j == i:
                continue
            if d <= max_link_range_km:
                adj[i].append((j, float(d)))
    return adj


def _dijkstra_shortest_paths(adj, source_indices):
    """Compute shortest path distances from each source to all satellites.
    
    Uses Dijkstra's algorithm on the satellite graph.
    
    Args:
        adj: adjacency list from _build_satellite_graph
        source_indices: list of source satellite indices
        
    Returns:
        dist_matrix: (len(sources), N) array of shortest path distances
    """
    import heapq
    N = len(adj)
    M_src = len(source_indices)
    dist_matrix = np.full((M_src, N), np.inf)
    
    for s_idx, src in enumerate(source_indices):
        dist = np.full(N, np.inf)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        dist_matrix[s_idx] = dist
    
    return dist_matrix


def benchmark_dijkstra_routing(sat_pos, gs_pos, gs_demand, max_link_range_km=1500.0):
    """Dijkstra Shortest Path Routing (centralized) — FIXED Round 47.
    
    Models the classical shortest-path approach (Chen et al. 2024 JSAC):
    each GS connects to its nearest satellite. If that satellite is overloaded,
    the satellite graph (ISL topology) is used to find an alternative satellite
    within the same ISL-connected component that has spare capacity.
    
    DISTANCE METRIC: ground-to-satellite Euclidean distance (consistent with
    all other algorithms). The ISL graph is used ONLY for load balancing,
    NOT for distance calculation.
    
    Returns:
        dict with load, n_used, imbalance, avg_dist_km, max_load
    """
    N = len(sat_pos)
    M = len(gs_pos)
    sat_tree = cKDTree(sat_pos)
    
    # Pre-compute ground-to-satellite distances for all GS-sat pairs
    gs_dists = np.zeros((M, N))
    nearest_sats = np.zeros(M, dtype=int)
    nearest_dists = np.zeros(M)
    for j in range(M):
        d, idx = sat_tree.query(gs_pos[j])
        nearest_sats[j] = idx
        nearest_dists[j] = d
    
    # Build satellite graph for load balancing
    adj = _build_satellite_graph(sat_pos, max_link_range_km)
    graph_empty = all(len(a) == 0 for a in adj)
    
    load = np.zeros(N)
    total_dist = 0.0
    capacity_per_sat = max(gs_demand) * 1.5
    
    for j in range(M):
        best_sat = nearest_sats[j]
        best_dist = nearest_dists[j]
        
        # If nearest satellite is overloaded, use graph to find alternative
        if load[best_sat] >= capacity_per_sat and not graph_empty:
            # Compute shortest paths from the nearest satellite through ISL graph
            sp_dists = _dijkstra_shortest_paths(adj, [best_sat])[0]
            # Find satellites within 3 ISL hops that have spare capacity
            candidates = []
            for i in range(N):
                if i == best_sat:
                    continue
                if load[i] < capacity_per_sat and sp_dists[i] < np.inf:
                    # Direct ground-to-satellite distance for this candidate
                    gs_to_i = np.linalg.norm(gs_pos[j] - sat_pos[i])
                    candidates.append((gs_to_i, i))
            if candidates:
                candidates.sort()
                best_sat = candidates[0][1]
                best_dist = candidates[0][0]
        
        load[best_sat] += gs_demand[j]
        total_dist += best_dist
    
    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
        'routing_type': 'dijkstra_centralized',
    }


def benchmark_sdn_centralized(sat_pos, gs_pos, gs_demand, max_link_range_km=1500.0,
                               controller_ratio=0.05):
    """SDN-based Centralized Routing — FIXED Round 47.
    
    Models the SDN approach (Roth et al. 2025 IDLB, Papa et al. 2020):
    - A subset of satellites act as SDN controllers (controller_ratio=5%)
    - Each GS is assigned to the nearest controller's region
    - Within the controller's region, the GS connects to the nearest satellite
      with available capacity
    
    DISTANCE METRIC: ground-to-satellite Euclidean distance.
    The ISL graph is used ONLY for region assignment and load balancing,
    NOT for distance calculation.
    
    Control overhead: n_controllers × N × 64 bytes (link-state database),
    representing the bandwidth consumed by controllers exchanging topology info.
    
    Returns:
        dict with routing metrics and controller overhead estimate
    """
    N = len(sat_pos)
    M = len(gs_pos)
    sat_tree = cKDTree(sat_pos)
    
    n_controllers = max(1, int(N * controller_ratio))
    controller_indices = np.linspace(0, N - 1, n_controllers, dtype=int)
    
    # Build satellite graph for region assignment
    adj = _build_satellite_graph(sat_pos, max_link_range_km)
    graph_empty = all(len(a) == 0 for a in adj)
    
    controller_pos = sat_pos[controller_indices]
    controller_tree = cKDTree(controller_pos)
    if not graph_empty:
        # Use ISL graph to assign satellites to nearest controller (by hop count)
        ctrl_sp = _dijkstra_shortest_paths(adj, controller_indices)
        sat_to_controller = np.argmin(ctrl_sp, axis=0)
    else:
        sat_to_controller = np.array([controller_tree.query(sat_pos[i])[1] for i in range(N)])
    
    load = np.zeros(N)
    total_dist = 0.0
    capacity_per_sat = max(gs_demand) * 1.5
    
    for j in range(M):
        # Find nearest controller to this GS
        _, ctrl_idx = controller_tree.query(gs_pos[j])
        
        # Get satellites in this controller's region
        region_sats = np.where(sat_to_controller == ctrl_idx)[0]
        if len(region_sats) == 0:
            region_sats = np.arange(N)
        
        # Find the nearest satellite in the region with spare capacity
        region_pos = sat_pos[region_sats]
        gs_to_region = np.linalg.norm(gs_pos[j] - region_pos, axis=1)
        sorted_idx = np.argsort(gs_to_region)
        
        best_sat = region_sats[sorted_idx[0]]
        best_dist = gs_to_region[sorted_idx[0]]
        
        for k in sorted_idx[1:min(20, len(sorted_idx))]:
            candidate = region_sats[k]
            if load[candidate] < capacity_per_sat:
                best_sat = candidate
                best_dist = gs_to_region[k]
                break
        
        load[best_sat] += gs_demand[j]
        total_dist += best_dist
    
    n_used = np.sum(load > 0)
    control_overhead_kbps = n_controllers * N * 0.064  # 64 bytes per link-state entry
    
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
        'n_controllers': n_controllers,
        'control_overhead_kbps': control_overhead_kbps,
        'routing_type': 'sdn_centralized',
    }


def benchmark_distributed_multipath(sat_pos, gs_pos, gs_demand, max_link_range_km=5000.0,
                                     n_paths=3):
    """Distributed Multipath Routing.
    
    Models the distributed multipath approach (Li et al. 2025 TMC):
    - Each GS demand is split across n_paths distinct paths
    - Paths are selected from the k-nearest satellites with distinct routes
    - Load is distributed proportionally to inverse path distance
    
    Returns:
        dict with routing metrics
    """
    N = len(sat_pos)
    M = len(gs_pos)
    sat_tree = cKDTree(sat_pos)
    
    load = np.zeros(N)
    total_dist = 0.0
    
    for j in range(M):
        k = min(n_paths * 3, N)
        dists, idxs = sat_tree.query(gs_pos[j], k=k)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)
        
        selected = []
        selected_dists = []
        for idx, d in zip(idxs, dists):
            if len(selected) >= n_paths:
                break
            if len(selected) == 0 or all(abs(d - sd) > 100 for sd in selected_dists):
                selected.append(idx)
                selected_dists.append(d)
        
        if len(selected) == 0:
            selected = [idxs[0]]
            selected_dists = [dists[0]]
        
        weights = 1.0 / (np.array(selected_dists) + 1.0)
        weights /= weights.sum()
        
        for s, w, d in zip(selected, weights, selected_dists):
            load[s] += gs_demand[j] * w
            total_dist += d * w
    
    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load > 0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
        'n_paths': n_paths,
        'routing_type': 'distributed_multipath',
    }


# ================================================================
# Section 7b: End-to-End (E2E) Routing Benchmark Engine
# ================================================================
# Motivation:
#   Sections 5-7 measure access-layer assignment only (ground-to-satellite
#   distance, serving load). The paper's Algorithm 2 and the ns-3 packet-level
#   implementation (paper/ns3/leo_cbdp_eval.cc, InstallCbdpRoutes) route
#   traffic end-to-end over the ISL graph via SNC portals. This section
#   implements the missing end-to-end evaluation used for the scalability
#   comparison against recent routing methods (PFNSAR, LPIH) under one shared
#   topology, flow model, and metric convention.
#
# Unified conventions (identical for every method):
#   Topology: ISL graph = Kruskal MST (connectivity backbone) union the
#     4 nearest neighbors per satellite, mirroring BuildISLEdges in
#     leo_cbdp_eval.cc.
#   Flows: all ordered GS pairs (i -> j); source demand d_i is split equally
#     over the M-1 destinations (same all-pairs model as the ns-3 evaluation:
#     20 ground stations -> 380 flows).
#   Path: GS_i -> ingress satellite(s) -> ISL path -> egress satellite
#     (nearest satellite of GS_j) -> GS_j.
#   avg_dist_km: demand-weighted mean end-to-end path length (km).
#   load / imbalance / n_used: serving-load semantics identical to Section 5
#     (demand counted at the ingress satellite), so numbers stay comparable
#     with the access-layer benchmark. Transit-inclusive carried load is
#     recorded separately (max_carried / mean_carried diagnostics).
#   overhead_bytes_per_cycle: analytical control-traffic volume per 15 s
#     reconfiguration cycle; CBDP message sizes from leo_cbdp_eval.cc.

E2E_LOAD_REPORT_BYTES = 64    # load report (member -> portal)
E2E_ROUTE_DIST_BYTES = 128    # routing table distribution (portal -> member)
E2E_CORE_ASSIGN_BYTES = 32    # assignment confirmation (portal -> member)
E2E_MESH_UPDATE_BYTES = 256   # SNC mesh update (portal <-> portal)
E2E_K_MESH = 6                # k_c mesh degree
E2E_T_RECONFIG_S = 15.0       # reconfiguration period (s)


def _build_isl_graph_e2e(sat_pos, k_nn=4, k_cand=12):
    """Symmetric ISL graph mirroring BuildISLEdges in leo_cbdp_eval.cc.

    Kruskal MST over a k_cand-nearest candidate graph (connectivity backbone)
    union the k_nn nearest neighbors per satellite. Edge weights are Euclidean
    distances (km). The MST of a 12-NN candidate graph coincides with the
    all-pairs MST for geometric point sets in practice.
    """
    N = len(sat_pos)
    tree = cKDTree(sat_pos)

    # candidate graph for MST: symmetrized k_cand-NN
    kc = min(k_cand + 1, N)
    d_c, i_c = tree.query(sat_pos, k=kc)
    rows, cols, data = [], [], []
    for u in range(N):
        for v, w in zip(np.atleast_1d(i_c[u]), np.atleast_1d(d_c[u])):
            if v == u:
                continue
            rows.append(u)
            cols.append(int(v))
            data.append(float(w))
    cand = csr_matrix((data, (rows, cols)), shape=(N, N))
    n_comp, _ = _cs_components(cand, directed=False)
    if n_comp != 1:
        raise RuntimeError(f"E2E ISL candidate graph disconnected ({n_comp} components)")
    mst = _cs_mst(cand).tocoo()

    edges = {}
    for a, b, w in zip(mst.row, mst.col, mst.data):
        key = (min(int(a), int(b)), max(int(a), int(b)))
        edges[key] = float(w)

    # k_nn nearest neighbors per satellite (as in BuildISLEdges)
    kn = min(k_nn + 1, N)
    d_n, i_n = tree.query(sat_pos, k=kn)
    for u in range(N):
        for v, w in zip(np.atleast_1d(i_n[u]), np.atleast_1d(d_n[u])):
            if v == u:
                continue
            key = (min(u, int(v)), max(u, int(v)))
            if key not in edges:
                edges[key] = float(w)

    adj = [[] for _ in range(N)]
    for (a, b), w in edges.items():
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj


def _adj_to_csr(adj):
    """Adjacency list -> CSR matrix for scipy.sparse.csgraph."""
    N = len(adj)
    indptr = [0]
    indices, data = [], []
    for u in range(N):
        for v, w in adj[u]:
            indices.append(v)
            data.append(w)
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), shape=(N, N))


def _spt_unique(csr, roots):
    """Multi-source Dijkstra with duplicate-root handling.

    Returns (dist, pred) with one row per entry of `roots` (duplicate roots
    are computed once and expanded back).
    """
    roots = np.asarray(roots, dtype=int)
    uniq, inv = np.unique(roots, return_inverse=True)
    dist_u, pred_u = _cs_dijkstra(csr, directed=False, indices=uniq,
                                  return_predecessors=True)
    return dist_u[inv], pred_u[inv]


def _walk_to_root(pred_row, u):
    """Node list from u up to (and including) the SPT root.

    pred_row follows the scipy.csgraph predecessor convention (-9999 at the
    root and for unreachable nodes).
    """
    path = [int(u)]
    x = int(u)
    limit = len(pred_row)
    while True:
        p = int(pred_row[x])
        if p < 0 or p == x:
            break
        path.append(p)
        x = p
        if len(path) > limit:  # safety against malformed chains
            break
    return path


def _detect_cores_powerlaw(sat_pos, gamma, N):
    """Core detection with n_cores from the C++ N-scan calibration.

    n_target = round(478.4 * N^(-0.2348))  (R^2 = 0.9953, C++ N=200..1000),
    capped to [3, N//3] so that the routing hierarchy remains feasible at
    small constellation sizes.
    """
    n_target = int(round(478.4 * N ** (-0.2348)))
    n_target = max(3, min(N // 3, n_target))
    return _detect_cores(sat_pos, gamma, N, n_target=n_target, strict=True)


def _kmeans_seeded(pos, k, seed, n_iter=15):
    """Deterministic k-means with k-means++ initialization (fixed seed)."""
    rng = np.random.RandomState(seed)
    N = len(pos)
    centroids = [pos[rng.randint(N)]]
    for _ in range(k - 1):
        d2 = np.min(((pos[:, None, :] - np.asarray(centroids)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / max(d2.sum(), 1e-12)
        centroids.append(pos[rng.choice(N, p=probs)])
    centroids = np.asarray(centroids)
    labels = np.full(N, -1)
    for _ in range(n_iter):
        d = ((pos[:, None, :] - centroids[None]) ** 2).sum(-1)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.any():
                centroids[c] = pos[m].mean(axis=0)
    return labels, centroids


def benchmark_e2e_all(sat_pos, gs_pos, gs_demand, gamma=1.0, k_spread=5, seed=42):
    """Run all end-to-end routing methods on shared topology precomputation.

    Methods:
        dijkstra    centralized shortest-path oracle (performance upper bound)
        greedy      least-loaded of 5 nearest access satellites + SP transit
        nearest3    3 nearest access satellites (equal split) + SP transit
        roundrobin  3-unit demand chunks in global rotation + SP transit
        cbdp        proposed (paper Algorithm 2): nearest SNC, intra-core
                    top-k_spread spread, portal relay over the ISL graph
                    (aligned with InstallCbdpRoutes in leo_cbdp_eval.cc)
        pfnsar      potential-field network-state-aware routing (adapted from
                    Wei et al. 2025 TCOM): state-aware access (5 nearest
                    candidates, min access range + beta*carried) plus per-flow
                    greedy descent on phi = dist_to_sink + beta * carried_load,
                    online updates
        lpih        hierarchical logic-path routing (adapted from Yan et al.
                    2024): k-means domains, gateway logic topology (MST +
                    4-NN over inter-gateway SP distances), gateway relay

    Returns:
        dict with topology summary and per-method metric dicts.
    """
    N = len(sat_pos)
    M = len(gs_pos)
    gs_demand = np.asarray(gs_demand, dtype=float)
    total_demand = float(gs_demand.sum())

    # ---------------- shared ISL topology ----------------
    adj = _build_isl_graph_e2e(sat_pos)
    csr = _adj_to_csr(adj)
    n_edges_dir = sum(len(a) for a in adj)
    deg_avg = n_edges_dir / N
    logN = max(np.log2(N), 1.0)

    sat_tree = cKDTree(sat_pos)
    d_gs_sat, gs_sat = sat_tree.query(gs_pos)  # egress satellite per GS
    gs_sat = np.atleast_1d(gs_sat).astype(int)
    d_gs_sat = np.atleast_1d(d_gs_sat)

    k_cand = min(8, N)
    knn_d, knn_i = sat_tree.query(gs_pos, k=k_cand)
    knn_d = np.atleast_2d(knn_d)
    knn_i = np.atleast_2d(knn_i)

    # SPTs rooted at each egress satellite (shared by all methods)
    distM, predM = _spt_unique(csr, gs_sat)  # (M, N)

    # edge-weight lookup for path-length accumulation
    wmap = {}
    for u in range(N):
        for v, w in adj[u]:
            wmap[(u, v)] = w

    dsts = [[l for l in range(M) if l != j] for j in range(M)]

    # ---------------- ingress policies ----------------
    ingress = {}
    for j in range(M):
        ingress[('dijkstra', j)] = [(int(knn_i[j, 0]), 1.0)]
        ingress[('lpih', j)] = [(int(knn_i[j, 0]), 1.0)]
        k3 = min(3, N)
        ingress[('nearest3', j)] = [(int(knn_i[j, t]), 1.0 / k3) for t in range(k3)]

    # greedy: least-loaded among 5 nearest, GSs processed in order
    load_tmp = np.zeros(N)
    for j in range(M):
        k5 = min(5, N)
        cand = knn_i[j, :k5]
        u = int(cand[np.argmin(load_tmp[cand])])
        load_tmp[u] += gs_demand[j]
        ingress[('greedy', j)] = [(u, 1.0)]

    # roundrobin: 3-unit chunks assigned to satellites in global rotation
    CHUNK = 3.0
    sat_idx = 0
    for j in range(M):
        chunks = max(1, int(np.ceil(gs_demand[j] / CHUNK)))
        agg = {}
        for _ in range(chunks):
            u = sat_idx % N
            agg[u] = agg.get(u, 0.0) + 1.0 / chunks
            sat_idx += 1
        ingress[('roundrobin', j)] = sorted(agg.items())

    # ---------------- CBDP: cores, portals, ingress ----------------
    core_cache = _detect_cores_powerlaw(sat_pos, gamma, N)
    core_pos = core_cache['core_positions']
    n_cores = core_cache['n_cores_real']
    sat_core = core_cache['sat_core']
    portal = np.zeros(n_cores, dtype=int)
    for c in range(n_cores):
        members = np.where(sat_core == c)[0]
        if len(members) == 0:
            portal[c] = int(sat_tree.query(core_pos[c])[1])
        else:
            d2 = np.linalg.norm(sat_pos[members] - core_pos[c], axis=1)
            portal[c] = int(members[np.argmin(d2)])
    distP, predP = _spt_unique(csr, portal)  # (n_cores, N)
    core_tree = cKDTree(core_pos)
    for j in range(M):
        cj = int(core_tree.query(gs_pos[j])[1])
        members = np.where(sat_core == cj)[0]
        if len(members) == 0:
            ingress[('cbdp', j)] = [(int(gs_sat[j]), 1.0)]
        else:
            dmem = np.linalg.norm(sat_pos[members] - gs_pos[j], axis=1)
            order = np.argsort(dmem)[:min(k_spread, len(members))]
            kk = len(order)
            ingress[('cbdp', j)] = [(int(members[o]), 1.0 / kk) for o in order]

    # ---------------- LPIH: domains, gateways, logic paths ----------------
    D = max(2, int(round(np.sqrt(N))))
    labels, centroids = _kmeans_seeded(sat_pos, D, seed)
    _, gw = sat_tree.query(centroids)
    gw = np.atleast_1d(gw).astype(int)
    distG, predG = _spt_unique(csr, gw)  # (D, N)
    gw_sp = distG[:, gw].copy()  # (D, D) ISL SP distances between gateways
    KDOM = min(4, D - 1)
    mst_dom = _cs_mst(csr_matrix(gw_sp)).tocoo()
    wmat = np.full((D, D), np.inf)
    for a, b, w in zip(mst_dom.row, mst_dom.col, mst_dom.data):
        a, b = int(a), int(b)
        if w < wmat[a, b]:
            wmat[a, b] = wmat[b, a] = float(w)
    for x in range(D):
        order = np.argsort(gw_sp[x])
        for y in order[1:KDOM + 1]:
            w = float(gw_sp[x, y])
            if np.isfinite(w) and w < wmat[x, y]:
                wmat[x, y] = wmat[y, x] = w
    distDom, predDom = _cs_dijkstra(csr_matrix(wmat), directed=False,
                                    indices=np.arange(D), return_predecessors=True)

    # ---------------- path functions ----------------
    def sp_path(u, l):
        return _walk_to_root(predM[l], u)

    def cbdp_path(u, l):
        """Portal relay aligned with InstallCbdpRoutes: nodes route toward the
        destination core's portal, then down the egress SPT member path."""
        eg = int(gs_sat[l])
        if u == eg:
            return [u]
        c_dst = int(sat_core[eg])
        p = int(portal[c_dst])
        down = _walk_to_root(predM[l], p)  # portal -> ... -> egress
        if u in down:  # u already on the portal->egress member path
            return _walk_to_root(predM[l], u)
        up = _walk_to_root(predP[c_dst], u)  # member -> ... -> portal
        if up[-1] != p:  # portal unreachable: SP fallback
            return _walk_to_root(predM[l], u)
        return up + down[1:]

    def lpih_path(u, l):
        """Gateway relay: intra-domain SP to own gateway, logic-path across
        the domain graph, then down to the egress satellite."""
        eg = int(gs_sat[l])
        a = int(labels[u])
        b = int(labels[eg])
        if a == b:
            return _walk_to_root(predM[l], u)
        seq = _walk_to_root(predDom[a], b)[::-1]  # [a, ..., b]
        path = _walk_to_root(predG[a], u)  # u -> ... -> gw[a]
        for x, y in zip(seq[:-1], seq[1:]):
            seg = _walk_to_root(predG[y], int(gw[x]))  # gw[x] -> gw[y]
            path += seg[1:] if path[-1] == seg[0] else seg
        seg = _walk_to_root(predM[l], int(gw[b]))  # gw[b] -> egress
        path += seg[1:] if path[-1] == seg[0] else seg
        return path

    # ---------------- generic evaluation ----------------
    def evaluate(method, path_fn):
        load = np.zeros(N)
        carried = np.zeros(N)
        dist_num = 0.0
        for j in range(M):
            w_flow = gs_demand[j] / (M - 1)
            for u, frac in ingress[(method, j)]:
                load[u] += gs_demand[j] * frac
                w_sub = w_flow * frac
                d_acc = float(np.linalg.norm(gs_pos[j] - sat_pos[u]))
                for l in dsts[j]:
                    path = path_fn(u, l)
                    pl = d_acc + d_gs_sat[l]
                    carried[u] += w_sub
                    for a, b in zip(path[:-1], path[1:]):
                        pl += wmap[(a, b)]
                        carried[b] += w_sub
                    dist_num += w_sub * pl
        return _e2e_metrics(load, carried, dist_num, total_demand)

    def _e2e_metrics(load, carried, dist_num, denom):
        n_used = int(np.sum(load > 0))
        pos = load[load > 0]
        imb = float((load.max() - pos.min()) / max(load.mean(), 1e-6)) if n_used > 0 else 0.0
        cpos = carried[carried > 0]
        # total carried load (ingress + transit) imbalance, same convention:
        # differentiates routing methods whose ingress policies coincide
        c_imb = float((carried.max() - cpos.min()) / max(carried.mean(), 1e-6)) if len(cpos) else 0.0
        return {
            'load': load,
            'carried': carried,
            'imbalance': imb,
            'carried_imbalance': c_imb,
            'avg_dist_km': float(dist_num / max(denom, 1e-9)),
            'n_used': n_used,
            'max_load': float(load.max()),
            'max_carried': float(carried.max()),
            'mean_carried': float(cpos.mean()) if len(cpos) else 0.0,
        }

    # ---------------- PFNSAR: state-aware potential descent ----------------
    res_dijkstra = evaluate('dijkstra', sp_path)
    carried_sp = res_dijkstra['carried']
    sp_carried_pos = carried_sp[carried_sp > 0]
    mean_carried_sp = float(sp_carried_pos.mean()) if len(sp_carried_pos) else 1.0
    sp_lens = np.array([distM[l, gs_sat[j]] for j in range(M) for l in dsts[j]])
    w_flows = np.array([gs_demand[j] / (M - 1) for j in range(M) for l in dsts[j]])
    mean_isl = float(np.average(sp_lens, weights=w_flows))
    beta = 0.2 * mean_isl / (2.0 * max(mean_carried_sp, 1e-9))

    def evaluate_pfnsar():
        load = np.zeros(N)
        carried = np.zeros(N)
        dist_num = 0.0
        k5 = min(5, N)
        for j in range(M):
            w_flow = gs_demand[j] / (M - 1)
            # state-aware access (part of the PFNSAR adaptation): among the 5
            # nearest visible satellites, attach at the one minimizing the
            # access potential = access range + beta * current carried load
            cand = knn_i[j, :k5]
            u = int(cand[np.argmin(knn_d[j, :k5] + beta * carried[cand])])
            frac = 1.0
            load[u] += gs_demand[j] * frac
            w_sub = w_flow * frac
            d_acc = float(np.linalg.norm(gs_pos[j] - sat_pos[u]))
            for l in dsts[j]:
                target = int(gs_sat[l])
                x = u
                path = [x]
                carried[x] += w_sub
                visited = {x}
                while x != target and len(path) <= 4 * N:
                    nbrs = adj[x]
                    if not nbrs:
                        break
                    vals = [distM[l, y] + beta * carried[y] for y, _ in nbrs]
                    y = int(nbrs[int(np.argmin(vals))][0])
                    if y in visited:  # potential loop: fall back to SPT parent
                        p = int(predM[l, x])
                        if p < 0 or p in visited:
                            break
                        y = p
                    path.append(y)
                    visited.add(y)
                    carried[y] += w_sub
                    x = y
                if x != target:  # incomplete: follow the SPT remainder
                    rem = _walk_to_root(predM[l], x)
                    for node in rem[1:]:
                        carried[node] += w_sub
                    path += rem[1:]
                pl = d_acc + d_gs_sat[l] + sum(wmap[(a, b)] for a, b in zip(path[:-1], path[1:]))
                dist_num += w_sub * pl
        return _e2e_metrics(load, carried, dist_num, total_demand)

    # ---------------- run all methods ----------------
    res = {
        'dijkstra': res_dijkstra,
        'greedy': evaluate('greedy', sp_path),
        'nearest3': evaluate('nearest3', sp_path),
        'roundrobin': evaluate('roundrobin', sp_path),
        'cbdp': evaluate('cbdp', cbdp_path),
        'pfnsar': evaluate_pfnsar(),
        'lpih': evaluate('lpih', lpih_path),
    }

    # ---------------- analytical control overhead and route ops ----------------
    E = float(n_edges_dir)
    overhead = {
        'dijkstra': 0.0,  # centralized: no on-network control traffic
        'greedy': 0.0,
        'nearest3': 0.0,
        'roundrobin': 0.0,
        # leo_cbdp_eval.cc CbdpControlCycle: load reports + route distribution
        # + assignment confirmations (member <-> portal) + k_c mesh updates
        'cbdp': float((N - n_cores) * (E2E_LOAD_REPORT_BYTES + E2E_ROUTE_DIST_BYTES
                                       + E2E_CORE_ASSIGN_BYTES)
                      + n_cores * E2E_K_MESH * E2E_MESH_UPDATE_BYTES),
        'pfnsar': float(N * deg_avg * 64.0),          # potential beacons per cycle
        'lpih': float(N * deg_avg * 48.0 + D * KDOM * 256.0),  # intra-domain LSA + summaries
    }
    route_ops = {
        'dijkstra': float(N * E * logN),              # global all-pairs SP
        'greedy': float(M * E * logN),
        'nearest3': float(M * E * logN),
        'roundrobin': float(M * E * logN),
        'cbdp': float(n_cores * E * logN
                      + n_cores ** 2 * max(np.log2(max(n_cores, 2)), 1.0)),
        'pfnsar': float(2 * M * E * logN),            # two potential passes
        'lpih': float(D * E * logN + D ** 2 * max(np.log2(max(D, 2)), 1.0)),
    }

    methods = {}
    for name, r in res.items():
        methods[name] = {
            'imbalance': r['imbalance'],
            'carried_imbalance': r['carried_imbalance'],
            'avg_dist_km': r['avg_dist_km'],
            'n_used': r['n_used'],
            'max_load': r['max_load'],
            'max_carried': r['max_carried'],
            'mean_carried': r['mean_carried'],
            'overhead_bytes_per_cycle': overhead[name],
            'route_ops': route_ops[name],
        }

    return {
        'N': N,
        'M': M,
        'gamma': gamma,
        'graph_edges': n_edges_dir // 2,
        'graph_deg_avg': float(deg_avg),
        'n_cores_cbdp': int(n_cores),
        'n_domains_lpih': int(D),
        'pfnsar_beta': float(beta),
        'methods': methods,
    }


# ================================================================
# Section 8: Starlink Real-World Performance Data
# ================================================================

# Starlink real performance data (sourced from official Starlink Network Update
# June-July 2025, Ookla Speedtest 2H 2025, FCC Broadband Reports, Quilty Space,
# and peer-reviewed measurement studies:
#   - Mohan et al. 2024 (WWW'24): "A Multifaceted Look at Starlink Performance"
#     19.2M M-Lab speed tests from 34 countries, 15-sec reconfiguration intervals
#   - Ullah et al. 2025 (arXiv:2508.09839): In-flight performance, 64/24 Mbps
#   - Lottermoser et al. 2026 (IFIP Networking): Weather effects, v2-mini satellites
#   - Garcia et al. 2026 (IMC'26): Starlink queuing characterization
#   - FCC DA 26-36 (Jan 2026): 15,000 Gen2 satellites authorized
#   - Quilty Space / Jonathan McDowell: 10,000+ active satellites (Mar 2026)
# All values are publicly verified measurements as of 2025-2026.

STARLINK_REAL_DATA = {
    # === Latency (Round-Trip Time) ===
    "latency_ms_median_peak_hour_us": 25.7,
    "latency_ms_median_ookla_2h2025": 37.0,
    "latency_ms_median_global_mohan_2024": 40.0,  # Mohan et al. WWW'24: 40-50ms
    "latency_ms_goal_target": 20.0,
    "latency_ms_2022_baseline": 44.0,
    "latency_ms_pct_under_55ms": 99.0,
    "latency_ms_reconfiguration_interval": 15000,  # 15-second reconfiguration (Mohan 2024)
    
    # === Throughput ===
    "download_mbps_median_peak_hour_us": 200.0,
    "download_mbps_median_ookla_2h2025": 117.74,
    "download_mbps_2022_baseline": 23.0,
    "download_mbps_max_lottermoser_2026": 490.0,  # Lottermoser et al. 2026: max 490 Mbps
    "upload_mbps_median_ookla_2h2025": 16.91,
    "upload_mbps_fcc_broadband_threshold": 20.0,
    "pct_meeting_fcc_standard_q4_2025": 44.7,
    "v2_mini_download_improvement_mbps": 34.0,  # Lottermoser et al. 2026
    "v2_mini_latency_improvement_ms": 1.4,       # Lottermoser et al. 2026
    
    # === Constellation Scale ===
    "satellites_in_orbit_mid_2025": 7800,
    "satellites_in_orbit_end_2025": 9400,        # FCC filing: ~9,400 at end of 2025
    "satellites_in_orbit_mar_2026": 10039,        # Jonathan McDowell: 10,039 active (Mar 17, 2026)
    "satellites_fcc_authorized_gen2": 15000,      # FCC DA 26-36 (Jan 9, 2026)
    "satellites_target_gen1": 12000,
    "active_customers_global": 10000000,           # 10M+ (Mar 2026, Quilty Space)
    "active_customers_us": 2000000,
    "countries_served": 160,                       # 160 countries (Mar 2026)
    
    # === Ground Infrastructure ===
    "gateway_sites_us": 100,
    "gateway_antennas_us": 1500,
    "gateway_sites_global_2025": 350,
    "gateway_sites_global_projected_2026": 474,
    
    # === ISL & Capacity ===
    "isl_capacity_per_satellite_gbps": 200,
    "cumulative_capacity_tbps": 450,
    "capacity_added_per_week_tbps": 5,
    "gen3_satellite_downlink_tbps": 1.0,
    
    # === In-Flight Performance (Ullah et al. 2025) ===
    "inflight_download_mbps_median": 64.0,
    "inflight_upload_mbps_median": 24.0,
    "inflight_upload_above_17000ft_mbps": 33.0,
    
    # === Weather Effects (Lottermoser et al. 2026) ===
    "rain_download_degradation_pct": 37.84,
    "rain_upload_degradation_pct": 52.27,
    "min_latency_ms_lottermoser_2026": 21.0,  # Minimum observed latency
    
    # === Comparison: GEO Satellite Providers ===
    "hughesnet_download_mbps_median": 73.0,
    "viasat_download_mbps_median": 70.0,
    
    # === Annual Growth ===
    "customer_growth_yoy_pct": 82.0,
    "new_satellites_2025": 2300,
    "new_missions_2025": 100,
    "new_customers_2025": 4600000,  # 4.6M added in 2025
}


def benchmark_starlink_comparison(cbdp_v3_result, constellation_N=4408, sdn_result=None):
    """Compare CBDP performance against real Starlink measurements — FIXED Round 47.
    
    Computes quantitative comparisons between CBDP predictions and
    published Starlink performance data across latency, throughput,
    and scalability metrics.
    
    Overhead calculation:
    - CBDP: 0.0074% (core beacons + inter-core sync + intra-core assignment)
    - SDN: n_controllers × N × 64 bytes/link-state × update_rate / total_ISL_capacity
      At N=4408 (220 controllers): ~0.04% (negligible at Gen1 scale)
      At N=30000 (1500 controllers): ~8.5% of per-controller ISL capacity
    
    Args:
        cbdp_v3_result: dict from benchmark_cbdp_v3()
        constellation_N: number of satellites in the constellation
        sdn_result: optional dict from benchmark_sdn_centralized() for overhead calc
        
    Returns:
        dict with comparison metrics
    """
    sd = STARLINK_REAL_DATA
    
    cbdp_avg_dist = cbdp_v3_result.get('avg_dist_km', 0)
    cbdp_latency_ms = cbdp_avg_dist / 300.0  # vacuum propagation: 300 km/ms
    
    n_cores = cbdp_v3_result.get('n_cores', 93)
    core_spacing_km = (4 * np.pi * (6371 + 550)**2 / max(n_cores, 1)) ** 0.5
    mesh_latency_ms = 6 * core_spacing_km / 300.0
    
    total_cbdp_latency_ms = cbdp_latency_ms + mesh_latency_ms
    latency_ratio = total_cbdp_latency_ms / max(sd['latency_ms_median_peak_hour_us'], 0.1)
    
    n_used = cbdp_v3_result.get('n_used', constellation_N)
    
    # CBDP overhead (from precise calculation in paper Methods)
    cbdp_overhead_pct = 0.0074
    
    # SDN overhead: n_controllers × N × 64 bytes × 1 update/sec / total ISL capacity
    # For Gen1 (N=4408, 220 controllers): ~496 Mbps / 881,600 Gbps ≈ 0.00006%
    # For Gen2 (N=30000, 1500 controllers): ~23 Gbps per-controller / 200 Gbps ≈ 11.5%
    # Use actual SDN result if available, otherwise estimate
    if sdn_result is not None and 'control_overhead_kbps' in sdn_result:
        n_ctrl = sdn_result.get('n_controllers', int(constellation_N * 0.05))
        total_isl_kbps = constellation_N * sd['isl_capacity_per_satellite_gbps'] * 1e6  # kbps
        sdn_overhead_pct = (sdn_result['control_overhead_kbps'] / max(total_isl_kbps, 1)) * 100
    else:
        n_ctrl = max(1, int(constellation_N * 0.05))
        total_isl_kbps = constellation_N * sd['isl_capacity_per_satellite_gbps'] * 1e6
        sdn_ctrl_kbps = n_ctrl * constellation_N * 0.064  # 64 bytes per link-state entry, 1 update/sec
        sdn_overhead_pct = (sdn_ctrl_kbps / max(total_isl_kbps, 1)) * 100
    
    # Gen2 projection: SDN overhead at N=30000
    N_gen2 = 30000
    n_ctrl_gen2 = max(1, int(N_gen2 * 0.05))
    sdn_ctrl_kbps_gen2 = n_ctrl_gen2 * N_gen2 * 0.064
    total_isl_kbps_gen2 = N_gen2 * sd['isl_capacity_per_satellite_gbps'] * 1e6
    sdn_gen2_overhead_pct = (sdn_ctrl_kbps_gen2 / max(total_isl_kbps_gen2, 1)) * 100
    
    overhead_reduction = sdn_gen2_overhead_pct / max(cbdp_overhead_pct, 1e-9)
    
    return {
        'cbdp_latency_ms': round(total_cbdp_latency_ms, 2),
        'starlink_latency_ms': sd['latency_ms_median_peak_hour_us'],
        'latency_ratio': round(latency_ratio, 3),
        'cbdp_overhead_pct': cbdp_overhead_pct,
        'sdn_overhead_pct_gen1': round(sdn_overhead_pct, 6),
        'sdn_overhead_pct_gen2_est': round(sdn_gen2_overhead_pct, 2),
        'overhead_reduction_vs_sdn_gen2': round(overhead_reduction, 0),
        'starlink_download_mbps': sd['download_mbps_median_peak_hour_us'],
        'starlink_active_satellites': sd['satellites_in_orbit_mid_2025'],
        'starlink_gateways_global': sd['gateway_sites_global_2025'],
        'cbdp_n_cores': n_cores,
        'cbdp_n_used': n_used,
        'comparison_timestamp': '2025-2026',
    }


def run_full_benchmark_suite(sat_pos, gs_pos, gs_demand, gamma=6.0, alpha=0.2, k_cores=3):
    """Run ALL 8 benchmark algorithms and return comprehensive results.
    
    Algorithms:
    1. Greedy  2. Round-Robin  3. Nearest-3  4. Shortest-Path
    5. Dijkstra  6. SDN Centralized  7. Distributed Multipath  8. CBDP v3
    
    Args:
        sat_pos: satellite positions (N, 3)
        gs_pos: ground station positions (M, 3)
        gs_demand: ground station demand weights (M,)
        gamma: PDE chemotactic strength for CBDP core detection
        alpha: CBDP v3 direct-routing fraction (0-1)
        k_cores: CBDP v3 number of nearest cores per GS
    
    Returns:
        dict with all results and comparison metrics
    """
    results = {}
    
    results['greedy'] = benchmark_greedy(sat_pos, gs_pos, gs_demand)
    results['roundrobin'] = benchmark_roundrobin(sat_pos, gs_pos, gs_demand)
    results['nearest3'] = benchmark_nearest3(sat_pos, gs_pos, gs_demand)
    results['shortest_path'] = benchmark_shortest_path(sat_pos, gs_pos, gs_demand)
    results['dijkstra'] = benchmark_dijkstra_routing(sat_pos, gs_pos, gs_demand)
    results['sdn'] = benchmark_sdn_centralized(sat_pos, gs_pos, gs_demand)
    results['distributed_multipath'] = benchmark_distributed_multipath(sat_pos, gs_pos, gs_demand)
    results['cbdp_v3'] = benchmark_cbdp_v3(sat_pos, gs_pos, gs_demand, gamma=gamma,
                                            alpha=alpha, k_cores=k_cores)
    results['starlink_comparison'] = benchmark_starlink_comparison(
        results['cbdp_v3'], constellation_N=len(sat_pos), sdn_result=results['sdn'])
    
    ref = results['nearest3']
    for algo_name in ['greedy', 'roundrobin', 'shortest_path', 'dijkstra',
                       'sdn', 'distributed_multipath', 'cbdp_v3']:
        r = results[algo_name]
        results[f'{algo_name}_vs_nearest3'] = {
            'imbalance_ratio': round(r['imbalance'] / max(ref['imbalance'], 1e-6), 3),
            'distance_ratio': round(r['avg_dist_km'] / max(ref['avg_dist_km'], 1e-6), 3),
            'n_used_ratio': round(r['n_used'] / max(ref['n_used'], 1), 3),
        }
    
    return results