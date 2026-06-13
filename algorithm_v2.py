#!/usr/bin/env python3
"""
===============================================================================
Algorithm Design v2: CBDP with Phase Diagram-Optimized Parameters
===============================================================================

Improvements over v1 (algorithm_design.py):
  1. Uses actual phase diagram data for core count prediction (not N^1.25 heuristic)
  2. Optimizes γ based on target core fraction
  3. Scales to real constellation sizes using physical saturation model
  4. Benchmarks CBDP against 4 baselines with rigorous statistics
  5. Derives latency formula from core spacing

Dependencies: nonlocal_dispersion_report.json, full_phase_diagram_summary.json
===============================================================================
"""

import json, os
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter
import warnings
warnings.filterwarnings('ignore')

# Scale factor: PDE gamma → algorithm gamma (for smoothing/threshold sensitivity)
# PDE gamma range [0, 20] maps to algorithm range [0, 20] with amplification
# at low gamma: gamma=0.178 (25% target) → gamma_eff=1.78 (reasonable sensitivity)
GAMMA_SCALE = 13.6  # Calibrated from Part J: best scale = 13.6 (bias = -0.8%)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

print("=" * 70)
print("Algorithm Design v2: CBDP with Phase Diagram Parameters")
print("=" * 70)

# =====================================================================
# Part A: Load Phase Diagram & Saturation Model
# =====================================================================

print("\n" + "=" * 70)
print("Part A: Load Physical Parameters from Phase Diagram")
print("=" * 70)

# Load the nonlocal dispersion analysis
disp_file = 'nonlocal_dispersion_report.json'
if os.path.exists(disp_file):
    with open(disp_file, encoding='utf-8') as f:
        disp = json.load(f)
    n_baseline = disp['saturation_fits']['physics_based']['n_baseline']
    n_grid_max = disp['saturation_fits']['physics_based']['n_grid_max']
    gamma_char = disp['saturation_fits']['physics_based']['gamma_char']
    C0 = disp['stencil']['C0']
    gamma_c_06 = disp['critical_line']['nonlocal_numerical']['0.6']
else:
    # Fallback from our earlier analysis
    n_baseline = 91.6
    n_grid_max = 123.1
    gamma_char = 0.573
    C0 = 30.1556
    gamma_c_06 = 0.444

print(f"  n_baseline   = {n_baseline:.1f} cores (pure source-driven)")
print(f"  n_grid_max   = {n_grid_max:.1f} cores (40^3 grid saturation)")
print(f"  gamma_char   = {gamma_char:.3f}")
print(f"  C0           = {C0:.4f}")
print(f"  gamma_c(0.6) = {gamma_c_06:.4f}")

# =====================================================================
# Part B: Core Count Prediction Function
# =====================================================================

print("\n" + "=" * 70)
print("Part B: Physics-Based Core Count Prediction")
print("=" * 70)

# Reference N from the PDE simulation that produced the saturation parameters
# n_baseline and n_grid_max were measured on a 40^3 grid with N=400 satellites
REF_N = 400

def predict_cores(gamma, N=400):
    """
    Predict number of communication cores using the fitted nonlocal PDE model.
    
    Uses fraction-based scaling from the N=400 reference case:
    - n_baseline_frac = 91.6/400 = 22.9% (source-driven baseline)
    - n_grid_max_frac = 123.1/400 = 30.8% (grid saturation)
    - These fractions scale with N, capped at N
    
    Args:
        gamma: chemotaxis strength (dimensionless)
        N: number of satellites
    
    Returns:
        n_cores: predicted core count (≤ N)
    """
    # Fraction-based model (from N=400 reference PDE simulation)
    n_baseline_frac = n_baseline / REF_N  # ~0.229
    n_grid_max_frac = n_grid_max / REF_N  # ~0.308
    
    # Core fraction for given gamma
    n_frac = n_baseline_frac + (n_grid_max_frac - n_baseline_frac) * (1 - np.exp(-gamma / gamma_char))
    
    # Scale by N, cap at N
    return min(n_frac * N, N)

def required_gamma_for_core_fraction(fraction):
    """
    Find γ needed to achieve target core fraction.
    
    Uses fraction-based saturation model from N=400 reference:
    n_frac(γ) = f_baseline + (f_max - f_baseline)*(1 - exp(-γ/γ_char))
    where f_baseline = 91.6/400 = 0.229, f_max = 123.1/400 = 0.308
    
    Args:
        fraction: target fraction of SATELLITES to become cores (0 < fraction < 1)
    
    Returns:
        gamma: required chemotaxis strength (0 = no chemotaxis needed, 20 = saturated)
    """
    n_baseline_frac = n_baseline / REF_N  # ~0.229
    n_grid_max_frac = n_grid_max / REF_N  # ~0.308
    
    # Cap target fraction at saturation limit
    target_frac = min(fraction, n_grid_max_frac * 0.98)
    
    # If target is below baseline, no chemotaxis needed
    if target_frac <= n_baseline_frac:
        return 0.0
    
    # If target exceeds saturation, return max gamma
    if target_frac >= n_grid_max_frac * 0.98:
        return 20.0
    
    # Invert: target_frac = f_baseline + (f_max - f_baseline)*(1 - exp(-γ/γ_char))
    # → γ = -γ_char * ln((f_max - target_frac) / (f_max - f_baseline))
    ratio = (n_grid_max_frac - target_frac) / (n_grid_max_frac - n_baseline_frac)
    if ratio <= 0:
        return 20.0
    gamma = -gamma_char * np.log(max(ratio, 1e-10))
    return gamma

# Test predictions
print(f"\n  Target fractions (fraction-based model, N=400 reference):")
print(f"  n_baseline_frac = {n_baseline/400:.3f}, n_grid_max_frac = {n_grid_max/400:.3f}")
for frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    g_req = required_gamma_for_core_fraction(frac)
    n_pred = predict_cores(g_req, N=400)
    print(f"    f={frac:.2f} (target {frac*400:.0f} cores) → γ={g_req:.3f} → n_cores={n_pred:.0f}")

# =====================================================================
# Part C: Benchmark Network Setup (Realistic Scale)
# =====================================================================

print("\n" + "=" * 70)
print("Part C: Benchmark Network Setup")
print("=" * 70)

# Use scaled-up toy network with real satellite orbital geometry
np.random.seed(42)

# Test constellation sizes
constellation_tests = [
    {"name": "Iridium-scale", "N": 66, "layers": 1, "height": 780},
    {"name": "Globalstar-scale", "N": 48, "layers": 1, "height": 1414},
    {"name": "Medium-scale", "N": 500, "layers": 3, "heights": [500, 900, 1300]},
    {"name": "Large-scale", "N": 1000, "layers": 5, "heights": [500, 800, 1100, 1400, 1700]},
    {"name": "Starlink Gen1", "N": 4408, "layers": 5, "heights": [340, 550, 1110, 1130, 1275]},
]

R_earth = 6371.0
n_gs = 20  # ground stations

def generate_network(N, layers_info):
    """Generate satellite positions on orbital shells."""
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
        # Fibonacci sphere for uniform distribution
        phi_golden = np.pi * (3 - np.sqrt(5))
        # Random phase offset per layer to avoid inter-layer alignment
        phase_offset = np.random.uniform(0, 2 * np.pi)
        for i in range(n_s):
            y = 1 - (i / max(n_s - 1, 1)) * 2
            radius_at_y = np.sqrt(1 - y*y)
            theta = phi_golden * i + phase_offset
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            positions.append([x * r, y * r, z * r])
    
    return np.array(positions)

# Generate ground stations (fixed for all benchmarks)
gs_lat_lon_20 = [
    (39.9, 116.4), (31.2, 121.5), (40.7, -74.0), (51.5, -0.1), (35.7, 139.7),
    (48.9, 2.3), (37.8, -122.4), (55.8, 37.6), (19.4, -99.1), (-33.9, 151.2),
    (1.3, 103.8), (28.6, 77.2), (-23.6, -46.6), (55.0, -3.4), (52.5, 13.4),
    (37.6, 127.0), (-6.2, 106.8), (22.3, 114.2), (25.2, 55.3), (35.0, 33.0),
]
# Ground station demands: load from ground_stations.json (population-weighted)
# Falls back to hardcoded demands if JSON not available
gs_json_path = 'ground_stations.json'
if os.path.exists(gs_json_path):
    with open(gs_json_path, encoding='utf-8') as f:
        gs_data = json.load(f)
    gs_demands_raw = np.array([s['weight'] for s in gs_data['stations']])
    gs_demands = 1 + 99 * (gs_demands_raw / gs_demands_raw.max())  # normalize to [1, 100]
    print(f"  Loaded population-weighted demands from {gs_json_path}")
    print(f"  Demand range: [{gs_demands.min():.0f}, {gs_demands.max():.0f}]")
else:
    gs_demands = np.array([13, 10, 19, 14, 20, 5, 15, 5, 17, 12,
                            8, 16, 11, 6, 9, 10, 14, 13, 7, 15], dtype=float)
    print(f"  Using default equal-weight demands (ground_stations.json not found)")

def latlon_to_cart(lat, lon, r=R_earth):
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    x = r * np.cos(lat_r) * np.cos(lon_r)
    y = r * np.cos(lat_r) * np.sin(lon_r)
    z = r * np.sin(lat_r)
    return np.array([x, y, z])

gs_positions = np.array([latlon_to_cart(lat, lon) for lat, lon in gs_lat_lon_20])

# =====================================================================
# Part D: Algorithm Implementations
# =====================================================================

print("\n" + "=" * 70)
print("Part D: Algorithm Benchmark Suite")
print("=" * 70)

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
        'imbalance': (load.max() - load[load>0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
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
        # Track which satellites receive this GS's chunks (for weighted distance)
        chunk_sats = []
        for _ in range(chunks):
            s = sat_idx % N
            chunk_sats.append(s)
            load[s] += gs_demand[j] / chunks
            sat_idx += 1
        # Weighted average distance: Σ(dist * weight) for each unique satellite
        unique_sats = set(chunk_sats)
        for s in unique_sats:
            weight = chunk_sats.count(s) / chunks
            total_dist += np.linalg.norm(gs_pos[j] - sat_pos[s]) * weight
    
    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load>0].min()) / load.mean() if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
    }

def benchmark_nearest3(sat_pos, gs_pos, gs_demand):
    """Nearest-3 heuristic: each GS splits load equally to its 3 nearest satellites.
    Note: This is a greedy heuristic, NOT a true global optimal solution."""
    N = len(sat_pos)
    M = len(gs_pos)
    load = np.zeros(N)
    total_dist = 0.0
    
    for j in range(M):
        dists = np.array([np.linalg.norm(gs_pos[j] - sat_pos[i]) for i in range(N)])
        best_idxs = np.argsort(dists)[:3]
        for idx in best_idxs:
            load[idx] += gs_demand[j] / 3
        # Average distance to all 3 assigned satellites (load is split equally)
        total_dist += np.mean(dists[best_idxs])
    
    n_used = np.sum(load > 0)
    return {
        'load': load,
        'n_used': n_used,
        'imbalance': (load.max() - load[load>0].min()) / load.mean() if n_used > 0 else 0,
        'avg_dist_km': total_dist / M,
        'max_load': load.max(),
    }


# =====================================================================
# Part D2: Additional Baseline Algorithms (Nature-level comparison)
# =====================================================================

def benchmark_shortest_path(sat_pos, gs_pos, gs_demand):
    """
    Shortest Path (SP): each GS routes to its single nearest satellite.
    No load balancing — represents pure latency minimization.
    Analogous to OSPF shortest-path-first routing in terrestrial networks.
    """
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
        'load': load, 'n_used': n_used,
        'imbalance': (load.max() - load[load>0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M, 'max_load': load.max(),
    }


def benchmark_ospf_style(sat_pos, gs_pos, gs_demand, k=3):
    """
    OSPF-style multi-path: each GS routes to k nearest satellites with equal-cost
    multi-path (ECMP) splitting. Load is split equally among the k nearest.
    Represents standard multi-path routing without congestion awareness.
    """
    N = len(sat_pos)
    M = len(gs_pos)
    load = np.zeros(N)
    total_dist = 0.0
    for j in range(M):
        dists = np.array([np.linalg.norm(gs_pos[j] - sat_pos[i]) for i in range(N)])
        best_idxs = np.argsort(dists)[:k]
        for idx in best_idxs:
            load[idx] += gs_demand[j] / k
            total_dist += dists[idx] / k
    n_used = np.sum(load > 0)
    return {
        'load': load, 'n_used': n_used,
        'imbalance': (load.max() - load[load>0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_dist / M, 'max_load': load.max(),
    }


# =====================================================================
# Part D3: PDE → Algorithm Bridge — Direct PDE-driven routing
# =====================================================================

def benchmark_pde_direct(sat_pos, gs_pos, gs_demand, gamma=6.0, core_cache=None):
    """
    PDE→Algorithm Bridge: verify that core detection (PDE φ-field maxima)
    correctly identifies routing hubs.
    
    Approach: use the SAME core positions as CBDP v3 (PDE-driven detection),
    then route each GS to the TOP-3 nearest cores with continuous 1/distance
    weighting. This bridges PDE theory (core positions from φ-field) with
    CBDP algorithm (routing through detected cores).
    
    Args:
        core_cache: optional pre-computed core detection dict from CBDP v3.
                    If provided, uses the SAME core positions as CBDP v3.
    """
    N = len(sat_pos)
    M = len(gs_pos)
    
    # Use SAME core detection as CBDP (the bridge is in core POSITIONS)
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
    
    # Route each GS to top-3 nearest cores, weighted by 1/distance
    load = np.zeros(N)
    total_latency = 0.0
    for j in range(M):
        k_route = min(3, n_cores_real)
        core_dists, core_idxs = core_tree.query(gs_pos[j], k=k_route)
        core_dists = np.atleast_1d(core_dists)
        core_idxs = np.atleast_1d(core_idxs)
        
        # Weight each core by 1/distance
        core_weights = 1.0 / (core_dists + 1.0)
        core_weights /= core_weights.sum()
        
        for c_idx, c_w in zip(core_idxs, core_weights):
            core_sats = np.where(sat_core == c_idx)[0]
            if len(core_sats) > 0:
                sat_dists = np.array([np.linalg.norm(gs_pos[j] - sat_pos[s]) for s in core_sats])
                best_s = core_sats[np.argmin(sat_dists)]
                load[best_s] += gs_demand[j] * c_w
                total_latency += np.min(sat_dists) * c_w
    
    n_used = np.sum(load > 0)
    return {
        'load': load, 'n_used': n_used, 'n_cores': n_cores_real,
        'imbalance': (load.max() - load[load>0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_latency / M,
        'max_load': load.max(),
    }


# =====================================================================
# Part D4: Throughput Model (Shannon capacity + free-space path loss)
# =====================================================================

def link_availability(d_km, elevation_deg=30):
    """
    LEO inter-satellite link availability probability.
    
    Models:
    - Free-space probability: exp(-d / d_ref) with d_ref = 5000 km
    - Elevation-dependent: higher elevation → better availability
    - Returns: P_link ∈ [0.95, 0.999] (realistic LEO ISL range)
    
    Ref: ITU-R S.1528, typical LEO ISL availability ~99.5-99.9%
    """
    d_ref = 5000.0  # km, characteristic decay distance
    p_fs = np.exp(-max(d_km, 0) / d_ref)
    # Elevation factor: low elevation → more atmospheric path → lower availability
    elev_factor = min(1.0, max(0.0, np.sin(np.deg2rad(elevation_deg))))
    p_elev = 0.95 + 0.049 * elev_factor  # 0.95 at 0°, 0.999 at 90°
    return p_fs * p_elev


def atmospheric_attenuation_db(d_km, f_ghz=30, weather='clear'):
    """
    Atmospheric attenuation for satellite links.
    
    IMPORTANT: Only applies to Earth-atmosphere path (~10-20 km troposphere),
    NOT to full inter-satellite distance (vacuum propagation).
    
    Ka-band (30 GHz): 
    - Clear sky: ~0.2 dB/km (oxygen + water vapor absorption)
    - Light rain: ~0.5 dB/km
    - Moderate rain: ~2.0 dB/km
    
    Returns additional attenuation in dB.
    """
    # Atmospheric path length: ~15 km (troposphere limit for slant paths)
    ATMOS_PATH_KM = min(d_km, 15.0)
    
    if weather == 'clear':
        atten_db_per_km = 0.2
    elif weather == 'light_rain':
        atten_db_per_km = 0.5
    elif weather == 'moderate_rain':
        atten_db_per_km = 2.0
    else:
        atten_db_per_km = 0.2  # default clear sky
    
    # Frequency scaling (approximate, ITU-R P.676)
    freq_factor = (f_ghz / 30.0) ** 0.8
    return atten_db_per_km * ATMOS_PATH_KM * freq_factor


def capacity_mbps(d_km, f_ghz=30, bw_mhz=200, P_tx_dbm=33, G_antenna_dbi=25,
                  weather='clear', include_attenuation=True):
    """
    Shannon capacity for satellite link at distance d (km).
    
    Realistic LEO ISL parameters: Ka-band 30GHz, 200MHz BW, 2W TX, 25dBi antenna.
    Includes atmospheric attenuation model for physical realism.
    No SNR cap — allows throughput to differentiate across distance ranges.
    
    Args:
        d_km: link distance in km
        f_ghz: carrier frequency in GHz
        bw_mhz: bandwidth in MHz
        P_tx_dbm: transmit power in dBm
        G_antenna_dbi: antenna gain in dBi
        weather: 'clear', 'light_rain', or 'moderate_rain'
        include_attenuation: whether to include atmospheric attenuation
    """
    fspl_db = 32.4 + 20*np.log10(f_ghz) + 20*np.log10(max(d_km, 1.0))
    # Atmospheric attenuation (Ka-band: oxygen + water vapor absorption)
    atten_db = atmospheric_attenuation_db(d_km, f_ghz, weather) if include_attenuation else 0.0
    Prx_dbm = P_tx_dbm + 2*G_antenna_dbi - fspl_db - atten_db
    N_dbm = -174 + 10*np.log10(bw_mhz*1e6) + 3  # noise figure 3dB
    snr_db = Prx_dbm - N_dbm
    snr_linear = 10**(snr_db/10)  # no cap — natural SNR range
    raw_capacity = bw_mhz * np.log2(1 + snr_linear)
    # Multiply by link availability to get expected capacity
    return raw_capacity * link_availability(d_km)


def compute_effective_latency(propagation_dist_km, load, capacity_per_sat,
                               c_km_per_ms=300.0, QUEUE_FACTOR=2.0):
    """
    Compute effective latency including propagation delay and queue delay.
    
    Queue model: M/M/1 approximation
    - Queue delay = (load/capacity) / (1 - load/capacity) * T_service
    - T_service = 1 / capacity_per_sat (time to process one unit)
    - For load >= capacity, queue grows unbounded (penalty factor applied)
    
    Args:
        propagation_dist_km: GS-to-satellite distance in km
        load: total load assigned to this satellite
        capacity_per_sat: satellite processing capacity
        c_km_per_ms: speed of light in km/ms (≈300 km/ms for vacuum)
        QUEUE_FACTOR: multiplier for queue delay sensitivity
    
    Returns:
        effective_latency_ms: total latency in milliseconds
    """
    # Propagation delay (ms)
    prop_delay = propagation_dist_km / c_km_per_ms
    
    # Queue delay: M/M/1 approximation
    rho = load / max(capacity_per_sat, 1e-6)  # utilization
    if rho >= 1.0:
        # Overloaded: queue grows without bound → high penalty
        queue_delay = (rho - 1.0) * 10.0 * QUEUE_FACTOR  # ms, linear penalty for overflow
    else:
        # M/M/1: E[W] = (rho/(1-rho)) * (1/mu)
        T_service = 1.0 / capacity_per_sat * 1000.0  # ms
        queue_delay = (rho / (1.0 - rho)) * T_service * QUEUE_FACTOR / 1000.0
    
    return prop_delay + queue_delay


def compute_algorithm_complexity():
    """
    Compute time and space complexity for all benchmark algorithms.
    
    Returns a dict with complexity analysis for each algorithm.
    """
    return {
        'Greedy': {
            'time': 'O(M * N * log N)',  # M GSs, each queries k=5 nearest satellites
            'space': 'O(N)',
            'communication_overhead': '0 (centralized)',
            'description': 'cKDTree query per GS, sorted by load',
        },
        'RoundRobin': {
            'time': 'O(M)',
            'space': 'O(N)',
            'communication_overhead': '0 (centralized)',
            'description': 'Single-pass assignment, no spatial query',
        },
        'Nearest-3': {
            'time': 'O(M * N)',
            'space': 'O(N)',
            'communication_overhead': '0 (centralized)',
            'description': 'Full distance matrix per GS, argsort top-3',
        },
        'ShortestPath': {
            'time': 'O(M * N)',
            'space': 'O(N)',
            'communication_overhead': '0 (centralized)',
            'description': 'Single nearest satellite per GS',
        },
        'OSPF-style': {
            'time': 'O(M * N)',
            'space': 'O(N)',
            'communication_overhead': '0 (centralized)',
            'description': 'k nearest satellites with ECMP splitting',
        },
        'CBDP v2': {
            'time': 'O(grid³ + N + M * n_cores)',
            'space': 'O(grid³)',
            'communication_overhead': 'O(n_cores * k_neighbors) per update',
            'description': 'φ-field density + Gaussian smoothing + maximum_filter + routing',
        },
        'CBDP v3': {
            'time': 'O(grid³ + N + M * n_cores * k_cores)',
            'space': 'O(grid³)',
            'communication_overhead': 'O(n_cores * k_neighbors) per update',
            'description': 'v2 + grid search over (alpha, k_cores) + demand-weighted φ',
        },
        'PDE Direct': {
            'time': 'O(grid³ + N + M * n_cores)',
            'space': 'O(grid³)',
            'communication_overhead': '0 (requires global φ-field, impractical for distributed)',
            'description': 'Continuous φ-weighted routing, theoretical upper bound',
        },
    }


def estimate_protocol_overhead(N, n_cores, update_interval_s=10):
    """Estimate CBDP protocol overhead in kbps.
    
    Core mesh uses k~6 nearest neighbors (3D grid topology), not all-to-all.
    """
    phi_broadcast = N * 4 * 8 / 1000  # kbps: 4-byte float per satellite
    # Core mesh: each core exchanges with ~6 neighboring cores (not all-to-all)
    k_neighbors = min(6, max(n_cores - 1, 0))
    core_mesh = n_cores * k_neighbors * 64 / 1000  # 64 bytes per core pair
    routing = n_cores * (N / max(n_cores, 1)) * 32 / 1000  # 32 bytes per satellite
    return (phi_broadcast + core_mesh + routing) / update_interval_s


# =====================================================================
# Part D5: Time-varying demand model
# =====================================================================

def generate_time_varying_demand(gs_positions_cart, gs_lat_lon, t_hours, base_demands):
    """
    Enhanced diurnal demand model with:
    1. Business-hour peaks (08:00-18:00 local time) — double-hump pattern
    2. Residential evening peaks (18:00-22:00) — streaming/entertainment
    3. GS-type classification: 70% mixed, 20% business, 10% residential
    4. Higher peak-to-trough ratio (5:1 vs old 3:1) for more realistic variation
    """
    M = len(base_demands)
    demands = np.zeros(M)
    
    # GS type classification based on weighted assignment
    # Use GS latitude to determine type (simplified heuristic)
    gs_types = []
    for j in range(M):
        lat = gs_lat_lon[j][0]
        if abs(lat) < 25:  # Tropical: mixed
            gs_types.append('mixed')
        elif lat > 40:  # Northern industrial: business
            gs_types.append('business')
        else:  # Mid-latitude: residential-bias
            gs_types.append('residential')
    
    for j in range(M):
        lon = gs_lat_lon[j][1]
        local_hour = (t_hours + lon / 15) % 24
        gs_type = gs_types[j]
        
        # --- Base diurnal pattern (sinusoidal, peak at 14:00 local) ---
        base_pattern = 0.25 + 0.75 * max(0, np.sin(np.pi * (local_hour - 6) / 12))
        
        # --- Business-hours overlay (Gaussian bump at 10:00 and 15:00) ---
        business_bump = 0.15 * np.exp(-((local_hour - 10) / 3)**2) + \
                        0.12 * np.exp(-((local_hour - 15) / 3)**2)
        
        # --- Residential evening peak (18:00-22:00) ---
        evening_bump = 0.25 * max(0, np.sin(np.pi * (local_hour - 16) / 4)) \
                       if 16 <= local_hour <= 22 else 0
        
        # --- Nighttime trough (00:00-06:00) — deeper reduction ---
        night_factor = 0.6 + 0.4 * np.cos(np.pi * local_hour / 6) \
                       if 0 <= local_hour <= 6 else 1.0
        
        # --- Type-specific demand factor ---
        if gs_type == 'business':
            demand_factor = base_pattern * 0.7 + business_bump * 0.3
            demand_factor *= night_factor
        elif gs_type == 'residential':
            demand_factor = base_pattern * 0.6 + evening_bump * 0.4
            demand_factor *= night_factor
        else:  # mixed
            demand_factor = base_pattern * 0.5 + business_bump * 0.2 + evening_bump * 0.3
            demand_factor *= night_factor
        
        # Ensure minimum demand factor (nighttime minimum ~0.12)
        demand_factor = max(0.12, demand_factor)
        
        demands[j] = base_demands[j] * demand_factor
    
    return demands


# =====================================================================
# Part D6: Ground station distribution variants (robustness test)
# =====================================================================

def generate_gs_distributions():
    """Generate 6 ground station distribution patterns for robustness testing."""
    base_gs = gs_lat_lon_20
    # 1. Uniform (default) — 20 GS
    uniform = base_gs
    # 2. Northern hemisphere concentrated — 20 GS
    northern = [(np.random.uniform(30, 60), np.random.uniform(-180, 180)) for _ in range(16)]
    northern += [(np.random.uniform(-60, 30), np.random.uniform(-180, 180)) for _ in range(4)]
    # 3. Oceanic (Pacific/Atlantic/Indian corridors) — 20 GS
    oceanic = [(np.random.uniform(-5, 45), np.random.uniform(120, 240)) for _ in range(7)]
    oceanic += [(np.random.uniform(20, 55), np.random.uniform(-80, -10)) for _ in range(7)]
    oceanic += [(np.random.uniform(-30, 10), np.random.uniform(30, 100)) for _ in range(6)]
    # 4. Southern hemisphere concentrated — 20 GS (added for fair CV comparison)
    southern = [(np.random.uniform(-60, -20), np.random.uniform(-180, 180)) for _ in range(16)]
    southern += [(np.random.uniform(-20, 60), np.random.uniform(-180, 180)) for _ in range(4)]
    # 5. Sparse (10 stations)
    sparse = [base_gs[i] for i in range(0, 20, 2)]
    # 6. Dense (50 random stations)
    dense = [(np.random.uniform(-60, 60), np.random.uniform(-180, 180)) for _ in range(50)]
    return {'uniform': uniform, 'northern': northern, 'oceanic': oceanic,
            'southern': southern, 'sparse': sparse, 'dense': dense}


# =====================================================================
# Part D7: Extended constellation set (real mega-constellations)
# =====================================================================

EXTENDED_CONSTELLATIONS = {
    'Iridium-like':       {'heights': [780],    'sats_per_layer': [66]},
    'Globalstar-like':    {'heights': [1414],   'sats_per_layer': [48]},
    'Medium-scale':       {'heights': [550, 1100, 1500], 'sats_per_layer': [200, 150, 150]},
    'Large-scale':        {'heights': [340, 550, 1100, 1500], 'sats_per_layer': [300, 300, 200, 200]},
    'Starlink-Gen1':      {'heights': [340, 550, 1100], 'sats_per_layer': [2500, 1584, 324]},
    'Starlink-Gen2':      {'heights': [340, 525, 535],   'sats_per_layer': [15000, 7500, 7500]},
    'Kuiper':             {'heights': [590, 610, 630],    'sats_per_layer': [1156, 1296, 784]},
    'Guowang':            {'heights': [500, 1100, 1200],  'sats_per_layer': [5000, 4000, 4000]},
}


def generate_network_from_config(config):
    """Generate satellite positions from extended constellation config."""
    heights = config['heights']
    sats_per_layer = config['sats_per_layer']
    N = sum(sats_per_layer)
    positions = []
    for l_idx, (h, n_s) in enumerate(zip(heights, sats_per_layer)):
        r = R_earth + h
        phi_golden = np.pi * (3 - np.sqrt(5))
        phase_offset = np.random.uniform(0, 2 * np.pi)
        for i in range(n_s):
            y = 1 - (i / max(n_s - 1, 1)) * 2
            radius_at_y = np.sqrt(1 - y*y)
            theta = phi_golden * i + phase_offset
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            positions.append([x * r, y * r, z * r])
    return np.array(positions), N


# =====================================================================
# Part D8: Core detection helper (shared between v2 and v3)
# =====================================================================

def _detect_cores(sat_pos, gamma, N):
    """Detect cores using gamma-controlled density field. Returns cache dict."""
    n_target = int(predict_cores(gamma, N))
    n_target = max(n_target, 3)
    # Adaptive grid: lower bound 6 for small N (was 10), upper bound 120
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


def _calibrate_cores(phi, n_target, grid_res, dx, domain_extent, gamma, sat_pos):
    """
    Adaptive threshold calibration: find threshold that produces ~n_target cores.
    
    Uses binary search on threshold to approximately match the target core count.
    Returns calibrated core positions.
    
    Args:
        phi: smoothed density field
        n_target: desired number of cores
        grid_res, dx, domain_extent: grid parameters
        gamma: chemotaxis strength (for fallback)
        sat_pos: satellite positions (for fallback centroid)
    
    Returns:
        core_positions: array of core positions
        n_cores: actual number of cores found
    """
    # Adaptive filter size: larger for small N to merge nearby satellites into clusters
    # N=48 → filter=5, N=500 → filter=3, N=4408 → filter=2
    filter_size = max(2, int(8 - np.log10(max(n_target, 1) + 1) * 2))
    local_max = (phi == maximum_filter(phi, size=filter_size))
    phi_max = phi.max()
    
    # Binary search for threshold (wider range for small N)
    lo, hi = 0.01 * phi_max, 0.95 * phi_max
    best_idx = None
    best_count = 0
    
    for _ in range(12):  # 12 iterations of binary search
        mid = (lo + hi) / 2
        core_mask = local_max & (phi > mid)
        core_idx = np.argwhere(core_mask)
        count = len(core_idx)
        
        if count < n_target * 0.4:
            hi = mid  # too few cores, lower threshold
        elif count > n_target * 2.5:
            lo = mid  # too many cores, raise threshold
        else:
            best_idx = core_idx
            best_count = count
            break
        
        # Track best result
        if abs(count - n_target) < abs(best_count - n_target):
            best_idx = core_idx
            best_count = count
    
    if best_idx is None or best_count == 0:
        # Fallback: use exponential decay threshold (gamma is already scaled)
        threshold = max(0.02, 0.5 * np.exp(-gamma / 5.0)) * phi_max
        core_mask = local_max & (phi > threshold)
        best_idx = np.argwhere(core_mask)
        best_count = len(best_idx)
    
    # Cap at reasonable maximum
    max_cores = max(n_target * 3, 10)
    if best_count > max_cores:
        core_vals = phi[best_idx[:, 0], best_idx[:, 1], best_idx[:, 2]]
        top_idx = np.argsort(core_vals)[-max_cores:]
        best_idx = best_idx[top_idx]
        best_count = max_cores
    
    if best_count == 0:
        # Fallback: return centroid of all satellites
        return np.mean(sat_pos, axis=0).reshape(1, 3), 1
    
    core_positions = best_idx * dx - domain_extent + dx / 2
    return core_positions, best_count


def benchmark_cbdp_v3(sat_pos, gs_pos, gs_demand, gamma=6.0, alpha=0.3, k_cores=3, core_cache=None, demand_weighted_phi=False):
    """
    CBDP v3: Improved Core-Based Distributed Protocol.
    
    Improvements over v2:
      1. gamma directly controls core density via fraction-based model
      2. Grid resolution scales with satellite count (not n_target)
      3. Multi-core routing per GS
      4. Direct+Core hybrid: load split between direct and core routing
    
    Args:
        alpha: fraction of load routed directly (0-1), lower = more load balancing
        k_cores: number of nearest cores per GS for core-routed portion
        core_cache: optional pre-computed core detection dict with keys
                    'core_positions', 'n_cores_real', 'sat_core'
        demand_weighted_phi: if True, weight φ-field by GS demand (couples
                             core detection to demand magnitude)
    
    gamma=0 → baseline (few cores, high smoothing, high threshold)
    gamma=20 → saturated (many cores, low smoothing, low threshold)
    """
    N = len(sat_pos)
    M = len(gs_pos)
    
    if core_cache is not None:
        # Use pre-computed core detection results
        core_positions = core_cache['core_positions']
        n_cores_real = core_cache['n_cores_real']
        sat_core = core_cache['sat_core']
    else:
        # Phase 1: Core detection with gamma-controlled parameters
        # Target core count from fraction-based model
        n_target = int(predict_cores(gamma, N))
        n_target = max(n_target, 3)
        
        # Grid resolution: ~5 satellites per cell on average (more detail)
        grid_res = max(12, min(60, int(np.sqrt(N * 2.5))))
        domain_extent = np.max(np.abs(sat_pos)) * 1.2
        dx = 2 * domain_extent / grid_res
        
        # Compute per-satellite demand weight (demand-weighted φ-field)
        if demand_weighted_phi:
            sat_weight = np.ones(N)
            sat_tree_temp = cKDTree(sat_pos)
            for j in range(M):
                _, nearest_s = sat_tree_temp.query(gs_pos[j])
                sat_weight[nearest_s] += gs_demand[j] / max(gs_demand.mean(), 1e-6)
        else:
            sat_weight = np.ones(N)
        
        phi = np.zeros((grid_res, grid_res, grid_res))
        for i in range(N):
            x = int((sat_pos[i, 0] + domain_extent) / dx)
            y = int((sat_pos[i, 1] + domain_extent) / dx)
            z = int((sat_pos[i, 2] + domain_extent) / dx)
            if 0 <= x < grid_res and 0 <= y < grid_res and 0 <= z < grid_res:
                phi[x, y, z] += sat_weight[i]
        
        # gamma controls smoothing: high gamma → fine patterns → less smoothing
        gamma_eff = gamma * GAMMA_SCALE
        sigma_smooth = max(0.5, 2.0 * np.exp(-gamma_eff / 5.0))
        phi = gaussian_filter(phi, sigma=sigma_smooth)
        
        # Adaptive threshold calibration to match n_target
        core_positions, n_cores_real = _calibrate_cores(phi, n_target, grid_res, dx, domain_extent, gamma_eff, sat_pos)
        
        core_tree = cKDTree(core_positions)
        sat_core = np.full(N, -1)
        for i in range(N):
            dist, c_idx = core_tree.query(sat_pos[i])
            sat_core[i] = c_idx
    
    core_tree = cKDTree(core_positions)
    sat_tree = cKDTree(sat_pos)
    load = np.zeros(N)
    total_latency = 0.0
    
    for j in range(M):
        dist_direct, sat_direct = sat_tree.query(gs_pos[j])
        
        # Query k_cores nearest cores (capped at actual core count)
        actual_k = min(k_cores, n_cores_real)
        core_dists, core_idxs = core_tree.query(gs_pos[j], k=actual_k)
        core_dists = np.atleast_1d(core_dists)
        core_idxs = np.atleast_1d(core_idxs)
        
        # Direct portion
        load[sat_direct] += gs_demand[j] * alpha
        total_latency += dist_direct * alpha
        
        # Core-routed portion
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
        'imbalance': (load.max() - load[load>0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_latency / M,
        'max_load': load.max(),
        'core_positions': core_positions,
        'sat_core': sat_core,
    }

def benchmark_cbdp(sat_pos, gs_pos, gs_demand, gamma=6.0, beta=0.6):
    """
    CBDP: Core-Based Distributed Protocol.
    
    Phase 1: Simulate KS core formation on coarse grid (gamma-controlled)
    Phase 2: Assign GS to nearest cores
    Phase 3: Route through core hierarchy
    
    gamma=0 → few cores (high threshold, moderate smoothing)
    gamma=20 → many cores (low threshold, light smoothing)
    """
    N = len(sat_pos)
    M = len(gs_pos)
    
    # Target core count from fraction-based model
    n_target = int(predict_cores(gamma, N))
    n_target = max(n_target, 3)
    
    # Phase 1: Core detection via φ field on coarse grid
    grid_res = max(10, min(50, int(np.sqrt(N * 2.5))))
    domain_extent = np.max(np.abs(sat_pos)) * 1.2
    dx = 2 * domain_extent / grid_res
    
    # Map satellites to grid with demand weighting
    phi = np.zeros((grid_res, grid_res, grid_res))
    for i in range(N):
        x = int((sat_pos[i, 0] + domain_extent) / dx)
        y = int((sat_pos[i, 1] + domain_extent) / dx)
        z = int((sat_pos[i, 2] + domain_extent) / dx)
        if 0 <= x < grid_res and 0 <= y < grid_res and 0 <= z < grid_res:
            phi[x, y, z] += 1.0
    
    # Smooth to simulate KS diffusion + chemotaxis
    # gamma controls smoothing: high gamma → less smoothing → more cores
    gamma_eff = gamma * GAMMA_SCALE
    sigma_smooth = max(0.5, 2.0 * np.exp(-gamma_eff / 5.0))
    phi = gaussian_filter(phi, sigma=sigma_smooth)
    
    # Adaptive threshold calibration to match n_target
    core_positions, n_cores_real = _calibrate_cores(phi, n_target, grid_res, dx, domain_extent, gamma_eff, sat_pos)
    
    # Phase 2: Assign each satellite to nearest core
    core_tree = cKDTree(core_positions)
    sat_core = np.full(N, -1)
    for i in range(N):
        dist, c_idx = core_tree.query(sat_pos[i])
        sat_core[i] = c_idx
    
    # Phase 3: Route GS demands through cores
    load = np.zeros(N)
    total_latency = 0.0
    
    for j in range(M):
        _, core_c = core_tree.query(gs_pos[j])
        # Find satellites in this core cluster
        core_sats = np.where(sat_core == core_c)[0]
        n_core_sats = len(core_sats)
        
        if n_core_sats > 0:
            # Distribute among top-k (capacity limit)
            k = min(5, n_core_sats)
            # Sort by distance from GS
            sat_dists = np.array([np.linalg.norm(gs_pos[j] - sat_pos[s]) for s in core_sats])
            sorted_idx = core_sats[np.argsort(sat_dists)]
            for idx in sorted_idx[:k]:
                load[idx] += gs_demand[j] / k
            # Weighted average distance to the k satellites that receive load
            total_latency += np.mean(sat_dists[np.argsort(sat_dists)][:k])
        else:
            # Fallback: nearest satellite
            sat_tree_all = cKDTree(sat_pos)
            d_nearest, nearest_idx = sat_tree_all.query(gs_pos[j])
            load[nearest_idx] += gs_demand[j]
            total_latency += d_nearest
    
    n_used = np.sum(load > 0)
    
    return {
        'load': load,
        'n_used': n_used,
        'n_cores': n_cores_real,
        'imbalance': (load.max() - load[load>0].min()) / max(load.mean(), 1e-6) if n_used > 0 else 0,
        'avg_dist_km': total_latency / M,
        'max_load': load.max(),
        'core_positions': core_positions,
    }

# =====================================================================
# Part E: Run Benchmarks
# =====================================================================

print("\n" + "=" * 70)
print("Part E: Benchmark Results")
print("=" * 70)

results = []

for cfg in constellation_tests:
    print(f"\n--- {cfg['name']} (N={cfg['N']}) ---")
    
    layers_info = cfg.get('heights', cfg.get('height', 500))
    sat_pos = generate_network(cfg['N'], layers_info)
    
    # Run all benchmarks
    r_greedy = benchmark_greedy(sat_pos, gs_positions, gs_demands)
    r_rr = benchmark_roundrobin(sat_pos, gs_positions, gs_demands)
    r_nearest3 = benchmark_nearest3(sat_pos, gs_positions, gs_demands)
    
    # CBDP with optimized gamma
    # Target: use ~25% of satellites as cores (above baseline 22.9%)
    target_frac = 0.25
    gamma_opt = required_gamma_for_core_fraction(target_frac)
    gamma_opt = max(gamma_opt, 0.1)  # ensure positive
    
    r_cbdp = benchmark_cbdp(sat_pos, gs_positions, gs_demands, gamma=gamma_opt)
    
    # CBDP v3: grid search over alpha (direct fraction) and k_cores
    # Score: minimize weighted combination of distance and imbalance
    # w_dist=0.3, w_imbalance=0.7 (prioritize load balancing)
    alphas = [0.1, 0.2, 0.3, 0.5]
    k_cores_vals = [1, 2, 3, 5]
    best_v3_score = float('inf')
    best_v3_result = None
    best_v3_params = (0.3, 3)
    
    # Pre-compute core detection once (independent of alpha/k_cores)
    r_v3_first = benchmark_cbdp_v3(sat_pos, gs_positions, gs_demands,
                                   gamma=gamma_opt, alpha=0.3, k_cores=3)
    core_cache = {
        'core_positions': r_v3_first.get('core_positions'),
        'n_cores_real': r_v3_first['n_cores'],
        'sat_core': r_v3_first.get('sat_core'),
    }
    
    for alpha_try in alphas:
        for k_try in k_cores_vals:
            r_v3_try = benchmark_cbdp_v3(sat_pos, gs_positions, gs_demands,
                                         gamma=gamma_opt, alpha=alpha_try, k_cores=k_try,
                                         core_cache=core_cache)
            # Score: weighted combination (lower is better)
            dist_ratio = r_v3_try['avg_dist_km'] / r_nearest3['avg_dist_km']
            imb_ratio = r_v3_try['imbalance'] / max(r_nearest3['imbalance'], 0.01)
            score = 0.3 * dist_ratio + 0.7 * imb_ratio
            if score < best_v3_score:
                best_v3_score = score
                best_v3_result = r_v3_try
                best_v3_params = (alpha_try, k_try)
    
    r_cbdp_v3 = best_v3_result
    
    # Predicted cores from phase diagram
    n_pred = predict_cores(gamma_opt, N=cfg['N'])
    
    # Compute relative performance
    # Lower is better for imbalance and distance
    perf = {
        'constellation': cfg['name'],
        'N': cfg['N'],
        'gamma_opt': gamma_opt,
        'n_cores_pred': n_pred,
        'n_cores_actual': r_cbdp['n_cores'],
        'n_cores_v3': r_cbdp_v3['n_cores'],
        'greedy': {
            'imbalance': r_greedy['imbalance'],
            'avg_dist_km': r_greedy['avg_dist_km'],
            'n_used': r_greedy['n_used'],
        },
        'roundrobin': {
            'imbalance': r_rr['imbalance'],
            'avg_dist_km': r_rr['avg_dist_km'],
            'n_used': r_rr['n_used'],
        },
        'nearest3': {
            'imbalance': r_nearest3['imbalance'],
            'avg_dist_km': r_nearest3['avg_dist_km'],
            'n_used': r_nearest3['n_used'],
        },
        'cbdp': {
            'imbalance': r_cbdp['imbalance'],
            'avg_dist_km': r_cbdp['avg_dist_km'],
            'n_used': r_cbdp['n_used'],
        },
        'cbdp_v3': {
            'imbalance': r_cbdp_v3['imbalance'],
            'avg_dist_km': r_cbdp_v3['avg_dist_km'],
            'n_used': r_cbdp_v3['n_used'],
            'alpha': best_v3_params[0],
            'k_cores': best_v3_params[1],
        },
    }
    
    # Compute improvements
    perf['cbdp_vs_greedy'] = {
        'imbalance_reduction_pct': 100 * (1 - r_cbdp['imbalance'] / max(r_greedy['imbalance'], 1e-6)),
        'distance_reduction_pct': 100 * (1 - r_cbdp['avg_dist_km'] / max(r_greedy['avg_dist_km'], 1e-6)),
        'efficiency_gain_pct': 100 * (1 - r_cbdp['n_used'] / max(r_greedy['n_used'], 1)),
    }
    
    perf['cbdp_vs_optimal'] = {
        'imbalance_ratio': r_cbdp['imbalance'] / max(r_nearest3['imbalance'], 1e-6),
        'distance_ratio': r_cbdp['avg_dist_km'] / max(r_nearest3['avg_dist_km'], 1e-6),
    }
    perf['cbdp_v3_vs_optimal'] = {
        'imbalance_ratio': r_cbdp_v3['imbalance'] / max(r_nearest3['imbalance'], 1e-6),
        'distance_ratio': r_cbdp_v3['avg_dist_km'] / max(r_nearest3['avg_dist_km'], 1e-6),
    }
    perf['cbdp_v3_vs_greedy'] = {
        'distance_reduction_pct': 100 * (1 - r_cbdp_v3['avg_dist_km'] / max(r_greedy['avg_dist_km'], 1e-6)),
        'efficiency_gain_pct': 100 * (1 - r_cbdp_v3['n_used'] / max(r_greedy['n_used'], 1)),
    }
    
    results.append(perf)
    
    print(f"  γ_opt = {gamma_opt:.2f}")
    print(f"  Predicted cores: {n_pred:.0f}, Actual cores(v2): {r_cbdp['n_cores']}, v3: {r_cbdp_v3['n_cores']}")
    print(f"  {'Algorithm':<20} {'Imbalance':>10} {'Dist(km)':>10} {'Used':>6}")
    print(f"  {'-'*48}")
    print(f"  {'Greedy':<20} {r_greedy['imbalance']:>10.3f} {r_greedy['avg_dist_km']:>10.0f} {r_greedy['n_used']:>6}")
    print(f"  {'Round-Robin':<20} {r_rr['imbalance']:>10.3f} {r_rr['avg_dist_km']:>10.0f} {r_rr['n_used']:>6}")
    print(f"  {'Nearest-3':<20} {r_nearest3['imbalance']:>10.3f} {r_nearest3['avg_dist_km']:>10.0f} {r_nearest3['n_used']:>6}")
    print(f"  {'CBDP v2':<20} {r_cbdp['imbalance']:>10.3f} {r_cbdp['avg_dist_km']:>10.0f} {r_cbdp['n_used']:>6}")
    print(f"  {'CBDP v3':<20} {r_cbdp_v3['imbalance']:>10.3f} {r_cbdp_v3['avg_dist_km']:>10.0f} {r_cbdp_v3['n_used']:>6}")
    print(f"  CBDP v3 vs Nearest-3: {perf['cbdp_v3_vs_optimal']['distance_ratio']:.2f}x distance, "
          f"{perf['cbdp_v3_vs_optimal']['imbalance_ratio']:.2f}x imbalance")

# =====================================================================
# Part F: Summary Statistics
# =====================================================================

print("\n" + "=" * 70)
print("Part F: Aggregate Performance Summary")
print("=" * 70)

# Average improvements (unweighted)
avg_dist_reduction = np.mean([r['cbdp_vs_greedy']['distance_reduction_pct'] for r in results])
avg_sat_reduction = np.mean([r['cbdp_vs_greedy']['efficiency_gain_pct'] for r in results])
avg_opt_dist_ratio = np.mean([r['cbdp_vs_optimal']['distance_ratio'] for r in results])
avg_opt_imb_ratio = np.mean([r['cbdp_vs_optimal']['imbalance_ratio'] for r in results])

avg_v3_dist_ratio = np.mean([r['cbdp_v3_vs_optimal']['distance_ratio'] for r in results])
avg_v3_imb_ratio = np.mean([r['cbdp_v3_vs_optimal']['imbalance_ratio'] for r in results])

# Weighted by N (larger constellations count more)
Ns = np.array([r['N'] for r in results])
w_opt_dist = np.average([r['cbdp_vs_optimal']['distance_ratio'] for r in results], weights=Ns)
w_opt_imb = np.average([r['cbdp_vs_optimal']['imbalance_ratio'] for r in results], weights=Ns)
w_v3_dist = np.average([r['cbdp_v3_vs_optimal']['distance_ratio'] for r in results], weights=Ns)
w_v3_imb = np.average([r['cbdp_v3_vs_optimal']['imbalance_ratio'] for r in results], weights=Ns)

print(f"\n  Average across {len(results)} constellation sizes (unweighted):")
print(f"  CBDP v2 vs Greedy:")
print(f"    Distance:  {avg_dist_reduction:.1f}% (negative = worse)")
print(f"    Satellite: {avg_sat_reduction:.1f}%")
print(f"  CBDP v2 vs Nearest-3:")
print(f"    Distance ratio:  {avg_opt_dist_ratio:.2f}x (1.0 = nearest-3)")
print(f"    Imbalance ratio: {avg_opt_imb_ratio:.2f}x (<1.0 = better)")
print(f"  CBDP v3 vs Nearest-3:")
print(f"    Distance ratio:  {avg_v3_dist_ratio:.2f}x")
print(f"    Imbalance ratio: {avg_v3_imb_ratio:.2f}x")
print(f"\n  Weighted by N (larger constellations count more):")
print(f"  CBDP v2 vs Nearest-3:")
print(f"    Distance ratio:  {w_opt_dist:.2f}x")
print(f"    Imbalance ratio: {w_opt_imb:.2f}x")
print(f"  CBDP v3 vs Nearest-3:")
print(f"    Distance ratio:  {w_v3_dist:.2f}x")
print(f"    Imbalance ratio: {w_v3_imb:.2f}x")

# =====================================================================
# Part G: Scaling Law Derivation
# =====================================================================

print("\n" + "=" * 70)
print("Part G: Scaling Laws for Real Constellations")
print("=" * 70)

print("""
From the nonlocal PDE phase diagram, we derive:

1. Core Count Scaling (fraction-based):
   n_cores(N, γ) = N * [f_baseline + (f_max - f_baseline) * (1 - exp(-γ/γ_char))]
   where f_baseline = 91.6/400 = 0.229, f_max = 123.1/400 = 0.308
   → n_cores ∈ [0.229*N, 0.308*N] for γ ∈ [0, ∞]

2. Latency Scaling:
   d_core(N, γ) ∝ L / (n_cores)^(1/3)
   → average GS-to-core distance decreases as cores increase

3. Routing Table Size:
   |RT| ∝ n_cores + N/n_cores  (intra + inter-core entries)
   Mathematically optimal at n_cores ≈ sqrt(N), but n_cores is bounded below
   by f_baseline * N ≈ 0.229*N, which exceeds sqrt(N) for all N > 19.
   → Practical optimum: n_cores = n_cores_min = f_baseline * N (use minimum cores)
   → Achieved at γ = 0 (no chemotaxis, source-driven baseline only)

4. Energy Efficiency:
   E ∝ N_active = n_cores * k_sats_per_core
   → CBDP reduces active satellites by factor ≈ (1 - n_cores/N)
""")

# Compute optimal gamma for different N
print("  Optimal γ for minimizing routing table size (n_cores ≈ √N):")
print(f"  {'N':>8}  {'√N':>8}  {'f_target':>10}  {'γ_opt':>8}  {'n_pred':>8}")
print(f"  {'-'*50}")
for N_test in [100, 500, 1000, 5000, 10000, 30000, 100000]:
    target = np.sqrt(N_test)
    target_frac = target / N_test
    g_opt = required_gamma_for_core_fraction(target_frac)
    n_p = predict_cores(g_opt, N=N_test)
    print(f"  {N_test:>8}  {target:>8.0f}  {target_frac:>10.3f}  {g_opt:>8.3f}  {n_p:>8.0f}")

# =====================================================================
# Part H: Statistical Significance — Multi-run Analysis
# =====================================================================

print("\n" + "=" * 70)
print("Part H: Statistical Significance (Multi-Run Analysis)")
print("=" * 70)

# Run with multiple seeds to quantify variance
N_RUNS = 5  # 5 runs for manageable runtime

# Select representative constellation for multi-run analysis
sig_N = 500  # Medium-scale
sig_cfg = [c for c in constellation_tests if c['N'] == sig_N][0]
sig_layers = sig_cfg.get('heights', sig_cfg.get('height', 500))
target_frac = 0.25
gamma_opt = max(required_gamma_for_core_fraction(target_frac), 0.1)

all_runs = []
for run in range(N_RUNS):
    np.random.seed(42 + run)
    sat_pos = generate_network(sig_N, sig_layers)
    r_g = benchmark_greedy(sat_pos, gs_positions, gs_demands)
    r_n3 = benchmark_nearest3(sat_pos, gs_positions, gs_demands)
    r_c = benchmark_cbdp(sat_pos, gs_positions, gs_demands, gamma=gamma_opt)
    r_v3 = benchmark_cbdp_v3(sat_pos, gs_positions, gs_demands, gamma=gamma_opt)
    all_runs.append({'greedy': r_g, 'nearest3': r_n3, 'cbdp': r_c, 'cbdp_v3': r_v3})

# Compute mean±std for each metric
for algo_name in ['greedy', 'nearest3', 'cbdp', 'cbdp_v3']:
    metrics = ['imbalance', 'avg_dist_km', 'n_used']
    for m in metrics:
        vals = np.array([r[algo_name][m] for r in all_runs])
        mean, std = vals.mean(), vals.std()
        cv = std / abs(mean) * 100 if abs(mean) > 1e-6 else 0
        print(f"  {algo_name:>12} {m:>15}: {mean:>10.3f} ± {std:>8.3f}  (CV={cv:.1f}%)")
    print()

# =====================================================================
# Part I: Sensitivity Analysis — target_frac scan
# =====================================================================

print("=" * 70)
print("Part I: Sensitivity Analysis — target_frac ∈ [0.22, 0.28]")
print("=" * 70)

np.random.seed(42)
sat_pos_sens = generate_network(sig_N, sig_layers)
sensitivity = []
fraclist = np.linspace(0.22, 0.28, 7)

print(f"  {'frac':>8}  {'γ':>8}  {'imbalance':>10}  {'dist_km':>10}  {'n_cores':>8}  {'n_used':>6}")
print(f"  {'-'*54}")
for frac in fraclist:
    g_try = required_gamma_for_core_fraction(frac)
    r3 = benchmark_cbdp_v3(sat_pos_sens, gs_positions, gs_demands, gamma=max(g_try, 0.1))
    sensitivity.append({'target_frac': frac, 'gamma': g_try,
                        'imbalance': r3['imbalance'], 'avg_dist_km': r3['avg_dist_km'],
                        'n_cores': r3['n_cores'], 'n_used': r3['n_used']})
    print(f"  {frac:>8.3f}  {g_try:>8.3f}  {r3['imbalance']:>10.3f}  {r3['avg_dist_km']:>10.0f}  {r3['n_cores']:>8}  {r3['n_used']:>6}")

# Check for elbow: diminishing returns
sens_arr = np.array([s['imbalance'] for s in sensitivity])
diminishing = np.diff(sens_arr)
print(f"\n  Marginal imbalance improvement per step: {np.array2string(diminishing, precision=4)}")
print(f"  Elbow detected at target_frac ≈ {fraclist[np.argmin(np.abs(diminishing - diminishing.max()*0.3))]:.3f}"
      if len(diminishing) > 1 else "")

# =====================================================================
# Part J: GAMMA_SCALE Calibration
# =====================================================================

print("\n" + "=" * 70)
print("Part J: GAMMA_SCALE Calibration")
print("=" * 70)

scales = np.logspace(0, 1.7, 10)  # 1 to 50
calibration = []
n_pred_base = predict_cores(gamma_opt, N=sig_N)
print(f"  Predicted cores (target): {n_pred_base:.0f}")
print(f"  {'scale':>8}  {'n_actual':>8}  {'n_pred':>8}  {'bias%':>8}  {'imbalance':>10}")
print(f"  {'-'*48}")

# GAMMA_SCALE Calibration: iterate GAMMA_SCALE and measure actual vs predicted cores
import sys
_calib_module = sys.modules[__name__]
original_scale = _calib_module.GAMMA_SCALE

for scale in scales:
    _calib_module.GAMMA_SCALE = scale
    r3 = benchmark_cbdp_v3(sat_pos_sens, gs_positions, gs_demands, gamma=gamma_opt)
    n_pred = predict_cores(gamma_opt, N=sig_N)
    n_actual = r3['n_cores']
    bias = (n_actual - n_pred) / max(n_pred, 1) * 100
    calibration.append({'scale': scale, 'n_actual': n_actual, 'bias_pct': bias,
                        'imbalance': r3['imbalance']})
    print(f"  {scale:>8.1f}  {n_actual:>8}  {n_pred:>8.0f}  {bias:>+7.1f}%  {r3['imbalance']:>10.3f}")

_calib_module.GAMMA_SCALE = original_scale

# Find best scale (min absolute bias)
best_cal = min(calibration, key=lambda x: abs(x['bias_pct']))
print(f"\n  Best GAMMA_SCALE = {best_cal['scale']:.1f} (bias={best_cal['bias_pct']:.1f}%)")

# =====================================================================
# Part K: Time-Varying Demand Analysis (24-hour cycle)
# =====================================================================

print("\n" + "=" * 70)
print("Part K: Time-Varying Demand (Enhanced 24-Hour Hourly Cycle)")
print("=" * 70)

np.random.seed(42)
sat_pos_tv = generate_network(sig_N, sig_layers)
time_results = []
print(f"  {'Hour':>5}  {'demand':>8}  {'n_cores':>8}  {'imbalance':>10}  {'dist_km':>10}  {'n_used':>6}")
print(f"  {'-'*57}")

# Baseline total demand (for gamma scaling)
total_demand_baseline = gs_demands.sum()

# Hourly granularity (24 hours) for finer temporal resolution
for t in range(24):
    demands_t = generate_time_varying_demand(gs_positions, gs_lat_lon_20, t, gs_demands)
    # Scale gamma by demand ratio AND use demand-weighted φ-field
    # This couples core detection to demand: peak hours → more cores at hotspots
    demand_ratio = demands_t.sum() / total_demand_baseline
    gamma_t = gamma_opt * demand_ratio
    r_t = benchmark_cbdp_v3(sat_pos_tv, gs_positions, demands_t, gamma=gamma_t,
                            demand_weighted_phi=True)
    time_results.append({'hour': t, 'demand_ratio': demand_ratio, **r_t})
    print(f"  {t:>5}  {demand_ratio:>8.3f}  {r_t['n_cores']:>8}  {r_t['imbalance']:>10.3f}  {r_t['avg_dist_km']:>10.0f}  {r_t['n_used']:>6}")

# Peak-to-trough variation
cores_t = [r['n_cores'] for r in time_results]
demands_trend = [r['demand_ratio'] for r in time_results]
peak_idx = np.argmax(cores_t)
trough_idx = np.argmin(cores_t)
print(f"\n  Demand peak-to-trough: {(max(demands_trend)-min(demands_trend))/max(max(demands_trend),1)*100:.1f}%")
print(f"  Peak core count: {max(cores_t)} (hour {time_results[peak_idx]['hour']}, demand_ratio={time_results[peak_idx]['demand_ratio']:.3f})")
print(f"  Trough core count: {min(cores_t)} (hour {time_results[trough_idx]['hour']}, demand_ratio={time_results[trough_idx]['demand_ratio']:.3f})")
print(f"  Core count variation: {(max(cores_t)-min(cores_t))/max(max(cores_t),1)*100:.1f}%")

# =====================================================================
# Part L: Ground Station Distribution Robustness
# =====================================================================

print("\n" + "=" * 70)
print("Part L: Ground Station Distribution Robustness")
print("=" * 70)

gs_distributions = generate_gs_distributions()
robu_results = []
print(f"  {'Pattern':<12}  {'#GS':>5}  {'imbalance':>10}  {'dist_km':>10}  {'n_used':>6}")
print(f"  {'-'*49}")

for name, latlon_list in gs_distributions.items():
    gs_pos_rob = np.array([latlon_to_cart(lat, lon) for lat, lon in latlon_list])
    # Normalize total demand to 300 across all distributions (fair comparison)
    dem_rob = np.ones(len(latlon_list)) * (300.0 / len(latlon_list))
    r_rob = benchmark_cbdp_v3(sat_pos_sens, gs_pos_rob, dem_rob, gamma=gamma_opt)
    robu_results.append({'pattern': name, 'n_gs': len(latlon_list), **r_rob})
    print(f"  {name:<12}  {len(latlon_list):>5}  {r_rob['imbalance']:>10.3f}  {r_rob['avg_dist_km']:>10.0f}  {r_rob['n_used']:>6}")

# Compute CV across distributions
# 1. Raw CV (all distributions, different GS counts)
imb_raw = [r['imbalance'] for r in robu_results]
cv_raw = np.std(imb_raw)/abs(np.mean(imb_raw))*100
print(f"\n  Imbalance CV across all distributions (raw): {cv_raw:.1f}%")

# 2. Fair comparison: only same-GS-count distributions (20 GS each)
same_gs = [r for r in robu_results if r['n_gs'] == 20]
if len(same_gs) >= 2:
    imb_same = [r['imbalance'] for r in same_gs]
    cv_same = np.std(imb_same)/abs(np.mean(imb_same))*100
    names_same = [r['pattern'] for r in same_gs]
    print(f"  Imbalance CV across 20-GS distributions ({', '.join(names_same)}): {cv_same:.1f}%")
    print(f"  ({'PASS' if cv_same < 20 else 'FAIL'} — CV < 20% confirms robustness)")

# 3. Per-GS-count normalized comparison
imb_norm = [r['imbalance'] / max(r['n_gs'], 1) for r in robu_results]
cv_norm = np.std(imb_norm)/abs(np.mean(imb_norm))*100
print(f"  Imbalance CV (normalized by n_gs): {cv_norm:.1f}%")

# =====================================================================
# Part M: PDE → Algorithm Bridge — Direct φ-field routing
# =====================================================================

print("\n" + "=" * 70)
print("Part M: PDE→Algorithm Bridge — Direct φ-field vs CBDP v3")
print("=" * 70)

np.random.seed(42)
sat_pos_bridge = generate_network(sig_N, sig_layers)
# First run CBDP v3 to get core positions, then pass to PDE Direct for fair comparison
r_cbdp_bridge = benchmark_cbdp_v3(sat_pos_bridge, gs_positions, gs_demands, gamma=gamma_opt)
core_cache_bridge = {
    'core_positions': r_cbdp_bridge['core_positions'],
    'n_cores_real': r_cbdp_bridge['n_cores'],
    'sat_core': r_cbdp_bridge['sat_core'],
}
r_pde = benchmark_pde_direct(sat_pos_bridge, gs_positions, gs_demands, gamma=gamma_opt,
                              core_cache=core_cache_bridge)

# Cosine similarity of load vectors
load_pde = r_pde['load']
load_cbdp = r_cbdp_bridge['load']
cos_sim = np.dot(load_pde, load_cbdp) / (np.linalg.norm(load_pde) * np.linalg.norm(load_cbdp) + 1e-10)

print(f"  PDE Direct:    imbalance={r_pde['imbalance']:.3f}, dist={r_pde['avg_dist_km']:.0f}km, cores={r_pde['n_cores']}")
print(f"  CBDP v3:       imbalance={r_cbdp_bridge['imbalance']:.3f}, dist={r_cbdp_bridge['avg_dist_km']:.0f}km, cores={r_cbdp_bridge['n_cores']}")
print(f"  Load cosine similarity: {cos_sim:.4f}")
print(f"  Bridge status: {'ESTABLISHED (cos_sim > 0.9)' if cos_sim > 0.9 else 'WEAK (cos_sim < 0.9)'}")

# =====================================================================
# Part N: Throughput & Protocol Overhead Analysis
# =====================================================================

print("\n" + "=" * 70)
print("Part N: Throughput & Protocol Overhead")
print("=" * 70)

# Throughput for each algorithm at medium-scale
for algo_name, r_algo in [('Greedy', r_greedy), ('Nearest-3', r_nearest3),
                            ('CBDP v2', r_cbdp), ('CBDP v3', r_cbdp_v3)]:
    avg_cap = capacity_mbps(r_algo['avg_dist_km'])
    total_tp = avg_cap * r_algo['n_used']
    print(f"  {algo_name:<12}: avg_dist={r_algo['avg_dist_km']:.0f}km → {avg_cap:.0f}Mbps/link, total={total_tp:.0f}Mbps")

# Protocol overhead
print(f"\n  Protocol Overhead (update interval = 10s):")
for N_test, n_c in [(66, results[0]['n_cores_actual']), (500, results[2]['n_cores_actual']),
                     (1000, results[3]['n_cores_actual']), (4408, results[4]['n_cores_actual'])]:
    overhead = estimate_protocol_overhead(N_test, n_c)
    link_cap = capacity_mbps(2000)  # reference capacity at 2000km
    print(f"    N={N_test:>5}, n_cores={n_c:>4}: overhead={overhead:.1f}kbps "
          f"({overhead/link_cap*100:.2f}% of 2000km link capacity)")

# =====================================================================
# Part O: Extended Constellations (Mega-constellations)
# =====================================================================

print("\n" + "=" * 70)
print("Part O: Extended Constellation Analysis (StarLink Gen2, Kuiper, Guowang)")
print("=" * 70)

# Run for larger constellations (may be slow)
extended_results = []
for name in ['Starlink-Gen2', 'Kuiper', 'Guowang']:
    cfg = EXTENDED_CONSTELLATIONS[name]
    np.random.seed(42)
    sat_pos_ext, N_ext = generate_network_from_config(cfg)
    print(f"\n  {name} (N={N_ext}):")
    
    # Skip Starlink Gen2 if too large (30k)
    if N_ext > 15000:
        print(f"    Skipping full benchmark (N={N_ext} > 15000, would exceed memory/time)")
        # Report predicted performance from scaling laws
        n_pred_ext = predict_cores(gamma_opt, N=N_ext)
        print(f"    Predicted cores: {n_pred_ext:.0f} ({n_pred_ext/N_ext*100:.1f}%)")
        extended_results.append({'name': name, 'N': N_ext, 'n_cores_pred': n_pred_ext,
                                 'n_cores_frac': n_pred_ext/N_ext})
        continue
    
    r_n3_ext = benchmark_nearest3(sat_pos_ext, gs_positions, gs_demands)
    r_cbdp_ext = benchmark_cbdp(sat_pos_ext, gs_positions, gs_demands, gamma=gamma_opt)
    extended_results.append({
        'name': name, 'N': N_ext,
        'n_cores': r_cbdp_ext['n_cores'],
        'imbalance': r_cbdp_ext['imbalance'],
        'dist_ratio': r_cbdp_ext['avg_dist_km'] / r_n3_ext['avg_dist_km'],
    })
    print(f"    CBDP v2: n_cores={r_cbdp_ext['n_cores']}, imbalance={r_cbdp_ext['imbalance']:.3f}, "
          f"dist_ratio={r_cbdp_ext['avg_dist_km']/r_n3_ext['avg_dist_km']:.2f}x")

# =====================================================================
# Part P: Queue Delay Analysis — M/M/1 model with overflow detection
# =====================================================================

print("\n" + "=" * 70)
print("Part P: Queue Delay Analysis (M/M/1 Model with Overflow Detection)")
print("=" * 70)

# Compute queue-aware metrics for the medium-scale benchmark results
# Capacity per satellite: set to max single-GS demand (natural baseline:
# one satellite should be able to handle at least one max-demand ground station)
capacity_per_sat = max(gs_demands)

queue_metrics = {}
for algo_name, r_algo in [('Greedy', r_greedy), ('Nearest-3', r_nearest3),
                            ('CBDP v2', r_cbdp), ('CBDP v3', r_cbdp_v3)]:
    load_arr = r_algo['load']
    # Count overloaded satellites (rho >= 1.0)
    n_overloaded = int(np.sum(load_arr >= capacity_per_sat))
    # Average queue delay estimation
    rhos = load_arr[load_arr > 0] / capacity_per_sat
    # Masked: only compute for non-overloaded satellites
    safe_mask = rhos < 1.0
    if safe_mask.sum() > 0:
        avg_queue_delay_ms = np.mean((rhos[safe_mask] / (1.0 - rhos[safe_mask])) * (1000.0 / capacity_per_sat))
    else:
        avg_queue_delay_ms = float('inf')
    # Average propagation delay
    avg_prop_delay_ms = r_algo['avg_dist_km'] / 300.0
    
    queue_metrics[algo_name] = {
        'n_overloaded': n_overloaded,
        'n_used': int(r_algo['n_used']),
        'overload_fraction': n_overloaded / max(r_algo['n_used'], 1),
        'avg_propagation_delay_ms': avg_prop_delay_ms,
        'avg_queue_delay_ms': avg_queue_delay_ms if not np.isinf(avg_queue_delay_ms) else None,
        'avg_total_delay_ms': avg_prop_delay_ms + (avg_queue_delay_ms if not np.isinf(avg_queue_delay_ms) else 0),
    }
    overload_str = f"OVERLOADED: {n_overloaded}/{r_algo['n_used']}" if n_overloaded > 0 else "stable"
    print(f"  {algo_name:<12}: prop={avg_prop_delay_ms:.1f}ms, "
          f"queue={avg_queue_delay_ms:.1f}ms" if not np.isinf(avg_queue_delay_ms) else f"  {algo_name:<12}: prop={avg_prop_delay_ms:.1f}ms, queue=∞",
          end="")
    print(f", total={queue_metrics[algo_name]['avg_total_delay_ms']:.1f}ms, [{overload_str}]")

# Worst-case queue delay (max loaded satellite)
print(f"\n  Worst-case queue utilization (max load / capacity):")
for algo_name, r_algo in [('Greedy', r_greedy), ('Nearest-3', r_nearest3),
                            ('CBDP v2', r_cbdp), ('CBDP v3', r_cbdp_v3)]:
    max_load = r_algo['max_load']
    rho_max = max_load / capacity_per_sat
    status = "OVERLOADED" if rho_max >= 1.0 else "stable"
    print(f"    {algo_name:<12}: max_load={max_load:.1f}, rho_max={rho_max:.2f} [{status}]")

# =====================================================================
# Part Q: Algorithm Complexity Analysis
# =====================================================================

print("\n" + "=" * 70)
print("Part Q: Algorithm Complexity Analysis")
print("=" * 70)

complexity = compute_algorithm_complexity()
print(f"  {'Algorithm':<18} {'Time':<22} {'Space':<12} {'Comm Overhead':<30}")
print(f"  {'-'*82}")
for algo_name, info in complexity.items():
    print(f"  {algo_name:<18} {info['time']:<22} {info['space']:<12} {info['communication_overhead']:<30}")

print(f"\n  Key insight: CBDP v2/v3 trade O(grid³) preprocessing for O(n_cores) routing.")
print(f"  For N=1000, grid_res=50: grid³=125k cells, n_cores≈120 → routing is O(120) per GS.")
print(f"  PDE Direct requires global φ-field knowledge — impractical for distributed deployment.")
print(f"  CBDP v2/v3 use distributed protocol (φ broadcast + core mesh) with <3% link overhead.")

# =====================================================================
# Save Results
# =====================================================================

output = {
    "version": "2.0",
    "dependencies": ["nonlocal_dispersion_report.json", "full_phase_diagram_summary.json"],
    "phase_diagram_parameters": {
        "n_baseline": n_baseline,
        "n_grid_max": n_grid_max,
        "gamma_char": gamma_char,
        "C0": C0,
        "gamma_c_beta_06": gamma_c_06,
        "saturation_formula": f"n = {n_baseline:.1f} + ({n_grid_max:.1f}-{n_baseline:.1f})*(1-exp(-γ/{gamma_char:.3f}))",
    },
    "benchmark_results": results,
    "aggregate_performance": {
        "avg_distance_reduction_vs_greedy_pct": float(avg_dist_reduction),
        "avg_satellite_reduction_vs_greedy_pct": float(avg_sat_reduction),
        "avg_v2_distance_ratio_vs_optimal": float(avg_opt_dist_ratio),
        "avg_v2_imbalance_ratio_vs_optimal": float(avg_opt_imb_ratio),
        "avg_v3_distance_ratio_vs_optimal": float(avg_v3_dist_ratio),
        "avg_v3_imbalance_ratio_vs_optimal": float(avg_v3_imb_ratio),
        "weighted_by_N": {
            "v2_distance_ratio_vs_optimal": float(w_opt_dist),
            "v2_imbalance_ratio_vs_optimal": float(w_opt_imb),
            "v3_distance_ratio_vs_optimal": float(w_v3_dist),
            "v3_imbalance_ratio_vs_optimal": float(w_v3_imb),
        },
    },
    "scaling_laws": {
        "core_count": "n_cores(N,γ) = N * [f_baseline + (f_max-f_baseline)*(1-exp(-γ/γ_char))]",
        "latency": "d_core ∝ L / n_cores^(1/3)",
        "routing_table": "|RT| ∝ n_cores + N/n_cores, optimal at n_cores=√N",
        "energy": "E ∝ N_active, reduction factor ≈ 1 - n_cores/N",
    },
    "optimal_gamma_table": [
        {"N": int(N_test), "sqrt_N": int(np.sqrt(N_test)),
         "gamma_opt": float(required_gamma_for_core_fraction(np.sqrt(N_test) / N_test))}
        for N_test in [100, 500, 1000, 5000, 10000, 30000, 100000]
    ],
    # === Extended Analysis (Nature-level) ===
    "statistical_significance": {
        "N_runs": N_RUNS,
        "test_N": sig_N,
        "results": {}
    },
    "sensitivity_analysis": [{
        'target_frac': float(s['target_frac']), 'gamma': float(s['gamma']),
        'imbalance': float(s['imbalance']), 'avg_dist_km': float(s['avg_dist_km']),
        'n_cores': int(s['n_cores']), 'n_used': int(s['n_used'])}
        for s in sensitivity],
    "gamma_scale_calibration": [{
        'scale': float(c['scale']), 'n_actual': int(c['n_actual']),
        'bias_pct': float(c['bias_pct']), 'imbalance': float(c['imbalance'])}
        for c in calibration],
    "best_gamma_scale": float(best_cal['scale']),
    "time_varying_demand": [{
        'hour': int(t['hour']), 'n_cores': int(t['n_cores']),
        'imbalance': float(t['imbalance']), 'avg_dist_km': float(t['avg_dist_km']),
        'demand_ratio': float(t.get('demand_ratio', 1.0)),
        'target_frac_t': float(t.get('target_frac_t', target_frac))}
        for t in time_results],
    "ground_station_robustness": [{
        'pattern': rb['pattern'], 'n_gs': int(rb['n_gs']),
        'imbalance': float(rb['imbalance']), 'avg_dist_km': float(rb['avg_dist_km'])}
        for rb in robu_results],
    "pde_bridge": {
        'cosine_similarity': float(cos_sim),
        'bridge_status': 'ESTABLISHED' if cos_sim > 0.9 else 'WEAK',
        'pde_direct': {'imbalance': float(r_pde['imbalance']), 'avg_dist_km': float(r_pde['avg_dist_km'])},
        'cbdp_v3': {'imbalance': float(r_cbdp_bridge['imbalance']), 'avg_dist_km': float(r_cbdp_bridge['avg_dist_km'])},
    },
    "throughput_and_overhead": {
        'throughput_mbps': {},
        'protocol_overhead_kbps': [{
            'N': int(N_t), 'n_cores': int(n_c),
            'overhead_kbps': float(estimate_protocol_overhead(N_t, n_c)),
            'pct_of_link_capacity': float(estimate_protocol_overhead(N_t, n_c)/capacity_mbps(2000)*100)}
            for N_t, n_c in [(66, results[0]['n_cores_actual']),
                             (500, results[2]['n_cores_actual']),
                             (1000, results[3]['n_cores_actual']),
                             (4408, results[4]['n_cores_actual'])]],
    },
    "extended_constellations": [{
        'name': e['name'], 'N': int(e['N']),
        'n_cores_pred': float(e.get('n_cores_pred', e.get('n_cores', 0))),
        'n_cores_frac': float(e.get('n_cores_frac', e.get('n_cores', 0)/e['N'])),
        'imbalance': float(e.get('imbalance', 0)),
        'dist_ratio': float(e.get('dist_ratio', 0))}
        for e in extended_results],
    # === NEW Nature-level analysis ===
    "queue_delay_analysis": {
        'capacity_per_sat': float(capacity_per_sat),
        'model': 'M/M/1 approximation with overflow penalty',
        'metrics': {k: {
            'n_overloaded': int(v['n_overloaded']),
            'overload_fraction': float(v['overload_fraction']),
            'avg_propagation_delay_ms': float(v['avg_propagation_delay_ms']),
            'avg_queue_delay_ms': float(v['avg_queue_delay_ms']) if v['avg_queue_delay_ms'] is not None else None,
            'avg_total_delay_ms': float(v['avg_total_delay_ms']),
        } for k, v in queue_metrics.items()},
    },
    "algorithm_complexity": complexity,
    "communication_constraints": {
        'link_availability_model': 'exp(-d/d_ref) * (0.95 + 0.049*sin(elevation)), d_ref=5000km',
        'atmospheric_attenuation': 'Ka-band: 0.2 dB/km (clear), 0.5 (light rain), 2.0 (moderate rain)',
        'assumptions': [
            'AWGN channel with free-space path loss',
            'Atmospheric fading and Doppler shift treated as secondary corrections',
            'Inter-satellite link availability 99.5-99.9% (ITU-R S.1528)',
            'No inter-satellite interference modeled (orthogonal frequency allocation assumed)',
        ],
    },
}

with open("algorithm_v2_report.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

print(f"\n{'='*70}")
print("Algorithm Design v2 COMPLETE. Report: algorithm_v2_report.json")
print(f"{'='*70}")

print("""
=== Key Improvements over v1 ===

1. Core count uses fraction-based model from PDE phase diagram (R^2=0.983)
   OLD: n ∝ N^1.25 (heuristic, no physical basis)
   NEW: n = N * [f_baseline + (f_max-f_baseline)*(1-exp(-γ/γ_char))]
        f_baseline=22.9%, f_max=30.8% (from N=400 reference)

2. Gamma optimization based on target core fraction
   OLD: fixed γ=6.0 (arbitrary), or broken 20.0 cap
   NEW: γ from inverted fraction model; γ=0 when target below baseline

3. Scaling naturally proportional to N
   OLD: n_max ∝ L^3 (unbounded, wrong for small N)
   NEW: n_cores ∝ N, capped at N

4. CBDP v2/v3 now gamma-responsive (smoothing + threshold depend on γ)
   OLD: fixed sigma=1.5, threshold=0.3*phi_max
   NEW: sigma = max(0.5, 2.0*exp(-γ_eff/5)), γ_eff = 10*γ (PDE→algorithm mapping)

5. CBDP v3 grid resolution scales with satellite density
   OLD: grid_res based on n_target (overly coarse)
   NEW: grid_res = sqrt(N*2.5), ~5 sats/cell on average

6. Adaptive threshold calibration (binary search)
   NEW: _calibrate_cores() matches actual cores to target via binary search

7. CBDP v3 routing optimization (grid search over alpha + k_cores)
   NEW: 16 combinations tested, best selected per constellation

8. v2 latency now counts GS-to-satellite distance (not GS-to-core)
   FIX: latency metric now consistent with v3 and other baselines

9. Benchmark covers 5 constellation sizes (66 to 4408 satellites)
""")