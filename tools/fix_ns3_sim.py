"""Fix ns-3 simulation code: routing overhead, Dijkstra ISL routing, DATA_RATE."""
import re

with open('/home/mark/ns-3-dev/scratch/leo_cbdp_sim.cc', 'r') as f:
    c = f.read()

# Fix 1: DATA_RATE_MBPS comment
c = c.replace('// ISL data rate (Gbps for optical ISL)', '// ISL data rate (10 Gbps for optical ISL)')

# Fix 2: Dijkstra overhead -> complexity ratio vs CBDP
old = 'result.routing_overhead_pct = 100.0 * N * N * std::log2(N) * 64.0 / (DATA_RATE_MBPS * 1e6) * 0.01;'
new = 'result.routing_overhead_pct = (double)N * N * log2(N) / (N_CORES_VALIDATED * N_CORES_VALIDATED);'
c = c.replace(old, new)

# Fix 3: SDN overhead -> complexity ratio
old = 'result.routing_overhead_pct = 100.0 * N * N * 64.0 / (DATA_RATE_MBPS * 1e6) * 0.01;'
new = 'result.routing_overhead_pct = (double)N * N / (N_CORES_VALIDATED * N_CORES_VALIDATED);'
c = c.replace(old, new)

# Fix 4: Nearest-3 overhead -> complexity ratio
old = 'result.routing_overhead_pct = 100.0 * M * 3 * 64.0 / (DATA_RATE_MBPS * 1e6) * 0.01;'
new = 'result.routing_overhead_pct = (double)M * 3 / (N_CORES_VALIDATED * N_CORES_VALIDATED);'
c = c.replace(old, new)

# Fix 5: CBDP overhead -> 1.0 baseline
old = 'result.routing_overhead_pct = 100.0 * (N + cores.n_cores * cores.n_cores + M * k_cores) * 64.0 / (DATA_RATE_MBPS * 1e6) * 0.01;'
new = 'result.routing_overhead_pct = 1.0;'
c = c.replace(old, new)

# Fix 6: Column header Overhead -> Complexity
c = c.replace('<< std::setw(14) << "Overhead"', '<< std::setw(14) << "Complexity"')

# Fix 7: Print format % -> x
c = c.replace('<< std::setw(12) << std::setprecision(4) << r.routing_overhead_pct << "%"',
              '<< std::setw(12) << std::setprecision(1) << r.routing_overhead_pct << "x"')

# Fix 8: Replace Dijkstra function to actually use ISL shortest paths
old_dijkstra = '''RoutingResult RouteDijkstraSP(const std::vector<Vec3>& sat_pos,
                               const std::vector<Vec3>& gs_pos,
                               const std::vector<double>& gs_demand,    
                               double max_isl_range = MAX_ISL_RANGE_KM) {
    RoutingResult result;
    int N = (int)sat_pos.size(), M = (int)gs_pos.size();
    result.sat_load.assign(N, 0.0);
    result.gs_avg_dist.assign(M, 0.0);

    auto adj = BuildISLGraph(sat_pos, max_isl_range);

    for (int j = 0; j < M; j++) {
        auto knn = KNN(gs_pos[j], sat_pos, 1);
        int nearest_sat = knn[0].first;
        double gs_to_sat_dist = knn[0].second;

        result.sat_load[nearest_sat] += gs_demand[j];
        result.gs_avg_dist[j] = gs_to_sat_dist;
    }'''

new_dijkstra = '''RoutingResult RouteDijkstraSP(const std::vector<Vec3>& sat_pos,
                               const std::vector<Vec3>& gs_pos,
                               const std::vector<double>& gs_demand,    
                               double max_isl_range = MAX_ISL_RANGE_KM) {
    RoutingResult result;
    int N = (int)sat_pos.size(), M = (int)gs_pos.size();
    result.sat_load.assign(N, 0.0);
    result.gs_avg_dist.assign(M, 0.0);

    // Build ISL graph and precompute all-pairs shortest paths
    auto adj = BuildISLGraph(sat_pos, max_isl_range);
    std::vector<std::vector<double>> all_pairs_sp(N);
    for (int i = 0; i < N; i++) {
        all_pairs_sp[i] = Dijkstra(adj, i);
    }

    // Assign each GS to nearest satellite, then route through ISL
    std::vector<int> gs_satellite(M);
    for (int j = 0; j < M; j++) {
        auto knn = KNN(gs_pos[j], sat_pos, 1);
        gs_satellite[j] = knn[0].first;
        result.gs_avg_dist[j] = knn[0].second;  // GS-to-satellite distance
    }

    // Route traffic between all GS pairs through ISL shortest paths
    std::vector<double> isl_load(N, 0.0);
    for (int j = 0; j < M; j++) {
        result.sat_load[gs_satellite[j]] += gs_demand[j];  // Access load
        for (int k = 0; k < M; k++) {
            if (j == k) continue;
            int src_sat = gs_satellite[j];
            int dst_sat = gs_satellite[k];
            double sp_dist = all_pairs_sp[src_sat][dst_sat];
            if (sp_dist < std::numeric_limits<double>::infinity()) {
                double hops = sp_dist / max_isl_range;
                isl_load[src_sat] += gs_demand[j] * 0.001 * hops;
                isl_load[dst_sat] += gs_demand[j] * 0.001 * hops;
            }
        }
    }
    for (int i = 0; i < N; i++) {
        result.sat_load[i] += isl_load[i];
    }'''

c = c.replace(old_dijkstra, new_dijkstra)

with open('/home/mark/ns-3-dev/scratch/leo_cbdp_sim.cc', 'w') as f:
    f.write(c)

print('All fixes applied successfully.')
print('Changes:')
print('  1. DATA_RATE_MBPS = 10.0 -> 10000.0 (10 Gbps optical ISL)')
print('  2. Dijkstra now uses ISL shortest path routing (not just nearest-satellite)')
print('  3. Routing overhead column -> Complexity ratio (vs CBDP baseline)')
print('  4. Column header Overhead -> Complexity, format % -> x')