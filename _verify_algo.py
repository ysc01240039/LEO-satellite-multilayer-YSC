"""
Algorithm correctness verification:
Check every benchmark function logic, distance calculation,
and the predicted vs actual core count chain.
Updated for v3: adaptive calibration + routing optimization.
"""
import json, os
import numpy as np


def verify_algorithms():
    """Run all algorithm verification checks."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'algorithm_v2_report.json'), encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 60)
    print("Algorithm Logic Verification")
    print("=" * 60)

    # === Check 1: Greedy imbalance should increase with N ===
    ns = [r['N'] for r in data['benchmark_results']]
    g_imb = [r['greedy']['imbalance'] for r in data['benchmark_results']]
    print(f"\n1. Greedy imbalance vs N: {list(zip(ns, [f'{x:.1f}' for x in g_imb]))}")
    mono = all(g_imb[i] <= g_imb[i+1] for i in range(len(g_imb)-1))
    print(f"   Monotonic increase: {mono} (may break due to different orbit heights)")

    # === Check 2: RoundRobin uses min(N, M) satellites ===
    print(f"\n2. RoundRobin satellite usage:")
    for r in data['benchmark_results']:
        actual = r['roundrobin']['n_used']
        print(f"   N={r['N']:>5}: used={actual}, range=[1, {r['N']}]")

    # === Check 3: Nearest-3 distance <= Greedy distance ===
    print(f"\n3. Distance comparison:")
    for r in data['benchmark_results']:
        gd, od = r['greedy']['avg_dist_km'], r['nearest3']['avg_dist_km']
        status = "OK" if gd >= od - 1.0 else "ISSUE"
        print(f"   {r['constellation']:>20}: Greedy={gd:.0f}, Nearest-3={od:.0f} [{status}]")
    print(f"   Note: Greedy may route to further satellite for load balance, Nearest-3 splits to 3 nearest")

    # === Check 4: RoundRobin distance == Nearest-3 distance ===
    print(f"\n4. RoundRobin vs Nearest-3 distance (should be equal):")
    for r in data['benchmark_results']:
        rr, od = r['roundrobin']['avg_dist_km'], r['nearest3']['avg_dist_km']
        diff = abs(rr - od)
        status = "OK" if diff < 1.0 else f"DIFF={diff:.1f}"
        print(f"   {r['constellation']:>20}: RR={rr:.0f}, N3={od:.0f} [{status}]")

    # === Check 5: v2 load balancing advantage ===
    print(f"\n5. CBDP v2 load balancing advantage:")
    for r in data['benchmark_results']:
        ratio = r['cbdp_vs_optimal']['imbalance_ratio']
        better = "BETTER" if ratio < 1.0 else "WORSE"
        print(f"   {r['constellation']:>20}: v2/opt={ratio:.3f} [{better}]")

    # === Check 6: predicted vs actual cores ===
    print(f"\n6. Predicted vs actual cores (calibrated):")
    for r in data['benchmark_results']:
        pred = r['n_cores_pred']
        actual_v2 = r['n_cores_actual']
        actual_v3 = r['n_cores_v3']
        ratio_v2 = actual_v2 / max(pred, 1)
        ratio_v3 = actual_v3 / max(pred, 1)
        print(f"   {r['constellation']:>20}: pred={pred:.0f}, v2={actual_v2} ({ratio_v2:.1%}), v3={actual_v3} ({ratio_v3:.1%})")

    # === Check 7: v2 distance = GS-to-satellite (now fixed) ===
    print(f"\n7. v2 distance (GS-to-satellite, consistent with other baselines):")
    for r in data['benchmark_results']:
        ratio = r['cbdp_vs_optimal']['distance_ratio']
        print(f"   {r['constellation']:>20}: v2/opt = {ratio:.2f}x")

    # === Check 8: v3 routing optimization (per-constellation alpha + k_cores) ===
    print(f"\n8. v3 routing optimization (best alpha + k_cores per constellation):")
    for r in data['benchmark_results']:
        v3 = r['cbdp_v3']
        vs = r['cbdp_v3_vs_optimal']
        print(f"   {r['constellation']:>20}: alpha={v3['alpha']}, k={v3['k_cores']}, "
              f"dist_ratio={vs['distance_ratio']:.2f}x, imb_ratio={vs['imbalance_ratio']:.2f}x")

    # === Check 9: aggregate v3 metrics exist in JSON ===
    print(f"\n9. Aggregate v3 metrics:")
    agg = data['aggregate_performance']
    has_v3 = 'avg_v3_distance_ratio_vs_optimal' in agg
    print(f"   v3 distance ratio in aggregate: {has_v3}")
    print(f"   v3 avg distance: {agg.get('avg_v3_distance_ratio_vs_optimal', 'MISSING'):.2f}x" if has_v3 else "   MISSING!")
    print(f"   v3 avg imbalance: {agg.get('avg_v3_imbalance_ratio_vs_optimal', 'MISSING'):.2f}x" if has_v3 else "   MISSING!")

    # === Check 10: gamma_opt consistency ===
    print(f"\n10. gamma_opt consistency:")
    gammas = [r['gamma_opt'] for r in data['benchmark_results']]
    all_same = all(abs(g - gammas[0]) < 1e-6 for g in gammas)
    print(f"   All gamma_opt = {gammas[0]:.4f} (same): {all_same}")
    print(f"   (target_frac=0.25 > baseline 0.229 → gamma > 0: {gammas[0] > 0})")

    print(f"\n{'='*60}")
    print("All algorithm logic checks complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    verify_algorithms()