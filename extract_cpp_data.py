"""
extract_cpp_data.py
从 Project/Project/ 下的所有 C++ 输出 JSON 文件中提取真实统计量。
生成新的汇总文件，仅包含实际存在的 C++ 数据，不做任何合成。
"""
import json
import os
import glob
import numpy as np
from pathlib import Path

CPP_DIR = Path("Project/Project")
RESULTS_DIR = Path("results")

# 确保结果目录存在
RESULTS_DIR.mkdir(exist_ok=True)

def extract_n_cores_stats(filepath):
    """从 C++ 输出文件中提取 n_cores 时间序列统计量"""
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    
    ts = data.get("time_series", {})
    n_cores = np.array(ts.get("n_cores", []))
    t = np.array(ts.get("t", []))
    
    if len(n_cores) == 0:
        return None
    
    # 稳态分析：取后半段
    half = len(n_cores) // 2
    steady = n_cores[half:]
    early = n_cores[:half]
    
    stats = {
        "file": os.path.basename(filepath),
        "gamma": data.get("gamma"),
        "beta": data.get("beta"),
        "n_sats": data.get("n_sats", "unknown"),
        "n_timesteps": len(n_cores),
        "t_max": float(t[-1]) if len(t) > 0 else 0,
        "avg_cores": float(np.mean(n_cores)),
        "std_cores": float(np.std(n_cores)),
        "min_cores": int(np.min(n_cores)),
        "max_cores": int(np.max(n_cores)),
        "steady_avg_cores": float(np.mean(steady)),
        "steady_std_cores": float(np.std(steady)),
        "early_avg_cores": float(np.mean(early)),
        "drift_pct": float(abs(np.mean(steady) - np.mean(early)) / max(np.mean(early), 1) * 100),
        "source": "C++ finite-difference simulation (multilayer_sim_real.exe)",
    }
    return stats


def main():
    all_stats = []
    
    # 收集所有 C++ JSON 文件
    json_files = sorted(CPP_DIR.glob("multilayer_results_*.json"))
    
    print(f"Found {len(json_files)} C++ output files\n")
    print(f"{'File':<55} {'gamma':>6} {'beta':>6} {'N':>6} {'avg':>8} {'std':>8} {'min':>6} {'max':>6} {'steady':>8} {'drift%':>7}")
    print("-" * 120)
    
    for fp in json_files:
        stats = extract_n_cores_stats(fp)
        if stats is None:
            print(f"  SKIP: {fp.name} (no n_cores data)")
            continue
        all_stats.append(stats)
        print(f"{stats['file']:<55} {stats['gamma']:>6.3f} {stats['beta']:>6.3f} {stats['n_sats']:>6} "
              f"{stats['avg_cores']:>8.4f} {stats['std_cores']:>8.4f} {stats['min_cores']:>6} "
              f"{stats['max_cores']:>6} {stats['steady_avg_cores']:>8.4f} {stats['drift_pct']:>7.1f}")
    
    print(f"\nTotal: {len(all_stats)} valid C++ data files\n")
    
    # ============================================================
    # 1. 生成 gamma_scan_summary.json（仅包含实际存在的 gamma 扫描数据）
    # ============================================================
    gamma_scan = [s for s in all_stats if s['beta'] == 0.6 and s['n_sats'] == 1000 
                  and 'long' not in s['file'] and 'critical' not in s['file']]
    # 也包含 critical scan 的文件（它们也是 gamma 扫描）
    gamma_critical = [s for s in all_stats if s['beta'] == 0.6 and s['n_sats'] == 1000 
                      and 'critical' in s['file']]
    
    gamma_scan_summary = {
        "description": "C++ gamma scan: n_cores vs gamma (N=1000, beta=0.6, 40^3 grid)",
        "data_source": "C++ finite-difference simulation (multilayer_sim_real.exe)",
        "n_available": len(gamma_scan) + len(gamma_critical),
        "gamma_scan": sorted(gamma_scan, key=lambda x: x['gamma']),
        "gamma_critical_scan": sorted(gamma_critical, key=lambda x: x['gamma']),
        "note": "Only gamma values with actual C++ output files are included. "
                "The gamma_scan script (run_gamma_scan.ps1) specified 9 values (0.1-10.0) "
                "but only gamma=0.444 and gamma=0.5 have output files preserved. "
                "The gamma_critical_scan (run_gamma_critical_scan.ps1) specified 9 values "
                "(0.35-0.70) but only 7 values have output files (0.43-1.0, different from script).",
    }
    
    with open(RESULTS_DIR / "gamma_scan_summary_v2.json", 'w', encoding='utf-8') as f:
        json.dump(gamma_scan_summary, f, indent=2, ensure_ascii=False)
    print("[OK] gamma_scan_summary_v2.json written")
    
    # ============================================================
    # 2. 生成 beta_scan_summary.json（仅包含实际存在的 beta 扫描数据）
    # ============================================================
    beta_scan = [s for s in all_stats if s['gamma'] == 6.0 and s['n_sats'] == 1000 
                 and 'long' not in s['file'] and 'backup' not in s['file'] and 'critical' not in s['file']]
    beta_scan.extend([s for s in all_stats if 'beta' in s['file'] and 'long' not in s['file']])
    # 去重
    seen = set()
    beta_unique = []
    for s in beta_scan:
        key = (s['gamma'], s['beta'])
        if key not in seen:
            seen.add(key)
            beta_unique.append(s)
    
    beta_scan_summary = {
        "description": "C++ beta scan: n_cores vs beta (gamma=6.0, N=1000, 40^3 grid)",
        "data_source": "C++ finite-difference simulation (multilayer_sim_real.exe)",
        "n_available": len(beta_unique),
        "data": sorted(beta_unique, key=lambda x: x['beta']),
        "note": "Only beta=0.1 and beta=2.0 have actual C++ output files preserved. "
                "The beta_scan script (run_p1_beta_scan.ps1) may have specified more values.",
    }
    
    with open(RESULTS_DIR / "beta_scan_summary_v2.json", 'w', encoding='utf-8') as f:
        json.dump(beta_scan_summary, f, indent=2, ensure_ascii=False)
    print("[OK] beta_scan_summary_v2.json written")
    
    # ============================================================
    # 3. 生成 n_scaling_summary.json（N扫描 - 注意：没有 N 扫描的 C++ 输出文件存在）
    # ============================================================
    n_scan = [s for s in all_stats if 'N_' in s['file']]
    
    n_scan_summary = {
        "description": "C++ N scan: n_cores vs N",
        "data_source": "C++ finite-difference simulation (multilayer_sim_real.exe)",
        "n_available": len(n_scan),
        "data": sorted(n_scan, key=lambda x: x['n_sats']),
        "note": "CRITICAL: No N-scan C++ output files exist in Project/Project/. "
                "The run_n_scan.ps1 script specified N=200,400,600,800,1000 but "
                "none of the expected multilayer_results_N_*.json files were found. "
                "The N-scan was either never executed or the output files were not preserved. "
                "Any scaling law claims must be based on the available data (N=400 and N=1000 only).",
    }
    
    with open(RESULTS_DIR / "n_scaling_summary_v2.json", 'w', encoding='utf-8') as f:
        json.dump(n_scan_summary, f, indent=2, ensure_ascii=False)
    print("[OK] n_scaling_summary_v2.json written (WARNING: no N-scan data found)")
    
    # ============================================================
    # 4. 生成 long_run_summary.json（长时间收敛验证）
    # ============================================================
    long_runs = [s for s in all_stats if 'long' in s['file']]
    
    long_run_summary = {
        "description": "C++ long-run convergence verification (7200 dimensionless time units)",
        "data_source": "C++ finite-difference simulation (multilayer_sim_real.exe)",
        "n_available": len(long_runs),
        "data": long_runs,
        "note": "Two long runs: gamma=0.4 and gamma=1.0, both at beta=0.6, N=1000. "
                "Previous audit found these two files have identical n_cores time series, "
                "suggesting potential duplicate rather than independent runs.",
    }
    
    with open(RESULTS_DIR / "long_run_summary_v2.json", 'w', encoding='utf-8') as f:
        json.dump(long_run_summary, f, indent=2, ensure_ascii=False)
    print("[OK] long_run_summary_v2.json written")
    
    # ============================================================
    # 5. 生成 control_experiments.json（对照实验）
    # ============================================================
    controls = [s for s in all_stats if 'no_source' in s['file'] or 'uniform' in s['file']]
    
    control_summary = {
        "description": "C++ control experiments: uniform source and no-source",
        "data_source": "C++ finite-difference simulation (multilayer_sim_real.exe)",
        "n_available": len(controls),
        "data": controls,
        "note": "Uniform source: n_cores=1 (confirms topology protection). "
                "No source: n_cores=0 (confirms source-driven pattern formation).",
    }
    
    with open(RESULTS_DIR / "control_experiments_v2.json", 'w', encoding='utf-8') as f:
        json.dump(control_summary, f, indent=2, ensure_ascii=False)
    print("[OK] control_experiments_v2.json written")
    
    # ============================================================
    # 6. 生成 full_phase_diagram_summary.json（仅包含实际 C++ 数据）
    # ============================================================
    # 包含所有非长时运行的 C++ 数据
    phase_data = [s for s in all_stats if 'long' not in s['file'] and 'no_source' not in s['file'] 
                  and 'uniform' not in s['file'] and 'backup' not in s['file']]
    
    phase_summary = {
        "description": "C++ phase diagram data: n_cores vs (gamma, beta) from actual simulations",
        "data_source": "C++ finite-difference simulation (multilayer_sim_real.exe)",
        "total_points": len(phase_data),
        "n_cores_reported": float(np.mean([s['avg_cores'] for s in phase_data])),
        "n_cores_std": float(np.std([s['avg_cores'] for s in phase_data])),
        "data": sorted(phase_data, key=lambda x: (x['gamma'], x['beta'])),
        "note": "Only includes gamma/beta points with actual C++ output files preserved. "
                "The synthetic full_phase_diagram_summary.json (120 points, all n_cores=92.3) "
                "was generated by generate_sweep_data.py with hardcoded constants and should not "
                "be used as primary data source.",
    }
    
    with open(RESULTS_DIR / "full_phase_diagram_summary_v2.json", 'w', encoding='utf-8') as f:
        json.dump(phase_summary, f, indent=2, ensure_ascii=False)
    print("[OK] full_phase_diagram_summary_v2.json written")
    
    # ============================================================
    # 7. 生成综合数据报告
    # ============================================================
    # 计算 pooled mean（仅使用 C++ 数据）
    # 排除长时运行（gamma_long）和对照实验
    core_data = [s for s in all_stats if 'long' not in s['file'] and 'no_source' not in s['file'] 
                 and 'uniform' not in s['file'] and 'backup' not in s['file'] and 'real.json' not in s['file']]
    
    if core_data:
        core_means = [s['avg_cores'] for s in core_data]
        pooled_mean = float(np.mean(core_means))
        pooled_std = float(np.std(core_means))
        
        # 按 gamma 分组
        from collections import defaultdict
        by_gamma = defaultdict(list)
        for s in core_data:
            by_gamma[s['gamma']].append(s['avg_cores'])
        
        print("\n=== C++ DATA SUMMARY ===")
        print(f"Pooled mean n_cores (standard runs): {pooled_mean:.4f} ± {pooled_std:.4f}")
        print(f"Number of data points: {len(core_data)}")
        print(f"\nBy gamma value:")
        for g in sorted(by_gamma.keys()):
            vals = by_gamma[g]
            print(f"  gamma={g:.3f}: mean={np.mean(vals):.4f}, n={len(vals)}")
        
        # 长时间运行
        long_data = [s for s in all_stats if 'long' in s['file']]
        if long_data:
            long_mean = float(np.mean([s['avg_cores'] for s in long_data]))
            print(f"\nLong runs (2h): mean={long_mean:.4f}")
        
        # 对照实验
        ctrl = [s for s in all_stats if 'no_source' in s['file'] or 'uniform' in s['file']]
        for c in ctrl:
            print(f"Control ({c['file']}): n_cores={c['avg_cores']:.1f}")
    
    # ============================================================
    # 8. 生成完整原始数据清单
    # ============================================================
    inventory = {
        "description": "Complete inventory of C++ simulation output files",
        "generated_by": "extract_cpp_data.py",
        "total_files": len(all_stats),
        "files": all_stats,
        "missing_scans": {
            "n_scan": "run_n_scan.ps1 specified N=200,400,600,800,1000 but no multilayer_results_N_*.json files exist",
            "gamma_scan_full": "run_gamma_scan.ps1 specified 9 values (0.1-10.0) but only 2 files preserved (0.444, 0.5)",
            "beta_scan_full": "Only beta=0.1 and beta=2.0 files preserved. Full scan may not have been executed.",
        },
        "data_quality_notes": [
            "gamma=0.4 and gamma=1.0 long-run files have identical n_cores time series (potential duplicate)",
            "Multiple gamma_critical files have identical avg_cores=93.05577689243027 (same random seed likely)",
            "multilayer_results_real.json shows avg_cores=88.6 (different from other gamma=0.5 run at 93.06)",
        ]
    }
    
    with open(RESULTS_DIR / "cpp_data_inventory_v2.json", 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
    print("\n[OK] cpp_data_inventory_v2.json written")
    print("\n=== EXTRACTION COMPLETE ===")


if __name__ == "__main__":
    main()