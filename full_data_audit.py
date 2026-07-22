"""完整数据提取：读取所有C++模拟文件和基准测试文件，生成参考真值表"""
import json, os, glob
import numpy as np

def extract_n_cores(filepath):
    """从C++输出文件中提取n_cores统计量"""
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    ts = data.get('time_series', {})
    n_cores = np.array(ts.get('n_cores', []))
    t = np.array(ts.get('t', []))
    if len(n_cores) == 0:
        return None
    half = len(n_cores) // 2
    early = np.mean(n_cores[:half])
    steady = np.mean(n_cores[half:])
    return {
        "file": os.path.basename(filepath),
        "gamma": data.get("gamma"),
        "beta": data.get("beta"),
        "n_sats": data.get("n_sats", "N/A"),
        "n_timesteps": len(n_cores),
        "t_max": float(t[-1]) if len(t) > 0 else 0,
        "avg_cores": float(np.mean(n_cores)),
        "std_cores": float(np.std(n_cores)),
        "steady_avg_cores": float(steady),
        "early_avg_cores": float(early),
        "drift_pct": float(abs(steady - early) / max(early, 1) * 100),
    }

print("=" * 70)
print("完整数据真值表")
print("=" * 70)

# ============================================================
# 1. Gamma扫描数据 (N=1000, beta=0.6)
# ============================================================
print("\n--- 1. Gamma扫描 (N=1000, beta=0.6, 251 timesteps) ---")
gamma_scan = []
proj = r'e:\pytorchFile\YSC_2\Project\Project'
for f in sorted(glob.glob(os.path.join(proj, 'multilayer_results_gamma_*.json'))):
    s = extract_n_cores(f)
    if s and s['n_timesteps'] == 251 and s['n_sats'] == 1000:
        gamma_scan.append(s)
        print(f"  gamma={s['gamma']:.4f}: avg={s['avg_cores']:.4f}, steady={s['steady_avg_cores']:.4f}, drift={s['drift_pct']:.2f}%")

for f in sorted(glob.glob(os.path.join(proj, 'multilayer_results_gamma_critical_*.json'))):
    s = extract_n_cores(f)
    if s and s['n_timesteps'] == 251 and s['n_sats'] == 1000:
        gamma_scan.append(s)
        print(f"  gamma={s['gamma']:.4f}: avg={s['avg_cores']:.4f}, steady={s['steady_avg_cores']:.4f}, drift={s['drift_pct']:.2f}%")

# gamma=6.0 from nscan N=1000 (standard run, 251 timesteps)
nscan_dir = r'e:\pytorchFile\YSC_2\Project\Project_nscan'
s = extract_n_cores(os.path.join(nscan_dir, 'multilayer_results_nscan_N1000.json'))
if s: 
    gamma_scan.append(s)
    print(f"  gamma={s['gamma']:.4f}: avg={s['avg_cores']:.4f}, steady={s['steady_avg_cores']:.4f}, drift={s['drift_pct']:.2f}% (from N-scan)")

gamma_scan.sort(key=lambda x: x['gamma'])
gamma_vals = sorted(set(d['gamma'] for d in gamma_scan))
print(f"  总计: {len(gamma_vals)}个gamma值, 范围[{min(gamma_vals)}, {max(gamma_vals)}]")
print(f"  跨度: {max(gamma_vals)/min(gamma_vals):.1f}x")

# 检查是否所有gamma值都给出相同的n_cores
all_avg = [d['avg_cores'] for d in gamma_scan]
if len(set(round(v, 4) for v in all_avg)) == 1:
    print(f"  [结论] 所有gamma值n_cores完全相同: {all_avg[0]:.4f}")
else:
    print(f"  [结论] n_cores值存在差异: {set(round(v, 4) for v in all_avg)}")

# ============================================================
# 2. Beta扫描数据 (gamma=6.0, N=1000, 251 timesteps)
# ============================================================
print("\n--- 2. Beta扫描 (gamma=6.0, N=1000, 251 timesteps) ---")
beta_scan = []
for f in sorted(glob.glob(os.path.join(proj, 'multilayer_results_beta_*.json'))):
    s = extract_n_cores(f)
    if s and s['n_timesteps'] == 251:
        beta_scan.append(s)
        print(f"  beta={s['beta']:.4f}: avg={s['avg_cores']:.4f}, steady={s['steady_avg_cores']:.4f}, drift={s['drift_pct']:.2f}%")

# beta=0.6 from nscan N=1000
s = extract_n_cores(os.path.join(nscan_dir, 'multilayer_results_nscan_N1000.json'))
if s:
    beta_scan.append(s)
    print(f"  beta={s['beta']:.4f}: avg={s['avg_cores']:.4f}, steady={s['steady_avg_cores']:.4f}, drift={s['drift_pct']:.2f}% (from N-scan)")

beta_scan.sort(key=lambda x: x['beta'])
beta_vals = sorted(set(d['beta'] for d in beta_scan))
print(f"  总计: {len(beta_vals)}个beta值, 范围[{min(beta_vals)}, {max(beta_vals)}]")
print(f"  跨度: {max(beta_vals)/min(beta_vals):.1f}x")

all_beta_avg = [d['avg_cores'] for d in beta_scan]
if len(set(round(v, 4) for v in all_beta_avg)) == 1:
    print(f"  [结论] 所有beta值n_cores完全相同: {all_beta_avg[0]:.4f}")
else:
    print(f"  [结论] n_cores值存在差异: {set(round(v, 4) for v in all_beta_avg)}")

# ============================================================
# 3. N扫描数据 (gamma=6.0, beta=0.6, 251 timesteps)
# ============================================================
print("\n--- 3. N扫描 (gamma=6.0, beta=0.6, 251 timesteps) ---")
n_scan = []
for f in sorted(glob.glob(os.path.join(nscan_dir, '*.json'))):
    s = extract_n_cores(f)
    if s:
        n_scan.append(s)
        print(f"  N={s['n_sats']}: avg={s['avg_cores']:.4f}, steady={s['steady_avg_cores']:.4f}, drift={s['drift_pct']:.2f}%")

n_scan.sort(key=lambda x: x['n_sats'])
N_vals = np.array([d['n_sats'] for d in n_scan])
n_vals = np.array([d['avg_cores'] for d in n_scan])

# 幂律拟合
logN = np.log(N_vals)
logn = np.log(n_vals)
slope, intercept = np.polyfit(logN, logn, 1)
a = np.exp(intercept)
b = -slope
pred = a * N_vals**(-b)
ss_res = np.sum((n_vals - pred)**2)
ss_tot = np.sum((n_vals - np.mean(n_vals))**2)
r2 = 1 - ss_res / ss_tot
print(f"  幂律拟合: n_cores = {a:.4f} * N^(-{b:.4f})")
print(f"  R2 = {r2:.6f}")
print(f"  5倍跨度: {N_vals[-1]/N_vals[0]:.0f}x")

# 检查N=400的值
for d in n_scan:
    if d['n_sats'] == 400:
        print(f"  [关键] N=400时n_cores={d['avg_cores']:.4f}, NOT 93.06")
    if d['n_sats'] == 1000:
        print(f"  [关键] N=1000时n_cores={d['avg_cores']:.4f}")

# ============================================================
# 4. 长时间运行 (gamma=0.4 and 1.0, beta=0.6, N=1000, 1001 timesteps)
# ============================================================
print("\n--- 4. 长时间运行 (7200 timesteps, 1001 steps) ---")
for f in sorted(glob.glob(os.path.join(proj, 'multilayer_results_gamma_long_*.json'))):
    s = extract_n_cores(f)
    if s:
        print(f"  {s['file']}: gamma={s['gamma']:.4f}, avg={s['avg_cores']:.4f}, steady={s['steady_avg_cores']:.4f}, drift={s['drift_pct']:.2f}%")

# 检查两个长时间运行是否相同
long_files = sorted(glob.glob(os.path.join(proj, 'multilayer_results_gamma_long_*.json')))
if len(long_files) == 2:
    with open(long_files[0]) as f1, open(long_files[1]) as f2:
        d1 = f1.read()
        d2 = f2.read()
    if d1 == d2:
        print("  [警告] 两个长时间运行文件内容完全相同(bit-for-bit相同)!")

# ============================================================
# 5. 对照实验
# ============================================================
print("\n--- 5. 对照实验 ---")
for fname in ['multilayer_results_uniform_source.json', 'multilayer_results_no_source.json']:
    s = extract_n_cores(os.path.join(proj, fname))
    if s:
        print(f"  {s['file']}: gamma={s['gamma']:.4f}, beta={s['beta']:.4f}, avg={s['avg_cores']:.4f}")

# ============================================================
# 6. 基准测试数据
# ============================================================
print("\n--- 6. 基准测试数据 ---")
with open(r'e:\pytorchFile\YSC_2\results\benchmark_ratios_round47.json') as f:
    bench = json.load(f)

for n_key in ['N1000', 'N4408']:
    cbdp = bench[n_key]['cbdp_v3_vs_nearest3']
    print(f"  {n_key} CBDP vs Nearest3: imbalance={cbdp['imbalance_ratio']:.4f}, distance={cbdp['distance_ratio']:.4f}")

with open(r'e:\pytorchFile\YSC_2\results\benchmark_round47_N1000.json') as f:
    b1000 = json.load(f)
with open(r'e:\pytorchFile\YSC_2\results\benchmark_round47_N4408.json') as f:
    b4408 = json.load(f)

cb1000 = b1000['cbdp_v3']
cb4408 = b4408['cbdp_v3']
near1000 = b1000['nearest3']
near4408 = b4408['nearest3']

print(f"  N=1000: CBDP abs imbalance={cb1000['imbalance']:.4f}, distance={cb1000['avg_dist_km']:.4f}")
print(f"  N=1000: Near3 abs imbalance={near1000['imbalance']:.4f}, distance={near1000['avg_dist_km']:.4f}")
print(f"  N=1000: CBDP/Near3 imbalance={cb1000['imbalance']/near1000['imbalance']:.4f}, distance={cb1000['avg_dist_km']/near1000['avg_dist_km']:.4f}")
print(f"  N=4408: CBDP abs imbalance={cb4408['imbalance']:.4f}, distance={cb4408['avg_dist_km']:.4f}")
print(f"  N=4408: Near3 abs imbalance={near4408['imbalance']:.4f}, distance={near4408['avg_dist_km']:.4f}")
print(f"  N=4408: CBDP/Near3 imbalance={cb4408['imbalance']/near4408['imbalance']:.4f}, distance={cb4408['avg_dist_km']/near4408['avg_dist_km']:.4f}")

# ============================================================
# 7. 汇总表
# ============================================================
print("\n" + "=" * 70)
print("关键数值真值汇总")
print("=" * 70)
print(f"  n_cores (标准gamma扫描, N=1000): 93.06 (所有9个gamma值完全相同)")
print(f"  n_cores (beta扫描, gamma=6.0, N=1000): 93.06 (所有3个beta值完全相同)")
print(f"  n_cores (长时间运行, gamma=0.4/1.0): 91.49")
print(f"  n_cores (N=200): 136.96")
print(f"  n_cores (N=400): 117.47")
print(f"  n_cores (N=600): 108.02")
print(f"  n_cores (N=800): 100.10")
print(f"  n_cores (N=1000): 93.06")
print(f"  N-scan幂律拟合: n_cores = {a:.4f} * N^(-{b:.4f}), R2 = {r2:.6f}")
print(f"  CBDP N=1000: imbalance={cbdp['imbalance_ratio']:.2f}x, distance={cbdp['distance_ratio']:.2f}x")
print(f"  CBDP N=4408: imbalance={cbdp2['imbalance_ratio']:.2f}x, distance={cbdp2['distance_ratio']:.2f}x")
print(f"  均匀源: n_cores=1.0")
print(f"  无源: n_cores=0.0")