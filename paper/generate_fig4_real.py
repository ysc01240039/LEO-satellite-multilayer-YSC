#!/usr/bin/env python3
"""
Generate fig4_algorithm_benchmark.pdf from the end-to-end multi-seed benchmark
(algorithm_v2_e2e_g1p0_report.json). Reproducible, publication quality.

Data source: algorithm_v2_e2e_g1p0_report.json (benchmark_results_mean_std)
  10 seeds (42..51), population-weighted GS demands (config B),
  gamma = 1.0 (nominal operating point, same as fig5/fig6-center/fig8),
  shared ISL graph (Kruskal MST + 4-NN), all-pairs GS flow model.
Real scales: N = 48 (Globalstar), 66 (Iridium), 500, 1000, 4408 (Starlink Gen1)

Methods (per advisor-meeting baseline decision):
  Retained traditional : Dijkstra (centralized SP oracle), Nearest-3
  Added SOTA           : PFNSAR (potential-field state-aware routing),
                         LPIH (hierarchical logic-path routing)
  Removed              : Greedy, Round-Robin, SDN, distributed multipath
  Proposed             : CBDP (SNC hierarchy + portal relay, paper Algorithm 2,
                         aligned with InstallCbdpRoutes in leo_cbdp_eval.cc)

Panels:
  (a) Access-load imbalance across constellation scales (mean +/- std)
  (b) Demand-weighted end-to-end distance (km) across scales
  (c) Active satellites utilized across scales
  (d) Analytical control overhead per reconfiguration cycle (KB)
      Protocols only: Dijkstra is centralized and Nearest-3 is a static
      heuristic riding on the SP substrate, so both incur no on-network
      control traffic in this accounting and are omitted (log scale).
"""

import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 13,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.5, 'lines.markersize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Okabe-Ito colorblind-safe palette
C_BLUE      = '#0072B2'   # CBDP (proposed)
C_BLACK     = '#000000'   # Dijkstra oracle
C_ORANGE    = '#E69F00'   # Nearest-3
C_VERMILION = '#D55E00'   # PFNSAR
C_TEAL      = '#009E73'   # LPIH

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ================================================================
# Load real multi-seed E2E data
# ================================================================
with open(os.path.join(PROJECT_DIR, 'algorithm_v2_e2e_g1p0_report.json'),
          encoding='utf-8') as f:
    algo = json.load(f)
bench = algo['benchmark_results_mean_std']

ALGO_DEFS = [
    ('dijkstra', 'Dijkstra (oracle)', C_BLACK,     's', '--'),
    ('nearest3', 'Nearest-3',         C_ORANGE,    '^', '-'),
    ('pfnsar',   'PFNSAR',            C_VERMILION, 'D', '-'),
    ('lpih',     'LPIH',              C_TEAL,      'v', '-'),
    ('cbdp',     'CBDP (proposed)',   C_BLUE,      'o', '-'),
]

METRICS = ['imbalance', 'avg_dist_km', 'n_used', 'overhead_bytes_per_cycle']
series = {ak: {'N': []} for ak, *_ in ALGO_DEFS}
for ak, *_ in ALGO_DEFS:
    for m in METRICS:
        series[ak][m] = []
        series[ak][m + '_std'] = []
for b in bench:
    for ak, *_ in ALGO_DEFS:
        v = b.get(ak)
        if v is None:
            continue
        series[ak]['N'].append(b['N'])
        for m in METRICS:
            series[ak][m].append(v[m + '_mean'])
            series[ak][m + '_std'].append(v[m + '_std'])
for ak in series:
    idx = np.argsort(series[ak]['N'])
    for k in series[ak]:
        series[ak][k] = np.array(series[ak][k])[idx]

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
fig.subplots_adjust(hspace=0.50, wspace=0.34,
                    left=0.09, right=0.99, top=0.90, bottom=0.09)


def plot_panel(ax, metric, ylabel, title, defs, yscale='log', legend=False):
    for ak, label, color, marker, ls in defs:
        ax.errorbar(series[ak]['N'], series[ak][metric],
                    yerr=series[ak][metric + '_std'],
                    fmt=marker + ls, color=color, label=label,
                    linewidth=1.4, markersize=5.5,
                    capsize=2.5, elinewidth=0.8)
    ax.set_xlabel('Constellation size $N$')
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale(yscale)
    ax.set_title(title, fontsize=13)
    if legend:
        ax.legend(frameon=False, fontsize=9)


# (a) Access-load imbalance
plot_panel(axes[0, 0], 'imbalance', 'Load imbalance',
           '(a) Load imbalance', ALGO_DEFS)

# (b) Demand-weighted end-to-end distance
plot_panel(axes[0, 1], 'avg_dist_km', 'End-to-end distance (km)',
           '(b) Distance overhead', ALGO_DEFS)

# (c) Active satellites utilized
plot_panel(axes[1, 0], 'n_used', 'Active satellites $n_{\\rm used}$',
           '(c) Active satellites', ALGO_DEFS)

# shared legend for panels (a)-(c) at the figure top (avoids curve occlusion)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.55, 0.995),
           ncol=5, frameon=False, fontsize=9.5, columnspacing=1.2,
           handletextpad=0.4)

# (d) Control overhead per reconfiguration cycle (protocols only)
OH_DEFS = [d for d in ALGO_DEFS if d[0] in ('pfnsar', 'lpih', 'cbdp')]
ax = axes[1, 1]
for ak, label, color, marker, ls in OH_DEFS:
    ax.errorbar(series[ak]['N'], series[ak]['overhead_bytes_per_cycle'] / 1e3,
                yerr=series[ak]['overhead_bytes_per_cycle_std'] / 1e3,
                fmt=marker + ls, color=color, label=label,
                linewidth=1.4, markersize=5.5, capsize=2.5, elinewidth=0.8)
ax.set_xlabel('Constellation size $N$')
ax.set_ylabel('Control overhead (kB/cycle)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('(d) Control overhead per cycle', fontsize=13)
ax.legend(frameon=False, fontsize=9, loc='upper left')

fig.savefig(os.path.join(FIG_DIR, 'fig4_algorithm_benchmark.pdf'), dpi=300)
plt.close(fig)

# ================================================================
# Print verification table (for paper text sync)
# ================================================================
print('Done: fig4_algorithm_benchmark.pdf (E2E multi-seed real data)')
print()
print('Verification table (mean +/- std over 10 seeds):')
for b in bench:
    N = b['N']
    cb, dj = b['cbdp'], b['dijkstra']
    n3, pf, lp = b['nearest3'], b['pfnsar'], b['lpih']
    print(f"N={N}:")
    print(f"   imb:   CBDP {cb['imbalance_mean']:.2f}+-{cb['imbalance_std']:.2f} | "
          f"PFNSAR {pf['imbalance_mean']:.2f} | LPIH {lp['imbalance_mean']:.2f} | "
          f"N3 {n3['imbalance_mean']:.2f} | Dij {dj['imbalance_mean']:.2f}")
    print(f"   dist:  CBDP {cb['avg_dist_km_mean']:.0f} | "
          f"PFNSAR {pf['avg_dist_km_mean']:.0f} | LPIH {lp['avg_dist_km_mean']:.0f} | "
          f"N3 {n3['avg_dist_km_mean']:.0f} | Dij {dj['avg_dist_km_mean']:.0f}")
    print(f"   n_used: CBDP {cb['n_used_mean']:.1f} | others "
          f"{dj['n_used_mean']:.1f}-{n3['n_used_mean']:.1f} | "
          f"overhead kB: CBDP {cb['overhead_bytes_per_cycle_mean']/1e3:.0f} | "
          f"PFNSAR {pf['overhead_bytes_per_cycle_mean']/1e3:.0f} | "
          f"LPIH {lp['overhead_bytes_per_cycle_mean']/1e3:.0f}")
    print(f"   carried_imb: CBDP {cb['carried_imbalance_mean']:.2f} | "
          f"PFNSAR {pf['carried_imbalance_mean']:.2f} | "
          f"LPIH {lp['carried_imbalance_mean']:.2f} | "
          f"Dij {dj['carried_imbalance_mean']:.2f} | "
          f"n_cores={b['n_cores_cbdp_mean']:.1f}")
