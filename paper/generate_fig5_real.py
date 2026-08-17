#!/usr/bin/env python3
"""
Generate fig5_ablation.pdf from REAL ablation experiment data
(fig5_ablation_v2_results.json, config B: population-weighted demands).

Experiment: fig5_ablation_v2_experiment.py, N=1000, gamma=1.0,
10 seeds (42-51), mean +/- std. Routing base = paper Algorithm 2
(CBDP v2: nearest SNC + intra-core top-5 spread).

Variants shown (routing-layer ablations that actually exist in code):
  Full CBDP            PDE core detection + intra-core top-5 spread
  -PDE SNC             PDE core detection replaced by k-means (same k, same routing)
  -SNC hierarchy       core hierarchy removed; each GS -> single nearest satellite
  -Intra-core spread   PDE cores kept, but top-1 only (no intra-core load spread)

Note: "-Nonlinear Damping" is not a routing-layer component (kappa exists only
in the C++ PDE integrator) and is therefore not ablatable at routing level.
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
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 11, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
})

C_BLUE   = '#0072B2'
C_ORANGE = '#E69F00'
C_RED    = '#D55E00'
C_GRAY   = '#999999'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(PROJECT_DIR, 'fig5_ablation_v2_results.json'),
          encoding='utf-8') as f:
    data = json.load(f)

sa = data['summary']

variants = [
    ('full_cbdp',    'Full\nCBDP',              C_BLUE),
    ('no_pde_snc',   '$-$PDE SNC\n($k$-means)',   C_ORANGE),
    ('no_hierarchy', '$-$SNC\nhierarchy',         C_RED),
    ('no_spread',    '$-$Intra-core\nspread',     C_GRAY),
]

labels, means, stds, colors = [], [], [], []
full = sa['full_cbdp']['imb_mean']
for key, label, color in variants:
    v = sa[key]
    labels.append(label)
    means.append(v['imb_mean'])
    stds.append(v['imb_std'])
    colors.append(color)

degr = [(m - full) / full * 100.0 for m in means]

fig, ax = plt.subplots(figsize=(3.5, 2.9))
x = np.arange(len(labels))
bars = ax.bar(x, means, yerr=stds, color=colors, width=0.62,
              capsize=3.5, error_kw=dict(elinewidth=0.9))

for xi, m, s, d in zip(x, means, stds, degr):
    ax.text(xi, m + s + 2.0, f'{m:.1f}', ha='center', va='bottom', fontsize=9.5)
    if abs(d) > 0.5:
        sign = '+' if d > 0 else ''
        ax.text(xi, m / 2, f'{sign}{d:.0f}%', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('Load imbalance')
ax.set_ylim(0, max(np.array(means) + np.array(stds)) * 1.22)
ax.set_xlabel('Ablation variant')

fig.savefig(os.path.join(FIG_DIR, 'fig5_ablation.pdf'), dpi=300)
plt.close(fig)

print('Done: fig5_ablation.pdf (real ablation data)')
for (key, label, _), m, s, d in zip(variants, means, stds, degr):
    print(f'  {key:20s} imb={m:6.2f}+-{s:5.2f}  degradation={d:+.1f}%  '
          f'dist={sa[key]["dist_mean"]:.0f}+-{sa[key]["dist_std"]:.0f} km')
