#!/usr/bin/env python3
"""Analyze complete phase diagram results from all sweeps."""
import os, json, itertools
import numpy as np

R = 'results'

lo_cfg = {
    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 4.0],
    'beta': [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
}
hi_cfg = {
    'gamma': [6.0, 8.0, 10.0, 12.0, 16.0, 20.0],
    'beta': [0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
}

all_data = []
for tag, cfg in [('low_gamma', lo_cfg), ('high_gamma', hi_cfg)]:
    for g, b in itertools.product(cfg['gamma'], cfg['beta']):
        f = os.path.join(R, f'phase_diagram_{tag}_gamma{g}_beta{b}_N400.json')
        if os.path.exists(f):
            d = json.load(open(f))
            all_data.append({'gamma': g, 'beta': b, 'n_cores': d.get('avg_cores')})

print(f'Total data points: {len(all_data)}')
print(f"Expected: {12*7 + 6*6} = 120")

# Summary by gamma
gammas = sorted(set(d['gamma'] for d in all_data))
print()
print(f"{'gamma':>6}  {'n_cores':>8}  {'std':>6}  {'N':>4}  {'beta_range':>12}")
print('-' * 52)
for g in gammas:
    vals = [d['n_cores'] for d in all_data if d['gamma']==g and d['n_cores'] is not None]
    betas = [d['beta'] for d in all_data if d['gamma']==g and d['n_cores'] is not None]
    if vals:
        print(f'{g:>6.1f}  {np.mean(vals):>8.1f}  {np.std(vals):>6.1f}  {len(vals):>4}  {min(betas):>4.1f}-{max(betas):>4.1f}')

# Fit saturation curve for gamma 0-4
print('\n=== Saturation Fit (gamma 0-4) ===')
g_arr = np.array([d['gamma'] for d in all_data if d['gamma'] <= 4.0])
n_arr = np.array([d['n_cores'] for d in all_data if d['gamma'] <= 4.0 and d['n_cores'] is not None])

# Model: n = n_inf - A * exp(-gamma/gamma_char)
from scipy.optimize import curve_fit
def sat_model(x, n_inf, A, gamma_char):
    return n_inf - A * np.exp(-x / gamma_char)

popt, pcov = curve_fit(sat_model, g_arr, n_arr, p0=[140, 48, 0.3], maxfev=10000)
n_inf, A, gamma_char = popt
print(f'n_inf = {n_inf:.2f} (grid saturation limit)')
print(f'A     = {A:.2f}')
print(f'gamma_char = {gamma_char:.3f}')
print(f'R^2   = {1 - np.sum((n_arr - sat_model(g_arr, *popt))**2) / np.sum((n_arr - n_arr.mean())**2):.4f}')

# Show full gamma-beta matrix
print('\n=== Phase Diagram Matrix ===')
all_gammas = sorted(set(d['gamma'] for d in all_data))
all_betas = sorted(set(d['beta'] for d in all_data))
print('gamma\\beta', end='')
for b in all_betas:
    print(f'  {b:>5.1f}', end='')
print()
for g in all_gammas:
    print(f'{g:>10.1f}', end='')
    for b in all_betas:
        vals = [d['n_cores'] for d in all_data if d['gamma']==g and d['beta']==b and d['n_cores'] is not None]
        if vals:
            print(f'  {vals[0]:>5.1f}', end='')
        else:
            print(f'  {"-":>5}', end='')
    print()

# Save summary
summary = {
    'total_points': len(all_data),
    'sat_fit': {'n_inf': float(n_inf), 'A': float(A), 'gamma_char': float(gamma_char)},
    'gamma_summary': {str(g): {'mean': float(np.mean([d['n_cores'] for d in all_data if d['gamma']==g and d['n_cores'] is not None])),
                                'std': float(np.std([d['n_cores'] for d in all_data if d['gamma']==g and d['n_cores'] is not None])),
                                'n': len([d for d in all_data if d['gamma']==g])}
                      for g in all_gammas},
    'data': all_data
}
with open(os.path.join(R, 'full_phase_diagram_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSaved to results/full_phase_diagram_summary.json')