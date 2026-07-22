import json, os

PROJ = 'e:/pytorchFile/YSC_2/Project/Project'
files = [
    'multilayer_results_gamma_long_0.4.json',
    'multilayer_results_gamma_long_1.json',
    'multilayer_results_gamma_critical_0.43.json',
]

for fname in files:
    path = os.path.join(PROJ, fname)
    try:
        with open(path) as f:
            d = json.load(f)
        ts = d['time_series']
        t = ts['t']
        nc = ts['n_cores']
        print(f'{fname}: gamma={d["gamma"]}, beta={d["beta"]}, t_max={t[-1]:.1f}, n_sats={d.get("n_sats","N/A")}')
        print(f'  t range: {t[0]:.1f} to {t[-1]:.1f}, len={len(t)}')
        print(f'  n_cores: first={nc[0]}, last={nc[-1]}, min={min(nc)}, max={max(nc)}')
        mid = len(nc) // 2
        first_half = nc[:mid]
        second_half = nc[mid:]
        print(f'  first_half_mean={sum(first_half)/len(first_half):.2f}, second_half_mean={sum(second_half)/len(second_half):.2f}')
        drift = abs(sum(first_half)/len(first_half) - sum(second_half)/len(second_half)) / (sum(nc)/len(nc)) * 100
        print(f'  drift={drift:.2f}%')
        print()
    except Exception as e:
        print(f'{fname}: ERROR {e}')
        print()