import json, os, glob

files = glob.glob('Project/Project/multilayer_results_gamma*.json')
for f in sorted(files):
    with open(f) as fh:
        data = json.load(fh)
    g = data.get('gamma', '?')
    avg = data.get('avg_cores', '?')
    final = data.get('final_cores', '?')
    ts = data.get('time_series', [])
    ts_len = len(ts) if isinstance(ts, list) else 0
    dur = '2h' if 'long' in f else '0.5h'
    print(f'{os.path.basename(f):<55} gamma={str(g):>6}  avg_cores={str(avg):>8}  final_cores={str(final):>8}  ts_len={ts_len:>5}  dur={dur}')

print()
print('--- N scan files ---')
files2 = glob.glob('Project/Project_nscan/multilayer_results_n*.json')
for f in sorted(files2):
    with open(f) as fh:
        data = json.load(fh)
    N = data.get('N', '?')
    avg = data.get('avg_cores', '?')
    final = data.get('final_cores', '?')
    ts = data.get('time_series', [])
    ts_len = len(ts) if isinstance(ts, list) else 0
    print(f'{os.path.basename(f):<55} N={str(N):>6}  avg_cores={str(avg):>8}  final_cores={str(final):>8}  ts_len={ts_len:>5}')