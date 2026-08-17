#!/usr/bin/env python3
"""
Parse leo_cbdp_eval.cc batch logs (logs_eval/) and regenerate:
  - fig6_ns3_validation.pdf   (CBDP across gamma=0.8/1.0/1.5, N=1000, 3 seeds)
  - fig8_failure_recovery.pdf (failure injection, failFrac 0.01..0.10)
  - fig_cmp_overview.pdf      (CBDP vs OLSR/AODV/Dijkstra, N=200/400/600)
Also writes eval_results_summary.json for paper-text synchronization.

All numbers come from the RESULT/TPUT/FAIL/RECONFIG lines of real ns-3 runs.
No hand-picked values.
"""

import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as peff

_LAB_FX = [peff.withStroke(linewidth=2.2, foreground='white')]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.5, 'lines.markersize': 6,
    'axes.spines.top': False, 'axes.spines.right': False,
})

C_BLUE = '#0072B2'
C_ORANGE = '#E69F00'
C_TEAL = '#009E73'
C_RED = '#D55E00'
C_GRAY = '#999999'
C_PURPLE = '#CC79A7'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, 'logs_eval')
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')

RESULT_RE = re.compile(
    r'RESULT mode=(\S+) N=(\d+) proto=(\S+) seed=(\d+) nCores=(\d+) '
    r'avgDelay_s=([\d.]+) avgJitter_s=([\d.]+) throughput_mbps=([\d.]+) '
    r'offered_mbps=([\d.]+) loss_pct=([\d.]+) nFlows=(\d+) txPkts=(\d+) '
    r'rxPkts=(\d+) ctrlTxPkts=(\d+) ctrlTxBytes=(\d+)')
TPUT_RE = re.compile(r'TPUT t=([\d.]+) mbps=([\d.]+)')
FAIL_RE = re.compile(
    r'FAIL injected (?:mode=(\S+) )?n=(\d+) of (\d+) ISLs'
    r'(?:, isolated_sats=(\d+) isolated_portals=(\d+))?')
RECONF_RE = re.compile(
    r'RECONFIG done at ([\d.]+) s.*?reelection=(\d)'
    r'|RECONFIG done at ([\d.]+) s')

FIG6_GAMMAS = [('g0.8', 0.8), ('g1.0', 1.0), ('g1.5', 1.5)]
FIG8_FRS = ['0.01', '0.02', '0.05', '0.10']
CMP_NS = [200, 400, 600]
CMP_PROTOS = ['Dijkstra', 'CBDP', 'OLSR', 'AODV']


def parse_result(line):
    m = RESULT_RE.search(line)
    if not m:
        return None
    g = m.groups()
    return {
        'mode': g[0], 'N': int(g[1]), 'proto': g[2], 'seed': int(g[3]),
        'nCores': int(g[4]), 'delay_s': float(g[5]), 'jitter_s': float(g[6]),
        'throughput_mbps': float(g[7]), 'offered_mbps': float(g[8]),
        'loss_pct': float(g[9]), 'nFlows': int(g[10]), 'txPkts': int(g[11]),
        'rxPkts': int(g[12]), 'ctrlTxPkts': int(g[13]),
        'ctrlTxBytes': int(g[14]),
    }


def load_logs():
    """Return dict jobname -> {'result':..., 'tput':[(t,mbps)], 'fail':...}."""
    logs = {}
    for fn in sorted(os.listdir(LOG_DIR)):
        if not fn.endswith('.log'):
            continue
        job = fn[:-4]
        entry = {'result': None, 'tput': [], 'fail': None, 'reconfig_s': None,
                 'fail_mode': None, 'isolated_sats': None,
                 'isolated_portals': None, 'reelection': None}
        with open(os.path.join(LOG_DIR, fn), encoding='utf-8',
                  errors='replace') as f:
            for line in f:
                r = parse_result(line)
                if r:
                    entry['result'] = r
                m = TPUT_RE.search(line)
                if m:
                    entry['tput'].append((float(m.group(1)),
                                          float(m.group(2))))
                m = FAIL_RE.search(line)
                if m:
                    entry['fail'] = (int(m.group(2)), int(m.group(3)))
                    entry['fail_mode'] = m.group(1) or 'random'
                    entry['isolated_sats'] = (int(m.group(4))
                                              if m.group(4) else None)
                    entry['isolated_portals'] = (int(m.group(5))
                                                 if m.group(5) else None)
                m = RECONF_RE.search(line)
                if m:
                    entry['reconfig_s'] = float(m.group(1) or m.group(3))
                    entry['reelection'] = (int(m.group(2))
                                           if m.group(2) else None)
        logs[job] = entry
    return logs


def agg(rows, key):
    vals = [r[key] for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def stats_by(rows, group_key):
    out = {}
    for r in rows:
        out.setdefault(r[group_key], []).append(r)
    return out


# ================================================================
# Figure 6: CBDP across gamma regimes (N=1000)
# ================================================================
def make_fig6(logs, summary):
    gam, tput_m, tput_s, dly_m, dly_s, ctrl_m, ctrl_s, ncores = \
        [], [], [], [], [], [], [], []
    dij = []
    for tag, g in FIG6_GAMMAS:
        rows = [logs[f'fig6_{tag}_s{s}']['result'] for s in (42, 123, 456)
                if logs.get(f'fig6_{tag}_s{s}', {}).get('result')]
        if not rows:
            continue
        gam.append(g)
        m, s = agg(rows, 'throughput_mbps'); tput_m.append(m); tput_s.append(s)
        m, s = agg(rows, 'delay_s'); dly_m.append(m * 1000); dly_s.append(s * 1000)
        m, s = agg(rows, 'ctrlTxBytes'); ctrl_m.append(m / 1e6); ctrl_s.append(s / 1e6)
        ncores.append(int(np.mean([r['nCores'] for r in rows])))
        summary['fig6'][f'gamma_{g}'] = {
            'n_seeds': len(rows), 'nCores_per_seed': [r['nCores'] for r in rows],
            'throughput_mbps': [r['throughput_mbps'] for r in rows],
            'offered_mbps': [r['offered_mbps'] for r in rows],
            'delay_ms': [r['delay_s'] * 1000 for r in rows],
            'loss_pct': [r['loss_pct'] for r in rows],
            'ctrlTxPkts': [r['ctrlTxPkts'] for r in rows],
            'ctrlTxBytes': [r['ctrlTxBytes'] for r in rows]}
    dij_rows = [logs[f'fig6_dij_s{s}']['result'] for s in (42, 123, 456)
                if logs.get(f'fig6_dij_s{s}', {}).get('result')]
    if dij_rows:
        dm, ds = agg(dij_rows, 'delay_s')
        tm, ts = agg(dij_rows, 'throughput_mbps')
        dij = {'delay_ms': dm * 1000, 'delay_std_ms': ds * 1000,
               'throughput_mbps': tm, 'throughput_std': ts}
        summary['fig6']['dijkstra_ref'] = dij

    x = np.arange(len(gam))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    fig.subplots_adjust(wspace=0.36, left=0.07, right=0.99, top=0.84,
                        bottom=0.20)

    ax = axes[0]
    offered = [np.mean(summary['fig6'][f'gamma_{g}']['offered_mbps']) for g in gam]
    ax.plot(x, offered, '--', color=C_GRAY, linewidth=1.2, zorder=1,
            label='Offered load')
    ax.errorbar(x, tput_m, yerr=tput_s, fmt='o-', color=C_BLUE, capsize=3,
                linewidth=1.4, markersize=5.5, label='CBDP throughput')
    for xi, m in zip(x, tput_m):
        ax.annotate(f'{m:.1f}', (xi, m), textcoords='offset points',
                    xytext=(0, -11), ha='center', fontsize=8, color=C_BLUE)
    ax.text(0.97, 0.97, '0\\% loss', transform=ax.transAxes, fontsize=8.5,
            color=C_BLUE, ha='right', va='top')
    ax.set_xticks(x)
    ax.set_xticklabels([f'$\\gamma={g}$' for g in gam])
    ax.set_ylabel('Throughput (Mbps)')
    ax.set_ylim(min(offered) - 2.0, max(offered) + 2.0)
    ax.set_title('(a) Throughput vs. offered load', fontsize=12)
    ax.legend(frameon=False, loc='upper left', fontsize=8.5)
    ax.set_xlim(-0.4, 2.4)

    ax = axes[1]
    ax.errorbar(x, dly_m, yerr=dly_s, fmt='o-', color=C_TEAL, capsize=3,
                linewidth=1.4, markersize=5.5, label='CBDP')
    if dij:
        ax.axhline(dij['delay_ms'], color=C_RED, linestyle='--', linewidth=1.2,
                   label=f"Dijkstra ref. ({dij['delay_ms']:.1f} ms)")
        ax.legend(frameon=False, fontsize=8.5)
    for xi, m in zip(x, dly_m):
        ax.annotate(f'{m:.1f}', (xi, m), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=8, color=C_TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels([f'$\\gamma={g}$' for g in gam])
    ax.set_ylabel('Average end-to-end delay (ms)')
    ylo = min(min(dly_m) - 3.0, dij['delay_ms'] - 3.0 if dij else 0)
    ax.set_ylim(ylo, max(dly_m) + 3.0)
    ax.set_title('(b) End-to-end delay', fontsize=12)
    ax.set_xlim(-0.4, 2.4)

    ax = axes[2]
    ax.bar(x, ctrl_m, yerr=ctrl_s, width=0.5, color=C_ORANGE, capsize=3)
    for xi, m, s, nc in zip(x, ctrl_m, ctrl_s, ncores):
        ax.text(xi, (m + s) * 1.06, f'{m:.1f}\n($n_{{\\rm cores}}$={nc})',
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'$\\gamma={g}$' for g in gam])
    ax.set_ylabel('Control traffic (MB per run)')
    ax.set_ylim(0, (max(ctrl_m) + max(ctrl_s)) * 1.42)
    ax.set_title('(c) Control-traffic overhead', fontsize=12)
    ax.set_xlim(-0.5, 2.5)

    fig.savefig(os.path.join(FIG_DIR, 'fig6_ns3_validation.pdf'), dpi=300)
    plt.close(fig)
    print('[fig6] written;', json.dumps(summary['fig6'], indent=1)[:400])


# ================================================================
# Figure 8: failure recovery
# ================================================================
def recovery_time(trace, fail_t=40.0, baseline_win=(35.0, 39.6)):
    base = [m for t, m in trace if baseline_win[0] <= t <= baseline_win[1]]
    if not base:
        return None, None
    base_m = float(np.mean(base))
    for t, m in trace:
        if t > fail_t and m >= 0.95 * base_m:
            return t - fail_t, base_m
    return None, base_m


def make_fig8(logs, summary):
    frs, rec, loss, nfail = [], [], [], []
    traces = {}
    for fr in FIG8_FRS:
        e = logs.get(f'fig8_ff{fr}')
        if not e or not e['result']:
            continue
        # The TPUT trace is application-layer goodput (PacketSink payload,
        # 1024 B/packet), whereas fig6/fig_cmp report FlowMonitor IP-layer
        # throughput. Rescale the trace to the same IP-layer byte basis using
        # the per-run measured ratio rxBytes / (rxPkts * 1024), with
        # rxBytes = throughput_mbps * 1e6 * (simTime - flowStart) / 8
        # (failure mode: simTime=60, flowStart=10 -> 50 s window).
        r = e['result']
        rx_bytes = r['throughput_mbps'] * 1e6 * 50.0 / 8.0
        byte_factor = rx_bytes / (r['rxPkts'] * 1024.0)
        trace = [(t, m * byte_factor) for t, m in e['tput']]
        rt, base_m = recovery_time(trace)
        frs.append(fr)
        nfail.append(e['fail'][0] if e['fail'] else None)
        rec.append(rt)
        loss.append(e['result']['loss_pct'])
        traces[fr] = trace
        summary['fig8'][f'failFrac_{fr}'] = {
            'failed_isls': e['fail'][0] if e['fail'] else None,
            'total_isls': e['fail'][1] if e['fail'] else None,
            'reconfig_done_s': e['reconfig_s'],
            'recovery_time_s': rt,
            'baseline_mbps': base_m,
            'loss_pct_run_avg': e['result']['loss_pct'],
            'throughput_mbps': e['result']['throughput_mbps'],
            'delay_ms': e['result']['delay_s'] * 1000}

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.7))
    fig.subplots_adjust(left=0.14, right=0.88, top=0.84, bottom=0.19)

    x = np.arange(len(frs))
    ax2 = ax.twinx()
    b1 = ax.bar(x - 0.18, rec, width=0.36, color=C_BLUE,
                label='Recovery time (s)')
    b2 = ax2.bar(x + 0.18, loss, width=0.36, color=C_RED,
                 label='Packet loss (\\%, run avg.)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nfail])
    ax.set_xlabel('Number of failed ISLs (of 2,002)')
    ax.set_ylabel('Recovery time (s)', color=C_BLUE)
    ax2.set_ylabel('Packet loss (\\%)', color=C_RED)
    ax.tick_params(axis='y', colors=C_BLUE)
    ax2.tick_params(axis='y', colors=C_RED)
    ax.set_ylim(0, max(r for r in rec if r) * 1.35)
    ax2.set_ylim(0, max(loss) * 1.35)
    for xi, r in zip(x, rec):
        ax.text(xi - 0.18, r + max(rec) * 0.03, f'{r:.1f}', ha='center',
                fontsize=8.5, color=C_BLUE)
    for xi, l in zip(x, loss):
        ax2.text(xi + 0.18, l + max(loss) * 0.03, f'{l:.1f}', ha='center',
                 fontsize=8.5, color=C_RED)
    ax.set_title('Recovery time and packet loss', fontsize=12)
    ax2.spines['top'].set_visible(False)

    fig.savefig(os.path.join(FIG_DIR, 'fig8_failure_recovery.pdf'), dpi=300)
    plt.close(fig)
    print('[fig8] written;', json.dumps(summary['fig8'], indent=1)[:400])


# ================================================================
# Figure cmp: protocol comparison
# ================================================================
def make_figcmp(logs, summary):
    data = {p: {n: [] for n in CMP_NS} for p in CMP_PROTOS}
    for n in CMP_NS:
        for p in CMP_PROTOS:
            for s in (42, 123, 456):
                e = logs.get(f'cmp_N{n}_{p}_s{s}', {}).get('result')
                if e:
                    data[p][n].append(e)
    ns = [n for n in CMP_NS if any(data[p][n] for p in CMP_PROTOS)]
    if not ns:
        print('[fig_cmp] no data, skipped')
        return

    def series(p, key, scale=1.0):
        m_, s_ = [], []
        for n in ns:
            rows = data[p][n]
            if rows:
                m, s = agg(rows, key)
                m_.append(m * scale)
                s_.append(s * scale)
            else:
                m_.append(np.nan)
                s_.append(0.0)
        return np.array(m_), np.array(s_)

    style = {'Dijkstra': (C_GRAY, 's'), 'CBDP': (C_BLUE, 'o'),
             'OLSR': (C_ORANGE, '^'), 'AODV': (C_RED, 'v')}
    # Horizontal offsets (in N units) so protocols with identical/close values
    # (e.g. Dijkstra and OLSR coincide at line rate) remain visually separable.
    xoffs = {'Dijkstra': -25.0, 'CBDP': -8.0, 'OLSR': +8.0, 'AODV': +25.0}
    ls = {'Dijkstra': '--', 'CBDP': '-', 'OLSR': ':', 'AODV': '-.'}

    for n in ns:
        summary['fig_cmp'][f'N{n}'] = {
            p: {'n_seeds': len(data[p][n]),
                'throughput_mbps': [r['throughput_mbps'] for r in data[p][n]],
                'delay_ms': [r['delay_s'] * 1000 for r in data[p][n]],
                'loss_pct': [r['loss_pct'] for r in data[p][n]],
                'ctrlTxPkts': [r['ctrlTxPkts'] for r in data[p][n]]}
            for p in CMP_PROTOS if data[p][n]}

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6))
    fig.subplots_adjust(wspace=0.42, left=0.065, right=0.995, top=0.84,
                        bottom=0.20)
    metrics = [
        ('throughput_mbps', 1.0, 'Throughput (Mbps)', '(a) Throughput', None),
        ('delay_s', 1000.0, 'Avg. end-to-end delay (ms)', '(b) Delay', None),
        ('ctrlTxPkts', 1.0, 'Control packets (log)', '(c) Control overhead',
         'log'),
        ('loss_pct', 1.0, 'Packet loss (%)', '(d) Packet loss', None),
    ]
    for ax, (key, scale, ylabel, title, yscale) in zip(axes, metrics):
        for p in CMP_PROTOS:
            m, s = series(p, key, scale)
            c, mk = style[p]
            # Dijkstra has zero control overhead: on the log-scale control panel
            # it cannot be drawn, so omit it from that panel's legend.
            lab = p
            if key == 'ctrlTxPkts' and p == 'Dijkstra':
                lab = '_nolegend_'
            ax.errorbar([n + xoffs[p] for n in ns], m, yerr=s,
                        fmt=mk + ls[p], color=c, label=lab,
                        capsize=2.0, elinewidth=0.7, markersize=4.2)
        ax.set_xlabel('Constellation size $N$')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(ns)
        if yscale == 'log':
            ax.set_yscale('log')
        ax.legend(frameon=False, fontsize=8)

    fig.savefig(os.path.join(FIG_DIR, 'fig_cmp_overview.pdf'), dpi=300)
    plt.close(fig)
    print('[fig_cmp] written for N =', ns)


# ================================================================
# Figure 9: core-concentrated vs non-core-concentrated failure
# ================================================================
def make_fig9(logs, summary):
    modes = [('noncore', 'Non-core-concentrated', C_TEAL),
             ('core', 'Core-concentrated', C_RED)]
    frs = FIG8_FRS
    # byte rescaling factor identical to fig8 (IP-layer byte basis)
    def scaled(entry):
        r = entry['result']
        rx_bytes = r['throughput_mbps'] * 1e6 * 50.0 / 8.0
        bf = rx_bytes / (r['rxPkts'] * 1024.0)
        return [(t, m * bf) for t, m in entry['tput']]

    rec = {m: [] for m, _, _ in modes}
    loss = {m: [] for m, _, _ in modes}
    nfail = []
    traces = {}
    for fr in frs:
        for m, _, _ in modes:
            e = logs.get(f'fig9_{m}_ff{fr}')
            if not e or not e['result']:
                continue
            trace = scaled(e)
            rt, base_m = recovery_time(trace)
            rec[m].append(rt)
            loss[m].append(e['result']['loss_pct'])
            traces[(m, fr)] = trace
            summary['fig9'][f'{m}_failFrac_{fr}'] = {
                'failed_isls': e['fail'][0] if e['fail'] else None,
                'isolated_sats': e['isolated_sats'],
                'isolated_portals': e['isolated_portals'],
                'reelection': e['reelection'],
                'reconfig_done_s': e['reconfig_s'],
                'recovery_time_s': rt,
                'loss_pct_run_avg': e['result']['loss_pct'],
                'throughput_mbps': e['result']['throughput_mbps'],
                'delay_ms': e['result']['delay_s'] * 1000}
        e0 = logs.get(f'fig9_core_ff{fr}')
        nfail.append(e0['fail'][0] if e0 and e0['fail'] else None)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.98, top=0.82,
                        bottom=0.20)

    # (a) goodput time series around failure at the largest extent (10%)
    ax = axes[0]
    e_r = logs.get('fig8_ff0.10')
    if e_r and e_r['result']:
        r = e_r['result']
        rx_bytes = r['throughput_mbps'] * 1e6 * 50.0 / 8.0
        bf = rx_bytes / (r['rxPkts'] * 1024.0)
        tr = [(t, m * bf) for t, m in e_r['tput']]
        ax.plot([t for t, _ in tr], [m for _, m in tr], color=C_GRAY,
                ls='--', lw=1.2, label='Random (Fig.~8)')
    for m, lab, col in modes:
        tr = traces.get((m, '0.10'))
        if tr:
            ax.plot([t for t, _ in tr], [m2 for _, m2 in tr], color=col,
                    label=lab)
    ax.axvline(40.0, color='k', ls=':', lw=1.0)
    ax.text(40.4, 6, 'failure', fontsize=8.5, va='bottom', ha='left')
    ax.set_xlim(36, 55)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Throughput (Mbps)')
    ax.set_title('(a) Throughput around failure (200 ISLs)', fontsize=12)
    ax.legend(fontsize=8.5, frameon=False, loc='lower right')

    # (b) run-average packet loss (bars) + isolated portal count (twin axis).
    # Recovery time is 3.5 s for both modes at every extent (quantized by the
    # 3 s detection delay + 0.5 s polling), so it carries no discriminating
    # information and is stated in text instead of plotted.
    ax = axes[1]
    x = np.arange(len(frs))
    w = 0.36
    for i, (m, lab, col) in enumerate(modes):
        ax.bar(x + (i - 0.5) * w, loss[m], width=w, color=col, label=lab)
        for xi, v in zip(x + (i - 0.5) * w, loss[m]):
            ax.text(xi, v + 0.05, f'{v:.1f}', ha='center', fontsize=8,
                    color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nfail])
    ax.set_xlabel('Number of failed ISLs (of 2,002)')
    ax.set_ylabel('Packet loss (\\%, run avg.)')
    ax.set_ylim(0, max(max(v) for v in loss.values()) * 1.4)
    ax2 = ax.twinx()
    iso_p = [summary['fig9'][f'core_failFrac_{fr}']['isolated_portals']
             for fr in frs]
    ax2.plot(x, iso_p, color=C_PURPLE, marker='s', ls='--', lw=1.2,
             label='Isolated portals (core)')
    for xi, v in zip(x, iso_p):
        ax2.annotate(str(v), (xi, v), textcoords='offset points',
                     xytext=(0, 6), ha='center', fontsize=8, color=C_PURPLE)
    ax2.set_ylabel('Isolated portal satellites')
    ax2.set_ylim(0, max(iso_p) * 1.4)
    ax2.spines['top'].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc='upper left')
    ax.set_title('(b) Packet loss and portal isolation', fontsize=12)

    fig.savefig(os.path.join(FIG_DIR, 'fig9_core_failure.pdf'), dpi=300)
    plt.close(fig)
    print('[fig9] written;', json.dumps(summary['fig9'], indent=1)[:400])


# ================================================================
# Merged failure figure: single-panel loss bars + portal line
# (replaces the separate fig8_failure_recovery / fig9_core_failure)
# ================================================================
def make_fig_failure_merged(logs, summary):
    modes = [('random', 'Random', C_BLUE),
             ('core', 'Core-concentrated', C_RED),
             ('noncore', 'Non-core-concentrated', C_TEAL)]
    frs = FIG8_FRS

    # Recovery time is 3.5 s for all three modes at every extent (set by the
    # 3 x 1 s heartbeat detection bound + 0.5 s polling cycle, independent of
    # failure extent and placement), and the throughput traces differ only in
    # dip depth (212/122/92 Mbps from the 389 Mbps baseline); both are stated
    # in the paper text, so only the discriminating quantities are drawn.
    loss = {m: [] for m, _, _ in modes}
    nfail, iso_p = [], []
    for fr in frs:
        for m, _, _ in modes:
            job = f'fig8_ff{fr}' if m == 'random' else f'fig9_{m}_ff{fr}'
            e = logs.get(job)
            if not e or not e['result']:
                continue
            loss[m].append(e['result']['loss_pct'])
            if m == 'core':
                iso_p.append(e['isolated_portals'])
        e0 = logs.get(f'fig8_ff{fr}')
        nfail.append(e0['fail'][0] if e0 and e0['fail'] else None)

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.7))
    fig.subplots_adjust(left=0.13, right=0.87, top=0.97, bottom=0.20)

    x = np.arange(len(frs))
    w = 0.26
    for i, (m, lab, col) in enumerate(modes):
        ax.bar(x + (i - 1) * w, loss[m], width=w, color=col, label=lab)
        for xi, v in zip(x + (i - 1) * w, loss[m]):
            ax.text(xi, v + 0.05, f'{v:.1f}', ha='center', fontsize=8,
                    color=col, path_effects=_LAB_FX)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nfail])
    ax.set_xlabel('Number of failed ISLs (of 2,002)')
    ax.set_ylabel('Packet loss (\\%, run avg.)')
    ax.set_ylim(0, max(max(v) for v in loss.values()) * 1.35)

    # Right axis: isolated portals (core mode). Axis label and ticks use the
    # line color so the axis-to-series binding is unambiguous; per-point
    # values (4/8/22/43) are given in the paper text, not annotated here.
    ax2 = ax.twinx()
    ax2.plot(x, iso_p, color=C_PURPLE, marker='s', ls='--', lw=1.2,
             label='Isolated portals (core)')
    ax2.set_ylabel('Isolated portal satellites', color=C_PURPLE)
    ax2.tick_params(axis='y', colors=C_PURPLE)
    # Fixed round upper bound lifts the line above the bar value labels
    # (43 -> 86% of the axis height, clear of the 4.6% label at 75%).
    ax2.set_ylim(0, 50)
    ax2.spines['top'].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc='upper left')

    fig.savefig(os.path.join(FIG_DIR, 'fig8_failure_merged.pdf'), dpi=300)
    plt.close(fig)
    print('[fig_failure_merged] written')


def main():
    logs = load_logs()
    n_res = sum(1 for e in logs.values() if e['result'])
    print(f'parsed {len(logs)} logs, {n_res} RESULT lines')
    summary = {'fig6': {}, 'fig8': {}, 'fig_cmp': {}, 'fig9': {}}
    make_fig6(logs, summary)
    make_fig8(logs, summary)
    make_figcmp(logs, summary)
    make_fig9(logs, summary)
    make_fig_failure_merged(logs, summary)
    out = os.path.join(SCRIPT_DIR, 'eval_results_summary.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1)
    print('[saved]', out)


if __name__ == '__main__':
    main()
