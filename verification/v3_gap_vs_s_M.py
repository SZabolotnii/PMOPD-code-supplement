#!/usr/bin/env python3
"""V3: Undecided rate of pairwise scheme vs. s and M -- data for main letter figure.

Measures on IDENTICAL configurations across all s (paired design):
    basis   mixed power {x, ..., x^s}, whitened by pooled covariance
    s       1..6
    (a)     fixed scenarios SCENARIOS + WORST, 3 seeds
    (b)     random: 30 configurations for each M in {3, 4, 5, 6}, rng(300+M)

Metrics per (configuration, s):
    gap      P(no Condorcet winner)        -- scheme failure
    cycle    P(tournament is intransitive) -- gap subset of cycle, coincidence only for M<=3
    cond_raw condition number of raw basis pooled covariance (before whitening)
    cond_F   max_{m<n} cond(C^(m)+C^(n)) after whitening
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, raw_basis, pairwise_rules      # noqa: E402
from p5_diagnostic_aggregation import SCENARIOS                        # noqa: E402

SEED = 20260817
S_LIST = [1, 2, 3, 4, 5, 6]
M_LIST = [3, 4, 5, 6]
N_CFG = 30
N_MOM = 100_000
N_EV = 120_000
N_WHITEN = 60_000

WORST = Scenario([0.0, 0.04, 1.28, 1.39, 1.45], [0.05, 1.83, 1.36, 0.39, 1.39])


def evaluate(sc, s, n_mom=N_MOM, n_ev=N_EV, seed=SEED):
    """Returns dict with gap, cycle, cond_raw, cond_F (or err)."""
    rng = np.random.default_rng(seed)
    f = lambda x: raw_basis(x, 'power', s)                              # noqa: E731
    Xw = np.concatenate([sc.sample(m, N_WHITEN // sc.M, rng) for m in range(sc.M)])
    P = f(Xw)
    mean = P.mean(0)
    C = np.cov(P, rowvar=False, ddof=1).reshape(s, s)
    cond_raw = float(np.linalg.cond(C))
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(np.maximum(w, 1e-300)**-0.5) @ V.T
    tr = lambda x: (f(x) - mean) @ W                                    # noqa: E731

    mus, Cs = [], []
    for m in range(sc.M):
        Q = tr(sc.sample(m, n_mom, rng))
        mus.append(Q.mean(0))
        Cs.append(np.cov(Q, rowvar=False, ddof=1).reshape(s, s))
    mus, Cs = np.array(mus), np.array(Cs)
    try:
        K = pairwise_rules(mus, Cs)
    except np.linalg.LinAlgError as e:
        return {'err': str(e), 'cond_raw': cond_raw}
    cond_F = float(max(v[2] for v in K.values()))

    gaps, cycs = [], []
    M = sc.M
    for m in range(M):
        Phi = tr(sc.sample(m, n_ev, rng))
        out = np.zeros((n_ev, M), int)
        for (i, j), (k, c, _) in K.items():
            win = (Phi - c) @ k > 0
            out[:, j] += win
            out[:, i] += ~win
        gaps.append((~(out == M - 1).any(1)).mean())          # no Condorcet winner
        cycs.append((~(np.sort(out, 1) == np.arange(M)).all(1)).mean())   # intransitive
    return {'gap': float(np.mean(gaps)), 'cycle': float(np.mean(cycs)),
            'cond_raw': cond_raw, 'cond_F': cond_F}


def random_configs(M, n=N_CFG):
    rng = np.random.default_rng(300 + M)
    cfgs = []
    for _ in range(n):
        lv = np.sort(rng.uniform(0, 2.5, M))
        lv -= lv[0]
        cfgs.append(Scenario(lv, rng.uniform(0, 1.9, M)))
    return cfgs


def summarize(vals):
    v = np.array(vals, float)
    return {'n': int(len(v)), 'median': float(np.median(v)),
            'mean': float(v.mean()), 'p90': float(np.percentile(v, 90)),
            'max': float(v.max()), 'n_gt_1pct': int((v > 0.01).sum()),
            'n_gt_5pct': int((v > 0.05).sum()), 'n_gt_0': int((v > 0).sum())}


if __name__ == '__main__':
    t0 = time.time()
    out = {'params': {'S': S_LIST, 'M': M_LIST, 'n_cfg': N_CFG, 'n_mom': N_MOM,
                      'n_ev_per_class': N_EV, 'n_whiten': N_WHITEN, 'basis': 'power (mixed), whitened'}}

    # ------------------------------------------------------------ (a) fixed
    print('=== (a) Fixed scenarios, 3 seeds (mean; min..max in parens) ===\n')
    fixed = [(name, sc) for name, sc, _ in SCENARIOS]
    fixed.append(('WORST (M=5) [= SCENARIOS[3]]', WORST))
    out['fixed'] = {}
    hdr = f'{"scenario":>40}' + ''.join(f'{f"s={s}":>10}' for s in S_LIST)
    print('gap (P no Condorcet winner):')
    print(hdr)
    for name, sc in fixed:
        rec = {'levels': sc.levels, 'skews': sc.skews, 'M': sc.M, 'by_s': {}}
        line = f'{name[:40]:>40}'
        for s in S_LIST:
            rs = [evaluate(sc, s, seed=SEED + 77 * i) for i in range(3)]
            g = [r['gap'] for r in rs]; c = [r['cycle'] for r in rs]
            rec['by_s'][s] = {'gap_mean': float(np.mean(g)), 'gap_min': float(min(g)),
                              'gap_max': float(max(g)), 'cycle_mean': float(np.mean(c)),
                              'cond_raw': float(np.mean([r['cond_raw'] for r in rs])),
                              'cond_F': float(np.mean([r['cond_F'] for r in rs]))}
            line += f'{np.mean(g):>10.5f}'
        print(line)
        out['fixed'][name] = rec
    print('\ncycle (P intransitive tournament):')
    print(hdr)
    for name, sc in fixed:
        print(f'{name[:40]:>40}' + ''.join(
            f'{out["fixed"][name]["by_s"][s]["cycle_mean"]:>10.5f}' for s in S_LIST))
    print('\ncond_F after whitening (max across pairs):')
    print(hdr)
    for name, sc in fixed:
        print(f'{name[:40]:>40}' + ''.join(
            f'{out["fixed"][name]["by_s"][s]["cond_F"]:>10.1f}' for s in S_LIST))
    print('\ncond of raw basis (before whitening):')
    print(hdr)
    for name, sc in fixed:
        print(f'{name[:40]:>40}' + ''.join(
            f'{out["fixed"][name]["by_s"][s]["cond_raw"]:>10.2e}' for s in S_LIST))
    print(f'\n[{time.time()-t0:.0f} s]\n')

    # ------------------------------------------------------------ (b) random
    print('=== (b) Random configurations, paired design (same configs for all s) ===\n')
    out['random'] = {}
    for M in M_LIST:
        cfgs = random_configs(M)
        rows = []
        for ci, sc in enumerate(cfgs):
            row = {'levels': sc.levels, 'skews': sc.skews, 'by_s': {}}
            for s in S_LIST:
                row['by_s'][s] = evaluate(sc, s)
            rows.append(row)
        out['random'][M] = rows
        print(f'  M={M}: {len(rows)} configurations, {time.time()-t0:.0f} s')

    # ------------------------------------------------------------ summary
    print('\n=== Summary of gap across (M, s) ===')
    print(f'{"M":>3}{"s":>3}{"median":>10}{"mean":>10}{"p90":>10}{"max":>10}'
          f'{">1%":>6}{">5%":>6}{">0":>5}{"err":>5}')
    out['summary_gap'] = {}
    out['summary_cycle'] = {}
    out['summary_cond'] = {}
    for M in M_LIST:
        for s in S_LIST:
            rs = [r['by_s'][s] for r in out['random'][M]]
            ok = [r for r in rs if 'gap' in r]
            g = summarize([r['gap'] for r in ok])
            c = summarize([r['cycle'] for r in ok])
            cd = {'cond_raw_median': float(np.median([r['cond_raw'] for r in rs])),
                  'cond_raw_max': float(np.max([r['cond_raw'] for r in rs])),
                  'cond_F_median': float(np.median([r['cond_F'] for r in ok])),
                  'cond_F_max': float(np.max([r['cond_F'] for r in ok])),
                  'n_err': len(rs) - len(ok)}
            out['summary_gap'][f'M{M}_s{s}'] = g
            out['summary_cycle'][f'M{M}_s{s}'] = c
            out['summary_cond'][f'M{M}_s{s}'] = cd
            print(f'{M:>3}{s:>3}{g["median"]:>10.5f}{g["mean"]:>10.5f}{g["p90"]:>10.5f}'
                  f'{g["max"]:>10.5f}{g["n_gt_1pct"]:>6}{g["n_gt_5pct"]:>6}'
                  f'{g["n_gt_0"]:>5}{cd["n_err"]:>5}')

    print('\n=== Summary of cycle across (M, s) ===')
    print(f'{"M":>3}{"s":>3}{"median":>10}{"mean":>10}{"p90":>10}{"max":>10}'
          f'{">1%":>6}{">5%":>6}')
    for M in M_LIST:
        for s in S_LIST:
            c = out['summary_cycle'][f'M{M}_s{s}']
            print(f'{M:>3}{s:>3}{c["median"]:>10.5f}{c["mean"]:>10.5f}{c["p90"]:>10.5f}'
                  f'{c["max"]:>10.5f}{c["n_gt_1pct"]:>6}{c["n_gt_5pct"]:>6}')

    print('\n=== Conditioning across (M, s) ===')
    print(f'{"M":>3}{"s":>3}{"cond_raw med":>14}{"cond_raw max":>14}'
          f'{"cond_F med":>12}{"cond_F max":>12}')
    for M in M_LIST:
        for s in S_LIST:
            cd = out['summary_cond'][f'M{M}_s{s}']
            print(f'{M:>3}{s:>3}{cd["cond_raw_median"]:>14.2e}{cd["cond_raw_max"]:>14.2e}'
                  f'{cd["cond_F_median"]:>12.2f}{cd["cond_F_max"]:>12.2f}')

    # ------------------------------------------------------------ paired trends over s
    print('\n=== Paired trend over s (same config): fraction gap(s+1) > gap(s) ===')
    out['paired_s'] = {}
    for M in M_LIST:
        line = f'  M={M}: '
        rec = {}
        for s in S_LIST[:-1]:
            a = np.array([r['by_s'][s].get('gap', np.nan) for r in out['random'][M]])
            b = np.array([r['by_s'][s + 1].get('gap', np.nan) for r in out['random'][M]])
            ok = ~np.isnan(a) & ~np.isnan(b)
            up = int((b[ok] > a[ok]).sum()); dn = int((b[ok] < a[ok]).sum())
            tie = int(ok.sum()) - up - dn
            rec[f'{s}->{s+1}'] = {'up': up, 'down': dn, 'tie': tie,
                                  'median_diff': float(np.median(b[ok] - a[ok]))}
            line += f'{s}->{s+1}: {up}↑/{dn}↓/{tie}=  '
        out['paired_s'][M] = rec
        print(line)

    # per-config max over s>=2 and argmax
    print('\n=== For each configuration: degree s maximizing undecided rate (s>=2) ===')
    out['argmax_s'] = {}
    for M in M_LIST:
        am = []
        for r in out['random'][M]:
            g = [r['by_s'][s].get('gap', -1) for s in S_LIST[1:]]
            am.append(S_LIST[1:][int(np.argmax(g))])
        cnt = {s: int(sum(1 for a in am if a == s)) for s in S_LIST[1:]}
        out['argmax_s'][M] = cnt
        print(f'  M={M}: ' + '  '.join(f's={s}:{cnt[s]}' for s in S_LIST[1:]))

    # any-s pooled: max over s per config
    print('\n=== max_s gap per configuration (s>=2): distribution by M ===')
    out['max_over_s'] = {}
    for M in M_LIST:
        mx = [max(r['by_s'][s].get('gap', 0) for s in S_LIST[1:]) for r in out['random'][M]]
        sm = summarize(mx)
        out['max_over_s'][M] = sm
        print(f'  M={M}: median {sm["median"]:.5f}  p90 {sm["p90"]:.5f}  max {sm["max"]:.5f}'
              f'  >1%: {sm["n_gt_1pct"]}/{sm["n"]}  >5%: {sm["n_gt_5pct"]}/{sm["n"]}')

    # s=1 exact zero check
    z = [r['by_s'][1]['gap'] for M in M_LIST for r in out['random'][M]]
    zc = [r['by_s'][1]['cycle'] for M in M_LIST for r in out['random'][M]]
    zf = [out['fixed'][n]['by_s'][1]['gap_max'] for n in out['fixed']]
    out['s1_check'] = {'max_gap_random': float(max(z)), 'max_cycle_random': float(max(zc)),
                       'max_gap_fixed': float(max(zf)),
                       'all_exactly_zero': bool(max(z) == 0 and max(zc) == 0 and max(zf) == 0)}
    print(f'\ns=1: max gap (random) = {max(z)}, max cycle = {max(zc)}, '
          f'max gap (fixed) = {max(zf)}  -> identically zero: {out["s1_check"]["all_exactly_zero"]}')

    # JSON: int keys -> str
    def fix(o):
        if isinstance(o, dict):
            return {str(k): fix(v) for k, v in o.items()}
        if isinstance(o, list):
            return [fix(v) for v in o]
        return o
    json.dump(fix(out), open('verification/results_v3_gap_vs_s_M.json', 'w'), indent=1)
    print(f'\nSaved verification/results_v3_gap_vs_s_M.json  [{time.time()-t0:.0f} s]')
