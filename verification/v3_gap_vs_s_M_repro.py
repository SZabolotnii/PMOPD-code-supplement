#!/usr/bin/env python3
"""V3-repro: reproducibility of v3_gap_vs_s_M.py figures and verification
that large-s undecided events are not an artifact of moment estimation error.

(a) Rerun of random configurations (identical configurations, independent RNG seed
    for whitening/moments/evaluation) -> summary across (M, s), cross-seed rank
    correlation per configuration, median absolute difference.
(b) Sensitivity to n_mom: WORST + 3 random configurations at M=6, s in 2..6,
    n_mom in {25k, 100k, 400k, 1.6M} (n_ev = 120k per class). If gap decays
    substantially with n_mom, part of the undecided events would be driven by
    estimation noise in mu and C rather than intrinsic scheme geometry.
"""
import json
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, 'verification')
from v3_gap_vs_s_M import (evaluate, random_configs, summarize, WORST,   # noqa: E402
                           S_LIST, M_LIST, SEED)

if __name__ == '__main__':
    t0 = time.time()
    out = {}
    base = json.load(open('verification/results_v3_gap_vs_s_M.json'))

    # ------------------------------------------------------------ (a)
    print('=== (a) Independent rerun, seed SEED+1 ===\n')
    out['rerun'] = {}
    out['rerun_summary'] = {}
    out['seed_agreement'] = {}
    for M in M_LIST:
        cfgs = random_configs(M)
        rows = []
        for sc in cfgs:
            rows.append({s: evaluate(sc, s, seed=SEED + 1) for s in S_LIST})
        out['rerun'][M] = rows
        print(f'  M={M}: {time.time()-t0:.0f} s')
    print(f'\n{"M":>3}{"s":>3}{"med. seed0":>11}{"med. seed1":>11}{"max0":>9}{"max1":>9}'
          f'{">1% s0":>8}{">1% s1":>8}{"Spearman":>10}{"med|Δ|":>10}{"med gap":>10}')
    for M in M_LIST:
        for s in S_LIST[1:]:
            g0 = np.array([r['by_s'][str(s)]['gap'] for r in base['random'][str(M)]])
            g1 = np.array([r[s]['gap'] for r in out['rerun'][M]])
            sm = summarize(g1)
            rho = float(spearmanr(g0, g1).correlation)
            rec = {'spearman': rho, 'median_absdiff': float(np.median(np.abs(g0 - g1))),
                   'median_gap_pooled': float(np.median((g0 + g1) / 2)),
                   'n_gt_1pct_seed0': int((g0 > .01).sum()), 'n_gt_1pct_seed1': int((g1 > .01).sum()),
                   'n_gt_1pct_both': int(((g0 > .01) & (g1 > .01)).sum())}
            out['rerun_summary'][f'M{M}_s{s}'] = sm
            out['seed_agreement'][f'M{M}_s{s}'] = rec
            print(f'{M:>3}{s:>3}{np.median(g0):>11.5f}{np.median(g1):>11.5f}'
                  f'{g0.max():>9.4f}{g1.max():>9.4f}{(g0>.01).sum():>8}{(g1>.01).sum():>8}'
                  f'{rho:>10.3f}{rec["median_absdiff"]:>10.5f}{rec["median_gap_pooled"]:>10.5f}')

    # paired trend over s averaged across two seeds
    print('\n=== Paired trend over s averaged across two seeds ===')
    out['paired_s_2seeds'] = {}
    for M in M_LIST:
        line = f'  M={M}: '
        rec = {}
        for s in S_LIST[:-1]:
            a = np.array([(r['by_s'][str(s)]['gap'] + q[s]['gap']) / 2
                          for r, q in zip(base['random'][str(M)], out['rerun'][M])])
            b = np.array([(r['by_s'][str(s + 1)]['gap'] + q[s + 1]['gap']) / 2
                          for r, q in zip(base['random'][str(M)], out['rerun'][M])])
            up, dn = int((b > a).sum()), int((b < a).sum())
            rec[f'{s}->{s+1}'] = {'up': up, 'down': dn, 'median_diff': float(np.median(b - a))}
            line += f'{s}->{s+1}: {up}↑/{dn}↓  '
        out['paired_s_2seeds'][M] = rec
        print(line)

    # ------------------------------------------------------------ (b)
    print('\n=== (b) Sensitivity to n_mom (n_ev = 120k/class) ===\n')
    NM = [25_000, 100_000, 400_000, 1_600_000]
    probes = [('WORST M=5', WORST)] + [(f'rand M=6 #{i}', c) for i, c in
                                       enumerate(random_configs(6)[:3])]
    out['nmom'] = {}
    print(f'{"configuration":>14}{"s":>3}' + ''.join(f'{f"n_mom={n//1000}k":>13}' for n in NM)
          + f'{"cond_F@1.6M":>13}')
    for name, sc in probes:
        out['nmom'][name] = {}
        for s in S_LIST[1:]:
            rs = [evaluate(sc, s, n_mom=n, seed=SEED + 5) for n in NM]
            out['nmom'][name][s] = {str(n): r for n, r in zip(NM, rs)}
            print(f'{name:>14}{s:>3}' + ''.join(f'{r["gap"]:>13.5f}' for r in rs)
                  + f'{rs[-1]["cond_F"]:>13.1f}')
        print(f'   [{time.time()-t0:.0f} s]')

    def fix(o):
        if isinstance(o, dict):
            return {str(k): fix(v) for k, v in o.items()}
        if isinstance(o, list):
            return [fix(v) for v in o]
        return o
    json.dump(fix(out), open('verification/results_v3_gap_vs_s_M_repro.json', 'w'), indent=1)
    print(f'\nSaved verification/results_v3_gap_vs_s_M_repro.json  [{time.time()-t0:.0f} s]')
