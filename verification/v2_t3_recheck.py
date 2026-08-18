#!/usr/bin/env python3
"""V2: Verification of T3 (parity_followup.py) preserving PER-CONFIGURATION values.

Investigates:
  In T3 at s=2, marginal column medians are 0.60194 (mixed+agg) vs 0.59060 (odd+agg)
  (odd is lower), yet the paired median difference is -0.00573 (favoring mixed) with
  odd winning in only 5/45 configs. This script resolves whether this is a standard statistical
  divergence (median of differences != difference of medians) or a computation/formatting artifact.

Protocols:
  A. Exact reproduction of T3: RNG seed 99, 45 configurations via rand_scen(), evaluate()
     with n_mom=100_000, n_ev=120_000, bases mixed {x^i} and odd {sgn(x)|x|^(2i-1)},
     s in {2, 3, 4}. Preserves gap, copeland, pooled, bayes per configuration.
  B. Same evaluation at s=3 on the 120-configuration generator (rng 2026) -- independent
     replication of Q-C under different sample sizes (100k/120k instead of 150k/200k);
     compared against results_basis_parity_sweep.json configuration-by-configuration.
"""
import json
import sys
import time

import numpy as np
from scipy import stats as sps

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario                                # noqa: E402
from parity_followup import rand_scen, make, sgnpow                   # noqa: E402
from basis_parity_sweep import evaluate                               # noqa: E402

N_MOM, N_EV = 100_000, 120_000
KEYS = ('gap', 'copeland', 'pooled', 'bayes')


def bases_for(s):
    mixed = make([(lambda x, i=i: x**i) for i in range(1, s + 1)])
    odd = make([sgnpow(2 * i - 1) for i in range(1, s + 1)])
    return mixed, odd


def run_config(sc, s):
    mixed, odd = bases_for(s)
    try:
        rm = evaluate(sc, mixed, n_mom=N_MOM, n_ev=N_EV)
        ro = evaluate(sc, odd, n_mom=N_MOM, n_ev=N_EV)
    except np.linalg.LinAlgError:
        return None
    return {'M': sc.M, 'levels': [float(v) for v in sc.levels],
            'skews': [float(v) for v in sc.skews],
            'mixed': dict(zip(KEYS, rm)), 'odd': dict(zip(KEYS, ro))}


def summarize(rows, label):
    """Column medians, paired differences, win counts; reported separately for errors and gap."""
    n = len(rows)
    mc = np.array([r['mixed']['copeland'] for r in rows])
    oc = np.array([r['odd']['copeland'] for r in rows])
    mp = np.array([r['mixed']['pooled'] for r in rows])
    op = np.array([r['odd']['pooled'] for r in rows])
    mb = np.array([r['mixed']['bayes'] for r in rows])
    ob = np.array([r['odd']['bayes'] for r in rows])
    mg = np.array([r['mixed']['gap'] for r in rows])
    og = np.array([r['odd']['gap'] for r in rows])
    Ms = np.array([r['M'] for r in rows])
    d = mc - oc                       # <0 => mixed+agg better (lower error)
    dg = mg - og                      # >0 => odd has lower undecided rate
    # sign test (two-sided) for d
    n_nz = int((d != 0).sum())
    p_sign = float(sps.binomtest(int((d < 0).sum()), n_nz, 0.5).pvalue) if n_nz else 1.0
    try:
        p_wilc = float(sps.wilcoxon(d).pvalue)
    except ValueError:
        p_wilc = 1.0
    out = {
        'label': label, 'n': n, 'M_counts': {int(k): int(v) for k, v in
                                             zip(*np.unique(Ms, return_counts=True))},
        'col_median_mixed_cop': float(np.median(mc)),
        'col_median_odd_cop': float(np.median(oc)),
        'col_median_diff_of_medians': float(np.median(mc) - np.median(oc)),
        'paired_median_diff_cop': float(np.median(d)),
        'paired_mean_diff_cop': float(np.mean(d)),
        'paired_sd_diff_cop': float(np.std(d, ddof=1)),
        'paired_min_diff': float(d.min()), 'paired_max_diff': float(d.max()),
        'odd_wins_cop': int((oc < mc).sum()), 'mixed_wins_cop': int((mc < oc).sum()),
        'ties_cop': int((mc == oc).sum()),
        'sign_test_p': p_sign, 'wilcoxon_p': p_wilc,
        'col_median_mixed_pool': float(np.median(mp)),
        'col_median_odd_pool': float(np.median(op)),
        'paired_median_diff_pool': float(np.median(mp - op)),
        'odd_wins_pool': int((op < mp).sum()),
        'bayes_median': float(np.median(mb)),
        'bayes_mixed_vs_odd_max_absdiff': float(np.abs(mb - ob).max()),
        'gap_to_bayes_median_mixed_cop': float(np.median(mc - mb)),
        'gap_to_bayes_median_odd_cop': float(np.median(oc - mb)),
        'gap_to_bayes_median_mixed_pool': float(np.median(mp - mb)),
        'gap_to_bayes_median_odd_pool': float(np.median(op - mb)),
        # aggregation vs pooling within basis
        'agg_gain_mixed_median': float(np.median(mp - mc)),
        'agg_gain_mixed_wins': int((mc < mp).sum()),
        'agg_gain_odd_median': float(np.median(op - oc)),
        'agg_gain_odd_wins': int((oc < op).sum()),
        # undecided gap
        'gap_median_mixed': float(np.median(mg)), 'gap_median_odd': float(np.median(og)),
        'gap_max_mixed': float(mg.max()), 'gap_max_odd': float(og.max()),
        'gap_gt1pct_mixed': int((mg > .01).sum()), 'gap_gt1pct_odd': int((og > .01).sum()),
        'gap_gt5pct_mixed': int((mg > .05).sum()), 'gap_gt5pct_odd': int((og > .05).sum()),
        'gap_paired_median_diff': float(np.median(dg)),
        'gap_paired_mean_diff': float(np.mean(dg)),
        'gap_odd_lower': int((og < mg).sum()), 'gap_mixed_lower': int((mg < og).sum()),
        'gap_ties': int((mg == og).sum()),
        'gap_odd_lower_among_mixed_gt1pct': int(((og < mg) & (mg > .01)).sum()),
        'gap_ratio_median_where_mixed_gt0': float(np.median(
            og[mg > 0] / mg[mg > 0])) if (mg > 0).any() else float('nan'),
    }
    # breakdown by M
    by_M = {}
    for Mv in sorted(set(Ms.tolist())):
        sel = Ms == Mv
        by_M[int(Mv)] = {'n': int(sel.sum()),
                         'col_med_mixed': float(np.median(mc[sel])),
                         'col_med_odd': float(np.median(oc[sel])),
                         'paired_med_diff': float(np.median(d[sel])),
                         'odd_wins': int((oc[sel] < mc[sel]).sum()),
                         'gap_med_mixed': float(np.median(mg[sel])),
                         'gap_med_odd': float(np.median(og[sel]))}
    out['by_M'] = by_M
    # configurations where odd wins, and top-5 by |d|
    idx_odd = np.where(oc < mc)[0]
    out['odd_win_configs'] = [{'idx': int(i), 'M': int(Ms[i]), 'mixed_cop': float(mc[i]),
                               'odd_cop': float(oc[i]), 'diff': float(d[i]),
                               'gap_mixed': float(mg[i]), 'gap_odd': float(og[i])}
                              for i in idx_odd]
    top = np.argsort(-np.abs(d))[:5]
    out['top5_absdiff'] = [{'idx': int(i), 'M': int(Ms[i]), 'mixed_cop': float(mc[i]),
                            'odd_cop': float(oc[i]), 'diff': float(d[i])} for i in top]
    return out


def show(S):
    print(f"\n--- {S['label']} (n={S['n']}, M: {S['M_counts']}) ---")
    print(f"  Copeland error: col.median mixed={S['col_median_mixed_cop']:.5f} "
          f"odd={S['col_median_odd_cop']:.5f} (diff of medians {S['col_median_diff_of_medians']:+.5f})")
    print(f"  paired mixed-odd: median={S['paired_median_diff_cop']:+.5f} "
          f"mean={S['paired_mean_diff_cop']:+.5f} sd={S['paired_sd_diff_cop']:.5f} "
          f"range=[{S['paired_min_diff']:+.5f}, {S['paired_max_diff']:+.5f}]")
    print(f"  wins: mixed {S['mixed_wins_cop']}, odd {S['odd_wins_cop']}, ties {S['ties_cop']}; "
          f"sign p={S['sign_test_p']:.2e}, Wilcoxon p={S['wilcoxon_p']:.2e}")
    print(f"  Pooled error: col.median mixed={S['col_median_mixed_pool']:.5f} "
          f"odd={S['col_median_odd_pool']:.5f}; paired median={S['paired_median_diff_pool']:+.5f}; "
          f"odd wins {S['odd_wins_pool']}")
    print(f"  Bayes median={S['bayes_median']:.5f}; gap-to-Bayes medians: mixed+agg "
          f"{S['gap_to_bayes_median_mixed_cop']:.5f}, odd+agg {S['gap_to_bayes_median_odd_cop']:.5f}, "
          f"mixed+pool {S['gap_to_bayes_median_mixed_pool']:.5f}, odd+pool {S['gap_to_bayes_median_odd_pool']:.5f}")
    print(f"  agg gain (pool-cop): mixed median {S['agg_gain_mixed_median']:+.5f} ({S['agg_gain_mixed_wins']} wins), "
          f"odd median {S['agg_gain_odd_median']:+.5f} ({S['agg_gain_odd_wins']} wins)")
    print(f"  GAP: median mixed={S['gap_median_mixed']:.5f} odd={S['gap_median_odd']:.5f}; "
          f"max {S['gap_max_mixed']:.5f}/{S['gap_max_odd']:.5f}; >1%: {S['gap_gt1pct_mixed']}/{S['gap_gt1pct_odd']}; "
          f">5%: {S['gap_gt5pct_mixed']}/{S['gap_gt5pct_odd']}")
    print(f"  GAP paired (mixed-odd): median {S['gap_paired_median_diff']:+.5f} mean {S['gap_paired_mean_diff']:+.5f}; "
          f"odd lower in {S['gap_odd_lower']}, mixed lower in {S['gap_mixed_lower']}, ties {S['gap_ties']}; "
          f"odd lower among mixed>1%: {S['gap_odd_lower_among_mixed_gt1pct']}/{S['gap_gt1pct_mixed']}; "
          f"median ratio odd/mixed (mixed>0): {S['gap_ratio_median_where_mixed_gt0']:.3f}")
    for Mv, b in S['by_M'].items():
        print(f"    M={Mv}: n={b['n']} col.med mixed={b['col_med_mixed']:.5f} odd={b['col_med_odd']:.5f} "
              f"paired med={b['paired_med_diff']:+.5f} odd wins {b['odd_wins']}; "
              f"gap med {b['gap_med_mixed']:.5f}/{b['gap_med_odd']:.5f}")
    if S['odd_win_configs']:
        print('  configs where odd+agg wins:')
        for c in S['odd_win_configs']:
            print(f"    idx={c['idx']:>3} M={c['M']} mixed={c['mixed_cop']:.5f} odd={c['odd_cop']:.5f} "
                  f"diff={c['diff']:+.5f} gap {c['gap_mixed']:.5f}/{c['gap_odd']:.5f}")
    print('  top-5 |diff|:')
    for c in S['top5_absdiff']:
        print(f"    idx={c['idx']:>3} M={c['M']} mixed={c['mixed_cop']:.5f} odd={c['odd_cop']:.5f} diff={c['diff']:+.5f}")


if __name__ == '__main__':
    t0 = time.time()
    out = {'params': {'n_mom': N_MOM, 'n_ev': N_EV, 'seed_T3': 99, 'n_T3': 45,
                      'seed_120': 2026, 'n_120': 120}}

    # ---------------------------------------------------------------- A: T3 exact replay
    print('=== A. Exact T3 Replay (seed 99, 45 configurations), s = 2, 3, 4 ===')
    A = {}
    for s in (2, 3, 4):
        rng = np.random.default_rng(99)
        rows, skipped = [], 0
        for t in range(45):
            sc = rand_scen(rng)
            r = run_config(sc, s)
            if r is None:
                skipped += 1
                continue
            r['idx'] = t
            rows.append(r)
            if (t + 1) % 15 == 0:
                print(f'  s={s}: {t+1}/45 done, {time.time()-t0:.0f}s')
        S = summarize(rows, f'T3 s={s}')
        S['skipped'] = skipped
        show(S)
        A[str(s)] = {'summary': S, 'rows': rows}
    out['A_T3'] = A

    # comparison with values printed in results_parity_followup.json
    try:
        old = json.load(open('verification/results_parity_followup.json'))['T3']
        print('\n  Comparison against results_parity_followup.json T3:')
        for o in old:
            S = A[str(o['s'])]['summary']
            print(f"    s={o['s']}: old mixed_cop={o['mixed_cop']:.5f} odd_cop={o['odd_cop']:.5f} "
                  f"odd_wins={o['odd_wins']}/{o['n']}  | new mixed={S['col_median_mixed_cop']:.5f} "
                  f"odd={S['col_median_odd_cop']:.5f} odd_wins={S['odd_wins_cop']}/{S['n']}")
    except FileNotFoundError:
        pass

    # ---------------------------------------------------------------- B: 120-config generator, s=3
    print('\n=== B. 120-Configuration Generator (seed 2026), s = 3, Q-C Replication ===')
    rng = np.random.default_rng(2026)
    rows, skipped = [], 0
    for t in range(120):
        M = int(rng.integers(3, 6))
        lv = np.sort(rng.uniform(0, 2.5, M))
        lv -= lv[0]
        sc = Scenario(lv, rng.uniform(0.0, 1.9, M))
        r = run_config(sc, 3)
        if r is None:
            skipped += 1
            continue
        r['idx'] = t
        rows.append(r)
        if (t + 1) % 30 == 0:
            print(f'  {t+1}/120 done, {time.time()-t0:.0f}s')
    S = summarize(rows, '120-config s=3')
    S['skipped'] = skipped
    show(S)
    out['B_120'] = {'summary': S, 'rows': rows}

    # config-by-config comparison with results_basis_parity_sweep.json (150k/200k)
    try:
        old = json.load(open('verification/results_basis_parity_sweep.json'))
        if len(old) == len(rows):
            omc = np.array([r['mixed']['copeland'] for r in old])
            ooc = np.array([r['odd']['copeland'] for r in old])
            nmc = np.array([r['mixed']['copeland'] for r in rows])
            noc = np.array([r['odd']['copeland'] for r in rows])
            omg = np.array([r['mixed']['gap'] for r in old])
            nmg = np.array([r['mixed']['gap'] for r in rows])
            cmp = {'n': len(old),
                   'old_median_diff': float(np.median(omc - ooc)),
                   'new_median_diff': float(np.median(nmc - noc)),
                   'old_odd_wins': int((ooc < omc).sum()), 'new_odd_wins': int((noc < nmc).sum()),
                   'max_abs_change_mixed_cop': float(np.abs(omc - nmc).max()),
                   'max_abs_change_odd_cop': float(np.abs(ooc - noc).max()),
                   'sign_agreement_of_diff': int(((omc - ooc) * (nmc - noc) > 0).sum()),
                   'max_abs_change_mixed_gap': float(np.abs(omg - nmg).max()),
                   'spearman_mixed_gap_old_new': float(sps.spearmanr(omg, nmg).correlation)}
            print('\n  Comparison against results_basis_parity_sweep.json (150k/200k):')
            for k, v in cmp.items():
                print(f'    {k}: {v}')
            out['B_120']['vs_old_sweep'] = cmp
    except FileNotFoundError:
        pass

    json.dump(out, open('verification/results_v2_t3_recheck.json', 'w'), indent=1)
    print(f'\nSaved verification/results_v2_t3_recheck.json  ({time.time()-t0:.0f}s)')
