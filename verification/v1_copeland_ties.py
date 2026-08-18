#!/usr/bin/env python3
"""V1: Copeland ties and decomposition of the "aggregation gain".

Investigates two questions:

  (i)  For M <= 4, does the absence of a Condorcet winner imply a TIE at the top
       of the Copeland score, meaning that aggregation on undecided events reduces
       to a tie-breaking rule (in basis_parity_sweep.py: argmax -> lowest index,
       introducing a bias towards hypothesis 0 = lowest signal level)?
  (ii) Does the bulk of the measured Copeland-vs-pooled gain originate from
       DECIDED events (pair-specific metrics vs. shared S_W) rather than
       undecided events?

Part 1: Exhaustive tournament enumeration M=3..6: among tournaments without a
        Condorcet winner, fraction with a UNIQUE Copeland maximizer.
Part 2: Replay of 120 sweep configurations (mixed {x,x^2,x^3}, s=3), event-by-event
        recording of all decision variants, and gain decomposition into decided / undecided events.
Part 3: Conservativeness verification: on decided events, all Copeland variants
        coincide with the Condorcet winner (exact check).
"""
import itertools
import json
import sys
import time

import numpy as np

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, pairwise_rules            # noqa: E402
from basis_parity_sweep import BASES, logpdf                       # noqa: E402

SEED = 20260817
N_MOM = 100_000
N_EV = 120_000
BASIS = 'mixed'


# ---------------------------------------------------------------- Part 1

def enumerate_ties(M):
    P = [(m, n) for m in range(M) for n in range(m + 1, M)]
    npat = 1 << len(P)
    n_nocw = n_nocw_unique = 0
    tie_hist = {}
    for pat in range(npat):
        out = np.zeros(M, int)
        for b, (m, n) in enumerate(P):
            if pat >> b & 1:
                out[n] += 1
            else:
                out[m] += 1
        if out.max() == M - 1:
            continue                                   # Condorcet winner exists
        n_nocw += 1
        ties = int((out == out.max()).sum())
        tie_hist[ties] = tie_hist.get(ties, 0) + 1
        if ties == 1:
            n_nocw_unique += 1
    return {'M': M, 'n_tournaments': npat, 'n_no_condorcet': n_nocw,
            'frac_no_condorcet': n_nocw / npat,
            'n_no_cw_unique_copeland_max': n_nocw_unique,
            'frac_unique_max_given_no_cw': (n_nocw_unique / n_nocw if n_nocw else 0.0),
            'top_tie_size_hist_given_no_cw': {str(k): v for k, v in sorted(tie_hist.items())}}


# ---------------------------------------------------------------- Part 2

def replay_configs():
    """Exact reproduction of the configuration generation loop from basis_parity_sweep.py."""
    rng = np.random.default_rng(2026)
    cfgs = []
    for t in range(120):
        M = int(rng.integers(3, 6))
        lv = np.sort(rng.uniform(0, 2.5, M))
        lv -= lv[0]
        cfgs.append(Scenario(lv, rng.uniform(0.0, 1.9, M)))
    return cfgs


def fit(sc, f, rng):
    Xw = np.concatenate([sc.sample(m, 60_000 // sc.M, rng) for m in range(sc.M)])
    P = f(Xw)
    mean = P.mean(0)
    C = np.cov(P, rowvar=False, ddof=1)
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(np.maximum(w, 1e-300)**-0.5) @ V.T
    tr = lambda x: (f(x) - mean) @ W                              # noqa: E731
    mus, Cs = [], []
    for m in range(sc.M):
        Q = tr(sc.sample(m, N_MOM, rng))
        mus.append(Q.mean(0))
        Cs.append(np.cov(Q, rowvar=False, ddof=1))
    mus, Cs = np.array(mus), np.array(Cs)
    K = pairwise_rules(mus, Cs)
    # margin standardization scale: sqrt(K^T F K), F = C^(m)+C^(n)
    scale = {(m, n): float(np.sqrt(k @ (Cs[m] + Cs[n]) @ k)) for (m, n), (k, c, _) in K.items()}
    SW = Cs.mean(0)
    A = np.linalg.solve(SW, mus.T).T
    b = -0.5 * np.einsum('ms,ms->m', A, mus)
    return tr, K, scale, A, b


RULES = ('pooled', 'cop_lowest', 'cop_random', 'cop_pooledtb', 'cop_margin',
         'margin_std', 'margin_raw', 'bayes')


def decide_all(Phi, x, sc, K, scale, A, b, rng):
    """Decision for all candidate rules on an array of observations; returns dict name->int array,
    along with Condorcet winner indicator and top tie size."""
    N, M = Phi.shape[0], sc.M
    out = np.zeros((N, M), np.int16)
    ms = np.zeros((N, M))          # standardized margins
    mr = np.zeros((N, M))          # raw margins
    for (i, j), (k, c, _) in K.items():
        lam = (Phi - c) @ k
        win = lam > 0
        out[:, j] += win
        out[:, i] += ~win
        mr[:, j] += lam
        mr[:, i] -= lam
        ls = lam / scale[(i, j)]
        ms[:, j] += ls
        ms[:, i] -= ls
    top = out.max(1)
    cw = top == M - 1
    tied = out == top[:, None]
    tie_size = tied.sum(1)
    pooled_sc = Phi @ A.T + b
    d = {}
    d['pooled'] = pooled_sc.argmax(1)
    d['cop_lowest'] = out.argmax(1)
    d['cop_random'] = (out + rng.uniform(0.0, 0.5, size=out.shape)).argmax(1)
    d['cop_pooledtb'] = np.where(tied, pooled_sc, -np.inf).argmax(1)
    d['cop_margin'] = np.where(tied, ms, -np.inf).argmax(1)
    d['margin_std'] = ms.argmax(1)
    d['margin_raw'] = mr.argmax(1)
    L = np.stack([logpdf(x, sc.levels[j], sc.skews[j]) for j in range(M)], 1)
    d['bayes'] = L.argmax(1)
    return d, cw, tie_size, out.argmax(1)


def evaluate(sc, f, seed=SEED):
    rng = np.random.default_rng(seed)
    tr, K, scale, A, b = fit(sc, f, rng)
    rng_tb = np.random.default_rng(seed + 1)          # fixed RNG for random tie-break
    M = sc.M
    n_all = n_fail = 0
    wrong = {r: 0 for r in RULES}
    wrong_fail = {r: 0 for r in RULES}
    n_fail_unique = 0                # failure, but unique Copeland maximizer (M>=5)
    n_dec_mismatch = {r: 0 for r in ('cop_lowest', 'cop_random', 'cop_pooledtb',
                                     'cop_margin', 'margin_std', 'margin_raw', 'pooled')}
    fail_true_class = np.zeros(M, int)
    fail_dec_class = {r: np.zeros(M, int) for r in ('cop_lowest', 'cop_random',
                                                    'cop_pooledtb', 'cop_margin', 'pooled')}
    per_class_err = {r: [] for r in RULES}
    for m in range(M):
        x = sc.sample(m, N_EV, rng)
        Phi = tr(x)
        d, cw, tie_size, cop = decide_all(Phi, x, sc, K, scale, A, b, rng_tb)
        fail = ~cw
        n_all += len(x)
        n_fail += int(fail.sum())
        n_fail_unique += int((fail & (tie_size == 1)).sum())
        fail_true_class[m] += int(fail.sum())
        for r in RULES:
            w = d[r] != m
            wrong[r] += int(w.sum())
            wrong_fail[r] += int((w & fail).sum())
            per_class_err[r].append(float(w.mean()))
        for r in n_dec_mismatch:
            n_dec_mismatch[r] += int((d[r][cw] != cop[cw]).sum())
        for r in fail_dec_class:
            fail_dec_class[r] += np.bincount(d[r][fail], minlength=M)
    n_dec = n_all - n_fail
    rec = {'M': M, 'levels': [float(v) for v in sc.levels],
           'skews': [float(v) for v in sc.skews],
           'p_fail': n_fail / n_all,
           'p_fail_unique_copeland_max': (n_fail_unique / n_fail if n_fail else 0.0),
           'n_events': n_all, 'n_fail_events': n_fail,
           'err': {r: wrong[r] / n_all for r in RULES},
           'err_fail': {r: (wrong_fail[r] / n_fail if n_fail else float('nan')) for r in RULES},
           'err_dec': {r: (wrong[r] - wrong_fail[r]) / n_dec for r in RULES},
           'per_class_err': per_class_err,
           'dec_mismatch_vs_condorcet': n_dec_mismatch,
           'fail_true_class_counts': fail_true_class.tolist(),
           'fail_decision_counts': {r: v.tolist() for r, v in fail_dec_class.items()}}
    return rec


def med(a):
    return float(np.median(a))


def summarize(rows, label):
    cop_variants = ('cop_lowest', 'cop_random', 'cop_pooledtb', 'cop_margin',
                    'margin_std', 'margin_raw')
    ep = np.array([r['err']['pooled'] for r in rows])
    pf = np.array([r['p_fail'] for r in rows])
    S = {'n_configs': len(rows), 'median_p_fail': med(pf),
         'median_err_pooled': med(ep),
         'median_err_bayes': med([r['err']['bayes'] for r in rows])}
    for v in cop_variants:
        ev = np.array([r['err'][v] for r in rows])
        gain = ep - ev
        # gain components: undecided / decided
        g_fail = np.array([r['p_fail'] * (r['err_fail']['pooled'] - r['err_fail'][v])
                           if r['n_fail_events'] else 0.0 for r in rows])
        g_dec = gain - g_fail
        with np.errstate(divide='ignore', invalid='ignore'):
            share = np.where(np.abs(gain) > 1e-9, g_fail / gain, np.nan)
        S[v] = {'median_err': med(ev),
                'median_gain_vs_pooled': med(gain),
                'wins_vs_pooled': int((gain > 0).sum()),
                'losses_vs_pooled': int((gain < 0).sum()),
                'median_gain_fail_component': med(g_fail),
                'median_gain_decisive_component': med(g_dec),
                'sum_gain': float(gain.sum()),
                'sum_gain_fail_component': float(g_fail.sum()),
                'aggregate_share_from_failures': float(g_fail.sum() / gain.sum()) if gain.sum() else float('nan'),
                'median_share_from_failures': float(np.nanmedian(share)),
                'n_share_defined': int(np.isfinite(share).sum()),
                'median_err_fail_cond': float(np.nanmedian([r['err_fail'][v] for r in rows])),
                'median_err_fail_cond_pooled_minus_this': float(np.nanmedian(
                    [r['err_fail']['pooled'] - r['err_fail'][v] for r in rows])),
                'median_err_dec_pooled_minus_this': med(
                    [r['err_dec']['pooled'] - r['err_dec'][v] for r in rows])}
    # cross-variant comparisons
    el = np.array([r['err']['cop_lowest'] for r in rows])
    er = np.array([r['err']['cop_random'] for r in rows])
    em = np.array([r['err']['cop_margin'] for r in rows])
    ept = np.array([r['err']['cop_pooledtb'] for r in rows])
    S['lowest_minus_random'] = {'median': med(el - er), 'max': float((el - er).max()),
                                'min': float((el - er).min()),
                                'lowest_better': int((el < er).sum()),
                                'random_better': int((er < el).sum())}
    S['lowest_minus_margin'] = {'median': med(el - em),
                                'lowest_better': int((el < em).sum()),
                                'margin_better': int((em < el).sum())}
    S['lowest_minus_pooledtb'] = {'median': med(el - ept),
                                  'lowest_better': int((el < ept).sum()),
                                  'pooledtb_better': int((ept < el).sum())}
    # conditional error on undecided events
    S['median_err_fail_cond'] = {r: float(np.nanmedian([r_['err_fail'][r] for r_ in rows]))
                                 for r in RULES}
    S['median_err_dec_cond'] = {r: med([r_['err_dec'][r] for r_ in rows]) for r in RULES}
    print(f'\n--- {label}: {len(rows)} configs, median P(fail)={S["median_p_fail"]:.5f} ---')
    print(f'{"rule":>14}{"med.error":>13}{"med.gain":>12}{"win/loss":>10}'
          f'{"med.gain.fail":>15}{"med.gain.dec":>14}{"share fail(agg)":>18}{"med.share":>12}')
    for v in cop_variants:
        s = S[v]
        print(f'{v:>14}{s["median_err"]:>13.5f}{s["median_gain_vs_pooled"]:>+12.5f}'
              f'{s["wins_vs_pooled"]:>5}/{s["losses_vs_pooled"]:<4}'
              f'{s["median_gain_fail_component"]:>+15.5f}{s["median_gain_decisive_component"]:>+14.5f}'
              f'{s["aggregate_share_from_failures"]:>18.3f}{s["median_share_from_failures"]:>12.3f}')
    print(f'{"pooled":>14}{S["median_err_pooled"]:>13.5f}    bayes {S["median_err_bayes"]:.5f}')
    print('Conditional error on UNDECIDED (median): ' +
          ', '.join(f'{r}={S["median_err_fail_cond"][r]:.4f}' for r in RULES))
    print('Conditional error on DECIDED (median): ' +
          ', '.join(f'{r}={S["median_err_dec_cond"][r]:.4f}' for r in RULES))
    print(f'lowest - random: median {S["lowest_minus_random"]["median"]:+.5f}, '
          f'lowest better in {S["lowest_minus_random"]["lowest_better"]}, '
          f'random better in {S["lowest_minus_random"]["random_better"]}')
    return S


if __name__ == '__main__':
    t0 = time.time()
    res = {}

    print('=== Part 1. Exhaustive Tournament Enumeration ===')
    print(f'{"M":>3}{"tournaments":>12}{"no CW":>8}{"fraction":>10}{"unique max Copeland | no CW":>32}'
          f'{"top tie size distribution":>30}')
    enum = []
    for M in (3, 4, 5, 6):
        e = enumerate_ties(M)
        enum.append(e)
        print(f'{M:>3}{e["n_tournaments"]:>12}{e["n_no_condorcet"]:>8}'
              f'{e["frac_no_condorcet"]:>10.4f}'
              f'{e["n_no_cw_unique_copeland_max"]:>8} ({e["frac_unique_max_given_no_cw"]:.4f})'
              f'{"":>10}{e["top_tie_size_hist_given_no_cw"]}')
    res['part1_enumeration'] = enum
    res['part1_analytic'] = (
        'No CW => max out-degree <= M-2. Unique max => others <= M-3 => total <= (M-2)+(M-1)(M-3) '
        '= M^2-3M+1, but total = M(M-1)/2; inequality M^2-3M+1 >= M(M-1)/2 <=> M^2-5M+2 >= 0 '
        '<=> M >= 4.56. Hence for M<=4 every no-CW tournament has a top tie of size >= 2 '
        '(M=3: 3-cycle, all scores 1; M=4: score sequences (2,2,1,1) or (2,2,2,0)).')

    print('\n=== Part 2. Replay of 120 configurations, mixed basis {x,x^2,x^3} ===')
    cfgs = replay_configs()
    old = json.load(open('verification/results_basis_parity_sweep.json'))
    assert [c.M for c in cfgs] == [r['M'] for r in old], 'M sequence mismatch'
    print('Sequence of M values matches results_basis_parity_sweep.json')
    f = BASES[BASIS]
    rows = []
    for t, sc in enumerate(cfgs):
        try:
            rec = evaluate(sc, f)
        except np.linalg.LinAlgError:
            print(f'  cfg {t}: LinAlgError, skipped')
            continue
        rec['cfg'] = t
        rec['old_sweep'] = old[t][BASIS]
        rows.append(rec)
        if t % 10 == 0 or t == 119:
            print(f'  cfg {t:3d} M={sc.M} P(fail)={rec["p_fail"]:.5f} '
                  f'pooled={rec["err"]["pooled"]:.5f} cop_lowest={rec["err"]["cop_lowest"]:.5f} '
                  f'bayes={rec["err"]["bayes"]:.5f}  [{time.time()-t0:.0f}s]')

    # comparison with original sweep (different n, same seed -> close but not identical)
    dg = np.array([r['p_fail'] - r['old_sweep']['gap'] for r in rows])
    dc = np.array([r['err']['cop_lowest'] - r['old_sweep']['copeland'] for r in rows])
    dp = np.array([r['err']['pooled'] - r['old_sweep']['pooled'] for r in rows])
    res['replay_check'] = {'max_abs_diff_gap': float(np.abs(dg).max()),
                           'max_abs_diff_copeland': float(np.abs(dc).max()),
                           'max_abs_diff_pooled': float(np.abs(dp).max()),
                           'gap_gt_1pct_here': int(sum(r['p_fail'] > 0.01 for r in rows)),
                           'gap_gt_1pct_old': int(sum(r['old_sweep']['gap'] > 0.01 for r in rows))}
    print(f'\nVerification against previous sweep: max|Δgap|={np.abs(dg).max():.5f} '
          f'max|Δcopeland|={np.abs(dc).max():.5f} max|Δpooled|={np.abs(dp).max():.5f}; '
          f'gap>1%: here {res["replay_check"]["gap_gt_1pct_here"]}, in sweep {res["replay_check"]["gap_gt_1pct_old"]}')

    print('\n=== Part 3. Conservativeness on Decided Events ===')
    tot = {}
    for r in rows:
        for k, v in r['dec_mismatch_vs_condorcet'].items():
            tot[k] = tot.get(k, 0) + v
    n_dec_total = sum(r['n_events'] - r['n_fail_events'] for r in rows)
    res['part3_conservativeness'] = {'n_decisive_events_total': n_dec_total,
                                     'mismatch_counts_vs_condorcet_winner': tot,
                                     'mismatch_frac': {k: v / n_dec_total for k, v in tot.items()}}
    for k, v in tot.items():
        print(f'  {k:>14}: mismatches with Condorcet winner on decided events: {v} '
              f'out of {n_dec_total} ({v / n_dec_total:.6f})')

    # fraction of failures with unique Copeland maximum (only M=5 possible)
    for M in (3, 4, 5):
        sub = [r for r in rows if r['M'] == M and r['n_fail_events'] > 0]
        fu = [r['p_fail_unique_copeland_max'] for r in sub]
        wsum = sum(r['n_fail_events'] * r['p_fail_unique_copeland_max'] for r in sub)
        nf = sum(r['n_fail_events'] for r in sub)
        res.setdefault('fail_unique_max_by_M', {})[str(M)] = {
            'n_configs_with_failures': len(sub),
            'weighted_frac_unique_max_given_fail': (wsum / nf if nf else None),
            'median_frac_unique_max_given_fail': (med(fu) if fu else None)}
        print(f'  M={M}: among failure events, fraction with UNIQUE Copeland max '
              f'(weighted by events) = {wsum / nf if nf else float("nan"):.4f} '
              f'({len(sub)} configurations with failures)')

    print('\n=== Summary ===')
    res['summary_all'] = summarize(rows, 'all configurations')
    sub = [r for r in rows if r['p_fail'] > 0.01]
    res['summary_gap_gt_1pct'] = summarize(sub, 'gap > 1%')
    sub_old = [r for r in rows if r['old_sweep']['gap'] > 0.01]
    res['summary_gap_gt_1pct_oldsel'] = summarize(sub_old, 'gap > 1% under old sweep (29)')

    # bias on failures: direction of decisions vs true class
    print('\n=== Lowest-index tie-break bias on failure events ===')
    agg_true = {}
    agg_dec = {r: {} for r in ('cop_lowest', 'cop_random', 'cop_pooledtb', 'cop_margin', 'pooled')}
    for r in rows:
        M = r['M']
        agg_true.setdefault(M, np.zeros(M))
        agg_true[M] += np.array(r['fail_true_class_counts'])
        for k in agg_dec:
            agg_dec[k].setdefault(M, np.zeros(M))
            agg_dec[k][M] += np.array(r['fail_decision_counts'][k])
    bias = {}
    for M in sorted(agg_true):
        n = agg_true[M].sum()
        bias[str(M)] = {'true_class_dist': (agg_true[M] / n).tolist(),
                        **{k: (agg_dec[k][M] / n).tolist() for k in agg_dec}}
        print(f'  M={M}: true class on failures {np.round(agg_true[M]/n, 3)}')
        for k in agg_dec:
            print(f'        {k:>13} chooses {np.round(agg_dec[k][M]/n, 3)}')
    res['failure_event_class_bias'] = bias

    # gain where P(fail) = 0 (pure effect of pair-specific metrics)
    print('\n=== Copeland gain when P(fail)=0 (zero failures -> decided events only) ===')
    zero = [r for r in rows if r['n_fail_events'] == 0]
    g0 = np.array([r['err']['pooled'] - r['err']['cop_lowest'] for r in zero])
    res['zero_failure_subset'] = {'n_configs': len(zero), 'median_gain': med(g0) if len(zero) else None,
                                  'wins': int((g0 > 0).sum()), 'losses': int((g0 < 0).sum())}
    print(f'  Configurations with zero failures: {len(zero)}; median gain '
          f'{med(g0) if len(zero) else float("nan"):+.5f}; win/loss {(g0>0).sum()}/{(g0<0).sum()}')
    from scipy import stats as sps
    pf = np.array([r['p_fail'] for r in rows])
    gl = np.array([r['err']['pooled'] - r['err']['cop_lowest'] for r in rows])
    rho = sps.spearmanr(pf, gl)
    res['spearman_pfail_vs_gain_lowest'] = {'rho': float(rho.statistic), 'p': float(rho.pvalue)}
    print(f'  Spearman P(fail) <-> gain(cop_lowest): rho={rho.statistic:+.3f} p={rho.pvalue:.3g}')
    byM = {}
    for M in (3, 4, 5):
        sub = [r for r in rows if r['M'] == M]
        g = np.array([r['err']['pooled'] - r['err']['cop_lowest'] for r in sub])
        gr = np.array([r['err']['pooled'] - r['err']['cop_random'] for r in sub])
        byM[str(M)] = {'n': len(sub), 'median_gain_lowest': med(g), 'wins_lowest': int((g > 0).sum()),
                       'median_gain_random': med(gr), 'median_p_fail': med([r['p_fail'] for r in sub])}
        print(f'  M={M}: n={len(sub)} median gain lowest {med(g):+.5f} (win {(g>0).sum()}), '
              f'random {med(gr):+.5f}, median P(fail) {byM[str(M)]["median_p_fail"]:.5f}')
    res['by_M'] = byM

    res['rows'] = rows
    res['settings'] = {'basis': BASIS, 's': 3, 'n_mom': N_MOM, 'n_ev_per_class': N_EV,
                       'seed': SEED, 'config_rng_seed': 2026, 'n_configs': len(rows),
                       'copeland_random_tiebreak_rng_seed': SEED + 1}
    json.dump(res, open('verification/results_v1_copeland_ties.json', 'w'), indent=1)
    print(f'\nSaved verification/results_v1_copeland_ties.json  [{time.time()-t0:.0f}s]')
