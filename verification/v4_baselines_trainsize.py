#!/usr/bin/env python3
"""V4: Baselines and robustness to training sample size (for Section V of the SPL letter).

Part 1 (baselines): 120 configurations (rng 2026, identical configuration loop as in
basis_parity_sweep.py), power basis {x, ..., x^s}. On IDENTICAL sample splits
(whitening, moments, test) for s in 1..4, computes:
    gap_s      P(no Condorcet winner) for pairwise scheme of degree s
    cop_s      error of pairwise + Copeland (tie -> lowest index)
    pool_s     error of consistent pooled rule on shared S_W
    bayes      Bayes optimal bound from true densities
The primary baseline of the letter is the linear rule s = 1, which coincides with the
nearest-mean classifier on x; exact agreement is checked numerically per configuration.

Part 2 (train-size): 6 configurations with the highest gap_3 + 3 with gap_3 < 1e-4.
Rules (whitening + moments) are re-estimated from realistic training sample sizes
n_train per class in {500, 2000, 10000} (same training set for whitening and moments),
20 replicas, and evaluated on a fixed large test set. Evaluates whether the pairwise
undecided behavior persists under finite training, whether estimation noise amplifies
undecided rates, and whether the ordering Copeland < pooled is maintained at n = 500.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, pairwise_rules            # noqa: E402
from basis_parity_sweep import logpdf                             # noqa: E402

SEED = 20260817
S_MAX = 4
N_MOM = 100_000
N_EV = 120_000
N_WHITEN = 60_000          # as in basis_parity_sweep.evaluate
OUT = 'verification/results_v4_baselines_trainsize.json'


def replay_configs(n=120):
    rng = np.random.default_rng(2026)
    cfgs = []
    for t in range(n):
        M = int(rng.integers(3, 6))
        lv = np.sort(rng.uniform(0, 2.5, M))
        lv -= lv[0]
        cfgs.append(Scenario(lv, rng.uniform(0.0, 1.9, M)))
    return cfgs


def powers(x, s):
    return np.stack([x**i for i in range(1, s + 1)], 1)


class Rules:
    """Whitening + moments + pairwise and pooled decision rules of degree s."""

    def __init__(self, s, Xw, Xtr):
        """Xw -- pooled sample for whitening, Xtr -- list of per-class training samples."""
        self.s = s
        P = powers(Xw, s)
        self.mean = P.mean(0)
        C = np.cov(P, rowvar=False, ddof=1).reshape(s, s)
        w, V = np.linalg.eigh(C)
        self.W = V @ np.diag(np.maximum(w, 1e-300)**-0.5) @ V.T
        mus, Cs = [], []
        for x in Xtr:
            Q = self.tr(x)
            mus.append(Q.mean(0))
            Cs.append(np.cov(Q, rowvar=False, ddof=1).reshape(s, s))
        self.mus, self.Cs = np.array(mus), np.array(Cs)
        self.M = len(Xtr)
        self.K = pairwise_rules(self.mus, self.Cs)
        SW = self.Cs.mean(0)
        self.A = np.linalg.solve(SW, self.mus.T).T
        self.b = -0.5 * np.einsum('ms,ms->m', self.A, self.mus)

    def tr(self, x):
        return (powers(x, self.s) - self.mean) @ self.W

    def decide(self, x):
        """-> (copeland_winner, pooled_winner, no_condorcet_flag) on array x."""
        Phi = self.tr(x)
        out = np.zeros((len(x), self.M), np.int8)
        for (i, j), (k, c, _) in self.K.items():
            win = (Phi - c) @ k > 0
            out[:, j] += win
            out[:, i] += ~win
        gap = ~(out == self.M - 1).any(1)
        cop = out.argmax(1)                       # tie -> lowest index
        pool = (Phi @ self.A.T + self.b).argmax(1)
        return cop, pool, gap


def part1(cfgs):
    rows = []
    t0 = time.time()
    for idx, sc in enumerate(cfgs):
        rng = np.random.default_rng(SEED)
        Xw = np.concatenate([sc.sample(m, N_WHITEN // sc.M, rng) for m in range(sc.M)])
        Xtr = [sc.sample(m, N_MOM, rng) for m in range(sc.M)]
        Xte = [sc.sample(m, N_EV, rng) for m in range(sc.M)]
        rec = {'idx': idx, 'M': sc.M, 'levels': [float(v) for v in sc.levels],
               'skews': [float(v) for v in sc.skews]}
        # Bayes optimal bound
        e_bay = []
        for m, x in enumerate(Xte):
            L = np.stack([logpdf(x, sc.levels[j], sc.skews[j]) for j in range(sc.M)], 1)
            e_bay.append((L.argmax(1) != m).mean())
        rec['bayes'] = float(np.mean(e_bay))
        # s = 1..4 on identical samples
        for s in range(1, S_MAX + 1):
            R = Rules(s, Xw, Xtr)
            gaps, e_cop, e_pool, agree = [], [], [], []
            for m, x in enumerate(Xte):
                cop, pool, gap = R.decide(x)
                gaps.append(gap.mean())
                e_cop.append((cop != m).mean())
                e_pool.append((pool != m).mean())
                if s == 1:
                    # nearest sample mean on x (identical in whitened 1D space)
                    xbar = np.array([xx.mean() for xx in Xtr])
                    near = np.abs(x[:, None] - xbar[None, :]).argmin(1)
                    agree.append((cop == near).mean())
            rec[f'gap_{s}'] = float(np.mean(gaps))
            rec[f'cop_{s}'] = float(np.mean(e_cop))
            rec[f'pool_{s}'] = float(np.mean(e_pool))
            if s == 1:
                rec['s1_nearest_mean_agree'] = float(np.mean(agree))
        # noise shape heterogeneity
        sk = np.array(sc.skews)
        rec['skew_range'] = float(sk.max() - sk.min())
        rec['skew_std'] = float(sk.std())
        rec['skew_mean'] = float(sk.mean())
        rec['min_level_gap'] = float(np.min(np.diff(sc.levels)))
        rows.append(rec)
        if idx % 10 == 9:
            print(f'  [part1] {idx + 1}/120 configurations, {time.time() - t0:.0f} s', flush=True)
    return rows


def spearman(a, b):
    from scipy import stats as sps
    r = sps.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def summarize_part1(rows):
    g = lambda key: np.array([r[key] for r in rows])                # noqa: E731
    out = {}
    print('\n=== Part 1: Error medians (120 configurations) ===')
    print(f'{"rule":>22}{"median":>10}{"mean":>10}')
    keys = ['cop_1', 'pool_1'] + [f'{k}_{s}' for s in (2, 3, 4) for k in ('cop', 'pool')] + ['bayes']
    for k in keys:
        v = g(k)
        out[f'median_{k}'] = float(np.median(v))
        out[f'mean_{k}'] = float(v.mean())
        print(f'{k:>22}{np.median(v):>10.5f}{v.mean():>10.5f}')
    print(f'\n{"gap by s":>10}{"median":>10}{"max":>10}{">1%":>6}{">5%":>6}')
    for s in range(1, S_MAX + 1):
        v = g(f'gap_{s}')
        out[f'gap_{s}'] = {'median': float(np.median(v)), 'max': float(v.max()),
                           'n_gt_1pct': int((v > 0.01).sum()), 'n_gt_5pct': int((v > 0.05).sum())}
        print(f'{s:>10}{np.median(v):>10.5f}{v.max():>10.5f}{(v > 0.01).sum():>6}{(v > 0.05).sum():>6}')

    ag = g('s1_nearest_mean_agree')
    out['s1_nearest_mean_agree'] = {'min': float(ag.min()), 'n_exact_1': int((ag == 1.0).sum())}
    print(f'\nAgreement of s=1 with nearest-mean rule: min={ag.min():.6f}, '
          f'exact 1.0 in {(ag == 1.0).sum()}/120; gap_1 max={g("gap_1").max():.6f}')

    lin, bay = g('cop_1'), g('bayes')
    print('\n=== Gain vs linear rule s=1 (paired, error_s1 - error_X) ===')
    print(f'{"rule":>10}{"median":>10}{"mean":>10}{"wins":>10}{"losses":>10}'
          f'{"frac closed":>15}')
    for k in ['cop_2', 'cop_3', 'cop_4', 'pool_2', 'pool_3', 'pool_4']:
        d = lin - g(k)
        frac = d / np.maximum(lin - bay, 1e-12)
        out[f'gain_vs_lin_{k}'] = {'median': float(np.median(d)), 'mean': float(d.mean()),
                                   'wins': int((d > 0).sum()), 'losses': int((d < 0).sum()),
                                   'median_frac_of_gap_closed': float(np.median(frac))}
        print(f'{k:>10}{np.median(d):>+10.5f}{d.mean():>+10.5f}{(d > 0).sum():>10}'
              f'{(d < 0).sum():>10}{np.median(frac):>15.3f}')
    d = g('pool_3') - g('cop_3')
    out['gain_cop3_vs_pool3'] = {'median': float(np.median(d)), 'wins': int((d > 0).sum())}
    print(f'\nCopeland s=3 vs pooled s=3: median {np.median(d):+.5f}, wins in {(d > 0).sum()}/120')
    print(f'Median gap of s=1 to Bayes: {np.median(lin - bay):.5f}, '
          f'Median gap of cop_3 to Bayes: {np.median(g("cop_3") - bay):.5f}')

    print('\n=== Correlation of cop_3 gain over s=1 with noise shape heterogeneity ===')
    gain = lin - g('cop_3')
    out['gain_correlates'] = {}
    for key in ('skew_range', 'skew_std', 'skew_mean', 'min_level_gap', 'M', 'gap_3'):
        rho, p = spearman(g(key), gain)
        out['gain_correlates'][key] = {'spearman': rho, 'p': p}
        print(f'  Spearman(gain, {key:>13}) = {rho:+.3f}  (p={p:.2g})')
    print('\n  Comparison: correlates of gap_3 itself:')
    out['gap3_correlates'] = {}
    for key in ('skew_range', 'skew_std', 'skew_mean', 'min_level_gap', 'M'):
        rho, p = spearman(g(key), g('gap_3'))
        out['gap3_correlates'][key] = {'spearman': rho, 'p': p}
        print(f'  Spearman(gap_3, {key:>13}) = {rho:+.3f}  (p={p:.2g})')
    # terciles by skewness range
    sr = g('skew_range')
    q = np.quantile(sr, [1 / 3, 2 / 3])
    out['gain_by_skew_range_tercile'] = []
    print(f'\n  Terciles by skewness range (max-min): bounds {q[0]:.2f}, {q[1]:.2f}')
    print(f'{"tercile":>10}{"n":>4}{"median gain":>17}{"wins":>10}{"median gap_3":>15}')
    for lo, hi, name in ((-1, q[0], 'low'), (q[0], q[1], 'mid'), (q[1], 9, 'high')):
        sel = (sr > lo) & (sr <= hi)
        rec = {'tercile': name, 'n': int(sel.sum()), 'median_gain': float(np.median(gain[sel])),
               'wins': int((gain[sel] > 0).sum()), 'median_gap3': float(np.median(g('gap_3')[sel]))}
        out['gain_by_skew_range_tercile'].append(rec)
        print(f'{name:>10}{sel.sum():>4}{np.median(gain[sel]):>+17.5f}{(gain[sel] > 0).sum():>10}'
              f'{np.median(g("gap_3")[sel]):>15.5f}')
    # also by mean skewness (degree of non-Gaussianity)
    sm = g('skew_mean')
    q2 = np.quantile(sm, [1 / 3, 2 / 3])
    out['gain_by_skew_mean_tercile'] = []
    print(f'\n  Terciles by mean skewness: bounds {q2[0]:.2f}, {q2[1]:.2f}')
    for lo, hi, name in ((-1, q2[0], 'low'), (q2[0], q2[1], 'mid'), (q2[1], 9, 'high')):
        sel = (sm > lo) & (sm <= hi)
        rec = {'tercile': name, 'n': int(sel.sum()), 'median_gain': float(np.median(gain[sel])),
               'wins': int((gain[sel] > 0).sum())}
        out['gain_by_skew_mean_tercile'].append(rec)
        print(f'{name:>10}{sel.sum():>4}{np.median(gain[sel]):>+17.5f}{(gain[sel] > 0).sum():>10}')
    return out


# ------------------------------------------------------------- part 2

N_TEST = 150_000
N_TRAIN = (500, 2000, 10000)
N_REP = 20
N_REF = 100_000


def part2(cfgs, rows):
    gap3 = np.array([r['gap_3'] for r in rows])
    top = [int(i) for i in np.argsort(-gap3)[:6]]
    small = [i for i in range(len(rows)) if gap3[i] < 1e-4]
    # three clean configurations, spanning distinct M if possible
    chosen = []
    for M in (3, 4, 5):
        c = [i for i in small if rows[i]['M'] == M and i not in chosen]
        if c:
            chosen.append(c[0])
    for i in small:
        if len(chosen) >= 3:
            break
        if i not in chosen:
            chosen.append(i)
    sel = top + chosen[:3]
    print(f'\n=== Part 2: Configurations {sel} (6 highest gap_3 + 3 with gap_3<1e-4) ===')
    res = []
    t0 = time.time()
    for idx in sel:
        sc = cfgs[idx]
        rng_te = np.random.default_rng(SEED + 7)
        Xte = [sc.sample(m, N_TEST, rng_te) for m in range(sc.M)]
        rec = {'idx': idx, 'M': sc.M, 'levels': rows[idx]['levels'], 'skews': rows[idx]['skews'],
               'gap_3_part1': rows[idx]['gap_3'], 'runs': {}}
        for n_tr in list(N_TRAIN) + [N_REF]:
            reps = N_REP if n_tr != N_REF else 3
            acc = {'gap3': [], 'cop3': [], 'pool3': [], 'lin1': [], 'gap1': [], 'fail': 0}
            for r in range(reps):
                rng = np.random.default_rng(SEED + 1000 * n_tr + r)
                Xtr = [sc.sample(m, n_tr, rng) for m in range(sc.M)]
                Xw = np.concatenate(Xtr)
                try:
                    R3 = Rules(3, Xw, Xtr)
                    R1 = Rules(1, Xw, Xtr)
                except np.linalg.LinAlgError:
                    acc['fail'] += 1
                    continue
                g3, c3, p3, l1, g1 = [], [], [], [], []
                for m, x in enumerate(Xte):
                    cop, pool, gap = R3.decide(x)
                    g3.append(gap.mean()); c3.append((cop != m).mean()); p3.append((pool != m).mean())
                    cop1, _, gap1 = R1.decide(x)
                    l1.append((cop1 != m).mean()); g1.append(gap1.mean())
                acc['gap3'].append(np.mean(g3)); acc['cop3'].append(np.mean(c3))
                acc['pool3'].append(np.mean(p3)); acc['lin1'].append(np.mean(l1))
                acc['gap1'].append(np.mean(g1))
            summ = {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                        'min': float(np.min(v)), 'max': float(np.max(v))}
                    for k, v in acc.items() if k != 'fail'}
            summ['n_rep'] = reps - acc['fail']
            summ['cop_beats_pool_reps'] = int(np.sum(np.array(acc['cop3']) < np.array(acc['pool3'])))
            summ['cop_beats_lin_reps'] = int(np.sum(np.array(acc['cop3']) < np.array(acc['lin1'])))
            rec['runs'][str(n_tr)] = summ
        res.append(rec)
        print(f'\n  cfg {idx} (M={sc.M}, gap_3={rows[idx]["gap_3"]:.5f}) '
              f'levels={np.round(sc.levels, 2).tolist()} skews={np.round(sc.skews, 2).tolist()}')
        print(f'{"n_train":>8}{"P(fail) mean±sd":>20}{"cop3 mean±sd":>20}{"pool3 mean±sd":>20}'
              f'{"lin1 mean±sd":>20}{"cop<pool":>9}{"cop<lin":>8}')
        for n_tr, sm in rec['runs'].items():
            print(f'{n_tr:>8}'
                  f'{sm["gap3"]["mean"]:>12.5f}±{sm["gap3"]["sd"]:<7.5f}'
                  f'{sm["cop3"]["mean"]:>12.5f}±{sm["cop3"]["sd"]:<7.5f}'
                  f'{sm["pool3"]["mean"]:>12.5f}±{sm["pool3"]["sd"]:<7.5f}'
                  f'{sm["lin1"]["mean"]:>12.5f}±{sm["lin1"]["sd"]:<7.5f}'
                  f'{sm["cop_beats_pool_reps"]:>5}/{sm["n_rep"]:<3}'
                  f'{sm["cop_beats_lin_reps"]:>5}/{sm["n_rep"]:<3}')
        print(f'    [{time.time() - t0:.0f} s]', flush=True)
    return res


def summarize_part2(res):
    print('\n=== Part 2: Summary ===')
    out = {}
    for group, ids in (('top6', range(6)), ('clean3', range(6, 9))):
        print(f'\n  group {group}')
        print(f'{"n_train":>8}{"mean P(fail)":>14}{"ratio to ref":>14}{"cop3-pool3":>12}'
              f'{"cop<pool cfgs":>14}{"cop3-lin1":>12}{"mean sd(cop3)":>14}')
        out[group] = {}
        for n_tr in [str(v) for v in list(N_TRAIN) + [N_REF]]:
            gf = np.array([res[i]['runs'][n_tr]['gap3']['mean'] for i in ids])
            gr = np.array([res[i]['runs'][str(N_REF)]['gap3']['mean'] for i in ids])
            dc = np.array([res[i]['runs'][n_tr]['cop3']['mean'] - res[i]['runs'][n_tr]['pool3']['mean']
                           for i in ids])
            dl = np.array([res[i]['runs'][n_tr]['cop3']['mean'] - res[i]['runs'][n_tr]['lin1']['mean']
                           for i in ids])
            sd = np.array([res[i]['runs'][n_tr]['cop3']['sd'] for i in ids])
            ratio = float(np.mean(gf) / np.mean(gr)) if np.mean(gr) > 0 else float('nan')
            out[group][n_tr] = {'mean_pfail': float(gf.mean()), 'ratio_pfail_to_ref': ratio,
                                'mean_cop_minus_pool': float(dc.mean()),
                                'n_cfg_cop_beats_pool': int((dc < 0).sum()),
                                'mean_cop_minus_lin': float(dl.mean()),
                                'n_cfg_cop_beats_lin': int((dl < 0).sum()),
                                'mean_sd_cop3': float(sd.mean())}
            print(f'{n_tr:>8}{gf.mean():>14.5f}{ratio:>14.3f}{dc.mean():>+12.5f}'
                  f'{(dc < 0).sum():>10}/{len(dc):<3}{dl.mean():>+12.5f}{sd.mean():>14.5f}')
    return out


if __name__ == '__main__':
    t_all = time.time()
    cfgs = replay_configs()
    rows = part1(cfgs)
    s1 = summarize_part1(rows)
    res = part2(cfgs, rows)
    s2 = summarize_part2(res)
    json.dump({'settings': {'seed': SEED, 'n_mom': N_MOM, 'n_ev': N_EV, 'n_whiten': N_WHITEN,
                            'n_test_part2': N_TEST, 'n_train': list(N_TRAIN), 'n_rep': N_REP,
                            'n_ref': N_REF, 'basis': 'power', 'copeland_tie': 'lowest index'},
               'part1_rows': rows, 'part1_summary': s1,
               'part2_configs': res, 'part2_summary': s2},
              open(OUT, 'w'), indent=1)
    print(f'\nSaved {OUT}; total time {time.time() - t_all:.0f} s')
