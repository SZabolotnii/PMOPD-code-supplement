#!/usr/bin/env python3
"""Targeted sweep: does basis conditioning predict the undecided probability?

Motivation: Two mechanistic hypotheses have already been tested (magnitude of
nonlinearity; basis parity). One candidate remained: with nearly collinear basis
functions, the whitened geometry is ill-conditioned, and minor differences in
C^(m) could flip pairwise comparisons into inconsistent cycles. Previously, this
rested on a correlation over only six points.

Design: Consider a ONE-PARAMETER family where only the exponent spread varies
while parity remains strictly odd:
        phi_i(x) = sgn(x)|x|^{p_i},   p = (1, 1+d, 1+2d),   d > 0.
All functions are odd for any d, eliminating parity as a confounding factor by
construction. d -> 0 yields a nearly collinear basis, whereas d = 2 yields (1, 3, 5).

This measures both predictor and response within a single parametric family rather
than across disparate basis families.

Measured quantities per (configuration, d) pair:
    cond_raw  condition number of basis covariance on pooled sample
              -- intrinsic basis property, independent of hypotheses;
    cond_F    median condition number of F_{mn} = C^(m) + C^(n) after whitening
              -- extent of covariance discrepancy between hypotheses;
    het       relative dispersion of C^(m);
    gap       P(no Condorcet winner) -- frequency of pairwise failure.

Statistics: Cross-configuration correlation conflates "scenario hardness" with
basis effects. Hence the primary metric is the WITHIN-configuration rank correlation
(over d), averaged across configurations. Pooled correlation is reported as a secondary metric.
"""
import json
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, pairwise_rules              # noqa: E402
from basis_parity_sweep import logpdf                               # noqa: E402

SEED = 20260818
DELTAS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]


def family(d):
    ps = [1.0, 1.0 + d, 1.0 + 2 * d]
    return (lambda x: np.stack([np.sign(x) * np.abs(x)**p for p in ps], 1)), ps


def measure(sc, f, n_mom=90_000, n_ev=120_000, seed=SEED):
    rng = np.random.default_rng(seed)
    Xw = np.concatenate([sc.sample(m, 60_000 // sc.M, rng) for m in range(sc.M)])
    P = f(Xw)
    mean, C = P.mean(0), np.cov(P, rowvar=False, ddof=1)
    cond_raw = float(np.linalg.cond(C))
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(np.maximum(w, 1e-300) ** -0.5) @ V.T
    tr = lambda x: (f(x) - mean) @ W                                # noqa: E731

    mus, Cs = [], []
    for m in range(sc.M):
        Q = tr(sc.sample(m, n_mom, rng))
        mus.append(Q.mean(0))
        Cs.append(np.cov(Q, rowvar=False, ddof=1))
    mus, Cs = np.array(mus), np.array(Cs)
    Cb = Cs.mean(0)
    het = float(np.mean([np.linalg.norm(c - Cb) for c in Cs]) / np.linalg.norm(Cb))
    K = pairwise_rules(mus, Cs)
    cond_F = float(np.median([np.linalg.cond(Cs[m] + Cs[n])
                              for m in range(sc.M) for n in range(m + 1, sc.M)]))

    X = np.concatenate([sc.sample(m, n_ev // sc.M, rng) for m in range(sc.M)])
    Phi = tr(X)
    o = np.zeros((len(X), sc.M), int)
    for (i, j), (k, c, _) in K.items():
        win = (Phi - c) @ k > 0
        o[:, j] += win
        o[:, i] += ~win
    gap = float((~(o == sc.M - 1).any(1)).mean())
    return cond_raw, cond_F, het, gap


if __name__ == '__main__':
    rng = np.random.default_rng(4242)
    scens = []
    while len(scens) < 40:
        M = int(rng.integers(3, 6))
        lv = np.sort(rng.uniform(0, 2.5, M))
        lv -= lv[0]
        scens.append(Scenario(lv, rng.uniform(0.0, 1.9, M)))

    recs = []
    for ci, sc in enumerate(scens):
        for d in DELTAS:
            f, ps = family(d)
            try:
                cr, cf, h, g = measure(sc, f)
            except np.linalg.LinAlgError:
                continue
            recs.append({'cfg': ci, 'M': sc.M, 'd': d, 'p': ps,
                         'cond_raw': cr, 'cond_F': cf, 'het': h, 'gap': g})
    print(f'Measurements: {len(recs)} ({len(scens)} configurations × {len(DELTAS)} d)\n')

    print('=== Family Profile: Variation across exponent spread d ===\n')
    print(f'{"d":>6}{"exponents":>20}{"cond_raw":>11}{"cond_F":>9}{"het":>8}'
          f'{"med. gap":>11}{">1%":>8}')
    for d in DELTAS:
        r = [x for x in recs if x['d'] == d]
        g = np.array([x['gap'] for x in r])
        ps = r[0]['p']
        print(f'{d:>6.2f}{f"1, {ps[1]:.2f}, {ps[2]:.2f}":>20}'
              f'{np.median([x["cond_raw"] for x in r]):>11.1f}'
              f'{np.median([x["cond_F"] for x in r]):>9.2f}'
              f'{np.median([x["het"] for x in r]):>8.3f}'
              f'{np.median(g):>11.5f}{(g > .01).sum():>5}/{len(g)}')

    print('\n=== WITHIN-configuration Correlation (Spearman over d) ===\n')
    res = {}
    for key in ('cond_raw', 'cond_F', 'het'):
        rs = []
        for ci in range(len(scens)):
            r = [x for x in recs if x['cfg'] == ci]
            if len(r) < 5:
                continue
            g = [x['gap'] for x in r]
            if len(set(g)) < 3:
                continue
            rs.append(sps.spearmanr([x[key] for x in r], g).statistic)
        rs = np.array([v for v in rs if np.isfinite(v)])
        res[key] = {'mean': float(rs.mean()), 'median': float(np.median(rs)),
                    'n': int(len(rs)), 'frac_neg': float((rs < 0).mean())}
        print(f'{key:>9}: mean rho = {rs.mean():+.3f}   median {np.median(rs):+.3f}'
              f'   negative in {(rs<0).mean():.0%} of configs   (n={len(rs)})')

    print('\n=== Pooled Correlation (secondary metric) ===\n')
    for key in ('cond_raw', 'cond_F', 'het'):
        v = np.array([x[key] for x in recs])
        g = np.array([x['gap'] for x in recs])
        print(f'{key:>9}: Spearman {sps.spearmanr(v, g).statistic:+.3f}')

    print('\n=== Control: Exponent spread d as direct predictor ===\n')
    rs = []
    for ci in range(len(scens)):
        r = [x for x in recs if x['cfg'] == ci]
        g = [x['gap'] for x in r]
        if len(r) < 5 or len(set(g)) < 3:
            continue
        rs.append(sps.spearmanr([x['d'] for x in r], g).statistic)
    rs = np.array([v for v in rs if np.isfinite(v)])
    print(f'        d: mean rho = {rs.mean():+.3f}   median {np.median(rs):+.3f}'
          f'   negative in {(rs<0).mean():.0%}   (n={len(rs)})')
    res['d'] = {'mean': float(rs.mean()), 'median': float(np.median(rs)),
                'frac_neg': float((rs < 0).mean()), 'n': int(len(rs))}

    json.dump({'records': recs, 'within': res},
              open('verification/results_cond_predictor.json', 'w'), indent=1)
    print('\nSaved verification/results_cond_predictor.json')
