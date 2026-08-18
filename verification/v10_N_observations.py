#!/usr/bin/env python3
"""Dependence of undecided probability on observation sample size N (Fig. 2(b)).

For N i.i.d. observations, §II dictates: phi -> sample mean, C^(m) -> C^(m)/N;
directions K_mn and thresholds remain invariant.

Hypothesis: the undecided event decays exponentially with N because the true
hypothesis wins every pairwise duel with probability approaching 1, thus
becoming a Condorcet winner. This clarifies that the letter's domain is single-
observation or small-N discrimination.

Measured on 120 random configurations (seeded rng 2026), s=3, mixed power basis,
and on the worst-case configuration WORST.
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, pairwise_rules            # noqa: E402

SEED = 20260817
S = 3
BASIS = lambda x: np.stack([x**i for i in range(1, S + 1)], 1)    # noqa: E731
NS = (1, 2, 4, 8, 16, 32)


def gap_vs_N(sc, Ns=NS, n_mom=150_000, n_ev=40_000, seed=SEED):
    """P(no Condorcet winner) for each sample size N."""
    rng = np.random.default_rng(seed)
    M = sc.M
    Xw = np.concatenate([sc.sample(m, 60_000 // M, rng) for m in range(M)])
    P = BASIS(Xw)
    mean, C = P.mean(0), np.cov(P, rowvar=False, ddof=1)
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(np.maximum(w, 1e-300)**-0.5) @ V.T
    tr = lambda x: (BASIS(x) - mean) @ W                          # noqa: E731

    mus, Cs1 = [], []
    for m in range(M):
        Q = tr(sc.sample(m, n_mom, rng))
        mus.append(Q.mean(0))
        Cs1.append(np.cov(Q, rowvar=False, ddof=1))
    mus, Cs1 = np.array(mus), np.array(Cs1)

    out = {}
    for N in Ns:
        K = pairwise_rules(mus, Cs1 / N)          # C -> C/N, directions invariant
        gaps = []
        for m in range(M):
            x = sc.sample(m, n_ev * N, rng)
            Phi = tr(x).reshape(n_ev, N, S).mean(1)
            wins = np.zeros((n_ev, M), int)
            for (i, j), (k, c, _) in K.items():
                win = (Phi - c) @ k > 0
                wins[:, j] += win
                wins[:, i] += ~win
            gaps.append((~(wins == M - 1).any(1)).mean())
        out[N] = float(np.mean(gaps))
    return out


WORST = Scenario([0.0, 0.04, 1.28, 1.39, 1.45], [0.05, 1.83, 1.36, 0.39, 1.39])
EASY = Scenario([0.0, 1.5, 3.0], [0.2, 1.0, 1.8])

if __name__ == '__main__':
    res = {'worst': gap_vs_N(WORST, n_ev=60_000),
           'easy': gap_vs_N(EASY, n_ev=60_000)}
    print('Fixed scenarios, s=3:\n')
    print(f'{"N":>5}' + ''.join(f'{n:>12}' for n in NS))
    for name in ('worst', 'easy'):
        print(f'{name:>5}' + ''.join(f'{res[name][n]:>12.5f}' for n in NS))

    # sweep across 120 configurations -- filtering those with non-negligible gap at N=1
    rng = np.random.default_rng(2026)
    rows = []
    for t in range(120):
        M = int(rng.integers(3, 6))
        lv = np.sort(rng.uniform(0, 2.5, M))
        lv -= lv[0]
        sc = Scenario(lv, rng.uniform(0.0, 1.9, M))
        try:
            g1 = gap_vs_N(sc, Ns=(1,), n_ev=30_000)[1]
        except np.linalg.LinAlgError:
            continue
        if g1 <= 0.01:                    # focus on configurations with noticeable gap
            continue
        try:
            rows.append({'idx': t, 'M': M, **{str(k): v for k, v in
                                              gap_vs_N(sc, n_ev=30_000).items()}})
        except np.linalg.LinAlgError:
            continue
        print(f'  cfg {t} (M={M}): ' + ' '.join(f'{rows[-1][str(n)]:.4f}' for n in NS),
              flush=True)

    print(f'\nConfigurations with P(undecided) > 1% at N=1: {len(rows)}')
    print(f'{"N":>5}{"median":>12}{"maximum":>12}{"fraction of N=1":>19}')
    base = np.array([r['1'] for r in rows])
    for n in NS:
        v = np.array([r[str(n)] for r in rows])
        print(f'{n:>5}{np.median(v):>12.5f}{v.max():>12.5f}'
              f'{np.median(v / base):>19.4f}')
    res['sweep'] = rows
    json.dump(res, open('verification/results_v10_N.json', 'w'), indent=1)
    print('\nSaved verification/results_v10_N.json')
