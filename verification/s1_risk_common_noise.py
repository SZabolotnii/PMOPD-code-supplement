#!/usr/bin/env python3
"""Risk S1 check for the SPL letter: is the effect merely heteroscedastic discriminant analysis?

Potential reviewer objection: "Covariance heterogeneity is well known; your result is
simply a special case of quadratic/heteroscedastic discriminant analysis."

Response demonstrated constructively: here the covariance matrices of basis functions
differ across hypotheses EVEN WHEN THE NOISE LAW IS STRICTLY IDENTICAL FOR ALL HYPOTHESES.
The cause is the nonlinearity of the basis: shifting the signal changes the higher-order
moments of phi_i(x) even under an invariant noise distribution. At s=1 this cannot occur.

Experiment: identical noise across all hypotheses (identical skewness parameter),
varying only signal levels.
Measured quantities:
    het   -- relative covariance dispersion of C^(m) around the mean in basis space;
    gap   -- P(no Condorcet winner), i.e., frequency of pairwise scheme failure.
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, Whitener, moments, pairwise_rules  # noqa: E402

SEED = 20260817


def measure(sc, s, kind='power', n_mom=300_000, n_eval=400_000, seed=SEED):
    rng = np.random.default_rng(seed)
    wh = Whitener(sc, kind, s, 120_000, rng)
    mus, Cs = moments(sc, wh, kind, s, n_mom, rng)
    Cb = Cs.mean(0)
    het = float(np.mean([np.linalg.norm(C - Cb) for C in Cs]) / np.linalg.norm(Cb))
    K = pairwise_rules(mus, Cs)
    X = np.concatenate([sc.sample(m, n_eval // sc.M, rng) for m in range(sc.M)])
    Phi = wh(X, kind, s)
    out = np.zeros((len(X), sc.M), int)
    for (m, n), (k, c, _) in K.items():
        win_n = (Phi - c) @ k > 0
        out[:, n] += win_n
        out[:, m] += ~win_n
    gap = float((~(out == sc.M - 1).any(axis=1)).mean())
    return het, gap


if __name__ == '__main__':
    out = {}
    SKEW = 1.2                      # COMMON noise across all hypotheses
    CONFIGS = [
        ('M=4 levels 0/0.4/0.8/1.2', [0.0, 0.4, 0.8, 1.2]),
        ('M=4 levels 0/0.2/0.4/0.6', [0.0, 0.2, 0.4, 0.6]),
        ('M=5 levels 0..0.8', [0.0, 0.2, 0.4, 0.6, 0.8]),
        ('M=5 levels 0..0.4', [0.0, 0.1, 0.2, 0.3, 0.4]),
    ]
    print(f'COMMON noise for all hypotheses (skew={SKEW}); differing only in signal levels\n')
    print(f'{"configuration":>26}{"s":>3}{"dispersion C⁽ᵐ⁾":>18}{"P(gap)":>14}')
    rows = []
    for name, lv in CONFIGS:
        sc = Scenario(lv, [SKEW] * len(lv))
        for s in (1, 2, 3, 4):
            het, gap = measure(sc, s)
            rows.append({'cfg': name, 's': s, 'het': het, 'gap': gap})
            print(f'{name[:26]:>26}{s:>3}{het:>18.5f}{gap:>14.5f}')
        print()
    out['common_noise'] = rows

    print('Control: at s=1 dispersion is zero (1D space, variance invariant to shift),')
    print('and the undecided gap is identically zero.')
    json.dump(out, open('verification/results_s1_risk.json', 'w'), indent=1)
    print('\nSaved verification/results_s1_risk.json')
