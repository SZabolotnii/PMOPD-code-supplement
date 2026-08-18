#!/usr/bin/env python3
"""Linear baseline control s=1: verification that the linear rule is free from incompleteness.

Claim under test:
  At s=1 the pairwise moment-optimal rule reduces to comparing the sample
  mean against the mid-point of the pair's means (nearest-mean classification).
  This defines a total order on the line, ensuring that the tournament is
  always transitive, and no undecided events can occur for any configuration.

Hence incompleteness is a purely nonlinear phenomenon: it is induced by the very
mechanism that delivers the higher-order performance gain.

Verified via two independent methods:
  (A) numerically -- fraction of cycles on representative scenarios;
  (B) analytically/empirically -- exact equivalence between s=1 decision and nearest-mean.
"""
import sys

import numpy as np

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, Whitener, moments, pairwise_rules, \
    intransitive, raw_basis                                    # noqa: E402
from p5_diagnostic_aggregation import SCENARIOS                # noqa: E402

SEED = 20260817


def cycles_at(sc, s, kind='power', n_mom=300_000, n_eval=400_000, seed=SEED):
    rng = np.random.default_rng(seed)
    wh = Whitener(sc, kind, s, 120_000, rng)
    mus, Cs = moments(sc, wh, kind, s, n_mom, rng)
    K = pairwise_rules(mus, Cs)
    X = np.concatenate([sc.sample(m, n_eval // sc.M, rng) for m in range(sc.M)])
    return float(intransitive(wh(X, kind, s), K, sc.M).mean())


def agrees_with_nearest_mean(sc, n=200_000, seed=SEED):
    """(B) Checks whether the s=1 pairwise decision matches the nearest-mean rule."""
    rng = np.random.default_rng(seed)
    kind, s = 'power', 1
    wh = Whitener(sc, kind, s, 120_000, rng)
    mus, Cs = moments(sc, wh, kind, s, 200_000, rng)
    K = pairwise_rules(mus, Cs)

    X = np.concatenate([sc.sample(m, n // sc.M, rng) for m in range(sc.M)])
    Phi = wh(X, kind, s)

    # tournament winner under pairwise rules
    wins = np.zeros((len(X), sc.M), int)
    for (m, n_), (k, c, _) in K.items():
        won = (Phi - c) @ k > 0
        wins[:, n_] += won
        wins[:, m] += ~won
    tour = wins.argmax(1)

    # nearest mean rule in 1D whitened space
    near = np.abs(Phi[:, [0]] - mus[:, 0][None, :]).argmin(1)
    return float((tour == near).mean())


if __name__ == '__main__':
    print('(A) Fraction of observations with cycles, power basis\n')
    print(f'{"scenario":>38}{"M":>3}{"s=1":>10}{"s=2":>10}{"s=3":>10}')
    for name, sc, _ in SCENARIOS:
        row = [cycles_at(sc, s) for s in (1, 2, 3)]
        print(f'{name[:38]:>38}{sc.M:>3}' + ''.join(f'{v:>10.5f}' for v in row))

    print('\n(B) Agreement between s=1 rule and nearest-mean classifier\n')
    print(f'{"scenario":>38}{"M":>3}{"agreement":>10}')
    for name, sc, _ in SCENARIOS:
        print(f'{name[:38]:>38}{sc.M:>3}{agrees_with_nearest_mean(sc):>10.5f}')
