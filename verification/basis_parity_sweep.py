#!/usr/bin/env python3
"""Summary sweep: basis parity × undecided events × aggregation gain.

Three questions governing the framing of the letter:

  Q-A. Is parity confirmed as the primary mechanism across an extensive sweep
       (120 configurations instead of 22, as in the initial P2 setup)?
  Q-B. Does the Copeland aggregation gain hold on the MIXED basis -- where
       undecided events actually occur, hence where aggregation is necessary?
  Q-C. Which strategy is preferred practically: retain the mixed basis and aggregate,
       or simply transition to an odd basis? This determines whether aggregation
       serves as the primary contribution or a fallback.

Three bases of identical dimension s=3 are compared:
    mixed  {x, x^2, x^3}          -- classical polynomial, mixed parity
    odd    {x, x^3, x^5}          -- odd-only
    patp1  {x, sgn(x)x^2, x^3}    -- signed PATP at alpha=1, odd-only

Metrics per configuration and basis:
    gap       P(no Condorcet winner) -- frequency of pairwise scheme failure
    copeland  error rate of tournament score aggregation
    pooled    error rate of pooled covariance detector on shared S_W
    bayes     Bayes error rate using true densities (lower bound)
"""
import json
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, pairwise_rules            # noqa: E402

SEED = 20260817
S = 3

BASES = {
    'mixed': lambda x: np.stack([x, x**2, x**3], 1),
    'odd':   lambda x: np.stack([x, x**3, x**5], 1),
    'patp1': lambda x: np.stack([x, np.sign(x) * x**2, x**3], 1),
}


def logpdf(x, a, skew):
    if skew == 0.0:
        return sps.norm.logpdf(x, loc=a, scale=1.0)
    k = 4.0 / skew**2
    z = (x - a) * np.sqrt(k) + k
    out = np.full_like(x, -np.inf)
    ok = z > 0
    out[ok] = sps.gamma.logpdf(z[ok], k) + 0.5 * np.log(k)
    return out


def evaluate(sc, f, n_mom=150_000, n_ev=200_000, seed=SEED):
    rng = np.random.default_rng(seed)
    Xw = np.concatenate([sc.sample(m, 60_000 // sc.M, rng) for m in range(sc.M)])
    P = f(Xw)
    mean = P.mean(0)
    C = np.cov(P, rowvar=False, ddof=1)
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(np.maximum(w, 1e-300)**-0.5) @ V.T
    tr = lambda x: (f(x) - mean) @ W                              # noqa: E731

    mus, Cs = [], []
    for m in range(sc.M):
        Q = tr(sc.sample(m, n_mom, rng))
        mus.append(Q.mean(0))
        Cs.append(np.cov(Q, rowvar=False, ddof=1))
    mus, Cs = np.array(mus), np.array(Cs)
    K = pairwise_rules(mus, Cs)

    SW = Cs.mean(0)
    A = np.linalg.solve(SW, mus.T).T
    b = -0.5 * np.einsum('ms,ms->m', A, mus)

    gaps, e_cop, e_pool, e_bay = [], [], [], []
    for m in range(sc.M):
        x = sc.sample(m, n_ev, rng)
        Phi = tr(x)
        out = np.zeros((len(x), sc.M), int)
        for (i, j), (k, c, _) in K.items():
            win = (Phi - c) @ k > 0
            out[:, j] += win
            out[:, i] += ~win
        gaps.append((~(out == sc.M - 1).any(1)).mean())
        e_cop.append((out.argmax(1) != m).mean())
        e_pool.append(((Phi @ A.T + b).argmax(1) != m).mean())
        L = np.stack([logpdf(x, sc.levels[j], sc.skews[j]) for j in range(sc.M)], 1)
        e_bay.append((L.argmax(1) != m).mean())
    return (float(np.mean(gaps)), float(np.mean(e_cop)),
            float(np.mean(e_pool)), float(np.mean(e_bay)))


if __name__ == '__main__':
    rng = np.random.default_rng(2026)
    rows = []
    for t in range(120):
        M = int(rng.integers(3, 6))
        lv = np.sort(rng.uniform(0, 2.5, M))
        lv -= lv[0]
        sc = Scenario(lv, rng.uniform(0.0, 1.9, M))
        rec = {'M': M}
        ok = True
        for name, f in BASES.items():
            try:
                g, c, p, bay = evaluate(sc, f)
            except np.linalg.LinAlgError:
                ok = False
                break
            rec[name] = {'gap': g, 'copeland': c, 'pooled': p, 'bayes': bay}
        if ok:
            rows.append(rec)
    print(f'Valid configurations: {len(rows)} out of 120\n')

    print('=== Q-A. Parity as Mechanism ===\n')
    print(f'{"basis":>8}{"median gap":>14}{"maximum":>11}{"> 1%":>8}{"> 5%":>7}')
    for name in BASES:
        g = np.array([r[name]['gap'] for r in rows])
        print(f'{name:>8}{np.median(g):>14.5f}{g.max():>11.5f}'
              f'{(g > 0.01).sum():>8}{(g > 0.05).sum():>7}')

    print('\n=== Q-B. Aggregation Gain vs. Pooling by Basis ===\n')
    print(f'{"basis":>8}{"median gain":>18}{"better in":>11}{"max":>10}')
    for name in BASES:
        d = np.array([r[name]['pooled'] - r[name]['copeland'] for r in rows])
        print(f'{name:>8}{np.median(d):>+18.5f}{f"{(d>0).sum()}/{len(d)}":>11}'
              f'{d.max():>+10.5f}')

    print('\n   Same comparison on configurations with non-negligible gap (gap > 1%):\n')
    sub = [r for r in rows if r['mixed']['gap'] > 0.01]
    print(f'   Configurations with gap > 1%: {len(sub)}')
    if sub:
        for name in BASES:
            d = np.array([r[name]['pooled'] - r[name]['copeland'] for r in sub])
            print(f'{name:>8}{np.median(d):>+18.5f}{f"{(d>0).sum()}/{len(d)}":>11}'
                  f'{d.max():>+10.5f}')

    print('\n=== Q-C. Mixed + Aggregation vs. Odd Basis ===\n')
    mc = np.array([r['mixed']['copeland'] for r in rows])
    mp = np.array([r['mixed']['pooled'] for r in rows])
    oc = np.array([r['odd']['copeland'] for r in rows])
    op = np.array([r['odd']['pooled'] for r in rows])
    bay = np.array([r['mixed']['bayes'] for r in rows])
    print(f'{"strategy":>34}{"median error":>18}{"gap to Bayes":>19}')
    for lbl, v in (('mixed + pooled', mp), ('mixed + Copeland', mc),
                   ('odd + pooled', op), ('odd + Copeland', oc)):
        print(f'{lbl:>34}{np.median(v):>18.5f}{np.median(v - bay):>19.5f}')
    print(f'\n{"Bayes lower bound":>34}{np.median(bay):>18.5f}')
    print(f'\nodd + Copeland is better than mixed + Copeland in '
          f'{(oc < mc).sum()}/{len(rows)} configurations')
    print(f'median difference (mixed+Copeland - odd+Copeland): {np.median(mc - oc):+.5f}')

    json.dump(rows, open('verification/results_basis_parity_sweep.json', 'w'), indent=1)
    print('\nSaved verification/results_basis_parity_sweep.json')
