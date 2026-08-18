#!/usr/bin/env python3
"""Three follow-up investigations after the sweep in basis_parity_sweep.py.

T1. Re-evaluate the P2 sweep with corrected PATP and report both quantities separately:
    P(cycle) -- tournament intransitivity (previously measured metric),
    P(undecided) -- absence of a Condorcet winner (decision-theoretic incompleteness).

T2. Why does patp1 {x, sgn·x², x³} yield 14 configurations > 1%, whereas odd {x, x³, x⁵}
    yields only 2, despite both consisting of odd functions? Parity dominates but does
    not exhaust the effect. Three explanatory candidates are evaluated:
      mono  -- monotonicity of E[phi_i(a+xi)] with respect to signal level a;
      het   -- covariance dispersion among hypotheses C^(m);
      cond  -- condition number of the normal system matrix.

T3. Stability of conclusion Q-C (mixed + aggregation vs. odd + aggregation)
    across polynomial degrees s in {2, 3, 4}.
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, raw_basis, pairwise_rules   # noqa: E402
from basis_parity_sweep import evaluate, logpdf                     # noqa: E402

SEED = 20260817


def sgnpow(p):
    return lambda x: np.sign(x) * np.abs(x)**p


def make(fs):
    return lambda x: np.stack([f(x) for f in fs], 1)


def whiten_stats(sc, f, n=120_000, seed=SEED):
    """Return (het, cond) -- dispersion of C^(m) and median condition number of F_{mn}."""
    rng = np.random.default_rng(seed)
    Xw = np.concatenate([sc.sample(m, n // sc.M, rng) for m in range(sc.M)])
    P = f(Xw)
    mean, C = P.mean(0), np.cov(P, rowvar=False, ddof=1)
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(np.maximum(w, 1e-300)**-0.5) @ V.T
    tr = lambda x: (f(x) - mean) @ W                                # noqa: E731
    mus, Cs = [], []
    for m in range(sc.M):
        Q = tr(sc.sample(m, 150_000, rng))
        mus.append(Q.mean(0))
        Cs.append(np.cov(Q, rowvar=False, ddof=1))
    mus, Cs = np.array(mus), np.array(Cs)
    Cb = Cs.mean(0)
    het = float(np.mean([np.linalg.norm(c - Cb) for c in Cs]) / np.linalg.norm(Cb))
    conds = [np.linalg.cond(Cs[m] + Cs[n])
             for m in range(sc.M) for n in range(m + 1, sc.M)]
    return het, float(np.median(conds))


def monotone_score(f, s, skew=1.2, n=200_000, seed=1):
    """Fraction of basis coordinates whose E[phi_i(a+xi)] is monotonic in a."""
    rng = np.random.default_rng(seed)
    k = 4.0 / skew**2
    xi = (rng.gamma(k, 1.0, n) - k) / np.sqrt(k)
    a = np.linspace(0.0, 2.0, 25)
    means = np.stack([f(av + xi).mean(0) for av in a])              # (25, s)
    d = np.diff(means, axis=0)
    return float(np.mean([(np.all(c > 0) or np.all(c < 0)) for c in d.T]))


BASES = {
    'mixed {x,x²,x³}':      make([lambda x: x, lambda x: x**2, lambda x: x**3]),
    'patp1 {x,sgn·x²,x³}':  make([lambda x: x, sgnpow(2), lambda x: x**3]),
    'odd {x,x³,x⁵}':        make([lambda x: x, lambda x: x**3, lambda x: x**5]),
    'signed frac {1,1.5,3}': make([sgnpow(1), sgnpow(1.5), sgnpow(3)]),
    'even {x,x²,x⁴}':       make([lambda x: x, lambda x: x**2, lambda x: x**4]),
    'compress {1,.5,.33}':  make([sgnpow(1), sgnpow(0.5), sgnpow(1/3)]),
}


def rand_scen(rng):
    M = int(rng.integers(3, 6))
    lv = np.sort(rng.uniform(0, 2.5, M))
    lv -= lv[0]
    return Scenario(lv, rng.uniform(0.0, 1.9, M))


if __name__ == '__main__':
    out = {}

    # ---------------------------------------------------------------- T1
    print('=== T1. P2 Sweep with Corrected PATP: Cycle vs. Undecided ===\n')
    print(f'{"basis":>8}{"s":>3}{"med. cycle":>12}{"med. gap":>14}'
          f'{"max cycle":>12}{"max gap":>14}{"gap >1%":>12}')
    rows = []
    for kind in ('power', 'patp'):
        for s in (2, 3, 4):
            rng = np.random.default_rng(7)
            cyc, gp = [], []
            for _ in range(40):
                sc = rand_scen(rng)
                f = (lambda x, k=kind, ss=s: raw_basis(x, k, ss, 0.65))
                try:
                    g, _, _, _ = evaluate(sc, f, n_mom=100_000, n_ev=120_000)
                except np.linalg.LinAlgError:
                    continue
                # cycle frequency separately
                r2 = np.random.default_rng(SEED)
                Xw = np.concatenate([sc.sample(m, 60_000 // sc.M, r2)
                                     for m in range(sc.M)])
                P = f(Xw); mn = P.mean(0); C = np.cov(P, rowvar=False, ddof=1)
                w, V = np.linalg.eigh(C)
                W = V @ np.diag(np.maximum(w, 1e-300)**-0.5) @ V.T
                tr = lambda x: (f(x) - mn) @ W                      # noqa: E731
                mus, Cs = [], []
                for m in range(sc.M):
                    Q = tr(sc.sample(m, 100_000, r2))
                    mus.append(Q.mean(0)); Cs.append(np.cov(Q, rowvar=False, ddof=1))
                try:
                    K = pairwise_rules(np.array(mus), np.array(Cs))
                except np.linalg.LinAlgError:
                    continue
                X = np.concatenate([sc.sample(m, 120_000 // sc.M, r2)
                                    for m in range(sc.M)])
                Phi = tr(X); o = np.zeros((len(X), sc.M), int)
                for (i, j), (kk, c, _) in K.items():
                    win = (Phi - c) @ kk > 0
                    o[:, j] += win; o[:, i] += ~win
                cyc.append(float((~(np.sort(o, 1) == np.arange(sc.M)).all(1)).mean()))
                gp.append(g)
            cyc, gp = np.array(cyc), np.array(gp)
            rows.append({'kind': kind, 's': s, 'med_cycle': float(np.median(cyc)),
                         'med_gap': float(np.median(gp)), 'max_cycle': float(cyc.max()),
                         'max_gap': float(gp.max()), 'n_gap_1pct': int((gp > .01).sum()),
                         'n': len(gp)})
            print(f'{kind:>8}{s:>3}{np.median(cyc):>12.5f}{np.median(gp):>14.5f}'
                  f'{cyc.max():>12.5f}{gp.max():>14.5f}{(gp>.01).sum():>9}/{len(gp)}')
    out['T1'] = rows

    # ---------------------------------------------------------------- T2
    print('\n=== T2. Explaining Performance Differences among Odd Bases ===\n')
    print(f'{"basis":>22}{"monotonicity":>14}{"het":>9}{"cond":>10}'
          f'{"med. gap":>14}{">1%":>7}')
    rows = []
    for name, f in BASES.items():
        rng = np.random.default_rng(2026)
        gaps, hets, conds = [], [], []
        for _ in range(45):
            sc = rand_scen(rng)
            try:
                g, _, _, _ = evaluate(sc, f, n_mom=100_000, n_ev=120_000)
                h, cd = whiten_stats(sc, f)
            except np.linalg.LinAlgError:
                continue
            gaps.append(g); hets.append(h); conds.append(cd)
        gaps = np.array(gaps)
        mono = monotone_score(f, 3)
        rows.append({'basis': name, 'mono': mono, 'het': float(np.median(hets)),
                      'cond': float(np.median(conds)), 'med_gap': float(np.median(gaps)),
                      'n1': int((gaps > .01).sum()), 'n': len(gaps)})
        print(f'{name:>22}{mono:>14.2f}{np.median(hets):>9.3f}{np.median(conds):>10.1f}'
              f'{np.median(gaps):>14.5f}{(gaps>.01).sum():>4}/{len(gaps)}')
    out['T2'] = rows
    g = np.array([r['n1'] / r['n'] for r in rows])
    for key in ('mono', 'het', 'cond'):
        v = np.array([r[key] for r in rows])
        print(f'corr(fraction of configs >1%, {key:>5}) = '
              f'{np.corrcoef(v, g)[0,1]:+.3f}')

    # ---------------------------------------------------------------- T3
    print('\n=== T3. Robustness of Q-C: Mixed + Aggregation vs. Odd + Aggregation ===\n')
    print(f'{"s":>3}{"mixed+Cop.":>13}{"odd+Cop.":>14}{"diff":>11}'
          f'{"odd wins":>16}')
    rows = []
    for s in (2, 3, 4):
        mixed = make([(lambda x, i=i: x**i) for i in range(1, s + 1)])
        odd = make([sgnpow(2 * i - 1) for i in range(1, s + 1)])
        rng = np.random.default_rng(99)
        mc, oc = [], []
        for _ in range(45):
            sc = rand_scen(rng)
            try:
                _, c1, _, _ = evaluate(sc, mixed, n_mom=100_000, n_ev=120_000)
                _, c2, _, _ = evaluate(sc, odd, n_mom=100_000, n_ev=120_000)
            except np.linalg.LinAlgError:
                continue
            mc.append(c1); oc.append(c2)
        mc, oc = np.array(mc), np.array(oc)
        rows.append({'s': s, 'mixed_cop': float(np.median(mc)),
                     'odd_cop': float(np.median(oc)),
                     'odd_wins': int((oc < mc).sum()), 'n': len(mc)})
        print(f'{s:>3}{np.median(mc):>13.5f}{np.median(oc):>14.5f}'
              f'{np.median(mc-oc):>+11.5f}{f"{(oc<mc).sum()}/{len(mc)}":>16}')
    out['T3'] = rows

    json.dump(out, open('verification/results_parity_followup.json', 'w'), indent=1)
    print('\nSaved verification/results_parity_followup.json')
