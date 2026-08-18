#!/usr/bin/env python3
"""P2: whether intransitivity arises in pairwise moment-optimal rules under M hypotheses.

Formulation:
    F_{mn} = C^(m) + C^(n),  Y_{mn} = mu^(n) - mu^(m),  K_{mn} = F^{-1} Y
    Lambda_{mn}(x) = K^T (phi(x) - (mu^(m)+mu^(n))/2)  > 0  =>  decision in favor of H_n

The decision rule is invariant under invertible affine transformation of the basis
(phi -> A phi + b, K -> A^{-T} K), hence the basis is whitened against the pooled
covariance: cond(F) drops from ~1e6 to ~1e1, verifying that cycles are geometric
rather than numerical artifacts.

Controls:
  C1  s=1 -- sign(K) is independent of C, cycle probability is identically zero;
  C2  enforced homoscedasticity C^(m) := C_pooled -- Theorem T2 holds;
  C3  multi-seed stability -- rules out MC sample noise in estimated mu, C.
"""
import json
import numpy as np

SEED = 20260817


def noise(n, skew, rng):
    """Standardized gamma noise: zero mean, unit variance, skewness = skew."""
    if skew == 0.0:
        return rng.standard_normal(n)
    k = 4.0 / skew**2
    z = rng.gamma(k, 1.0, size=n)
    return (z - k) / np.sqrt(k)


def raw_basis(x, kind, s, alpha=0.6):
    if kind == 'power':
        return np.stack([x**i for i in range(1, s + 1)], axis=1)
    if kind == 'patp':
        # Published quadratic exponent map for PATP (Form-B), arXiv:2605.14610 eq. 29:
        #   p_i(a) = 1/i + (4 - i - 3/i) a + (2i - 4 + 2/i) a^2 ;  p_1 = 1 identically.
        ps = [(1.0 / i) + (4.0 - i - 3.0 / i) * alpha + (2.0 * i - 4.0 + 2.0 / i) * alpha**2
              for i in range(1, s + 1)]
        return np.stack([np.sign(x) * np.abs(x)**p for p in ps], axis=1)
    raise ValueError(kind)


class Scenario:
    """M hypotheses: signal level a_m and noise skewness skew_m."""

    def __init__(self, levels, skews):
        self.levels, self.skews = list(levels), list(skews)
        self.M = len(levels)

    def sample(self, m, n, rng):
        return self.levels[m] + noise(n, self.skews[m], rng)


class Whitener:
    """Affine basis whitening against pooled covariance -- for conditioning only."""

    def __init__(self, sc, kind, s, n, rng):
        X = np.concatenate([sc.sample(m, n // sc.M, rng) for m in range(sc.M)])
        P = raw_basis(X, kind, s)
        self.mean = P.mean(axis=0)
        C = np.cov(P, rowvar=False, ddof=1).reshape(s, s)
        w, V = np.linalg.eigh(C)
        w = np.maximum(w, 1e-300)
        self.W = V @ np.diag(w**-0.5) @ V.T

    def __call__(self, x, kind, s):
        return (raw_basis(x, kind, s) - self.mean) @ self.W


def moments(sc, wh, kind, s, n, rng):
    mus, Cs = [], []
    for m in range(sc.M):
        P = wh(sc.sample(m, n, rng), kind, s)
        mus.append(P.mean(axis=0))
        Cs.append(np.cov(P, rowvar=False, ddof=1).reshape(s, s))
    return np.array(mus), np.array(Cs)


def pairwise_rules(mus, Cs):
    K = {}
    for m in range(len(mus)):
        for n in range(m + 1, len(mus)):
            F = Cs[m] + Cs[n]
            K[(m, n)] = (np.linalg.solve(F, mus[n] - mus[m]),
                         0.5 * (mus[m] + mus[n]), np.linalg.cond(F))
    return K


def intransitive(Phi, K, M):
    """Tournament is transitive <=> out-degrees form a permutation of 0..M-1."""
    W = np.zeros((Phi.shape[0], M, M), dtype=bool)
    for (m, n), (k, c, _) in K.items():
        win_n = (Phi - c) @ k > 0
        W[:, n, m] = win_n
        W[:, m, n] = ~win_n
    out = W.sum(axis=2)
    return ~(np.sort(out, axis=1) == np.arange(M)).all(axis=1)


def run(sc, kind, s, n_mom=400_000, n_eval=600_000, force_homo=False,
        whiten=True, seed=SEED):
    rng = np.random.default_rng(seed)
    if whiten:
        wh = Whitener(sc, kind, s, 200_000, rng)
    else:
        wh = lambda x, k, ss: raw_basis(x, k, ss)          # noqa: E731
    mus, Cs = moments(sc, wh, kind, s, n_mom, rng)
    if force_homo:
        Cs = np.repeat(Cs.mean(axis=0)[None], sc.M, axis=0)
    K = pairwise_rules(mus, Cs)

    per = n_eval // sc.M
    xs = np.concatenate([sc.sample(m, per, rng) for m in range(sc.M)])
    Phi = wh(xs, kind, s)
    bad = intransitive(Phi, K, sc.M)

    Cbar = Cs.mean(axis=0)
    het = float(np.mean([np.linalg.norm(C - Cbar) for C in Cs]) / np.linalg.norm(Cbar))
    return {'p_cycle': float(bad.mean()), 'heterosced': het,
            'cond': float(max(v[2] for v in K.values()))}


def multi_seed(sc, kind, s, seeds=(1, 2, 3, 4, 5), **kw):
    rs = [run(sc, kind, s, seed=SEED + i, **kw) for i in seeds]
    p = np.array([r['p_cycle'] for r in rs])
    return {'p_mean': float(p.mean()), 'p_min': float(p.min()),
            'p_max': float(p.max()), 'heterosced': rs[0]['heterosced'],
            'cond': rs[0]['cond']}


if __name__ == '__main__':
    out = {}
    shift = lambda d, sk: Scenario([0, d, 2.1 * d], [sk] * 3)      # noqa: E731

    print('=== C1: linear basis s=1 (cycles impossible) ===')
    r = [run(shift(d, 1.0), 'power', 1) for d in (0.5, 1.0, 2.0)]
    for d, x in zip((0.5, 1.0, 2.0), r):
        print(f'  Delta={d}: P(cycle)={x["p_cycle"]:.6f}')
    out['C1_linear'] = r

    print('\n=== C2: enforced homoscedasticity (Theorem T2) ===')
    r = [run(shift(d, 1.0), 'power', 3, force_homo=True) for d in (0.5, 1.0, 2.0)]
    for d, x in zip((0.5, 1.0, 2.0), r):
        print(f'  Delta={d}: P(cycle)={x["p_cycle"]:.6f}')
    out['C2_homosced'] = r

    print('\n=== A: classes differ in SHIFT only, power basis s=3, M=3 ===')
    print(f'{"skew":>6}{"Delta":>7}{"heterosc.":>11}{"P(cycle) mean":>14}'
          f'{"[min..max]":>22}{"cond":>9}')
    grid = []
    for sk in (0.0, 1.0, 1.5):
        for d in (0.5, 1.0, 2.0, 3.0):
            x = multi_seed(shift(d, sk), 'power', 3)
            grid.append({'skew': sk, 'delta': d, **x})
            print(f'{sk:>6.1f}{d:>7.2f}{x["heterosced"]:>11.3f}{x["p_mean"]:>14.6f}'
                  f'   [{x["p_min"]:.6f}..{x["p_max"]:.6f}]{x["cond"]:>9.1f}')
    out['A_shift_only'] = grid

    print('\n=== B: classes differ in NOISE SHAPE, power basis s=3, M=3 ===')
    print(f'{"description":>26}{"heterosc.":>11}{"P(cycle) mean":>14}{"[min..max]":>22}{"cond":>9}')
    shapes = [
        ('skew 0.2/1.0/1.8, d=1.0', Scenario([0, 1.0, 2.1], [0.2, 1.0, 1.8])),
        ('skew 0.2/1.0/1.8, d=0.5', Scenario([0, 0.5, 1.05], [0.2, 1.0, 1.8])),
        ('skew 0.0/1.0/1.9, d=2.0', Scenario([0, 2.0, 4.2], [0.0, 1.0, 1.9])),
        ('skew 0.3/1.9/0.3, d=1.0', Scenario([0, 1.0, 2.1], [0.3, 1.9, 0.3])),
        ('skew 1.9/0.2/1.9, d=0.5', Scenario([0, 0.5, 1.05], [1.9, 0.2, 1.9])),
    ]
    grid = []
    for name, sc in shapes:
        x = multi_seed(sc, 'power', 3)
        grid.append({'name': name, **x})
        print(f'{name:>26}{x["heterosced"]:>11.3f}{x["p_mean"]:>14.6f}'
              f'   [{x["p_min"]:.6f}..{x["p_max"]:.6f}]{x["cond"]:>9.1f}')
    out['B_shape'] = grid

    print('\n=== C: PATP basis s=3, classes differ in shape ===')
    grid = []
    for name, sc in shapes[:3]:
        x = multi_seed(sc, 'patp', 3)
        grid.append({'name': name, **x})
        print(f'{name:>26}{x["heterosced"]:>11.3f}{x["p_mean"]:>14.6f}'
              f'   [{x["p_min"]:.6f}..{x["p_max"]:.6f}]')
    out['C_patp'] = grid

    print('\n=== D: M=4, heterogeneous shapes ===')
    sc4 = Scenario([0, 1.0, 2.1, 3.4], [0.2, 1.2, 0.4, 1.8])
    x = multi_seed(sc4, 'power', 3)
    print(f'  heterosc.={x["heterosced"]:.3f}  P(cycle)={x["p_mean"]:.6f}'
          f'  [{x["p_min"]:.6f}..{x["p_max"]:.6f}]')
    out['D_M4'] = x

    # --- worst-case scenario discovered via random search across configurations ---
    WORST = Scenario([0.0, 0.04, 1.28, 1.39, 1.45], [0.05, 1.83, 1.36, 0.39, 1.39])

    print('\n=== E: Worst-case identified configuration (M=5, s=3) with controls ===')
    x = multi_seed(WORST, 'power', 3, n_mom=600_000, n_eval=800_000)
    y = multi_seed(WORST, 'power', 3, n_mom=600_000, n_eval=800_000, force_homo=True)
    z = multi_seed(WORST, 'power', 1, n_mom=600_000, n_eval=800_000)
    print(f'  as-is:                   P={x["p_mean"]:.4f} '
          f'[{x["p_min"]:.4f}..{x["p_max"]:.4f}]  cond={x["cond"]:.1f}')
    print(f'  homoscedasticity (T2):   P={y["p_mean"]:.6f}')
    print(f'  linear basis s=1:        P={z["p_mean"]:.6f}')
    out['E_worst'] = {'as_is': x, 'homosced': y, 'linear': z}

    print('\n=== F: Sweep across class separation (fixed noise shapes) ===')
    base = np.array([0.0, 0.04, 1.28, 1.39, 1.45])
    sk = [0.05, 1.83, 1.36, 0.39, 1.39]
    grid = []
    print(f'{"scale":>10}{"heterosc.":>11}{"P(cycle)":>11}')
    for c in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        r = multi_seed(Scenario(base * c, sk), 'power', 3,
                       n_mom=250_000, n_eval=300_000)
        grid.append({'scale': c, **r})
        print(f'{c:>10.2f}{r["heterosced"]:>11.3f}{r["p_mean"]:>11.4f}')
    out['F_separation'] = grid

    print('\n=== G: Dependence on hypothesis count M (close classes, distinct shapes) ===')
    rng = np.random.default_rng(11)
    grid = []
    print(f'{"M":>3}{"P(cycle) mean":>14}')
    for M in (3, 4, 5, 6, 7):
        ps = []
        for _ in range(6):
            lv = np.sort(rng.uniform(0, 1.5, M))
            lv -= lv[0]
            try:
                ps.append(run(Scenario(lv, rng.uniform(0.05, 1.9, M)), 'power', 3,
                              n_mom=150_000, n_eval=200_000)['p_cycle'])
            except np.linalg.LinAlgError:
                pass
        grid.append({'M': M, 'p_mean': float(np.mean(ps))})
        print(f'{M:>3}{np.mean(ps):>14.4f}')
    out['G_vs_M'] = grid

    json.dump(out, open('verification/results_p2.json', 'w'), indent=1)
    print('\nSaved verification/results_p2.json')
