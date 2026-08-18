#!/usr/bin/env python3
"""Table I of the SPL letter: all decision rules on identical 120 configurations.

Configurations -- generated via basis_parity_sweep.py sweep (rng 2026), mixed
power basis {x, x^2, x^3}, whitening via pooled covariance,
n_mom=100k, n_ev=120k per class, seed 20260817 (as in v1/v4).

Decision rules (all computed strictly from moments mu^(m), C^(m), except Bayes oracle):
    lin         s=1, pairwise scheme = nearest mean (baseline D16)
    pooled      shared S_W = mean C^(m), argmax of linear scores (Rao/LDA)
    qda         Gaussian QDA in whitened basis space (consistent by design)
    coupling    pairwise coupling of Wu-Lin-Weng (Method 2) with r_mn from Gaussian
                approximation of Lambda_mn under both hypotheses (consistent by design)
    cop_margin  Copeland, tie broken by sum of standardized margins among leaders
    cop_random  Copeland, tie broken uniformly at random among leaders
    bayes       Bayes oracle from true underlying densities
Additionally: P(fail) -- fraction of events without a Condorcet winner; error rate
of each rule conditioned on failure events (demonstrating intrinsic ambiguity).
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, pairwise_rules            # noqa: E402
from basis_parity_sweep import logpdf                             # noqa: E402

SEED = 20260817
S = int(sys.argv[1]) if len(sys.argv) > 1 else 3
BASIS = lambda x: np.stack([x**i for i in range(1, S + 1)], 1)     # noqa: E731


def gauss_logpdf(z, m, v):
    return -0.5 * np.log(2 * np.pi * v) - 0.5 * (z - m)**2 / v


def wlw_coupling(R, M):
    """Wu-Lin-Weng (2004) Method 2. R[:, i, j] = r_ij = estimate of P(i | i or j)."""
    n = R.shape[0]
    Q = np.zeros((n, M, M))
    for i in range(M):
        for j in range(M):
            if i == j:
                Q[:, i, i] = sum(R[:, j2, i]**2 for j2 in range(M) if j2 != i)
            else:
                Q[:, i, j] = -R[:, j, i] * R[:, i, j]
    A = np.zeros((n, M + 1, M + 1))
    A[:, :M, :M] = Q
    A[:, :M, M] = 1.0
    A[:, M, :M] = 1.0
    b = np.zeros((n, M + 1))
    b[:, M] = 1.0
    p = np.linalg.solve(A, b[..., None])[:, :M, 0]
    return p.argmax(1)


def evaluate(sc, n_mom=100_000, n_ev=120_000, seed=SEED):
    rng = np.random.default_rng(seed)
    M = sc.M
    # whitening
    Xw = np.concatenate([sc.sample(m, 60_000 // M, rng) for m in range(M)])
    P = BASIS(Xw)
    mean, C = P.mean(0), np.cov(P, rowvar=False, ddof=1)
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(np.maximum(w, 1e-300)**-0.5) @ V.T
    tr = lambda x: (BASIS(x) - mean) @ W                              # noqa: E731
    # moments
    mus, Cs = [], []
    for m in range(M):
        Q = tr(sc.sample(m, n_mom, rng))
        mus.append(Q.mean(0))
        Cs.append(np.cov(Q, rowvar=False, ddof=1))
    mus, Cs = np.array(mus), np.array(Cs)
    K = pairwise_rules(mus, Cs)
    # linear rule s=1: nearest mean on x (identical to pairwise s=1)
    a = np.array([sc.sample(m, n_mom, np.random.default_rng(seed + 1 + m)).mean()
                  for m in range(M)])
    # pooled
    SW = Cs.mean(0)
    A = np.linalg.solve(SW, mus.T).T
    b = -0.5 * np.einsum('ms,ms->m', A, mus)
    # QDA in basis space
    Cinv = np.array([np.linalg.inv(c) for c in Cs])
    logdet = np.array([np.linalg.slogdet(c)[1] for c in Cs])
    # parameters of Lambda_mn for standardization and coupling
    par = {}
    for (m, n), (k, c, _) in K.items():
        J = k @ (mus[n] - mus[m])
        par[(m, n)] = (k, c, J, k @ Cs[m] @ k, k @ Cs[n] @ k)

    res = {}
    for m in range(M):
        x = sc.sample(m, n_ev, rng)
        Phi = tr(x)
        n_ = len(x)
        # pairwise statistics
        wins = np.zeros((n_, M), int)
        margin = np.zeros((n_, M))
        R = np.full((n_, M, M), 0.5)
        for (i, j), (k, c, J, vi, vj) in par.items():
            lam = (Phi - c) @ k
            win_j = lam > 0
            wins[:, j] += win_j
            wins[:, i] += ~win_j
            std = lam / np.sqrt(J)                    # K^T F K = J
            margin[:, j] += std
            margin[:, i] -= std
            # Gaussian calibration: under H_i Lambda ~ N(-J/2, vi), under H_j ~ N(+J/2, vj)
            li = gauss_logpdf(lam, -0.5 * J, vi)
            lj = gauss_logpdf(lam, +0.5 * J, vj)
            rj = 1.0 / (1.0 + np.exp(np.clip(li - lj, -700, 700)))
            R[:, j, i] = rj
            R[:, i, j] = 1.0 - rj
        top = wins.max(1, keepdims=True)
        tied = wins == top
        fail = ~(wins == M - 1).any(1)
        # Copeland + margins
        cop_margin = np.where(tied, margin, -np.inf).argmax(1)
        # Copeland + random
        rr = np.random.default_rng(seed + 100 + m).random((n_, M))
        cop_random = np.where(tied, rr, -1.0).argmax(1)
        lin = np.abs(x[:, None] - a[None, :]).argmin(1)
        pooled = (Phi @ A.T + b).argmax(1)
        d = Phi[:, None, :] - mus[None, :, :]
        qda = (-0.5 * logdet[None, :]
               - 0.5 * np.einsum('nms,mst,nmt->nm', d, Cinv, d)).argmax(1)
        coup = wlw_coupling(R, M)
        L = np.stack([logpdf(x, sc.levels[j], sc.skews[j]) for j in range(M)], 1)
        bayes = L.argmax(1)
        dec = {'lin': lin, 'pooled': pooled, 'qda': qda, 'coupling': coup,
               'cop_margin': cop_margin, 'cop_random': cop_random, 'bayes': bayes}
        for name, dd in dec.items():
            r = res.setdefault(name, {'err': [], 'err_fail': [], 'err_dec': []})
            r['err'].append(float((dd != m).mean()))
            r['err_fail'].append(float((dd[fail] != m).mean()) if fail.any() else np.nan)
            r['err_dec'].append(float((dd[~fail] != m).mean()))
        res.setdefault('fail', []).append(float(fail.mean()))
        # consistency of coupling/QDA with Condorcet winner on decided events
        cw = wins.argmax(1)
        res.setdefault('coup_vs_cw', []).append(float((coup[~fail] != cw[~fail]).mean()))
        res.setdefault('qda_vs_cw', []).append(float((qda[~fail] != cw[~fail]).mean()))
        res.setdefault('pooled_vs_cw', []).append(float((pooled[~fail] != cw[~fail]).mean()))
    out = {'M': M, 'fail': float(np.mean(res['fail']))}
    for k in ('coup_vs_cw', 'qda_vs_cw', 'pooled_vs_cw'):
        out[k] = float(np.mean(res[k]))
    for name in ('lin', 'pooled', 'qda', 'coupling', 'cop_margin', 'cop_random', 'bayes'):
        out[name] = float(np.mean(res[name]['err']))
        out[name + '_dec'] = float(np.mean(res[name]['err_dec']))
        ef = np.array(res[name]['err_fail'], float)
        out[name + '_fail'] = float(np.nanmean(ef)) if np.isfinite(ef).any() else None
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(2026)
    rows = []
    for t in range(120):
        M = int(rng.integers(3, 6))
        lv = np.sort(rng.uniform(0, 2.5, M))
        lv -= lv[0]
        sk = rng.uniform(0.0, 1.9, M)
        sc = Scenario(lv, sk)
        try:
            r = evaluate(sc)
        except np.linalg.LinAlgError:
            print('LinAlgError at', t)
            continue
        r['idx'] = t
        r['levels'] = [float(v) for v in lv]
        r['skews'] = [float(v) for v in sk]
        rows.append(r)
        if t % 20 == 19:
            print(f'{t+1}/120 ...', flush=True)

    names = ('lin', 'pooled', 'qda', 'coupling', 'cop_margin', 'cop_random', 'bayes')
    get = lambda k, sub=rows: np.array([r[k] for r in sub])       # noqa: E731
    fail = get('fail')
    hi = [r for r in rows if r['fail'] > 0.01]
    print(f'\nValid: {len(rows)}; P(fail): median {np.median(fail):.5f}, '
          f'max {fail.max():.4f}, >1%: {len(hi)}, >5%: {(fail>0.05).sum()}')
    print(f'\n{"rule":>12}{"med. error":>14}{"med. gain vs lin":>20}'
          f'{"vs pooled":>11}{"wins/120 vs pooled":>24}{"on >1% (med)":>14}')
    lin, pool = get('lin'), get('pooled')
    for nm in names:
        e = get(nm)
        d_lin = np.median(lin - e)
        d_pool = np.median(pool - e)
        wins = int((e < pool).sum())
        e_hi = np.median(get(nm, hi))
        print(f'{nm:>12}{np.median(e):>14.5f}{d_lin:>+20.5f}{d_pool:>+11.5f}'
              f'{wins:>24}{e_hi:>14.5f}')
    print('\nError on FAILURE events (median across configs with failures) / on DECIDED events:')
    withf = [r for r in rows if r['fail'] > 0]
    for nm in names:
        ef = np.array([r[nm + '_fail'] for r in withf if r[nm + '_fail'] is not None])
        ed = get(nm + '_dec')
        print(f'{nm:>12}  fail {np.median(ef):.4f}   dec {np.median(ed):.4f}')
    print('\nFraction of decided events where rule disagrees with Condorcet winner (median):')
    for k in ('coup_vs_cw', 'qda_vs_cw', 'pooled_vs_cw'):
        print(f'{k:>14} {np.median(get(k)):.4f}')
    print(f'\ncop_margin vs coupling: median diff {np.median(get("coupling")-get("cop_margin")):+.5f}, '
          f'cop wins in {(get("cop_margin")<get("coupling")).sum()}/120')
    print(f'cop_margin vs qda:      median diff {np.median(get("qda")-get("cop_margin")):+.5f}, '
          f'cop wins in {(get("cop_margin")<get("qda")).sum()}/120')
    print(f'cop_margin vs cop_random: median diff {np.median(get("cop_random")-get("cop_margin")):+.6f}')
    gap_lin = np.median(lin - get('bayes'))
    print(f'\nMedian gap lin -> Bayes {gap_lin:.4f}; fraction closed by cop_margin: '
          f'{np.median((lin-get("cop_margin"))/(lin-get("bayes"))):.3f}')
    json.dump(rows, open(f'verification/results_v8_table1_s{S}.json' if S != 3 else 'verification/results_v8_table1.json', 'w'), indent=1)
    print(f'\nSaved verification/results_v8_table1{"" if S == 3 else f"_s{S}"}.json')
