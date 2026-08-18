#!/usr/bin/env python3
"""Robustness of the ranking coupling < Copeland < pooled < lin under small training samples.

Identical 120 configurations (rng 2026), s=3, mixed power basis; whitening and moments
are estimated from a SINGLE training sample of size n_train per class (realistic protocol),
with evaluation performed on a fixed large test sample (60k per class). 10 replicas per config.
Gaussian calibration of coupling also derives variances from the same training sample.
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, pairwise_rules            # noqa: E402
from basis_parity_sweep import logpdf                             # noqa: E402
import v8_table1                                                   # noqa: E402
from v8_table1 import gauss_logpdf, wlw_coupling                  # noqa: E402
BASIS = lambda x: np.stack([x**i for i in range(1, 4)], 1)         # noqa: E731  (s=3)

SEED = 20260817
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 500
REPS = 10


def fit_and_test(sc, n_train, Xtest, rng):
    M = sc.M
    tr_x = [sc.sample(m, n_train, rng) for m in range(M)]
    P = BASIS(np.concatenate(tr_x))
    mean, C = P.mean(0), np.cov(P, rowvar=False, ddof=1)
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(np.maximum(w, 1e-300)**-0.5) @ V.T
    tr = lambda x: (BASIS(x) - mean) @ W                              # noqa: E731
    mus = np.array([tr(t).mean(0) for t in tr_x])
    Cs = np.array([np.cov(tr(t), rowvar=False, ddof=1) for t in tr_x])
    K = pairwise_rules(mus, Cs)
    a = np.array([t.mean() for t in tr_x])
    SW = Cs.mean(0)
    A = np.linalg.solve(SW, mus.T).T
    b = -0.5 * np.einsum('ms,ms->m', A, mus)
    par = {}
    for (m, n), (k, c, _) in K.items():
        J = k @ (mus[n] - mus[m])
        par[(m, n)] = (k, c, J, k @ Cs[m] @ k, k @ Cs[n] @ k)
    err = {k: [] for k in ('lin', 'pooled', 'coupling', 'cop_margin')}
    fails = []
    for m in range(M):
        x = Xtest[m]
        Phi = tr(x)
        n_ = len(x)
        wins = np.zeros((n_, M), int)
        margin = np.zeros((n_, M))
        R = np.full((n_, M, M), 0.5)
        for (i, j), (k, c, J, vi, vj) in par.items():
            lam = (Phi - c) @ k
            win_j = lam > 0
            wins[:, j] += win_j
            wins[:, i] += ~win_j
            std = lam / np.sqrt(max(J, 1e-12))
            margin[:, j] += std
            margin[:, i] -= std
            li = gauss_logpdf(lam, -0.5 * J, max(vi, 1e-12))
            lj = gauss_logpdf(lam, +0.5 * J, max(vj, 1e-12))
            rj = 1.0 / (1.0 + np.exp(np.clip(li - lj, -700, 700)))
            R[:, j, i] = rj
            R[:, i, j] = 1.0 - rj
        tied = wins == wins.max(1, keepdims=True)
        fails.append(float((~(wins == M - 1).any(1)).mean()))
        dec = {'cop_margin': np.where(tied, margin, -np.inf).argmax(1),
               'lin': np.abs(x[:, None] - a[None, :]).argmin(1),
               'pooled': (Phi @ A.T + b).argmax(1),
               'coupling': wlw_coupling(R, M)}
        for k, d in dec.items():
            err[k].append(float((d != m).mean()))
    return {k: float(np.mean(v)) for k, v in err.items()}, float(np.mean(fails))


if __name__ == '__main__':
    rng_cfg = np.random.default_rng(2026)
    rows = []
    for t in range(120):
        M = int(rng_cfg.integers(3, 6))
        lv = np.sort(rng_cfg.uniform(0, 2.5, M))
        lv -= lv[0]
        sk = rng_cfg.uniform(0.0, 1.9, M)
        sc = Scenario(lv, sk)
        rt = np.random.default_rng(SEED + 7)
        Xtest = [sc.sample(m, 60_000, rt) for m in range(M)]
        reps = []
        for r in range(REPS):
            try:
                e, f = fit_and_test(sc, N_TRAIN, Xtest, np.random.default_rng(1000 * t + r))
            except np.linalg.LinAlgError:
                continue
            e['fail'] = f
            reps.append(e)
        if not reps:
            continue
        agg = {k: float(np.mean([e[k] for e in reps])) for k in reps[0]}
        agg['idx'] = t
        agg['M'] = M
        agg['n_reps'] = len(reps)
        rows.append(agg)
        if t % 20 == 19:
            print(f'{t+1}/120', flush=True)

    get = lambda k: np.array([r[k] for r in rows])                   # noqa: E731
    print(f'\nn_train = {N_TRAIN} per class, {REPS} replicas, {len(rows)} configurations')
    print(f'P(fail): median {np.median(get("fail")):.5f}, max {get("fail").max():.4f}, '
          f'>1%: {(get("fail")>0.01).sum()}')
    lin = get('lin')
    for k in ('lin', 'pooled', 'cop_margin', 'coupling'):
        e = get(k)
        print(f'{k:>11}: median {np.median(e):.5f}; med. gain vs lin {np.median(lin-e):+.5f}; '
              f'better than lin in {(e<lin).sum()}/{len(rows)}')
    c, cp, po = get('coupling'), get('cop_margin'), get('pooled')
    print(f'coupling vs cop_margin: med. {np.median(cp-c):+.5f}, coupling wins in {(c<cp).sum()}/{len(rows)}')
    print(f'cop_margin vs pooled:   med. {np.median(po-cp):+.5f}, cop wins in {(cp<po).sum()}/{len(rows)}')
    print(f'coupling vs pooled:     med. {np.median(po-c):+.5f}, coupling wins in {(c<po).sum()}/{len(rows)}')
    json.dump(rows, open(f'verification/results_v9_trainsize_n{N_TRAIN}.json', 'w'), indent=1)
    print(f'Saved verification/results_v9_trainsize_n{N_TRAIN}.json')
