#!/usr/bin/env python3
"""Threshold ablation and One-Versus-Mixture baseline verification.

A. THRESHOLD ABLATION: Criterion (3) is invariant to K -> aK, making threshold choice
   a convention. We verify that undecided events are not an artifact of the midpoint
   threshold (c = (mu_m + mu_n)/2). Three threshold conventions are compared on 120 configs:
     mid-mean   Lambda > 0, threshold at c = (mu_m + mu_n)/2 (letter baseline)
     eqerr      Gaussian equal-error threshold:
                t = (J/2)(sigma_m - sigma_n)/(sigma_m + sigma_n),
                where sigma_m^2 = K' C^(m) K, sigma_n^2 = K' C^(n) K
     gmap       Gaussian MAP: hypothesis n wins if
                logN(Lambda; +J/2, v_n) > logN(Lambda; -J/2, v_m)
   (Under H_m, E[Lambda] = -J/2; under H_n, E[Lambda] = +J/2, and v_m + v_n = J.)

B. ONE-VERSUS-MIXTURE (OvM) BASELINE: A complete competitor directly derived from Prop. 1:
   project each p_m/qbar onto span{1, phi_1..phi_s} in L^2(qbar), qbar = M^-1 sum_m p_m,
   and take argmax. Because
       <p_m/qbar, psi_i>_qbar = E_m[psi_i],
   coefficients are E_qbar[psi psi']^-1 E_m[psi] for psi = (1, phi) -- computed from the
   exact same moments mu^(m), C^(m). This rule is linear in feature space, hence potential-
   generated and complete by construction.

Both runs use the 120-configuration sweep (rng 2026), s=3, mixed power basis,
whitened by pooled covariance, seed 20260817, matching v8_table1.py.
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, pairwise_rules            # noqa: E402
from basis_parity_sweep import logpdf                             # noqa: E402

SEED = 20260817
S = 3
BASIS = lambda x: np.stack([x**i for i in range(1, S + 1)], 1)    # noqa: E731


def glp(z, m, v):
    return -0.5 * np.log(2 * np.pi * v) - 0.5 * (z - m)**2 / v


def evaluate(sc, n_mom=100_000, n_ev=120_000, seed=SEED):
    rng = np.random.default_rng(seed)
    M = sc.M
    Xw = np.concatenate([sc.sample(m, 60_000 // M, rng) for m in range(M)])
    P = BASIS(Xw)
    mean, C = P.mean(0), np.cov(P, rowvar=False, ddof=1)
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(np.maximum(w, 1e-300)**-0.5) @ V.T
    tr = lambda x: (BASIS(x) - mean) @ W                          # noqa: E731

    mus, Cs = [], []
    for m in range(M):
        Q = tr(sc.sample(m, n_mom, rng))
        mus.append(Q.mean(0))
        Cs.append(np.cov(Q, rowvar=False, ddof=1))
    mus, Cs = np.array(mus), np.array(Cs)
    K = pairwise_rules(mus, Cs)

    par = {}
    for (m, n), (k, c, _) in K.items():
        J = float(k @ (mus[n] - mus[m]))
        vm = float(k @ Cs[m] @ k)
        vn = float(k @ Cs[n] @ k)
        # equal-error threshold in Lambda coordinate (centered at c)
        t_eq = 0.5 * J * (np.sqrt(vm) - np.sqrt(vn)) / (np.sqrt(vm) + np.sqrt(vn))
        par[(m, n)] = (k, c, J, vm, vn, t_eq)

    # --- B: one-versus-mixture, coefficients from identical moments ---
    # psi = (1, phi);  E_qbar[psi psi'] computed from mus, Cs (equal priors)
    mbar = mus.mean(0)
    Sbar = (Cs + np.einsum('mi,mj->mij', mus, mus)).mean(0)       # E_qbar[phi phi']
    G = np.empty((S + 1, S + 1))
    G[0, 0] = 1.0
    G[0, 1:] = mbar
    G[1:, 0] = mbar
    G[1:, 1:] = Sbar
    E_psi = np.hstack([np.ones((M, 1)), mus])                     # E_m[psi]
    Aovm = np.linalg.solve(G, E_psi.T).T                          # (M, S+1)

    # --- other complete decision rules, matching v8 ---
    SW = Cs.mean(0)
    Apool = np.linalg.solve(SW, mus.T).T
    bpool = -0.5 * np.einsum('ms,ms->m', Apool, mus)

    out = {'M': M}
    err = {k: [] for k in ('lin', 'pooled', 'ovm', 'cop_margin', 'bayes')}
    gaps = {k: [] for k in ('midmean', 'eqerr', 'gmap')}
    a = np.array([sc.sample(m, n_mom, np.random.default_rng(seed + 1 + m)).mean()
                  for m in range(M)])

    for m in range(M):
        x = sc.sample(m, n_ev, rng)
        Phi = tr(x)
        n_ = len(x)
        wins = {k: np.zeros((n_, M), int) for k in gaps}
        margin = np.zeros((n_, M))
        for (i, j), (k, c, J, vm, vn, t_eq) in par.items():
            lam = (Phi - c) @ k
            w_mid = lam > 0
            w_eq = lam > t_eq
            w_map = glp(lam, +0.5 * J, vn) > glp(lam, -0.5 * J, vm)
            for key, wj in (('midmean', w_mid), ('eqerr', w_eq), ('gmap', w_map)):
                wins[key][:, j] += wj
                wins[key][:, i] += ~wj
            std = lam / np.sqrt(max(J, 1e-12))
            margin[:, j] += std
            margin[:, i] -= std
        for key in gaps:
            gaps[key].append(float((~(wins[key] == M - 1).any(1)).mean()))
        tied = wins['midmean'] == wins['midmean'].max(1, keepdims=True)
        err['cop_margin'].append(
            float((np.where(tied, margin, -np.inf).argmax(1) != m).mean()))
        err['lin'].append(float((np.abs(x[:, None] - a[None, :]).argmin(1) != m).mean()))
        err['pooled'].append(float(((Phi @ Apool.T + bpool).argmax(1) != m).mean()))
        Psi = np.hstack([np.ones((n_, 1)), Phi])
        err['ovm'].append(float(((Psi @ Aovm.T).argmax(1) != m).mean()))
        L = np.stack([logpdf(x, sc.levels[j], sc.skews[j]) for j in range(M)], 1)
        err['bayes'].append(float((L.argmax(1) != m).mean()))

    for k, v in gaps.items():
        out['gap_' + k] = float(np.mean(v))
    for k, v in err.items():
        out[k] = float(np.mean(v))
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(2026)
    rows = []
    for t in range(120):
        M = int(rng.integers(3, 6))
        lv = np.sort(rng.uniform(0, 2.5, M))
        lv -= lv[0]
        sc = Scenario(lv, rng.uniform(0.0, 1.9, M))
        try:
            r = evaluate(sc)
        except np.linalg.LinAlgError:
            continue
        r['idx'] = t
        rows.append(r)
        if t % 20 == 19:
            print(f'{t+1}/120', flush=True)

    g = lambda k: np.array([r[k] for r in rows])                  # noqa: E731

    print('\n=== A. Threshold Ablation: Are undecided events an artifact of midpoint threshold? ===\n')
    print(f'{"threshold convention":>26}{"median":>12}{"maximum":>11}{"> 1%":>8}{"> 5%":>7}')
    for key, name in (('midmean', 'mid-mean (as in letter)'),
                      ('eqerr', 'Gaussian equal-error'),
                      ('gmap', 'Gaussian MAP')):
        v = g('gap_' + key)
        print(f'{name:>26}{np.median(v):>12.5f}{v.max():>11.4f}'
              f'{(v > 0.01).sum():>8}{(v > 0.05).sum():>7}')

    print('\n=== B. One-Versus-Mixture Baseline (Projection from Prop. 1) ===\n')
    lin, bay = g('lin'), g('bayes')
    print(f'{"rule":>16}{"median":>11}{"gain vs lin":>16}{"wins":>10}'
          f'{"fraction of gap":>18}')
    for k, nm in (('lin', 'linear s=1'), ('pooled', 'pooled'),
                  ('ovm', 'one-versus-mixture'), ('cop_margin', 'pairwise+Copeland'),
                  ('bayes', 'Bayes')):
        e = g(k)
        print(f'{nm:>16}{np.median(e):>11.4f}{np.median(lin - e) * 100:>+16.2f}'
              f'{int((e < lin).sum()):>10}{np.median((lin - e) / (lin - bay)):>18.3f}')
    ovm, cop, pool = g('ovm'), g('cop_margin'), g('pooled')
    print(f'\nCopeland vs One-Versus-Mixture: median {np.median(ovm - cop)*100:+.2f} pp, '
          f'Copeland better in {int((cop < ovm).sum())}/{len(rows)}')
    print(f'One-Versus-Mixture vs Pooled: median {np.median(pool - ovm)*100:+.2f} pp, '
          f'better in {int((ovm < pool).sum())}/{len(rows)}')

    json.dump(rows, open('verification/results_v11_threshold_ovm.json', 'w'), indent=1)
    print('\nSaved verification/results_v11_threshold_ovm.json')
