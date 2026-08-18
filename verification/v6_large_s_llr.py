#!/usr/bin/env python3
"""V6: Large polynomial degree s -- pairwise polynomial rule vs. true LLR.

Theory:
  J(g) = (E1 g - E0 g)^2 / (Var0 g + Var1 g) is maximized over L2 by
  h = tanh(LLR/2) = (p1-p0)/(p0+p1) (up to affine transformation);
  the pairwise rule with midpoint threshold is an orthogonal projection of h onto
  span{1, phi_1..phi_s} in L2(q), q = (p_m+p_n)/2, with a positive scaling factor.
  Therefore, by basis completeness, Lambda_s -> h in L2(q), threshold -> LR = 1,
  and the undecided rate (no Condorcet winner) converges to 0.

Numerical validation: power basis moments are computed EXACTLY (standardized gamma
cumulants via mpmath with 60-digit precision), orthonormalized against the pooled
mixture (equivalent to whitening; invariant to affine coordinate shifts).
Evaluation via Monte Carlo. Additionally: sample-based version (Whitener from p2)
to verify empirical moment estimation effects.

Metrics per pair and degree s:
  spearman(Lambda_s, LLR), pearson(Lambda_s, tanh(LLR/2)), J_s / J*,
  P(sign Lambda_s != sign LLR), effective log-threshold t* (minimizing disagreement with
  [LLR > t]) and disagreement rate at t*.
Per scenario and degree s: gap = P(no Condorcet winner), P(cycle), Copeland error,
Bayes error, maximum cond(F).
"""
import json
import sys
import time
from math import comb, factorial

import numpy as np
import mpmath as mp
from scipy import stats as sps

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, Whitener, moments, pairwise_rules, noise  # noqa: E402
from basis_parity_sweep import logpdf                                              # noqa: E402

mp.mp.dps = 60
SEED = 20260818
S_MAX = 8


# ---------- exact moments ----------
def std_gamma_moments(skew, nmax):
    """Moments E y^n, n=0..nmax, of standardized gamma (mpmath)."""
    g = mp.mpf(skew)
    kap = [mp.mpf(0)] * (nmax + 1)
    kap[1] = mp.mpf(0)
    for j in range(2, nmax + 1):
        kap[j] = mp.factorial(j - 1) * (g / 2) ** (j - 2)   # (j-1)! (skew/2)^{j-2}
    m = [mp.mpf(1)] + [mp.mpf(0)] * nmax
    for n in range(1, nmax + 1):
        m[n] = sum(mp.binomial(n - 1, j - 1) * kap[j] * m[n - j] for j in range(1, n + 1))
    return m


def shifted_moments(a, skew, nmax):
    my = std_gamma_moments(skew, nmax)
    a = mp.mpf(a)
    return [sum(mp.binomial(i, j) * a ** (i - j) * my[j] for j in range(i + 1))
            for i in range(nmax + 1)]


def exact_rules(sc, s):
    """Returns: coefficients of orthonormalized (against mixture) polynomials P_i (i=1..s)
    in monomial basis (float64, for Horner evaluation), dictionary of pairwise rules K,
    mu, C in this basis (float64), and exact J_s."""
    M = sc.M
    mom = [shifted_moments(a, sk, 2 * s) for a, sk in zip(sc.levels, sc.skews)]
    # pooled mixture
    mix = [sum(mom[m][i] for m in range(M)) / M for i in range(2 * s + 1)]
    mean = mp.matrix([mix[i] for i in range(1, s + 1)])
    Cq = mp.matrix(s, s)
    for i in range(1, s + 1):
        for j in range(1, s + 1):
            Cq[i - 1, j - 1] = mix[i + j] - mix[i] * mix[j]
    L = mp.cholesky(Cq)                     # Cq = L L^T
    Linv = mp.inverse(L)                    # lower triangular: P = Linv (phi - mean)
    # monomial coefficients of P_i: P_i(x) = sum_j Linv[i,j] x^{j+1} - (Linv mean)_i
    const = Linv * mean
    coef = np.zeros((s, s + 1))             # coef[i, d] -- coefficient at x^d
    for i in range(s):
        coef[i, 0] = float(-const[i])
        for j in range(s):
            coef[i, j + 1] = float(Linv[i, j])
    mus, Cs = [], []
    for m in range(M):
        mu_raw = mp.matrix([mom[m][i] for i in range(1, s + 1)])
        C_raw = mp.matrix(s, s)
        for i in range(1, s + 1):
            for j in range(1, s + 1):
                C_raw[i - 1, j - 1] = mom[m][i + j] - mom[m][i] * mom[m][j]
        mu_w = Linv * (mu_raw - mean)
        C_w = Linv * C_raw * Linv.T
        mus.append(np.array([float(mu_w[i]) for i in range(s)]))
        Cs.append(np.array([[float(C_w[i, j]) for j in range(s)] for i in range(s)]))
    mus, Cs = np.array(mus), np.array(Cs)
    K = pairwise_rules(mus, Cs)             # float64 in orthonormal basis
    J = {mn: float((mus[mn[1]] - mus[mn[0]]) @ k) for mn, (k, c, _) in K.items()}
    return coef, K, mus, Cs, J


def eval_basis(x, coef):
    """P_i(x) for all i via Horner in float64."""
    s = coef.shape[0]
    out = np.empty((len(x), s))
    for i in range(s):
        c = coef[i]
        acc = np.full_like(x, c[-1])
        for d in range(len(c) - 2, -1, -1):
            acc = acc * x + c[d]
        out[:, i] = acc
    return out


# ---------- J* via quadrature ----------
def pdf(x, a, skew):
    return np.exp(logpdf(np.asarray(x, float), a, skew))


def j_star(a0, s0, a1, s1):
    lo = min(a0 - (np.sqrt(4 / s0**2) if s0 > 0 else 40), a1 - (np.sqrt(4 / s1**2) if s1 > 0 else 40))
    lo = max(lo, min(a0, a1) - 40)
    hi = max(a0, a1) + 60
    x = np.linspace(lo, hi, 2_000_001)
    p0, p1 = pdf(x, a0, s0), pdf(x, a1, s1)
    den = p0 + p1
    ok = den > 0
    delta = np.trapezoid((p1[ok] - p0[ok])**2 / den[ok], x[ok])
    return 2 * delta / (2 - delta), delta


# ---------- effective threshold ----------
def effective_threshold(lam, llr):
    """t* minimizing mean([lam>0] != [llr>t]); returns (t*, disagree(t*), disagree(0))."""
    fin = np.isfinite(llr)
    d0 = np.mean((lam > 0) != (llr > 0))
    l, y = llr[fin], (lam[fin] > 0)
    o = np.argsort(l)
    l, y = l[o], y[o]
    n = len(l)
    # threshold between l[i-1] and l[i]: points < i predicted 0, >= i predicted 1
    cum1 = np.concatenate([[0], np.cumsum(y)])          # count of y=1 in first i
    err = cum1 + (np.sum(~y) - (np.arange(n + 1) - cum1))  # y=1 below + y=0 above
    i = int(np.argmin(err))
    t = (l[i - 1] + l[i]) / 2 if 0 < i < n else (l[0] - 1 if i == 0 else l[-1] + 1)
    # infinite LLRs: -inf always < t, +inf > t
    err_inf = np.sum((lam[~fin] > 0) != (llr[~fin] > 0))
    return float(t), float((err[i] + err_inf) / len(llr)), float(d0)


# ---------- scenario evaluation ----------
def evaluate_exact(sc, s, n_per, rng):
    coef, K, mus, Cs, J = exact_rules(sc, s)
    M = sc.M
    xs = [sc.sample(m, n_per, rng) for m in range(M)]
    Phis = [eval_basis(x, coef) for x in xs]
    LL = [np.stack([logpdf(x, sc.levels[j], sc.skews[j]) for j in range(M)], 1) for x in xs]
    # accuracy check: mixture covariance of whitened basis ~ I
    Pall = np.concatenate(Phis)
    Cchk = np.cov(Pall, rowvar=False, ddof=1).reshape(s, s)
    whiten_err = float(np.abs(Cchk - np.eye(s)).max())

    gaps, cycles, e_cop, e_bay = [], [], [], []
    mism = {mn: [] for mn in K}
    for m in range(M):
        Phi, L = Phis[m], LL[m]
        out = np.zeros((n_per, M), int)
        for (i, j), (k, c, _) in K.items():
            lam = (Phi - c) @ k
            win = lam > 0
            out[:, j] += win
            out[:, i] += ~win
            dl = L[:, j] - L[:, i]
            okp = ~np.isnan(dl)            # both densities zero -> pair irrelevant for Bayes
            mism[(i, j)].append(np.mean(win[okp] != (dl[okp] > 0)))
        gaps.append((~(out == M - 1).any(1)).mean())
        cycles.append((~(np.sort(out, 1) == np.arange(M)).all(1)).mean())
        e_cop.append((out.argmax(1) != m).mean())
        e_bay.append((L.argmax(1) != m).mean())
    pairs = {}
    for (i, j), (k, c, cond) in K.items():
        Phi = np.concatenate([Phis[i], Phis[j]])
        L = np.concatenate([LL[i], LL[j]])
        lam = (Phi - c) @ k
        llr = L[:, j] - L[:, i]
        fin = np.isfinite(llr)
        rho = sps.spearmanr(lam[fin], llr[fin]).statistic
        h = np.tanh(np.clip(llr, -700, 700) / 2)
        pear = np.corrcoef(lam, h)[0, 1]
        t, d_t, d_0 = effective_threshold(lam, llr)
        # check identity "rule = projection of tanh(LLR/2)": OLS of h on centered basis
        Xc = Phi - Phi.mean(0)
        beta = np.linalg.lstsq(Xc, h - h.mean(), rcond=None)[0]
        cos = float(beta @ k / (np.linalg.norm(beta) * np.linalg.norm(k)))
        pairs[f'{i}{j}'] = {'proj_cosine': cos, 'spearman': float(rho), 'pearson_tanh': float(pear),
                            'J_s': J[(i, j)], 'cond': float(cond),
                            'p_sign_mismatch_mix': float(d_0),
                            'p_sign_mismatch_avg_over_hyp': float(np.mean(mism[(i, j)])),
                            'eff_log_threshold': t, 'disagree_at_eff': d_t,
                            'frac_llr_inf': float(np.mean(~fin))}
    return {'s': s, 'gap': float(np.mean(gaps)), 'p_cycle': float(np.mean(cycles)),
            'err_copeland': float(np.mean(e_cop)), 'err_bayes': float(np.mean(e_bay)),
            'cond_max': float(max(v[2] for v in K.values())),
            'whiten_check_maxdev': whiten_err,
            'sum_pair_mismatch_bound': float(sum(np.mean(v) for v in mism.values())),
            'pairs': pairs}


def evaluate_sample(sc, s, n_mom, n_per, seed):
    """Sample-based version: Whitener + moments from p2 (estimated moments)."""
    rng = np.random.default_rng(seed)
    wh = Whitener(sc, 'power', s, 200_000, rng)
    mus, Cs = moments(sc, wh, 'power', s, n_mom, rng)
    K = pairwise_rules(mus, Cs)
    M = sc.M
    gaps, rhos = [], []
    for m in range(M):
        x = sc.sample(m, n_per, rng)
        Phi = wh(x, 'power', s)
        out = np.zeros((n_per, M), int)
        for (i, j), (k, c, _) in K.items():
            win = (Phi - c) @ k > 0
            out[:, j] += win
            out[:, i] += ~win
        gaps.append((~(out == M - 1).any(1)).mean())
    return {'s': s, 'gap': float(np.mean(gaps)),
            'cond_max': float(max(v[2] for v in K.values()))}


if __name__ == '__main__':
    t0 = time.time()
    scen = {
        'M3_fixed': Scenario([0.0, 1.5, 3.0], [0.2, 1.0, 1.8]),
        'M5_worst': Scenario([0.0, 0.04, 1.28, 1.39, 1.45], [0.05, 1.83, 1.36, 0.39, 1.39]),
    }
    n_per = {'M3_fixed': 300_000, 'M5_worst': 200_000}
    out = {}
    for name, sc in scen.items():
        print(f'\n===== {name}: levels={sc.levels} skews={sc.skews} =====')
        # J* across pairs
        jstar = {}
        for i in range(sc.M):
            for j in range(i + 1, sc.M):
                js, dl = j_star(sc.levels[i], sc.skews[i], sc.levels[j], sc.skews[j])
                jstar[f'{i}{j}'] = {'J_star': js, 'delta_LeCam': dl}
        print('J* (full L2):', {k: round(v['J_star'], 4) for k, v in jstar.items()})
        rows = []
        for s in range(1, S_MAX + 1):
            rng = np.random.default_rng(SEED + s)
            r = evaluate_exact(sc, s, n_per[name], rng)
            for k in r['pairs']:
                r['pairs'][k]['J_ratio'] = r['pairs'][k]['J_s'] / jstar[k]['J_star']
                Js, dl = r['pairs'][k]['J_s'], jstar[k]['delta_LeCam']
                r['pairs'][k]['rho_L2_exact'] = float(np.sqrt(Js / ((1 + Js / 2) * dl)))
            if sc.M == 3:
                pr = r['pairs']
                r['cycle_logtau_sum'] = pr['01']['eff_log_threshold'] + pr['12']['eff_log_threshold'] - pr['02']['eff_log_threshold']
            rows.append(r)
            pr = r['pairs']
            print(f"s={s}: gap={r['gap']:.5f} cycle={r['p_cycle']:.5f} "
                  f"errCop={r['err_copeland']:.5f} errBayes={r['err_bayes']:.5f} "
                  f"cond={r['cond_max']:.1f} whitenchk={r['whiten_check_maxdev']:.2e} "
                  f"bound(sum mism)={r['sum_pair_mismatch_bound']:.5f}  [{time.time()-t0:.0f}s]")
            print('   pair  spearman  pearson_tanh  rho_L2exact  J_s/J*  mismatch  t*_eff  disagree@t*  projcos')
            for k, v in pr.items():
                print(f"   {k:>4}  {v['spearman']:.4f}    {v['pearson_tanh']:.4f}     {v['rho_L2_exact']:.4f}   "
                      f"{v['J_ratio']:.4f}  {v['p_sign_mismatch_mix']:.5f}  "
                      f"{v['eff_log_threshold']:+.4f}  {v['disagree_at_eff']:.5f}  {v['proj_cosine']:.4f}")
            if sc.M == 3:
                print(f"   cycle sum of effective log-thresholds t01+t12-t02 = {r['cycle_logtau_sum']:+.4f}")
        # sample-based version
        samp = []
        for s in range(1, S_MAX + 1):
            g = [evaluate_sample(sc, s, 200_000, 100_000, SEED + 100 * s + i) for i in range(2)]
            samp.append({'s': s, 'gap_mean': float(np.mean([x['gap'] for x in g])),
                         'gap_min': float(min(x['gap'] for x in g)),
                         'gap_max': float(max(x['gap'] for x in g)),
                         'cond_max': float(max(x['cond_max'] for x in g))})
            print(f"  [sample-based] s={s}: gap={samp[-1]['gap_mean']:.5f} "
                  f"[{samp[-1]['gap_min']:.5f}..{samp[-1]['gap_max']:.5f}] cond={samp[-1]['cond_max']:.1f}"
                  f"  [{time.time()-t0:.0f}s]")
        out[name] = {'levels': sc.levels, 'skews': sc.skews, 'J_star': jstar,
                     'exact_moments': rows, 'sample_moments': samp}
    json.dump(out, open('verification/results_v6_large_s.json', 'w'), indent=1)
    print(f'\nSaved verification/results_v6_large_s.json  ({time.time()-t0:.0f}s)')
