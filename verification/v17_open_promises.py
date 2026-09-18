#!/usr/bin/env python3
"""V17: the three measurements the response promised "on request".

  K  CALIBRATION vs RESIDUAL SKEWNESS (R3.3).  The response says the Gaussian
     calibration displaces the pairwise posteriors appreciably but rarely their
     argmax, without isolating the residual skewness as a covariate. Here, per
     PAIR: gamma1 = standardized third moment of Lambda_mn under H_m, against
     the calibration error |r_mn - p_m/(p_m+p_n)| on the same draws. Reported as
     a Spearman correlation over pairs plus the error in bins of |gamma1|, so the
     claim "the skewness is what the calibration misses" is either supported or not.

  M  LARGER M (R1.7).  The letter measures M = 3..6. Here M in {6, 8, 10},
     30 configurations each, s = 2..4, exact moments, power basis. Two separate
     quantities are reported and must not be confused:
        the measured undecided probability, and
        M / 2^(M-1), the fraction of SIGN PATTERNS with a Condorcet winner,
     which carries no probability model and is not a prediction of the first.

  P  VIOLATED PRIORS (R3.7).  The response answers analytically: r_mn approximates
     the EQUAL-PRIOR pairwise posterior, so the coupling returns the equal-prior
     posterior and its argmax is the ML rule; replacing z_mn by z_mn + log(pi_n/pi_m)
     restores MAP. That is now measured. Priors are geometric, pi_m ~ ratio^m,
     ratio in {1, 2, 4}; the error is the prior-weighted class error, which is what
     an unequal-prior problem actually costs. Each rule appears twice, uncorrected
     and corrected, with the prior-aware Bayes oracle as the floor.

Run:     python3 verification/v17_open_promises.py {K|M|P} [n_cfg]
Output:  verification/results_v17_{K,M,P}.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_argv, sys.argv = sys.argv, sys.argv[:1]
import v14_why_no_gain as v14                                          # noqa: E402
sys.argv = _argv
v13 = v14.v13
Law = v14.Law
gauss_logpdf = v13.gauss_logpdf
wlw_coupling = v13.wlw_coupling

N_EV = 40_000
SEED = 20260922
POWER = v13.BASES["power"]


def _whiten(ex):
    """Pooled-total whitening from exact moments; the rules are invariant to it."""
    mu_bar = np.mean([m for m, _ in ex], 0)
    S = np.mean([C + np.outer(m, m) for m, C in ex], 0) - np.outer(mu_bar, mu_bar)
    w, V = np.linalg.eigh(S)
    W = V @ np.diag(np.maximum(w, 1e-300) ** -0.5) @ V.T
    return mu_bar, W


def _setup(laws, basis=POWER):
    ex = [lw.exact_moments(basis) for lw in laws]
    mu_bar, W = _whiten(ex)
    mus = np.array([(m - mu_bar) @ W for m, _ in ex])
    Cs = np.array([W @ C @ W for _, C in ex])
    return v13.rules_from_moments(mus, Cs), (lambda x: (basis(x) - mu_bar) @ W)


# ------------------------------------------------------------------ K: calibration
def part_K(cfgs):
    """Per-pair: skewness of Lambda under H_m against the calibration error."""
    rows, t0 = [], time.time()
    for i, (lv, us) in enumerate(cfgs):
        laws = [Law("gamma", a, u) for a, u in zip(lv, us)]
        M = len(laws)
        (par, A, b), tr = _setup(laws)
        rng = np.random.default_rng(SEED + i)
        for m in range(M):
            x = laws[m].sample(N_EV, rng)
            Phi = tr(x)
            logp = np.stack([lw.logpdf(x) for lw in laws], 1)
            for (p, q), (k, c, J, vp, vq, _) in par.items():
                if m not in (p, q):
                    continue
                lam = (Phi - c) @ k
                sd = lam.std()
                if sd < 1e-12:
                    continue
                g1 = float(((lam - lam.mean()) ** 3).mean() / sd ** 3)
                lp = gauss_logpdf(lam, -0.5 * J, max(vp, 1e-300))
                lq = gauss_logpdf(lam, +0.5 * J, max(vq, 1e-300))
                r_q = 1.0 / (1.0 + np.exp(np.clip(lp - lq, -700, 700)))   # P(H_q | pair)
                other = q if m == p else p
                r_m = r_q if m == q else 1.0 - r_q
                r_ex = 1.0 / (1.0 + np.exp(np.clip(logp[:, other] - logp[:, m], -700, 700)))
                rows.append({"cfg": i, "m": m, "other": other, "J": J,
                             "skew_lam": g1, "calib_mae": float(np.abs(r_m - r_ex).mean()),
                             "calib_max": float(np.abs(r_m - r_ex).max())})
        if (i + 1) % 20 == 0:
            print(f"[{time.time()-t0:5.0f}s] K {i+1}/{len(cfgs)} cfgs, {len(rows)} pair-rows", flush=True)
    a = np.array([r["skew_lam"] for r in rows])
    e = np.array([r["calib_mae"] for r in rows])
    rk = lambda v: np.argsort(np.argsort(v))                              # noqa: E731
    sp = float(np.corrcoef(rk(np.abs(a)), rk(e))[0, 1])
    qs = np.quantile(np.abs(a), [0, .25, .5, .75, 1.0])
    bins = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        s = (np.abs(a) >= lo) & (np.abs(a) <= hi)
        bins.append({"abs_skew_range": [float(lo), float(hi)], "n": int(s.sum()),
                     "calib_mae_median": float(np.median(e[s]))})
    out = {"rows": rows, "n_pairs": len(rows),
           "spearman_abs_skew_vs_calib": sp,
           "calib_mae_overall": float(e.mean()),
           "abs_skew_median": float(np.median(np.abs(a))), "bins": bins}
    print(f"  pairs={len(rows)}  Spearman(|skew|, calib)={sp:+.3f}  "
          f"MAE={e.mean():.4f}  bins={[round(b['calib_mae_median'], 4) for b in bins]}", flush=True)
    (HERE / "results_v17_K.json").write_text(json.dumps(out))


# ------------------------------------------------------------------ M: larger M
def part_M(n_per_M=30):
    """Undecided frequency at M = 6, 8, 10. Raw moments (v14) so any degree works;
    z = (x - c)/h standardization, which matters for conditioning at large M."""
    res, t0 = {}, time.time()
    S_MAX = 4
    for M in (6, 8, 10):
        rng = np.random.default_rng(202600 + M)
        cfgs = []
        for _ in range(n_per_M):
            lv = np.sort(rng.uniform(0, 2.5, M)); lv[0] = 0.0
            cfgs.append((lv, rng.uniform(0.0, 1.9, M)))
        fails = {s: [] for s in range(2, S_MAX + 1)}
        for i, (lv, us) in enumerate(cfgs):
            laws = [Law("gamma", a, u) for a, u in zip(lv, us)]
            a = np.array([lw.a for lw in laws])
            c, h = float(a.mean()), float(np.sqrt(1.0 + a.var()))
            raw = [lw.raw_moments(2 * S_MAX, c, h) for lw in laws]
            r = np.random.default_rng(SEED + 1000 * M + i)
            xs = [lw.sample(N_EV, r) for lw in laws]
            for s in range(2, S_MAX + 1):
                idx = np.arange(1, s + 1)
                mus = np.array([q[idx] for q in raw])
                Cs = np.array([q[idx[:, None] + idx[None, :]] - np.outer(q[idx], q[idx]) for q in raw])
                mu_bar = mus.mean(0)
                S_tot = np.mean([C + np.outer(m, m) for m, C in zip(mus, Cs)], 0) - np.outer(mu_bar, mu_bar)
                w, V = np.linalg.eigh(S_tot)
                W = V @ np.diag(np.maximum(w, 1e-300) ** -0.5) @ V.T
                par, A, b = v13.rules_from_moments((mus - mu_bar) @ W,
                                                   np.array([W @ C @ W for C in Cs]))
                f = []
                for m in range(M):
                    z = (xs[m] - c) / h
                    Phi = (np.stack([z ** k for k in idx], 1) - mu_bar) @ W
                    f.append(float(v13.decide(Phi, par, A, b, M)[1].mean()))
                fails[s].append(float(np.mean(f)))
            if (i + 1) % 10 == 0:
                print(f"[{time.time()-t0:5.0f}s] M={M} {i+1}/{n_per_M}", flush=True)
        for s in range(2, S_MAX + 1):
            v = np.array(fails[s])
            res[f"M{M}/s{s}"] = {
                "n_cfg": n_per_M, "median": float(np.median(v)), "max": float(v.max()),
                "gt1pct": int((v > 0.01).sum()),
                "pattern_fraction_with_winner": M / 2.0 ** (M - 1)}
            q = res[f"M{M}/s{s}"]
            print(f"[{time.time()-t0:5.0f}s] M={M:<3d} s={s}  median={q['median']:.4f} "
                  f"max={q['max']:.4f} >1%={q['gt1pct']}/{n_per_M}  "
                  f"(M/2^(M-1) = {q['pattern_fraction_with_winner']:.4f})", flush=True)
        (HERE / "results_v17_M.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ P: priors
def decide_pri(Phi, par, A, b, M, logpi):
    """Rules with and without the prior correction. logpi = log prior, length M."""
    n_ = Phi.shape[0]
    wins = np.zeros((n_, M), int)
    wins_p = np.zeros((n_, M), int)
    marg = np.zeros((n_, M))
    marg_p = np.zeros((n_, M))
    R = np.full((n_, M, M), 0.5)
    Rp = np.full((n_, M, M), 0.5)
    for (i, j), (k, c, J, vi, vj, _) in par.items():
        lam = (Phi - c) @ k
        li = gauss_logpdf(lam, -0.5 * J, max(vi, 1e-300))
        lj = gauss_logpdf(lam, +0.5 * J, max(vj, 1e-300))
        d = logpi[j] - logpi[i]                      # the one-term correction
        for W_, Mg_, z in ((wins, marg, lj - li), (wins_p, marg_p, lj - li + d)):
            w_j = z > 0
            W_[:, j] += w_j
            W_[:, i] += ~w_j
            std = z / np.sqrt(max(J, 1e-300))
            Mg_[:, j] += std
            Mg_[:, i] -= std
        rj = 1.0 / (1.0 + np.exp(np.clip(li - lj, -700, 700)))
        rjp = 1.0 / (1.0 + np.exp(np.clip(li - lj - d, -700, 700)))
        R[:, j, i], R[:, i, j] = rj, 1.0 - rj
        Rp[:, j, i], Rp[:, i, j] = rjp, 1.0 - rjp
    tie = wins == wins.max(1, keepdims=True)
    tie_p = wins_p == wins_p.max(1, keepdims=True)
    sc = Phi @ A.T + b
    return {"pooled": sc.argmax(1), "pooled_pri": (sc + logpi).argmax(1),
            "copeland": np.where(tie, marg, -np.inf).argmax(1),
            "copeland_pri": np.where(tie_p, marg_p, -np.inf).argmax(1),
            "coupling": wlw_coupling(R, M), "coupling_pri": wlw_coupling(Rp, M)}


def part_P(cfgs, ratios=(1.0, 2.0, 4.0)):
    res, t0 = {}, time.time()
    names = ("pooled", "pooled_pri", "copeland", "copeland_pri", "coupling", "coupling_pri")
    for ratio in ratios:
        rows = []
        for i, (lv, us) in enumerate(cfgs):
            laws = [Law("gamma", a, u) for a, u in zip(lv, us)]
            M = len(laws)
            pi = ratio ** np.arange(M)
            pi = pi / pi.sum()
            logpi = np.log(pi)
            (par, A, b), tr = _setup(laws)
            rng = np.random.default_rng(SEED + 7919 * i)
            err = {k: 0.0 for k in names}
            err.update(bayes=0.0, bayes_pri=0.0, lin=0.0)
            a_lv = np.array([lw.a for lw in laws])
            for m in range(M):
                x = laws[m].sample(N_EV, rng)
                logp = np.stack([lw.logpdf(x) for lw in laws], 1)
                dec = decide_pri(tr(x), par, A, b, M, logpi)
                for k in names:                                  # prior-WEIGHTED error
                    err[k] += pi[m] * float((dec[k] != m).mean())
                err["bayes"] += pi[m] * float((logp.argmax(1) != m).mean())
                err["bayes_pri"] += pi[m] * float(((logp + logpi).argmax(1) != m).mean())
                err["lin"] += pi[m] * float((np.abs(x[:, None] - a_lv).argmin(1) != m).mean())
            rows.append(err)
        g = lambda k: np.array([r[k] for r in rows])             # noqa: E731
        s = {"n_cfg": len(rows), "bayes": float(np.median(g("bayes"))),
             "bayes_pri": float(np.median(g("bayes_pri"))), "lin": float(np.median(g("lin")))}
        for k in names:
            s[k] = {"err": float(np.median(g(k))),
                    "gain_vs_uncorrected_pp": None if k.endswith("_pri") is False else
                    float(np.median(g(k.replace("_pri", "")) - g(k)) * 100),
                    "corrected_better": None if not k.endswith("_pri") else
                    int((g(k) < g(k.replace("_pri", ""))).sum())}
        res[f"ratio{ratio:g}"] = {"rows": rows, **s}
        print(f"[{time.time()-t0:5.0f}s] ratio={ratio:<4g} bayes={s['bayes']:.4f} "
              f"bayes_pri={s['bayes_pri']:.4f} | "
              + " ".join(f"{k}={s[k]['err']:.4f}" for k in names), flush=True)
        for k in ("pooled_pri", "copeland_pri", "coupling_pri"):
            print(f"        {k}: {s[k]['gain_vs_uncorrected_pp']:+.2f} pp, "
                  f"better in {s[k]['corrected_better']}/{len(rows)}", flush=True)
        (HERE / "results_v17_P.json").write_text(json.dumps(res))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "K"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    if what == "M":
        part_M(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    else:
        {"K": part_K, "P": part_P}[what](v14.configs(n))
