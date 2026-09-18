#!/usr/bin/env python3
"""V13: the SPL letter on genuinely heavy-tailed noise, in a moderate regime, with exact moments.

Why (from the SPL-48519-2026 decision of 16.09.2026):
  R1.1, R4.4  gamma noise is light-tailed, yet Sec. II-A motivates fractional powers by
              heavy tails -- a motivation never actually tested;
  R4.3        Table I was measured only at a Bayes error of about 0.57;
  R1.8        exact-moment and estimated-moment results were not separated;
  R1.9        whether overruling the Condorcet winner by the soft rule improves accuracy;
  R1.3, R2.6, R4.7  error of the Gaussian calibration r_mn against the EXACT pairwise posterior;
  R4.7        the margin sum as a stand-alone completion.

Design. The same 120 configurations as v8_table1.py (rng 2026: M, levels, u ~ U[0,1.9]);
the shape parameter u maps into each family:
    gamma  skewness u                        (control: must reproduce Table I)
    cg     eps = 0.1 u/1.9, kappa = 10       eps-contaminated Gaussian mixture, variance 1
    t      nu = 15 - 8 u/1.9  (7 < nu <= 15) Student law, variance 1
    gcg    gamma with skewness u (the letter's core) + 5% impulsive contamination
           N(0, kappa^2), variance 1 -- does the letter's gain survive heavy tails
Levels are scaled by L/2.5, L in {2.5, 6, 10}: from the letter's extreme regime to a moderate one.
Bases (s = 3):  power {x, x^2, x^3};  frac sgn(x)|x|^p, p = (1, 1.25, 1.5).
Moments:  exact  -- quadrature of the densities (no sample anywhere in the rules);
          est1e5 -- 10^5 observations per hypothesis (as in the letter);
          est500 -- 500 observations per hypothesis.
Every moment mode and both bases are scored on ONE test sample, so comparisons are paired.
Whitening uses the exact pooled covariance; the rules are invariant to it, it only conditions.

Heteroscedastic QDA in the observation space (R3.6) needs NO separate row here: all laws are
standardized to unit variance, so it coincides identically with the linear rule.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import integrate, special, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
_argv, sys.argv = sys.argv, sys.argv[:1]          # v8 reads sys.argv[1] at import time
from v8_table1 import gauss_logpdf, wlw_coupling   # noqa: E402
sys.argv = _argv

SEED = 20260917
KAPPA = 10.0
EPS0 = 0.05
N_EV = 40_000
OUT = HERE / "results_v13_heavy_tails.json"

BASES = {
    "power": lambda x: np.stack([x, x**2, x**3], -1),
    "frac": lambda x: np.stack([np.sign(x) * np.abs(x)**p for p in (1.0, 1.25, 1.5)], -1),
}


# ------------------------------------------------------------------ noise families
class Law:
    """x = a + xi, E xi = 0, Var xi = 1."""

    def __init__(self, family, a, u):
        self.family, self.a, self.u = family, float(a), float(u)
        if family == "gamma":
            self.skew = self.u
            self.k = 4.0 / self.skew**2
        elif family == "cg":
            self.eps = 0.1 * self.u / 1.9
            self.s0 = 1.0 / np.sqrt(1 - self.eps + self.eps * KAPPA**2)
        elif family == "t":
            self.nu = 15.0 - 8.0 * self.u / 1.9
            self.c = np.sqrt((self.nu - 2.0) / self.nu)
        elif family == "gcg":
            # the letter's skewed core + impulsive contamination: with prob. 1-EPS0 a standardized
            # gamma (skewness u), with prob. EPS0 an N(0, kappa^2); all rescaled to variance 1
            self.skew = self.u
            self.k = 4.0 / self.skew**2
            self.sc = 1.0 / np.sqrt(1 - EPS0 + EPS0 * KAPPA**2)

    def sample(self, n, rng):
        if self.family == "gamma":
            return self.a + (rng.gamma(self.k, 1.0, n) - self.k) / np.sqrt(self.k)
        if self.family == "cg":
            sc = np.where(rng.random(n) < self.eps, KAPPA * self.s0, self.s0)
            return self.a + sc * rng.standard_normal(n)
        if self.family == "gcg":
            core = (rng.gamma(self.k, 1.0, n) - self.k) / np.sqrt(self.k)
            xi = np.where(rng.random(n) < EPS0, KAPPA * rng.standard_normal(n), core)
            return self.a + self.sc * xi
        return self.a + self.c * rng.standard_t(self.nu, n)

    def logpdf(self, x):
        if self.family == "gamma":
            z = (x - self.a) * np.sqrt(self.k) + self.k
            out = np.full_like(x, -np.inf, dtype=float)
            ok = z > 0
            out[ok] = stats.gamma.logpdf(z[ok], self.k) + 0.5 * np.log(self.k)
            return out
        if self.family == "cg":
            l0 = np.log1p(-self.eps) + stats.norm.logpdf(x, self.a, self.s0)
            if self.eps == 0:
                return l0
            return np.logaddexp(l0, np.log(self.eps) + stats.norm.logpdf(x, self.a, KAPPA * self.s0))
        if self.family == "gcg":
            y = (x - self.a) / self.sc
            z = y * np.sqrt(self.k) + self.k
            lg = np.full_like(y, -np.inf, dtype=float)
            ok = z > 0
            lg[ok] = stats.gamma.logpdf(z[ok], self.k) + 0.5 * np.log(self.k)
            return np.logaddexp(np.log1p(-EPS0) + lg,
                                np.log(EPS0) + stats.norm.logpdf(y, 0.0, KAPPA)) - np.log(self.sc)
        return stats.t.logpdf((x - self.a) / self.c, self.nu) - np.log(self.c)

    def pieces(self):
        """[(lo, hi)] -- integration ranges; breakpoints at 0 (|x|^p is not smooth) and at a."""
        if self.family == "gamma":
            lo, hi = max(self.a - 2.0 / self.skew, self.a - 14.0), self.a + 60.0
        elif self.family == "cg":
            lo, hi = self.a - 14 * KAPPA * self.s0, self.a + 14 * KAPPA * self.s0
        elif self.family == "gcg":
            lo, hi = self.a - 14 * KAPPA * self.sc, self.a + 14 * KAPPA * self.sc
        else:
            lo, hi = -np.inf, np.inf
        w = getattr(self, "s0", getattr(self, "sc", 1.0))
        extra = [self.a - self.sc * 2.0 / self.skew] if self.family == "gcg" else []
        cuts = sorted({lo, hi, *[c for c in (0.0, self.a, self.a - 6 * w, self.a + 6 * w, *extra)
                                 if lo < c < hi]})
        return list(zip(cuts[:-1], cuts[1:]))

    def exact_moments(self, basis):
        """E phi, Cov phi by quadrature of the density. Checks: mass 1, E x = a, Var x = 1."""
        def f(x):
            xs = np.atleast_1d(x)
            ph = basis(xs)[0]
            w = np.exp(self.logpdf(xs))[0]
            return np.concatenate([[1.0, xs[0], xs[0]**2], ph, np.outer(ph, ph)[np.triu_indices(3)]]) * w
        tot = sum(integrate.quad_vec(f, lo, hi, epsabs=1e-11, epsrel=1e-10, limit=400)[0]
                  for lo, hi in self.pieces())
        assert abs(tot[0] - 1) < 1e-7 and abs(tot[1] - self.a) < 1e-6 and \
            abs(tot[2] - self.a**2 - 1) < 1e-5, (self.family, self.a, self.u, tot[:3])
        mu = tot[3:6]
        E2 = np.zeros((3, 3))
        E2[np.triu_indices(3)] = tot[6:]
        E2 = E2 + E2.T - np.diag(np.diag(E2))
        return mu, E2 - np.outer(mu, mu)


# ------------------------------------------------------------------ rules
def rules_from_moments(mus, Cs):
    M = len(mus)
    par = {}
    for m in range(M):
        for n in range(m + 1, M):
            F = Cs[m] + Cs[n]
            k = np.linalg.solve(F, mus[n] - mus[m])
            par[(m, n)] = (k, 0.5 * (mus[m] + mus[n]), float(k @ (mus[n] - mus[m])),
                           float(k @ Cs[m] @ k), float(k @ Cs[n] @ k), float(np.linalg.cond(F)))
    SW = Cs.mean(0)
    A = np.linalg.solve(SW, mus.T).T
    return par, A, -0.5 * np.einsum("ms,ms->m", A, mus)


def decide(Phi, par, A, b, M):
    n_ = Phi.shape[0]
    wins = np.zeros((n_, M), int)
    margin = np.zeros((n_, M))
    R = np.full((n_, M, M), 0.5)
    for (i, j), (k, c, J, vi, vj, _) in par.items():
        lam = (Phi - c) @ k
        win_j = lam > 0
        wins[:, j] += win_j
        wins[:, i] += ~win_j
        std = lam / np.sqrt(max(J, 1e-300))
        margin[:, j] += std
        margin[:, i] -= std
        li = gauss_logpdf(lam, -0.5 * J, max(vi, 1e-300))
        lj = gauss_logpdf(lam, +0.5 * J, max(vj, 1e-300))
        rj = 1.0 / (1.0 + np.exp(np.clip(li - lj, -700, 700)))
        R[:, j, i] = rj
        R[:, i, j] = 1.0 - rj
    tied = wins == wins.max(1, keepdims=True)
    fail = ~(wins == M - 1).any(1)
    return {"pooled": (Phi @ A.T + b).argmax(1),
            "copeland": np.where(tied, margin, -np.inf).argmax(1),
            "margin_sum": margin.argmax(1),
            "coupling": wlw_coupling(R, M)}, fail, wins.argmax(1), R


def run_config(family, L, levels, us, seed):
    M = len(levels)
    laws = [Law(family, a * L / 2.5, u) for a, u in zip(levels, us)]
    rng = np.random.default_rng(seed)
    xs = [law.sample(N_EV, rng) for law in laws]
    logp = [np.stack([lw.logpdf(x) for lw in laws], 1) for x in xs]
    tr1 = [law.sample(100_000, rng) for law in laws]
    tr2 = [law.sample(500, rng) for law in laws]
    a = np.array([lw.a for lw in laws])
    out = {"bayes": float(np.mean([(lp.argmax(1) != m).mean() for m, lp in enumerate(logp)])),
           "lin": float(np.mean([(np.abs(x[:, None] - a).argmin(1) != m).mean()
                                 for m, x in enumerate(xs)]))}
    for bname, basis in BASES.items():
        ex = [law.exact_moments(basis) for law in laws]
        mu_bar = np.mean([m for m, _ in ex], 0)
        S_tot = np.mean([C + np.outer(m, m) for m, C in ex], 0) - np.outer(mu_bar, mu_bar)
        w, V = np.linalg.eigh(S_tot)
        W = V @ np.diag(w**-0.5) @ V.T
        tr = lambda x: (basis(x) - mu_bar) @ W                               # noqa: E731
        Phis = [tr(x) for x in xs]
        moms = {"exact": (np.array([(m - mu_bar) @ W for m, _ in ex]),
                          np.array([W @ C @ W for _, C in ex]))}
        for tag, trs in (("est1e5", tr1), ("est500", tr2)):
            Q = [tr(t) for t in trs]
            moms[tag] = (np.array([q.mean(0) for q in Q]),
                         np.array([np.cov(q, rowvar=False, ddof=1) for q in Q]))
        for mode, (mus, Cs) in moms.items():
            par, A, b = rules_from_moments(mus, Cs)
            acc = {k: [] for k in ("pooled", "copeland", "margin_sum", "coupling")}
            fails, ovr_n, ovr_coup_ok, ovr_cw_ok, dec_n, cal = [], 0, 0, 0, 0, []
            for m in range(M):
                dec, fail, cw, R = decide(Phis[m], par, A, b, M)
                for k, d in dec.items():
                    acc[k].append(float((d != m).mean()))
                fails.append(float(fail.mean()))
                o = (~fail) & (dec["coupling"] != cw)
                ovr_n += int(o.sum())
                dec_n += int((~fail).sum())
                ovr_coup_ok += int((dec["coupling"][o] == m).sum())
                ovr_cw_ok += int((cw[o] == m).sum())
                for j in range(M):                       # calibration of r against the exact posterior
                    if j != m:
                        r_exact = 1.0 / (1.0 + np.exp(np.clip(logp[m][:, j] - logp[m][:, m], -700, 700)))
                        cal.append(float(np.abs(R[:, m, j] - r_exact).mean()))
            cell = {k: float(np.mean(v)) for k, v in acc.items()}
            cell.update(fail=float(np.mean(fails)), override_share=ovr_n / max(dec_n, 1),
                        override_coupling_acc=ovr_coup_ok / max(ovr_n, 1),
                        override_cw_acc=ovr_cw_ok / max(ovr_n, 1), override_n=ovr_n,
                        calib_mae=float(np.mean(cal)),
                        cond_F_max=float(max(p[5] for p in par.values())),
                        J_min=float(min(p[2] for p in par.values())))
            out[f"{bname}/{mode}"] = cell
    return out


def summarize(rows, key):
    g = lambda f: np.array([f(r) for r in rows])                           # noqa: E731
    lin, bay = g(lambda r: r["lin"]), g(lambda r: r["bayes"])
    c = lambda k: g(lambda r: r[key][k])                                   # noqa: E731
    gap = lin - bay
    ok = gap > 1e-3
    # the closed fraction is meaningful only where a lin -> Bayes gap exists
    closed = lambda e: float(np.median(((lin - e) / gap)[ok])) if ok.sum() >= 10 else None  # noqa: E731
    ov_n = c("override_n").sum()
    return {"n": len(rows), "n_with_gap": int(ok.sum()),
            "bayes_med": float(np.median(bay)), "lin_med": float(np.median(lin)),
            **{f"{k}_med": float(np.median(c(k))) for k in ("pooled", "copeland", "margin_sum", "coupling")},
            **{f"{k}_gap_closed": closed(c(k)) for k in ("pooled", "copeland", "margin_sum", "coupling")},
            "coupling_beats_copeland": int((c("coupling") < c("copeland")).sum()),
            "margin_sum_beats_copeland": int((c("margin_sum") < c("copeland")).sum()),
            "fail_med": float(np.median(c("fail"))), "fail_max": float(c("fail").max()),
            "fail_gt_1pct": int((c("fail") > 0.01).sum()),
            "override_share_med": float(np.median(c("override_share"))),
            "override_acc_coupling_pooled": float((c("override_coupling_acc") * c("override_n")).sum() / max(ov_n, 1)),
            "override_acc_cw_pooled": float((c("override_cw_acc") * c("override_n")).sum() / max(ov_n, 1)),
            "calib_mae_med": float(np.median(c("calib_mae"))),
            "cond_F_max_med": float(np.median(c("cond_F_max"))),
            "cond_F_max_max": float(c("cond_F_max").max())}


def main():
    n_cfg = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    rng = np.random.default_rng(2026)                  # the same stream as v8_table1.py
    cfgs = []
    for _ in range(120):
        M = int(rng.integers(3, 6))
        lv = np.sort(rng.uniform(0, 2.5, M))
        cfgs.append((lv - lv[0], rng.uniform(0.0, 1.9, M)))
    cfgs = cfgs[:n_cfg]
    fams = sys.argv[2].split(",") if len(sys.argv) > 2 else ["gamma", "cg", "t", "gcg"]
    res = json.loads(OUT.read_text()) if OUT.exists() and len(sys.argv) > 2 else \
        {"n_ev": N_EV, "kappa": KAPPA, "eps0": EPS0, "seed": SEED, "cells": {}, "rows": {}}
    res["eps0"] = EPS0
    t0 = time.time()
    for family in fams:
        for L in (2.5, 6.0, 10.0):
            rows = []
            for i, (lv, us) in enumerate(cfgs):
                try:
                    rows.append(run_config(family, L, lv, us, SEED + i))
                except (np.linalg.LinAlgError, AssertionError) as e:
                    print(f"  skip {family} L={L} cfg={i}: {type(e).__name__} {e}", flush=True)
            res["rows"][f"{family}/L{L}"] = rows
            for bname in BASES:
                for mode in ("exact", "est1e5", "est500"):
                    res["cells"][f"{family}/L{L}/{bname}/{mode}"] = summarize(rows, f"{bname}/{mode}")
            s = res["cells"][f"{family}/L{L}/power/exact"]
            print(f"[{time.time()-t0:6.0f}s] {family:>5} L={L:<4} n={s['n']} bayes={s['bayes_med']:.3f} "
                  f"lin={s['lin_med']:.3f} | power/exact: cop={s['copeland_med']:.3f} "
                  f"coup={s['coupling_med']:.3f} fail>1%={s['fail_gt_1pct']} max={s['fail_max']:.3f}",
                  flush=True)
            OUT.write_text(json.dumps(res, indent=1))
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    main()
