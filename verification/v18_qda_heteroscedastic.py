#!/usr/bin/env python3
"""V18: heteroscedastic QDA in the observation space against the completions.

Why (SPL-48917-2026, second round, Reviewer 2). The first-round answer called the comparison
of the soft completion with heteroscedastic QDA "degenerate by design", and the reviewer was
not convinced. In every panel of the Letter the noise laws have unit variance, so the
per-class Gaussian model of x (mean a_m, variance 1) gives the nearest-mean rule, which is the
linear rule of Table I. That is a property of the design, not of QDA. This script removes it:
the same 120 configurations of Panel B (rng 2026: M, levels, skewness) receive per-class noise
scales sigma_m from an independent stream, so the class variances differ and QDA becomes a
genuine competitor.

Model: x = a_m * L/2.5 + sigma_m * xi_m, xi_m zero-mean and unit-variance.
    gauss  xi_m ~ N(0, 1): QDA is the Bayes rule here (the control for the comparison)
    gamma  xi_m standardized gamma with skewness u_m (the Letter's family)
Scales sigma_m = r ** U_m with U_m ~ U[-1, 1], drawn once per configuration and shared by
    unit  r = 1    control: QDA must reproduce the linear rule decision for decision
    het   r = 1.5  standard deviations in [2/3, 3/2]
    wide  r = 2    standard deviations in [1/2, 2]
L in {2.5, 10}: the Letter's hard regime and a moderate one.

Rules, all from exact moments (quadrature of the densities; no sample enters a rule):
    lin         s = 1, nearest mean (Proposition 3(b)); blind to the variances
    qda         Gaussian ML on x with per-class E_m x and Var_m x (moments of order <= 2)
    pooled, copeland, margin_sum, coupling   power basis, s = 2 and s = 3 (as in v13)
    bayes       true densities
Every rule is evaluated on one test sample per configuration (1.2e5 draws per hypothesis, as
in Panel B), so all comparisons are paired.

Run:  python3 verification/v18_qda_heteroscedastic.py [n_cfg [out.json]]
Out:  verification/results_v18_qda_heteroscedastic.json, written only by the full
      120-configuration run; a smoke test with n_cfg < 120 must name its own output file, so
      it cannot overwrite the archived result.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import integrate, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
_argv, sys.argv = sys.argv, sys.argv[:1]          # v8_table1 reads sys.argv[1] on import
import v14_why_no_gain as v14                      # noqa: E402
sys.argv = _argv
v13 = v14.v13

SEED = 20261028
N_EV = 120_000
LS = (2.5, 10.0)
RATIOS = {"unit": 1.0, "het": 1.5, "wide": 2.0}
FAMILIES = ("gauss", "gamma")
SS = (2, 3)
RULES = ("pooled", "copeland", "margin_sum", "coupling")
OUT = HERE / "results_v18_qda_heteroscedastic.json"


def power(s):
    return lambda x: np.stack([x**i for i in range(1, s + 1)], -1)


class Law:
    """x = a + sigma * xi, E xi = 0, Var xi = 1; xi Gaussian or standardized gamma."""

    def __init__(self, family, a, u, sigma):
        self.family, self.a, self.u, self.sigma = family, float(a), float(u), float(sigma)
        self.k = 4.0 / self.u**2 if family == "gamma" else None

    def sample(self, n, rng):
        if self.family == "gauss":
            xi = rng.standard_normal(n)
        else:
            xi = (rng.gamma(self.k, 1.0, n) - self.k) / np.sqrt(self.k)
        return self.a + self.sigma * xi

    def logpdf(self, x):
        y = (np.asarray(x, float) - self.a) / self.sigma
        if self.family == "gauss":
            return stats.norm.logpdf(y) - np.log(self.sigma)
        z = y * np.sqrt(self.k) + self.k
        out = np.full_like(y, -np.inf)
        ok = z > 0
        out[ok] = stats.gamma.logpdf(z[ok], self.k) + 0.5 * np.log(self.k) - np.log(self.sigma)
        return out

    def pieces(self):
        w = self.sigma
        if self.family == "gauss":
            lo, hi = self.a - 14 * w, self.a + 14 * w
        else:
            lo, hi = max(self.a - w * 2.0 / self.u, self.a - 14 * w), self.a + 60 * w
        cuts = sorted({lo, hi, *[c for c in (self.a, self.a - 6 * w, self.a + 6 * w) if lo < c < hi]})
        return list(zip(cuts[:-1], cuts[1:]))

    def exact_moments(self, s):
        """E phi, Cov phi of the power basis by quadrature. Checks: mass 1, E x = a, Var x = sigma^2."""
        basis, iu = power(s), np.triu_indices(s)

        def f(x):
            xs = np.atleast_1d(x)
            ph = basis(xs)[0]
            return np.concatenate([[1.0], ph, np.outer(ph, ph)[iu]]) * np.exp(self.logpdf(xs))[0]

        tot = sum(integrate.quad_vec(f, lo, hi, epsabs=1e-11, epsrel=1e-10, limit=400)[0]
                  for lo, hi in self.pieces())
        mu = tot[1:1 + s]
        E2 = np.zeros((s, s))
        E2[iu] = tot[1 + s:]
        E2 = E2 + E2.T - np.diag(np.diag(E2))
        C = E2 - np.outer(mu, mu)
        assert abs(tot[0] - 1) < 1e-7 and abs(mu[0] - self.a) < 1e-6 and \
            abs(C[0, 0] - self.sigma**2) < 1e-5 * max(1.0, self.sigma**2), \
            (self.family, self.a, self.u, self.sigma, tot[0], mu[0], C[0, 0])
        return mu, C


def run_config(family, L, levels, us, sigmas, seed):
    M = len(levels)
    laws = [Law(family, a * L / 2.5, u, sg) for a, u, sg in zip(levels, us, sigmas)]
    rng = np.random.default_rng(seed)
    xs = [lw.sample(N_EV, rng) for lw in laws]
    a = np.array([lw.a for lw in laws])
    rules = {}
    for s in SS:
        ex = [lw.exact_moments(s) for lw in laws]
        mu_bar = np.mean([m for m, _ in ex], 0)
        S_tot = np.mean([C + np.outer(m, m) for m, C in ex], 0) - np.outer(mu_bar, mu_bar)
        w, V = np.linalg.eigh(S_tot)
        W = V @ np.diag(w**-0.5) @ V.T                 # whitening: conditioning only
        mus = np.array([(m - mu_bar) @ W for m, _ in ex])
        Cs = np.array([W @ C @ W for _, C in ex])
        rules[s] = (mu_bar, W, *v13.rules_from_moments(mus, Cs))
        if s == SS[0]:                                 # QDA on x: order-2 moments of the same laws
            m1 = np.array([m[0] for m, _ in ex])
            v1 = np.array([C[0, 0] for _, C in ex])
    errs = {k: [] for k in ("bayes", "lin", "qda", *[f"{r}/s{s}" for s in SS for r in RULES],
                            *[f"fail/s{s}" for s in SS])}
    qda_lin_disagree = 0
    for m, x in enumerate(xs):
        lp = np.stack([lw.logpdf(x) for lw in laws], 1)
        lin = np.abs(x[:, None] - a).argmin(1)
        qda = (-0.5 * np.log(v1) - 0.5 * (x[:, None] - m1)**2 / v1).argmax(1)
        qda_lin_disagree += int((qda != lin).sum())
        for name, d in (("bayes", lp.argmax(1)), ("lin", lin), ("qda", qda)):
            errs[name].append(float((d != m).mean()))
        for s, (mu_bar, W, par, A, b) in rules.items():
            dec, fail, _, _ = v13.decide((power(s)(x) - mu_bar) @ W, par, A, b, M)
            for k, d in dec.items():
                errs[f"{k}/s{s}"].append(float((d != m).mean()))
            errs[f"fail/s{s}"].append(float(fail.mean()))
    out = {k: float(np.mean(v)) for k, v in errs.items()}
    out.update(M=M, sigmas=[float(v) for v in sigmas],
               qda_lin_disagree=qda_lin_disagree / (M * N_EV),
               J_min_s3=float(min(p[2] for p in rules[3][2].values())))
    return out


def task(args):
    family, rname, L, i, lv, us, U = args
    try:
        return (family, rname, L, i), run_config(family, L, lv, us, RATIOS[rname] ** U, SEED + i)
    except (np.linalg.LinAlgError, AssertionError) as e:
        return (family, rname, L, i), {"error": f"{type(e).__name__} {e}"}


def summarize(rows):
    g = lambda k: np.array([r[k] for r in rows])                       # noqa: E731
    med = lambda v: float(np.median(v))                                 # noqa: E731
    lin, bay, qda = g("lin"), g("bayes"), g("qda")
    gap = lin - bay
    ok = gap > 1e-3                         # the gap closed is defined only where there is one
    closed = lambda e: med(((lin - e) / gap)[ok]) if ok.sum() >= 10 else None   # noqa: E731
    out = {"n": len(rows), "n_with_gap": int(ok.sum()), "bayes": med(bay), "lin": med(lin),
           "qda": {"err": med(qda), "gain_pp": med(lin - qda) * 100, "closed": closed(qda),
                   "beats_lin": int((qda < lin).sum()), "above_bayes_pp": med(qda - bay) * 100},
           "qda_lin_disagree_max": float(g("qda_lin_disagree").max())}
    for s in SS:
        for k in RULES:
            e = g(f"{k}/s{s}")
            out[f"{k}/s{s}"] = {"err": med(e), "gain_pp": med(lin - e) * 100, "closed": closed(e),
                                "vs_qda_pp": med(qda - e) * 100, "beats_qda": int((e < qda).sum()),
                                "above_bayes_pp": med(e - bay) * 100}
        f = g(f"fail/s{s}")
        out[f"fail/s{s}"] = {"med": med(f), "max": float(f.max()), "gt1pct": int((f > 0.01).sum())}
    return out


def main():
    n_cfg = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    if n_cfg < 120 and len(sys.argv) < 3:
        sys.exit("a smoke test (n_cfg < 120) must name its own output file")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    cfgs = v14.configs(n_cfg)
    rng_u = np.random.default_rng(SEED)            # independent of the configuration stream
    Us = [rng_u.uniform(-1.0, 1.0, len(lv)) for lv, _ in v14.configs(120)][:n_cfg]
    tasks = [(f, r, L, i, lv, us, Us[i]) for f in FAMILIES for r in RATIOS for L in LS
             for i, (lv, us) in enumerate(cfgs)]
    t0 = time.time()
    with Pool(8) as pool:
        done = pool.map(task, tasks, chunksize=4)
    print(f"{len(done)} tasks in {time.time() - t0:.0f}s", flush=True)
    res = {"seed": SEED, "n_ev": N_EV, "ratios": RATIOS, "L": LS, "s": SS, "cells": {}, "rows": {}}
    for f in FAMILIES:
        for r in RATIOS:
            for L in LS:
                key = f"{f}/{r}/L{L:g}"
                rows = [dict(v, idx=k[3]) for k, v in done if k[:3] == (f, r, L)]
                bad = [x for x in rows if "error" in x]
                for x in bad:
                    print(f"  skip {key} cfg={x['idx']}: {x['error']}", flush=True)
                rows = [x for x in rows if "error" not in x]
                res["rows"][key] = rows
                c = res["cells"][key] = summarize(rows)
                print(f"{key:<16} n={c['n']} Bayes={c['bayes']:.3f} lin={c['lin']:.3f} "
                      f"qda={c['qda']['err']:.3f} ({c['qda']['closed']}) | s3: "
                      f"cop={c['copeland/s3']['err']:.3f} ({c['copeland/s3']['closed']}) "
                      f"coup={c['coupling/s3']['err']:.3f} ({c['coupling/s3']['closed']}) "
                      f"coup-vs-qda {c['coupling/s3']['vs_qda_pp']:+.2f}pp "
                      f"{c['coupling/s3']['beats_qda']}/{c['n']} | s2: "
                      f"coup={c['coupling/s2']['err']:.3f} vs-qda {c['coupling/s2']['vs_qda_pp']:+.2f}pp "
                      f"{c['coupling/s2']['beats_qda']}/{c['n']} | qda!=lin max "
                      f"{c['qda_lin_disagree_max']:.2e}", flush=True)
    out_path.write_text(json.dumps(res, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
