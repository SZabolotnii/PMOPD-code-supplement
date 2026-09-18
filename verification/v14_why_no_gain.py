#!/usr/bin/env python3
"""V14: WHY the pairwise scheme gained nothing on heavy tails in V13 -- disentangling it.

V13 concluded that the letter's gain comes from skewness, not from heavy tails. That
conflates two things: every heavy-tailed family in V13 was either SYMMETRIC (cg, t) or
had its skewness diluted by contamination (gcg). And at s <= 3, with unit-variance
noise, only the skewness enters the feature means:  E x = a,  E x^2 = a^2 + 1,
E x^3 = a^3 + 3a + gamma,  and only E x^4 = ... + kappa_4 carries the kurtosis. So on a
symmetric family Y_mn at s = 3 holds NO shape information at all -- whatever the tails.

A 2 x 2 design (shape x tail), swept over the degree s = 2..5, power basis, EXACT moments:
                     light tails                    heavy tails
    skewed           gamma (the letter)             logn -- lognormal, SAME skewness u
    symmetric        gg -- generalized Gaussian     cg   -- eps-contaminated Gaussian mixture
Predictions, if the explanation is right:
    gg   (symmetric, light): s=3 -- NO gain;  s=4 -- it appears      <- the decisive control
    logn (skewed, heavy):    s=3 -- gain IS there                    <- tails alone do not kill it
    cg   (symmetric, heavy): s=3 -- none;  s=4 -- ? (this is where the tails cost)
Pair-level diagnostics in the language of Proposition 1 of the letter:  J_1 (linear
criterion),  J_s,  and the ceiling 2 Delta/(2 - Delta). The ratio J_s / ceiling shows how
much of the available information the scheme extracts.

Part B: a dose-response for gcg (gamma core + contamination), s = 3:  eps0 x kappa,
together with the standardized skewness of the mixture -- to see WHAT exactly kills the gain.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import integrate, special, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_argv, sys.argv = sys.argv, sys.argv[:1]
import v13_heavy_tails as v13                                         # noqa: E402
sys.argv = _argv

SEED = 20260918
N_EV = 30_000
S_MAX = 5
OUT = HERE / "results_v14_why_no_gain.json"


class Law(v13.Law):
    def __init__(self, family, a, u):
        super().__init__(family, a, u)
        if family == "logn":                       # standardized lognormal with skewness u
            w = float(np.real([r for r in np.roots([1, 3, 0, -4 - u * u])
                               if abs(r.imag) < 1e-9 and r.real > 1][0]))   # (w+2)^2 (w-1) = u^2
            self.sg = np.sqrt(np.log(w))
            self.m0 = np.exp(0.5 * self.sg**2)
            self.sd = self.m0 * np.sqrt(w - 1.0)
        elif family == "gg":                       # generalized Gaussian, beta in (1, 6]
            self.beta = 1.0 + 5.0 * self.u / 1.9
            self.alpha = np.sqrt(special.gamma(1 / self.beta) / special.gamma(3 / self.beta))

    def sample(self, n, rng):
        if self.family == "logn":
            return self.a + (np.exp(self.sg * rng.standard_normal(n)) - self.m0) / self.sd
        if self.family == "gg":
            r = rng.gamma(1.0 / self.beta, 1.0, n)**(1.0 / self.beta)
            return self.a + self.alpha * r * rng.choice([-1.0, 1.0], n)
        return super().sample(n, rng)

    def logpdf(self, x):
        if self.family == "logn":
            y = (x - self.a) * self.sd + self.m0
            out = np.full_like(x, -np.inf, dtype=float)
            ok = y > 0
            ly = np.log(y[ok])
            out[ok] = -0.5 * (ly / self.sg)**2 - ly - np.log(self.sg * np.sqrt(2 * np.pi)) + np.log(self.sd)
            return out
        if self.family == "gg":
            return (np.log(self.beta) - np.log(2 * self.alpha) - special.gammaln(1 / self.beta)
                    - (np.abs(x - self.a) / self.alpha)**self.beta)
        return super().logpdf(x)

    def pieces(self):
        if self.family == "logn":
            lo = self.a - self.m0 / self.sd
            cuts = [lo, self.a, self.a + 6, self.a + 40, self.a + 400]
            return list(zip(cuts[:-1], cuts[1:]))
        if self.family == "gg":
            cuts = [self.a - 60, self.a - 6, self.a, self.a + 6, self.a + 60]
            return list(zip(cuts[:-1], cuts[1:]))
        return super().pieces()

    def raw_moments(self, K, c, h):
        """E z^k, k = 0..K, z = (x - c)/h. logn integrates in Z (Gaussian weight), the rest in x."""
        ks = np.arange(K + 1)
        if self.family == "logn":
            f = lambda t: (((self.a + (np.exp(self.sg * t) - self.m0) / self.sd - c) / h)**ks  # noqa: E731
                           * np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi))
            tot = sum(integrate.quad_vec(f, lo, hi, epsabs=1e-13, epsrel=1e-11, limit=400)[0]
                      for lo, hi in ((-12, 0), (0, 8), (8, 20), (20, 40)))
        else:
            f = lambda x: ((x - c) / h)**ks * np.exp(self.logpdf(np.atleast_1d(x)))[0]   # noqa: E731
            tot = sum(integrate.quad_vec(f, lo, hi, epsabs=1e-13, epsrel=1e-11, limit=400)[0]
                      for lo, hi in self.pieces())
        assert abs(tot[0] - 1) < 1e-6, (self.family, self.a, self.u, tot[0])
        m1 = tot[1] * h + c
        var = tot[2] * h * h - (tot[1] * h)**2
        assert abs(m1 - self.a) < 1e-5 and abs(var - 1) < 1e-4, (self.family, self.a, self.u, m1, var)
        return tot


def triangular(l1, l2):
    """Delta = int (p2 - p1)^2 / (p1 + p2) dx."""
    def f(x):
        xs = np.atleast_1d(x)
        p1, p2 = np.exp(l1.logpdf(xs))[0], np.exp(l2.logpdf(xs))[0]
        return 0.0 if p1 + p2 == 0 else (p2 - p1)**2 / (p1 + p2)
    cuts = sorted({c for lw in (l1, l2) for pc in lw.pieces() for c in pc if np.isfinite(c)})
    return float(sum(integrate.quad(f, lo, hi, limit=300, epsabs=1e-11)[0]
                     for lo, hi in zip(cuts[:-1], cuts[1:])))


def run_config(family, L, levels, us, seed, with_delta=True):
    M = len(levels)
    laws = [Law(family, a * L / 2.5, u) for a, u in zip(levels, us)]
    a = np.array([lw.a for lw in laws])
    c, h = float(a.mean()), float(np.sqrt(1.0 + a.var()))
    rng = np.random.default_rng(seed)
    xs = [lw.sample(N_EV, rng) for lw in laws]
    logp = [np.stack([lw.logpdf(x) for lw in laws], 1) for x in xs]
    bayes = float(np.mean([(lp.argmax(1) != m).mean() for m, lp in enumerate(logp)]))
    lin = float(np.mean([(np.abs(x[:, None] - a).argmin(1) != m).mean() for m, x in enumerate(xs)]))
    raw = [lw.raw_moments(2 * S_MAX, c, h) for lw in laws]
    out = {"bayes": bayes, "lin": lin, "M": M}
    if with_delta:
        dl = {(m, n): triangular(laws[m], laws[n]) for m in range(M) for n in range(m + 1, M)}
        out["ceiling"] = [2 * d / (2 - d) for d in dl.values()]
    for s in range(1, S_MAX + 1):
        idx = np.arange(1, s + 1)
        mus = np.array([r[idx] for r in raw])
        Cs = np.array([r[idx[:, None] + idx[None, :]] - np.outer(r[idx], r[idx]) for r in raw])
        mu_bar = mus.mean(0)
        S_tot = np.mean([C + np.outer(m, m) for m, C in zip(mus, Cs)], 0) - np.outer(mu_bar, mu_bar)
        w, V = np.linalg.eigh(S_tot)
        W = V @ np.diag(np.maximum(w, 1e-300)**-0.5) @ V.T
        mus_w = (mus - mu_bar) @ W
        Cs_w = np.array([W @ C @ W for C in Cs])
        par, A, b = v13.rules_from_moments(mus_w, Cs_w)
        errs = {k: [] for k in ("copeland", "coupling")}
        fails = []
        for m in range(M):
            z = (xs[m] - c) / h
            Phi = (np.stack([z**i for i in idx], 1) - mu_bar) @ W
            dec, fail, _, _ = v13.decide(Phi, par, A, b, M)
            for k in errs:
                errs[k].append(float((dec[k] != m).mean()))
            fails.append(float(fail.mean()))
        out[f"s{s}"] = {"copeland": float(np.mean(errs["copeland"])), "coupling": float(np.mean(errs["coupling"])),
                        "fail": float(np.mean(fails)), "J": [p[2] for p in par.values()],
                        "cond": float(max(p[5] for p in par.values()))}
    return out


def summarize(rows):
    lin, bay = np.array([r["lin"] for r in rows]), np.array([r["bayes"] for r in rows])
    gap = lin - bay
    ok = gap > 1e-3
    res = {"n": len(rows), "n_with_gap": int(ok.sum()), "bayes_med": float(np.median(bay)),
           "lin_med": float(np.median(lin)), "gap_med_pp": float(100 * np.median(gap))}
    J1 = np.concatenate([r["s1"]["J"] for r in rows])
    if "ceiling" in rows[0]:
        ceil = np.concatenate([r["ceiling"] for r in rows])
        res["ceiling_over_J1_med"] = float(np.median(ceil / J1))
    for s in range(1, S_MAX + 1):
        e = lambda k: np.array([r[f"s{s}"][k] for r in rows])                       # noqa: E731
        Js = np.concatenate([r[f"s{s}"]["J"] for r in rows])
        cell = {"copeland_med": float(np.median(e("copeland"))), "coupling_med": float(np.median(e("coupling"))),
                "closed_copeland": float(np.median(((lin - e("copeland")) / gap)[ok])) if ok.sum() >= 10 else None,
                "closed_coupling": float(np.median(((lin - e("coupling")) / gap)[ok])) if ok.sum() >= 10 else None,
                "fail_gt_1pct": int((e("fail") > 0.01).sum()), "fail_max": float(e("fail").max()),
                "Js_over_J1_med": float(np.median(Js / J1)), "cond_med": float(np.median(e("cond")))}
        if "ceiling" in rows[0]:
            cell["Js_over_ceiling_med"] = float(np.median(Js / ceil))
        res[f"s{s}"] = cell
    return res


def ff(v):
    return " n/a " if v is None else f"{v:+.2f}"


def configs(n):
    rng = np.random.default_rng(2026)
    out = []
    for _ in range(120):
        M = int(rng.integers(3, 6))
        lv = np.sort(rng.uniform(0, 2.5, M))
        out.append((lv - lv[0], rng.uniform(0.0, 1.9, M)))
    return out[:n]


def main():
    global OUT
    part, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 120
    OUT = HERE / f"results_v14_why_no_gain_{part}.json"       # parts are run in parallel
    cfgs = configs(n)
    res = {}
    t0 = time.time()
    if part == "A":
        for family in ("gamma", "logn", "gg", "cg"):
            rows = [run_config(family, 2.5, lv, us, SEED + i) for i, (lv, us) in enumerate(cfgs)]
            res[f"A/{family}"] = summarize(rows)
            s = res[f"A/{family}"]
            print(f"[{time.time()-t0:5.0f}s] {family:>6}: bayes={s['bayes_med']:.3f} lin={s['lin_med']:.3f} "
                  f"gap={s['gap_med_pp']:.2f}pp ceil/J1={s['ceiling_over_J1_med']:.2f} | closed(cop) "
                  + " ".join(f"s{k}={ff(s[f's{k}']['closed_copeland'])}" for k in range(2, S_MAX + 1)), flush=True)
            OUT.write_text(json.dumps(res, indent=1))
    else:
        for kappa in (3.0, 10.0):
            for eps0 in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
                v13.KAPPA, v13.EPS0 = kappa, eps0
                sc = 1.0 / np.sqrt(1 - eps0 + eps0 * kappa**2)
                rows = [run_config("gcg", 2.5, lv, us, SEED + i, with_delta=False)
                        for i, (lv, us) in enumerate(cfgs)]
                s = summarize(rows)
                s["skew_retained"] = float((1 - eps0) * sc**3)        # std. skewness of mixture / u
                s["kurt_excess_at_u1"] = float(((1 - eps0) * (3 + 1.5) + eps0 * 3 * kappa**4) * sc**4 - 3)
                res[f"B/kappa{kappa:g}/eps{eps0:g}"] = s
                print(f"[{time.time()-t0:5.0f}s] kappa={kappa:<4g} eps0={eps0:<6g} skew kept={s['skew_retained']:.3f} "
                      f"ex.kurt(u=1)={s['kurt_excess_at_u1']:6.1f} | gap={s['gap_med_pp']:.2f}pp closed s3: "
                      f"cop={ff(s['s3']['closed_copeland'])} coup={ff(s['s3']['closed_coupling'])} "
                      f"J3/J1={s['s3']['Js_over_J1_med']:.3f} fail>1%={s['s3']['fail_gt_1pct']}", flush=True)
                OUT.write_text(json.dumps(res, indent=1))
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    main()
