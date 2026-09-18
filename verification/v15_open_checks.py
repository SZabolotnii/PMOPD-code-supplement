#!/usr/bin/env python3
"""V15: the three checks V14 left open.

  S  SHIFT.  The fractional basis sgn(x)|x|^p is not shift-invariant, and the levels
     of the letter lie in [0, 2.5]. Same 120 configurations, levels shifted by
     c in {-5, -1.25, 0, +5} (c = -1.25 centres the level range on zero). The noise
     is the SAME realization (same seed), so the power basis is obliged to return
     identical numbers for every c: that is the control.
  R  ROWS for the bootstrap.  V14/A kept only the summaries; here are the per-
     configuration rows for four families, s = 1..5, to quantify the uncertainty of
     the medians.
  T  WHERE THE INFORMATION LIVES.  A direct test of V14's reading, that the tail of
     the lognormal law carries class information while contamination does not. For
     each pair, with the mixture q = (p_m + p_n)/2, the tail is what lies outside the
     central 99 % of the mass of q (0.5 % on each side):
        info_tail  -- share of the triangular discrimination Delta in the tail;
        var_tail   -- share of Var_q of the feature in the tail, for x^3 and
                      for sgn(x)|x|^1.5.
     A nuisance tail: var_tail large, info_tail small. An informative tail: both large.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_argv, sys.argv = sys.argv, sys.argv[:1]
import v14_why_no_gain as v14                                         # noqa: E402
sys.argv = _argv
v13 = v14.v13
Law = v14.Law
N_EV = 30_000


def eval_exact(laws, seed):
    """Exact moments, both V13 bases, one test sample. Returns one configuration row."""
    M = len(laws)
    rng = np.random.default_rng(seed)
    xi = [lw.sample(N_EV, rng) - lw.a for lw in laws]            # noise apart: identical under any shift
    xs = [lw.a + e for lw, e in zip(laws, xi)]
    a = np.array([lw.a for lw in laws])
    logp = [np.stack([lw.logpdf(x) for lw in laws], 1) for x in xs]
    row = {"bayes": float(np.mean([(lp.argmax(1) != m).mean() for m, lp in enumerate(logp)])),
           "lin": float(np.mean([(np.abs(x[:, None] - a).argmin(1) != m).mean() for m, x in enumerate(xs)]))}
    for bname, basis in v13.BASES.items():
        ex = [lw.exact_moments(basis) for lw in laws]
        mu_bar = np.mean([m for m, _ in ex], 0)
        S_tot = np.mean([C + np.outer(m, m) for m, C in ex], 0) - np.outer(mu_bar, mu_bar)
        w, V = np.linalg.eigh(S_tot)
        W = V @ np.diag(w**-0.5) @ V.T
        par, A, b = v13.rules_from_moments(np.array([(m - mu_bar) @ W for m, _ in ex]),
                                           np.array([W @ C @ W for _, C in ex]))
        err = {"copeland": [], "coupling": []}
        fails = []
        for m in range(M):
            dec, fail, _, _ = v13.decide((basis(xs[m]) - mu_bar) @ W, par, A, b, M)
            for k in err:
                err[k].append(float((dec[k] != m).mean()))
            fails.append(float(fail.mean()))
        row[bname] = {"copeland": float(np.mean(err["copeland"])), "coupling": float(np.mean(err["coupling"])),
                      "fail": float(np.mean(fails))}
    return row


def set_family(tag):
    if tag.startswith("gcg"):
        v13.KAPPA, v13.EPS0 = 10.0, float(tag.split("_")[1])
        return "gcg"
    return tag


def part_S(cfgs):
    res, t0 = {}, time.time()
    for tag in ("gamma", "logn", "gcg_0.002", "gcg_0.01"):
        fam = set_family(tag)
        for c in (-5.0, -1.25, 0.0, 5.0):
            rows = [eval_exact([Law(fam, a + c, u) for a, u in zip(lv, us)], 20260919 + i)
                    for i, (lv, us) in enumerate(cfgs)]
            res[f"{tag}/c{c:g}"] = rows
            print(f"[{time.time()-t0:5.0f}s] {tag:<10} c={c:<6g} done", flush=True)
            (HERE / "results_v15_S.json").write_text(json.dumps(res))


def part_R(cfgs):
    res, t0 = {}, time.time()
    for fam in ("gamma", "logn", "gg", "cg"):
        v13.KAPPA, v13.EPS0 = 10.0, 0.05
        res[fam] = [v14.run_config(fam, 2.5, lv, us, v14.SEED + i, with_delta=False)
                    for i, (lv, us) in enumerate(cfgs)]
        print(f"[{time.time()-t0:5.0f}s] {fam} done", flush=True)
        (HERE / "results_v15_R.json").write_text(json.dumps(res))


def part_T(cfgs):
    core = np.linspace(-25, 25, 200_001)
    far = np.geomspace(25, 4000, 4000)
    base = np.unique(np.concatenate([-far, core, far]))
    res, t0 = {}, time.time()
    feats = {"x3": lambda x: x**3, "frac1.5": lambda x: np.sign(x) * np.abs(x)**1.5}
    for tag in ("gamma", "logn", "gcg_0.002", "gcg_0.01", "gcg_0.05", "cg"):
        fam = set_family(tag)
        acc = {"info_tail": [], **{f"var_tail_{k}": [] for k in feats}}
        for lv, us in cfgs:
            laws = [Law(fam, a, u) for a, u in zip(lv, us)]
            for m in range(len(laws)):
                for n in range(m + 1, len(laws)):
                    g = base + 0.5 * (laws[m].a + laws[n].a)
                    pm, pn = np.exp(laws[m].logpdf(g)), np.exp(laws[n].logpdf(g))
                    q = 0.5 * (pm + pn)
                    tz = lambda f: float(np.trapezoid(f, g))                 # noqa: E731
                    assert abs(tz(q) - 1) < 2e-4, (tag, tz(q))
                    cdf = np.concatenate([[0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(g))])
                    tail = (cdf < 0.005) | (cdf > 0.995)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        d = np.where(pm + pn > 0, (pn - pm)**2 / (pm + pn), 0.0)
                    acc["info_tail"].append(tz(np.where(tail, d, 0)) / tz(d))
                    for k, f in feats.items():
                        ph = f(g)
                        mu = tz(ph * q)
                        v = (ph - mu)**2 * q
                        acc[f"var_tail_{k}"].append(tz(np.where(tail, v, 0)) / tz(v))
        res[tag] = {k: {"median": float(np.median(v)), "q25": float(np.quantile(v, .25)),
                        "q75": float(np.quantile(v, .75)), "n_pairs": len(v)} for k, v in acc.items()}
        r = res[tag]
        print(f"[{time.time()-t0:5.0f}s] {tag:<10} pairs={r['info_tail']['n_pairs']:<4} info_tail={r['info_tail']['median']:.3f} "
              f"var_tail(x^3)={r['var_tail_x3']['median']:.3f} var_tail(frac)={r['var_tail_frac1.5']['median']:.3f}", flush=True)
        (HERE / "results_v15_T.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"S": part_S, "R": part_R, "T": part_T}[sys.argv[1]](v14.configs(int(sys.argv[2]) if len(sys.argv) > 2 else 120))
