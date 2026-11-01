#!/usr/bin/env python3
"""V19: numbers printed in the SPL letter that no earlier script wrote to a JSON file.

Why (SPL-48917-2026, second round, Reviewer 3, point 1). A consistency audit of the Letter,
the supplement and the first-round response traced every number to its result file. Six
families of numbers had no such file; they had been computed in working notes. This script
recomputes each of them and stores the value the text now prints, so that the supplement's
statement "each script writes the JSON file its numbers are read from" holds.

Parts (each checks itself against the value printed in R0 and reports the difference):
  E  closed-form example (Letter Sec. III, S-II): Bayes error of the configuration, the
     linear rule's error, and on the undecided interval U the posteriors and the
     conditional Bayes error. Densities integrated by quadrature; no sampling.
  I  variance inflation behind "Var x^3 sixty-fold" and "fractional features less than
     twofold" (Letter Sec. V, S-IV): per pair, Var_q of x^3 and of sgn(x)|x|^1.5 under the
     pair mixture q; the mean over the 798 pairs of Panel B for each noise family divided
     by the same mean under the gamma laws of the same configurations. Same grid as
     v15_open_checks.py part T.
  P  the one-versus-mixture projection row of Table I: its error over the 30 configurations
     above 1% and its class-averaged disagreement with the Condorcet winner (CW). The draws
     replicate v11_threshold_and_ovm.evaluate exactly; the projection error of every
     configuration is checked against results_v11_threshold_ovm.json.
  C  95% interval of the mean within-configuration rank correlation of the undecided
     probability with each basis diagnostic (Letter Sec. V, "Basis effects"), from the
     records of cond_predictor_sweep.py: normal approximation and a bootstrap over
     configurations (B = 5000).

Run from the repository root:  python3 verification/v19_traceability.py [E,I,P,C]
Out: verification/results_v19_traceability.json (parts are merged into the existing file).
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import integrate, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results_v19_traceability.json"


# ------------------------------------------------------------------ E: closed-form example
def part_E():
    p = [lambda x: stats.norm.pdf(x),
         lambda x: np.where(x > -0.75, np.exp(-(x + 0.75)), 0.0),     # 1/4 + (E - 1), E ~ Exp(1)
         lambda x: stats.norm.pdf(x, 1.75)]
    quad = lambda f, a, b: integrate.quad(f, a, b, limit=400, epsabs=1e-13, epsrel=1e-12)[0]  # noqa: E731
    cuts = [-np.inf, -0.75, 0.125, 1.0, np.inf]
    pmax = lambda x: max(f(x) for f in p)                              # noqa: E731
    bayes = 1 - sum(quad(pmax, a, b) for a, b in zip(cuts[:-1], cuts[1:])) / 3
    # linear rule, s = 1: nearest mean among 0, 1/4, 7/4 -> thresholds 1/8 and 1
    correct = stats.norm.cdf(0.125) + (np.exp(-0.875) - np.exp(-1.75)) + stats.norm.sf(1.0, 1.75)
    lin = 1 - correct / 3
    lo, hi = (25 - np.sqrt(389)) / 8, 7 / 8
    mass = np.array([quad(f, lo, hi) for f in p])
    post = mass / mass.sum()
    cond_bayes = 1 - quad(pmax, lo, hi) / mass.sum()
    return {"U": [lo, hi], "bayes_error": bayes, "linear_error": lin,
            "posteriors_on_U": post.tolist(), "conditional_bayes_error_on_U": cond_bayes,
            "printed": {"bayes_error": 0.37, "linear_error": 0.478,
                        "posteriors_on_U": [0.39, 0.29, 0.32], "conditional_bayes_error_on_U": 0.611}}


# ------------------------------------------------------------------ I: variance inflation
def part_I():
    _argv, sys.argv = sys.argv, sys.argv[:1]
    import v15_open_checks as v15                                       # noqa: E402
    sys.argv = _argv
    v14 = v15.v14
    core = np.linspace(-25, 25, 200_001)
    far = np.geomspace(25, 4000, 4000)
    base = np.unique(np.concatenate([-far, core, far]))
    feats = {"x3": lambda x: x**3, "frac1.5": lambda x: np.sign(x) * np.abs(x)**1.5}

    def pair_vars(fam_tag, lv, us):
        fam = v15.set_family(fam_tag)
        laws = [v15.Law(fam, a, u) for a, u in zip(lv, us)]
        out = []
        for m in range(len(laws)):
            for n in range(m + 1, len(laws)):
                g = base + 0.5 * (laws[m].a + laws[n].a)
                q = 0.5 * (np.exp(laws[m].logpdf(g)) + np.exp(laws[n].logpdf(g)))
                tz = lambda f: float(np.trapezoid(f, g))                # noqa: E731
                row = {}
                for k, f in feats.items():
                    ph = f(g)
                    mu = tz(ph * q)
                    row[k] = tz((ph - mu)**2 * q)
                out.append(row)
        return out

    cfgs = v14.configs(120)
    ref = [pair_vars("gamma", lv, us) for lv, us in cfgs]
    # The printed inflation is the ratio of the MEAN Var_q over all pairs (family / gamma);
    # the median of the per-pair ratios is stored beside it and is larger under impulses.
    res = {}
    for tag in ("logn", "gcg_0.002", "gcg_0.01", "gcg_0.05"):
        num, den = {k: [] for k in feats}, {k: [] for k in feats}
        for (lv, us), r0 in zip(cfgs, ref):
            for a, b in zip(pair_vars(tag, lv, us), r0):
                for k in feats:
                    num[k].append(a[k])
                    den[k].append(b[k])
        res[tag] = {k: {"ratio_of_means": float(np.mean(num[k]) / np.mean(den[k])),
                        "median_of_ratios": float(np.median(np.array(num[k]) / np.array(den[k]))),
                        "n_pairs": len(num[k])} for k in feats}
        print(f"  I {tag:<10} Var x^3 x{res[tag]['x3']['ratio_of_means']:.2f}  "
              f"Var frac x{res[tag]['frac1.5']['ratio_of_means']:.2f}  (median of ratios "
              f"{res[tag]['x3']['median_of_ratios']:.2f}, {res[tag]['frac1.5']['median_of_ratios']:.2f})",
              flush=True)
    res["printed"] = {"logn": {"x3": 1.15}, "gcg_0.002": {"x3": 61, "frac1.5": 1.47},
                      "gcg_0.01": {"x3": 68, "frac1.5": 2.01}}
    return res


# ------------------------------------------------------------------ P: projection row
def _proj_config(args):
    t, M, lv, sk = args
    _argv, sys.argv = sys.argv, sys.argv[:1]
    import v11_threshold_and_ovm as v11                                 # noqa: E402
    sys.argv = _argv
    sc = v11.Scenario(np.array(lv), np.array(sk))
    seed, n_mom, n_ev, S = v11.SEED, 100_000, 120_000, v11.S
    rng = np.random.default_rng(seed)                                   # draw for draw as v11.evaluate
    Xw = np.concatenate([sc.sample(m, 60_000 // M, rng) for m in range(M)])
    P = v11.BASIS(Xw)
    mean, C = P.mean(0), np.cov(P, rowvar=False, ddof=1)
    w, V = np.linalg.eigh(C)
    W = V @ np.diag(np.maximum(w, 1e-300)**-0.5) @ V.T
    tr = lambda x: (v11.BASIS(x) - mean) @ W                            # noqa: E731
    mus, Cs = [], []
    for m in range(M):
        Q = tr(sc.sample(m, n_mom, rng))
        mus.append(Q.mean(0))
        Cs.append(np.cov(Q, rowvar=False, ddof=1))
    mus, Cs = np.array(mus), np.array(Cs)
    K = v11.pairwise_rules(mus, Cs)
    G = np.empty((S + 1, S + 1))
    G[0, 0] = 1.0
    G[0, 1:] = G[1:, 0] = mus.mean(0)
    G[1:, 1:] = (Cs + np.einsum('mi,mj->mij', mus, mus)).mean(0)
    Aovm = np.linalg.solve(G, np.hstack([np.ones((M, 1)), mus]).T).T
    err, cw_dis, fail = [], [], []
    for m in range(M):
        x = sc.sample(m, n_ev, rng)
        Phi = tr(x)
        wins = np.zeros((len(x), M), int)
        for (i, j), (k, c, _) in K.items():
            wj = (Phi - c) @ k > 0
            wins[:, j] += wj
            wins[:, i] += ~wj
        decided = (wins == M - 1).any(1)
        ovm = (np.hstack([np.ones((len(x), 1)), Phi]) @ Aovm.T).argmax(1)
        err.append(float((ovm != m).mean()))
        cw_dis.append(float((ovm[decided] != wins.argmax(1)[decided]).mean()))
        fail.append(float((~decided).mean()))
    return {"idx": t, "M": M, "ovm": float(np.mean(err)), "ovm_vs_cw": float(np.mean(cw_dis)),
            "fail": float(np.mean(fail))}


def part_P():
    rng = np.random.default_rng(2026)                                   # Panel B stream, as v8/v11
    tasks = []
    for t in range(120):
        M = int(rng.integers(3, 6))
        lv = np.sort(rng.uniform(0, 2.5, M))
        lv -= lv[0]
        tasks.append((t, M, lv.tolist(), rng.uniform(0.0, 1.9, M).tolist()))
    with Pool(8) as pool:
        rows = pool.map(_proj_config, tasks)
    v11rows = {r["idx"]: r for r in json.loads((HERE / "results_v11_threshold_ovm.json").read_text())}
    v8rows = {r["idx"]: r for r in json.loads((HERE / "results_v8_table1.json").read_text())}
    dev = max(abs(r["ovm"] - v11rows[r["idx"]]["ovm"]) for r in rows)
    worst = [r for r in rows if v8rows[r["idx"]]["fail"] > 0.01]
    return {"rows": rows, "max_abs_dev_from_v11": dev, "n_worst": len(worst),
            "ovm_err_median": float(np.median([r["ovm"] for r in rows])),
            "ovm_err_worst30_median": float(np.median([r["ovm"] for r in worst])),
            "ovm_cw_median": float(np.median([r["ovm_vs_cw"] for r in rows])),
            "printed": {"worst30": 0.649, "cw": 0.17}}


# ------------------------------------------------------------------ C: rank-correlation CI
def part_C():
    recs = json.loads((HERE / "results_cond_predictor.json").read_text())["records"]
    out = {}
    for key in ("d", "cond_F", "het", "cond_raw"):
        rs = []
        for ci in sorted({x["cfg"] for x in recs}):
            r = [x for x in recs if x["cfg"] == ci]
            g = [x["gap"] for x in r]
            if len(r) < 5 or len(set(g)) < 3:                           # as cond_predictor_sweep.py
                continue
            rs.append(stats.spearmanr([x[key] for x in r], g).statistic)
        rs = np.array([v for v in rs if np.isfinite(v)])
        boot = np.random.default_rng(20261028).choice(rs, (5000, len(rs))).mean(1)
        out[key] = {"n": int(len(rs)), "mean": float(rs.mean()),
                    "normal_halfwidth": float(1.96 * rs.std(ddof=1) / np.sqrt(len(rs))),
                    "bootstrap_95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    out["printed"] = {"halfwidth": 0.16, "means_within": 0.1}
    return out


if __name__ == "__main__":
    parts = sys.argv[1].split(",") if len(sys.argv) > 1 else ["E", "I", "P", "C"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    t0 = time.time()
    for p in parts:
        res[p] = {"E": part_E, "I": part_I, "P": part_P, "C": part_C}[p]()
        OUT.write_text(json.dumps(res, indent=1))
        print(f"[{time.time() - t0:5.0f}s] part {p} done", flush=True)
    for p in parts:
        r = res[p]
        if p == "E":
            print("E", {k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4))
                        for k, v in r.items() if k != "printed"})
        elif p == "P":
            print("P", {k: v for k, v in r.items() if k != "rows"})
        elif p == "C":
            print("C", {k: v for k, v in r.items()})
    print(f"-> {OUT}")
