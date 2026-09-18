#!/usr/bin/env python3
"""V16: the low-error regime (reviewer request: Bayes error between 0.05 and 0.25).

The request was to repeat the comparison of completions at Bayes errors between
0.05 and 0.25. V13 reached only 0.43 (L = 6) and 0.32 (L = 10), because levels are
scaled by L/2.5 and L = 10 was the top of its grid. Here: the same 120
configurations, the same gamma family, the power and fractional bases, exact and
estimated moments, but L in {16, 25, 40}.

Why this is not a formality. Two quantities move in OPPOSITE directions as L grows:
  * the undecided event disappears (it needs poorly separated pairs);
  * the linear-to-Bayes gap shrinks too, so the closed fraction is computed on an
    ever smaller denominator and becomes unstable.
Both forms are therefore reported below: the median of ratios (as in the letter)
and the aggregate fraction (ratio of medians), together with the number of
configurations whose gap is too small to divide by.

Run:     python3 verification/v16_low_error.py [n_cfg]
Output:  verification/results_v16_low_error.json
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

LS = (16.0, 25.0, 40.0)
SEED = 20260921


def summarize(rows, key):
    """key = 'power/exact' etc. Paired statistics; ratio form and aggregate form."""
    g = lambda f: np.array([f(r) for r in rows])                       # noqa: E731
    lin, bay = g(lambda r: r["lin"]), g(lambda r: r["bayes"])
    c = lambda k: g(lambda r: r[key][k])                               # noqa: E731
    gap = lin - bay
    ok = gap > 1e-3                                                    # exclude configs with nothing to close
    out = {"bayes": float(np.median(bay)), "lin": float(np.median(lin)),
           "n_ok": int(ok.sum()), "n": len(rows),
           "fail_gt1pct": int((c("fail") > 0.01).sum()),
           "fail_max": float(c("fail").max()), "fail_med": float(np.median(c("fail")))}
    for k in ("pooled", "copeland", "margin_sum", "coupling"):
        e = c(k)
        out[k] = {
            "err": float(np.median(e)),
            "gain_pp": float(np.median(lin - e) * 100),                # paired median
            "wins": int((e < lin).sum()),
            "closed_med": float(np.median(((lin - e) / gap)[ok])) if ok.any() else None,
            "closed_agg": float(np.median(lin - e) / np.median(gap)),
        }
    out["coupling_vs_copeland_pp"] = float(np.median(c("copeland") - c("coupling")) * 100)
    out["coupling_beats_copeland"] = int((c("coupling") < c("copeland")).sum())
    return out


def main(n_cfg=120):
    cfgs = v14.configs(n_cfg)
    res, t0 = {}, time.time()
    for L in LS:
        rows = [v13.run_config("gamma", L, lv, us, SEED + i) for i, (lv, us) in enumerate(cfgs)]
        res[f"L{L:g}"] = {"rows": rows,
                          **{key: summarize(rows, key)
                             for key in ("power/exact", "power/est1e5", "frac/exact")}}
        s = res[f"L{L:g}"]["power/exact"]
        print(f"[{time.time()-t0:5.0f}s] L={L:<4g} Bayes={s['bayes']:.3f} lin={s['lin']:.3f} "
              f"cop={s['copeland']['err']:.3f} coup={s['coupling']['err']:.3f} | "
              f"closed med {s['copeland']['closed_med']} / {s['coupling']['closed_med']} | "
              f"agg {s['copeland']['closed_agg']:.2f} / {s['coupling']['closed_agg']:.2f} | "
              f">1% {s['fail_gt1pct']}, max {s['fail_max']:.4f}, n_ok {s['n_ok']}", flush=True)
        (HERE / "results_v16_low_error.json").write_text(json.dumps(res))
    print("written verification/results_v16_low_error.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
