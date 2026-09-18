#!/usr/bin/env python3
"""V12: an analytical configuration with an undecided event (M=3, s=2, power basis).

Why. SPL-48519-2026, R4 item 2 and R2 item 10: in the submitted text the claim that the
truncated tournament can lack a Condorcet winner rested on Monte Carlo counts alone.
Here we look for ONE configuration in which everything is exact:

    x = a_m + xi_m,  xi_m a standardized gamma (mean 0, variance 1, skewness g_m),
    phi(x) = (x, x^2),  mu^(m), C^(m) EXACT (gamma moments up to order 4),
    F_mn = C^(m) + C^(n),  Y_mn = mu^(n) - mu^(m),  K_mn = F_mn^{-1} Y_mn,
    Lambda_mn(x) = K_mn^T (phi(x) - (mu^(m)+mu^(n))/2)  > 0  <=>  n beats m.

No sample enters the construction of the rules: mu and C come from closed forms, so a
cycle, if there is one, cannot be estimation noise. Each Lambda_mn is a quadratic in x,
so the undecided set is a finite union of intervals whose endpoints are roots, and its
probability is a difference of gamma CDFs.

Stages:
  A  continuous search for max P(undecided) over (a_2, a_3, g_1, g_2, g_3), a_1 = 0
     (a shift is an affine change of the basis (x, x^2), to which the rules are invariant);
  B  a sweep over "nice" rational parameters: levels on a grid, skewness from
     {0, 1/2, 1, 2} (Gaussian, Erlang-16, Erlang-4, exponential) -- these give a rational C;
  C  exact verification of the best nice configuration: rational K (fractions), roots as
     quadratic irrationals (sympy), signs evaluated exactly at a rational interior point,
     the probability in mpmath to 30 digits;
  D  an independent Monte Carlo control from SAMPLED data (rules from the exact moments).

Controls, without which the number cannot be believed:
  K1  s=1: the tournament must have a Condorcet winner (Proposition 3) -- P = 0 exactly;
  K2  common covariance C^(m) := the mean -- same thing, P = 0;
  K3  the identity J_mn = Y^T F^{-1} Y > 0 for every pair (otherwise the duel is degenerate).
"""
import itertools
import json
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sy
from scipy import optimize, stats

SEED = 20260917
OUT = Path(__file__).with_name("results_v12_analytic_cycle.json")
PAIRS = [(0, 1), (0, 2), (1, 2)]


# ---------------------------------------------------------------- exact moments
def moments(a, g):
    """mu, C for phi=(x, x^2), x = a + xi; xi: E=0, Var=1, skew=g, ex.kurt=1.5 g^2 (gamma)."""
    mu = (a, a * a + 1)
    c12 = 2 * a + g
    c22 = 4 * a * a + 4 * a * g + 2 + Fraction(3, 2) * g * g if isinstance(g, Fraction) \
        else 4 * a * a + 4 * a * g + 2 + 1.5 * g * g
    return mu, ((1, c12), (c12, c22))


def duel(a, g, m, n, common=None, s=2):
    """Coefficients (q2, q1, q0) of the quadratic Lambda_mn(x) = q2 x^2 + q1 x + q0, and J_mn."""
    mu_m, C_m = moments(a[m], g[m])
    mu_n, C_n = moments(a[n], g[n])
    if common is not None:
        C_m = C_n = common
    y1, y2 = mu_n[0] - mu_m[0], mu_n[1] - mu_m[1]
    c1, c2 = (mu_m[0] + mu_n[0]) / 2, (mu_m[1] + mu_n[1]) / 2
    if s == 1:
        f = C_m[0][0] + C_n[0][0]
        k1 = y1 / f
        return (0 * k1, k1, -k1 * c1), y1 * k1
    f11 = C_m[0][0] + C_n[0][0]
    f12 = C_m[0][1] + C_n[0][1]
    f22 = C_m[1][1] + C_n[1][1]
    det = f11 * f22 - f12 * f12
    k1 = (f22 * y1 - f12 * y2) / det
    k2 = (-f12 * y1 + f11 * y2) / det
    return (k2, k1, -(k1 * c1 + k2 * c2)), y1 * k1 + y2 * k2


def cdf(x, a, g):
    """CDF of x = a + xi; g = 0 is the normal law."""
    if g == 0:
        return stats.norm.cdf(x - a)
    k = 4.0 / g**2
    return stats.gamma.cdf((x - a) * np.sqrt(k) + k, k)


def winner(signs):
    """signs[(m,n)] = +1 if n beats m. Returns the index of the Condorcet winner, or None."""
    for h in range(3):
        ok = True
        for (m, n), sg in signs.items():
            if h == n and sg <= 0:
                ok = False
            if h == m and sg >= 0:
                ok = False
        if ok:
            return h
    return None


def undecided(a, g, common=None, s=2):
    """(P_undecided under the equal-prior mixture, [intervals], J per pair)."""
    polys, J = {}, {}
    for m, n in PAIRS:
        polys[(m, n)], J[(m, n)] = duel(a, g, m, n, common, s)
    roots = []
    for q2, q1, q0 in polys.values():
        r = np.roots([q2, q1, q0]) if abs(q2) > 1e-14 else np.roots([q1, q0])
        roots += [float(z.real) for z in r if abs(z.imag) < 1e-12]
    lo = min(a[i] - (2.0 / g[i] if g[i] > 0 else 12.0) for i in range(3))
    hi = max(a) + 40.0
    cuts = sorted(set([lo] + [z for z in roots if lo < z < hi] + [hi]))
    ivs, p = [], 0.0
    for l, r in zip(cuts[:-1], cuts[1:]):
        mid = 0.5 * (l + r)
        sg = {k: np.sign(np.polyval(v, mid)) for k, v in polys.items()}
        if winner(sg) is None:
            w = np.mean([cdf(r, a[i], g[i]) - cdf(l, a[i], g[i]) for i in range(3)])
            if w > 0:
                ivs.append((l, r, float(w)))
                p += w
    return float(p), ivs, {f"{m}{n}": float(v) for (m, n), v in J.items()}


# ------------------------------------------------------------------ stage A
def stage_a(rng, jmin, n_start=150):
    """max P(undecided) subject to min_pairs J_mn >= jmin.

    Without that constraint the maximum runs off into a degenerate corner (two levels
    coincide, J -> 0) where the duels are noise and "undecided" means nothing. The curve
    P_max(jmin) is itself the answer to how the event depends on pair separation.
    """
    def neg(t):
        a = [0.0, t[0], t[1]]
        g = [t[2], t[3], t[4]]
        p, _, J = undecided(a, g)
        short = jmin - min(J.values())
        return -p + (10.0 * short if short > 0 else 0.0)

    bounds = [(0.05, 4.0), (0.05, 4.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
    best = (0.0, None, None)
    hits = 0
    for _ in range(n_start):
        t0 = [rng.uniform(*b) for b in bounds]
        if undecided([0.0, t0[0], t0[1]], t0[2:])[0] > 1e-3:
            hits += 1
        res = optimize.minimize(neg, t0, method="Nelder-Mead",
                                options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 2000})
        t = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
        p, _, J = undecided([0.0, t[0], t[1]], [t[2], t[3], t[4]])
        if min(J.values()) >= jmin - 1e-9 and p > best[0]:
            best = (p, t.tolist(), J)
    return best, hits / n_start


# ------------------------------------------------------------------ stage B
def stage_b(level_step=Fraction(1, 4), level_max=Fraction(4)):
    skews = [Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2)]
    n_lv = int(level_max / level_step)
    levels = [level_step * i for i in range(1, n_lv + 1)]
    found = []
    for a2, a3 in itertools.combinations(levels, 2):
        for g in itertools.product(skews, repeat=3):
            a = [0.0, float(a2), float(a3)]
            p, ivs, J = undecided(a, [float(z) for z in g])
            if p > 1e-4:
                found.append((p, [Fraction(0), a2, a3], list(g), ivs, J))
    found.sort(key=lambda z: -z[0])
    return found


# ------------------------------------------------------------------ stage C
def stage_c(a, g):
    """Exact check: rational coefficients, irrational roots, signs, and P in mpmath."""
    mp.mp.dps = 30
    x = sy.symbols("x")
    polys = {}
    for m, n in PAIRS:
        (q2, q1, q0), J = duel(a, g, m, n)
        assert all(isinstance(z, Fraction) for z in (q2, q1, q0, J))
        polys[(m, n)] = (sy.Rational(q2.numerator, q2.denominator) * x**2
                         + sy.Rational(q1.numerator, q1.denominator) * x
                         + sy.Rational(q0.numerator, q0.denominator), J)
    roots = sorted({r for p, _ in polys.values() for r in sy.real_roots(sy.Poly(p, x))},
                   key=lambda r: float(r))
    lo = min(sy.Rational(a[i].numerator, a[i].denominator)
             - (sy.Rational(2) / sy.Rational(g[i].numerator, g[i].denominator) if g[i] > 0 else 50)
             for i in range(3))
    cuts = [lo] + [r for r in roots if r > lo] + [sy.oo]

    def mp_pdf(t, ai, gi):
        t, ai, gi = mp.mpf(t), mp.mpf(ai.numerator) / ai.denominator, \
            mp.mpf(gi.numerator) / gi.denominator
        if gi == 0:
            return mp.npdf(t - ai)
        k = 4 / gi**2
        z = (t - ai) * mp.sqrt(k) + k
        return mp.sqrt(k) * z**(k - 1) * mp.exp(-z) / mp.gamma(k) if z > 0 else mp.mpf(0)

    def mp_cdf(t, ai, gi):
        if t == sy.oo:
            return mp.mpf(1)
        t, ai, gi = mp.mpf(sy.N(t, 40)), mp.mpf(ai.numerator) / ai.denominator, \
            mp.mpf(gi.numerator) / gi.denominator
        if gi == 0:
            return mp.ncdf(t - ai)
        k = 4 / gi**2
        z = (t - ai) * mp.sqrt(k) + k
        return mp.gammainc(k, 0, z, regularized=True) if z > 0 else mp.mpf(0)

    out, total = [], mp.mpf(0)
    for l, r in zip(cuts[:-1], cuts[1:]):
        mid = float(l) + 1.0 if r == sy.oo else (float(l) + float(r)) / 2
        mid_q = sy.Rational(str(round(mid, 6)))                  # a rational interior point
        assert l < mid_q < r
        sg = {k: int(sy.sign(p.subs(x, mid_q))) for k, (p, _) in polys.items()}
        if winner(sg) is None:
            per_h = [mp_cdf(r, a[i], g[i]) - mp_cdf(l, a[i], g[i]) for i in range(3)]
            w = sum(per_h) / 3
            total += w
            # the exact (Bayes, equal-prior) winner on the interval: the tournament built from
            # exact likelihoods is transitive (Proposition 3), so the argmax of the density is well defined
            top = float(l) + 30.0 if r == sy.oo else float(r)
            grid = np.linspace(float(l), top, 203)[1:-1]
            bayes = sorted({int(np.argmax([mp_pdf(t, a[i], g[i]) for i in range(3)])) + 1
                            for t in grid})
            beats = {f"{m}{n}": (f"H{n+1} beats H{m+1}" if s > 0 else f"H{m+1} beats H{n+1}")
                     for (m, n), s in sg.items()}
            out.append({"left": str(l), "right": str(r), "left_float": float(l),
                        "right_float": float(r) if r != sy.oo else None,
                        "witness_x": str(mid_q), "duels": beats,
                        "bayes_winner_on_interval": [f"H{b}" for b in bayes],
                        "P_given_H": [mp.nstr(z, 15) for z in per_h], "P_mixture": mp.nstr(w, 15)})
    return {"polys": {f"{m}{n}": {"Lambda": str(sy.expand(p)), "J": str(J), "J_float": float(J)}
                      for (m, n), (p, J) in polys.items()},
            "undecided_intervals": out, "P_undecided": mp.nstr(total, 15)}


# ------------------------------------------------------------------ stage D
def stage_d(a, g, rng, n=4_000_000):
    af, gf = [float(z) for z in a], [float(z) for z in g]
    polys = {k: duel(af, gf, *k)[0] for k in PAIRS}
    rates = []
    for i in range(3):
        if gf[i] == 0:
            xi = rng.standard_normal(n)
        else:
            k = 4.0 / gf[i]**2
            xi = (rng.gamma(k, 1.0, n) - k) / np.sqrt(k)
        xs = af[i] + xi
        sg = {k: np.sign(np.polyval(v, xs)) for k, v in polys.items()}
        w0 = (sg[(0, 1)] < 0) & (sg[(0, 2)] < 0)
        w1 = (sg[(0, 1)] > 0) & (sg[(1, 2)] < 0)
        w2 = (sg[(0, 2)] > 0) & (sg[(1, 2)] > 0)
        rates.append(float(np.mean(~(w0 | w1 | w2))))
    p = float(np.mean(rates))
    return {"n_per_hypothesis": n, "rate_given_H": rates, "P_mixture": p,
            "se": float(np.sqrt(p * (1 - p) / (3 * n)))}


def report(tag, a, g, rng):
    exact = stage_c(a, g)
    print(f"[C:{tag}] a={[str(z) for z in a]} g={[str(z) for z in g]}"
          f"  P_undecided = {exact['P_undecided']}")
    for k, v in exact["polys"].items():
        print(f"    Lambda_{k}(x) = {v['Lambda']}   J = {v['J']} ~ {v['J_float']:.4f}")
    for iv in exact["undecided_intervals"]:
        print(f"    U = ({iv['left']}, {iv['right']})  witness x={iv['witness_x']}  {iv['duels']}"
              f"  Bayes: {iv['bayes_winner_on_interval']}  P={iv['P_mixture']}")
    mc = stage_d(a, g, rng)
    print(f"[D:{tag}] MC: P = {mc['P_mixture']:.6f} +- {mc['se']:.6f}")
    af, gf = [float(z) for z in a], [float(z) for z in g]
    Cbar = tuple(tuple(np.mean([moments(af[i], gf[i])[1][r][c] for i in range(3)])
                       for c in range(2)) for r in range(2))
    ctl = {"K1_s1": undecided(af, gf, s=1)[0],
           "K2_common_cov": undecided(af, gf, common=Cbar)[0],
           "K3_J_all_positive": all(v["J_float"] > 0 for v in exact["polys"].values()),
           "K4_exact_vs_mc_sigmas": abs(float(exact["P_undecided"]) - mc["P_mixture"]) / mc["se"]}
    print(f"[K:{tag}] s=1: {ctl['K1_s1']:.2e}; common C: {ctl['K2_common_cov']:.2e};"
          f" all J>0: {ctl['K3_J_all_positive']}; |exact-MC| = {ctl['K4_exact_vs_mc_sigmas']:.2f} se")
    return {"a": [str(z) for z in a], "g": [str(z) for z in g], **exact,
            "monte_carlo": mc, "controls": ctl}


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "A_continuous": {}}

    for jmin in (0.1, 0.25, 0.5, 1.0, 2.0):
        (pA, tA, JA), hit = stage_a(rng, jmin)
        res["A_continuous"][str(jmin)] = {"P_max": pA, "params_a2_a3_g1_g2_g3": tA, "J": JA,
                                          "share_of_random_starts_with_P>1e-3": hit}
        print(f"[A] min J >= {jmin:<4}: max P(undecided) = {pA:.5f}"
              f" at {np.round(tA, 3).tolist() if tA else None}")

    found = stage_b()
    print(f"[B] nice rational configurations with P>1e-4: {len(found)}")
    picks = {"max_P": found[0] if found else None,
             "separated_minJ>=0.5": next((f for f in found if min(f[4].values()) >= 0.5), None),
             "separated_minJ>=1": next((f for f in found if min(f[4].values()) >= 1.0), None),
             "paper_range_g<=3/2": next((f for f in found if max(f[2]) <= Fraction(3, 2)), None)}
    res["B_nice_grid"] = {"n_with_P>1e-4": len(found), "picks": {}}
    res["C_exact"] = {}
    seen = {}
    for tag, f in picks.items():
        if f is None:
            print(f"[B] {tag}: none")
            res["B_nice_grid"]["picks"][tag] = None
            continue
        p, a, g, _, J = f
        key = (tuple(a), tuple(g))
        res["B_nice_grid"]["picks"][tag] = {"P": p, "a": [str(z) for z in a],
                                            "g": [str(z) for z in g], "J": J}
        print(f"[B] {tag}: P={p:.5f} a={[str(z) for z in a]} g={[str(z) for z in g]}"
              f" minJ={min(J.values()):.3f}")
        if key in seen:
            res["C_exact"][tag] = {"same_as": seen[key]}
            continue
        seen[key] = tag
        res["C_exact"][tag] = report(tag, a, g, rng)

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    main()
