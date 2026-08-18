#!/usr/bin/env python3
"""V6b: Transitivity of pairwise LR tests with arbitrary thresholds (numerical illustration).

n beats m <=> p_n(x)/p_m(x) > tau_mn, tau_nm = 1/tau_mn.
Theorem: The tournament is transitive for all x (Condorcet winner exists a.s.)
if and only if tau_mn = c_m/c_n (product along every 3-cycle equals 1);
otherwise a cycle is possible and has positive probability if the LR vector law
charges the corresponding cone.
"""
import json
import sys
import numpy as np

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario                    # noqa: E402
from basis_parity_sweep import logpdf                     # noqa: E402

rng = np.random.default_rng(7)
out = {}
for name, sc in {'M3_fixed': Scenario([0.0, 1.5, 3.0], [0.2, 1.0, 1.8]),
                 'M3_close': Scenario([0.0, 0.3, 0.6], [0.2, 1.0, 1.8]),
                 'M5_worst': Scenario([0.0, 0.04, 1.28, 1.39, 1.45], [0.05, 1.83, 1.36, 0.39, 1.39])}.items():
    M = sc.M
    x = np.concatenate([sc.sample(m, 200_000, rng) for m in range(M)])
    L = np.stack([logpdf(x, sc.levels[j], sc.skews[j]) for j in range(M)], 1)
    res = {}
    def gap_for(logtau):          # logtau[m,n] = log tau_mn, skew-symmetric
        out_ = np.zeros((len(x), M), int)
        for m in range(M):
            for n in range(m + 1, M):
                win = (L[:, n] - L[:, m]) > logtau[m, n]
                out_[:, n] += win
                out_[:, m] += ~win
        return float((~(out_ == M - 1).any(1)).mean())
    res['tau=1 (ML)'] = gap_for(np.zeros((M, M)))
    c = rng.uniform(-1.5, 1.5, M)                    # potential thresholds log tau_mn = c_m - c_n
    res['potential (weighted MAP)'] = gap_for(c[:, None] - c[None, :])
    lt = rng.uniform(-1.5, 1.5, (M, M)); lt = np.triu(lt, 1); lt = lt - lt.T   # arbitrary
    res['arbitrary'] = gap_for(lt)
    # product along 3-cycles (sum of logs) for arbitrary thresholds
    cyc = [float(lt[a, b] + lt[b, cc] + lt[cc, a]) for a in range(M) for b in range(a + 1, M) for cc in range(b + 1, M)]
    res['arbitrary_cycle_logsums'] = cyc
    out[name] = res
    print(name, {k: (round(v, 6) if isinstance(v, float) else [round(z, 3) for z in v]) for k, v in res.items()})
json.dump(out, open('verification/results_v6_lr_threshold_check.json', 'w'), indent=1)


# --- M=3, close levels: cycle sum of log-thresholds vs gap ---
sc = Scenario([0.0, 0.3, 0.6], [0.2, 1.0, 1.8]); M = 3
rng = np.random.default_rng(3)
x = np.concatenate([sc.sample(m, 200_000, rng) for m in range(M)])
L = np.stack([logpdf(x, sc.levels[j], sc.skews[j]) for j in range(M)], 1)
rows = []
for t01, t12, t20 in [(0.5, 0.5, 0.5), (-0.5, -0.5, -0.5), (0.2, 0.2, 0.2), (-0.2, -0.2, -0.2),
                      (1.0, 1.0, -1.0), (0.5, 0.5, -1.0), (0.3, -0.7, 0.4)]:
    lt = np.zeros((M, M)); lt[0, 1], lt[1, 2], lt[0, 2] = t01, t12, -t20
    lt[1, 0], lt[2, 1], lt[2, 0] = -t01, -t12, t20
    o = np.zeros((len(x), M), int)
    for m in range(M):
        for n in range(m + 1, M):
            w = (L[:, n] - L[:, m]) > lt[m, n]; o[:, n] += w; o[:, m] += ~w
    g = float((~(o == M - 1).any(1)).mean())
    rows.append({'logtau01': t01, 'logtau12': t12, 'logtau20': t20, 'cycle_logsum': t01 + t12 + t20, 'gap': g})
    print(f"M3_close logtau=({t01:+.1f},{t12:+.1f},{t20:+.1f}) cycle-sum={t01+t12+t20:+.1f} gap={g:.5f}")
out['M3_close_cycle_sum_scan'] = rows
json.dump(out, open('verification/results_v6_lr_threshold_check.json', 'w'), indent=1)
