#!/usr/bin/env python3
"""Generalization of pairwise scheme incompleteness from M=3 to arbitrary M.

Palagin's scheme (2010, p. 28) for M=3 defines the selection rule:
    H0: 0 beats 1 and 0 beats 2;
    H1: 1 beats 0 and 1 beats 2;
    H2: 2 beats 0 and 2 beats 1,
i.e., a hypothesis is chosen iff it wins ALL its pairwise duels. In tournament theory,
this is a Condorcet winner -- a vertex with out-degree M-1.

Hence the general equivalence:
    the scheme reaches a decision <=> the tournament has a Condorcet winner.

Combinatorial counting: A Condorcet winner can be chosen in M ways, its M-1 duels
are fixed, the remaining C(M-1, 2) edges are arbitrary, and a Condorcet winner is unique:
    N_dec(M) = M * 2^C(M-1, 2),   fraction = M / 2^(M-1),
    undecided fraction = 1 - M / 2^(M-1) -> 1 as M -> infinity.

KEY DISTINCTION: An undecided event is the ABSENCE of a Condorcet winner, NOT intransitivity.
At M=3, these two events coincide, which can be misleading. For M >= 4, a tournament can contain
cycles and yet possess a Condorcet winner (e.g., hypothesis 0 beats all others, while a cycle
exists among the remaining M-1 hypotheses). Thus:
    {undecided event}  STRICT SUBSET OF  {intransitive tournament}   for M >= 4.
Therefore, the cycle frequency overestimates the undecided frequency.

Verified here: (1) combinatorial formula vs. exact enumeration; (2) divergence between
the two event sets; (3) empirical probabilities of both events on representative scenarios.
"""
import itertools
import json
import sys
from math import comb, factorial

import numpy as np

sys.path.insert(0, 'verification')
from p2_intransitivity import Scenario, Whitener, moments, pairwise_rules  # noqa: E402
from p5_diagnostic_aggregation import SCENARIOS                            # noqa: E402

SEED = 20260817


def enumerate_all(M):
    """Exhaustive tournament enumeration: count Condorcet winners and transitive tournaments."""
    pairs = [(m, n) for m in range(M) for n in range(m + 1, M)]
    dec = trans = both = 0
    for pat in range(1 << len(pairs)):
        out = np.zeros(M, int)
        for b, (m, n) in enumerate(pairs):
            out[n if pat >> b & 1 else m] += 1
        has_cw = (out == M - 1).any()
        is_trans = sorted(out) == list(range(M))
        dec += int(has_cw)
        trans += int(is_trans)
        both += int(has_cw and is_trans)
    return dec, trans, both, 1 << len(pairs)


def empirical(sc, s, kind='power', n_mom=250_000, n_eval=350_000, seed=SEED):
    """P(no Condorcet winner) vs. P(intransitive tournament) on synthetic model."""
    rng = np.random.default_rng(seed)
    wh = Whitener(sc, kind, s, 120_000, rng)
    mus, Cs = moments(sc, wh, kind, s, n_mom, rng)
    K = pairwise_rules(mus, Cs)
    X = np.concatenate([sc.sample(m, n_eval // sc.M, rng) for m in range(sc.M)])
    Phi = wh(X, kind, s)
    out = np.zeros((len(X), sc.M), int)
    for (m, n), (k, c, _) in K.items():
        win_n = (Phi - c) @ k > 0
        out[:, n] += win_n
        out[:, m] += ~win_n
    no_cw = ~(out == sc.M - 1).any(axis=1)
    intr = ~(np.sort(out, axis=1) == np.arange(sc.M)).all(axis=1)
    return float(no_cw.mean()), float(intr.mean())


if __name__ == '__main__':
    out = {}
    print('=== 1. Analytic Formula vs. Exact Enumeration ===\n')
    print(f'{"M":>3}{"tournaments":>14}{"decided":>12}{"M·2^C(M-1,2)":>15}'
          f'{"fraction":>10}{"undecided":>11}')
    rows = []
    for M in range(2, 8):
        dec, trans, both, tot = enumerate_all(M)
        formula = M * 2**comb(M - 1, 2)
        assert dec == formula, (M, dec, formula)
        frac = M / 2**(M - 1)
        assert abs(dec / tot - frac) < 1e-12
        rows.append({'M': M, 'total': tot, 'decided': dec, 'transitive': trans,
                     'frac_decided': frac, 'gap': 1 - frac})
        print(f'{M:>3}{tot:>14}{dec:>12}{formula:>15}{frac:>10.4f}{1-frac:>11.4f}')
    out['enumeration'] = rows

    print('\n=== 2. Undecided Event vs. Intransitivity ===\n')
    print(f'{"M":>3}{"no Condorcet":>14}{"intransitive":>16}{"coincide?":>13}')
    rows = []
    for M in range(2, 8):
        dec, trans, both, tot = enumerate_all(M)
        no_cw, intr = tot - dec, tot - trans
        same = (no_cw == intr) and (both == trans)
        rows.append({'M': M, 'no_condorcet': no_cw, 'intransitive': intr,
                     'coincide': bool(same)})
        print(f'{M:>3}{no_cw:>14}{intr:>16}{("YES" if same else "no"):>13}')
    out['gap_vs_intransitive'] = rows
    print('\nFor M=3 the two events coincide, which can obscure the distinction.')
    print('For M>=4 intransitive tournaments are a STRICT SUPERSET of undecided events.')

    print('\n=== 3. Empirical Rates on Representative Scenarios (s=3, power basis) ===\n')
    print(f'{"scenario":>38}{"M":>3}{"P(gap)":>14}{"P(cycle)":>11}{"ratio":>12}')
    rows = []
    for name, sc, _ in SCENARIOS:
        g, i = empirical(sc, 3)
        rows.append({'scenario': name, 'M': sc.M, 'p_gap': g, 'p_intr': i})
        ratio = f'{i/g:.2f}x' if g > 1e-9 else '—'
        print(f'{name[:38]:>38}{sc.M:>3}{g:>14.5f}{i:>11.5f}{ratio:>12}')
    out['empirical'] = rows

    print('\n=== 4. Basis Comparison: Power vs. PATP ===\n')
    print('(at s=1 the bases are identical since p_1(alpha)=1 for all alpha)\n')
    hdr = f'{"scenario":>34}{"M":>3}' + ''.join(f'{f"s={k}":>9}' for k in (2, 3, 4))
    rows = []
    for kind in ('power', 'patp'):
        print(f'-- basis: {kind} --')
        print(hdr)
        for name, sc, _ in SCENARIOS:
            vals = [empirical(sc, k, kind=kind)[0] for k in (2, 3, 4)]
            rows.append({'kind': kind, 'scenario': name, 'M': sc.M,
                         **{f's{k}': v for k, v in zip((2, 3, 4), vals)}})
            print(f'{name[:34]:>34}{sc.M:>3}' + ''.join(f'{v:>9.5f}' for v in vals))
        print()
    out['basis_comparison'] = rows

    json.dump(out, open('verification/results_incompleteness.json', 'w'), indent=1, default=int)
    print('\nSaved verification/results_incompleteness.json')
