# PMOPD Code Supplement

Reproducibility code for the letter

**Pairwise Moment-Optimal Polynomial Detectors Are Incomplete for M-ary Discrimination in Non-Gaussian Noise**

This repository is deliberately separate from the manuscript repository. It contains
only code, computed JSON summaries, rendered figures and the Lean formalization.

**This version corresponds to the resubmitted letter (September 2026).** Relative to
v1.0.0 it adds the five studies the reviewers asked for — an analytical example of the
undecided event, three further noise families, the low-error regimes, the shift
sensitivity of the fractional basis and a bootstrap over configurations — and the
supplemental material's Fig. S1. Nothing from v1.0.0 was removed or recomputed; the
`v1`–`v11` scripts and their JSON are unchanged, which is why the original Table I
numbers still reproduce exactly.

Section numbers below refer to the letter (L) and to its supplemental material (S).

## What the letter claims, and which script establishes it

| claim in the letter | script |
|---|---|
| §III  Identity `Λ = (1 + J/2)·Π_s tanh(ℓ/2)`; criterion ceiling `2Δ/(2−Δ)`; behaviour up to `s = 8` with exact moments | `verification/v6_large_s_llr.py` |
| §III  Transitivity of likelihood-ratio duels and the threshold condition `τ₁₂τ₂₃τ₃₁ = 1` | `verification/v6_lr_threshold_check.py` |
| §III  Counting `M·2^{C(M−1,2)}`; undecided ⊊ intransitive from `M = 4` | `verification/incompleteness_general_M.py` |
| §III  Covariance dispersion `0.003` at `s = 1` versus `0.14–0.64` at `s ≥ 2` under a **common** noise law | `verification/s1_risk_common_noise.py` |
| §III  `s = 1` is nearest-mean and never undecided | `verification/s1_linear_baseline.py` |
| §IV, Prop. 4  Copeland ties: undecided ⇒ tie at the top for `M ≤ 4`; conservativeness over `5.8·10⁷` decided events | `verification/v1_copeland_ties.py` |
| **L Fig. 1**  Fraction of configurations above 1 %, with intransitivity for comparison | `verification/v3_gap_vs_s_M.py`, `v3_gap_vs_s_M_repro.py` |
| **S Fig. S1**  Decay with the number of observations `N` per decision | `verification/v10_N_observations.py` |
| **L Table I**  Linear, pooled, Copeland, coupling, Bayes on the same 120 configurations | `verification/v8_table1.py` |
| L§V  Robustness to a training set of 500 and 5000 samples per hypothesis | `verification/v9_trainsize_coupling.py` |
| L§II  Threshold ablation: mid-mean versus Gaussian equal-error and Gaussian-MAP thresholds | `verification/v11_threshold_and_ovm.py` |
| **L Table I**  One-versus-mixture projection baseline | `verification/v11_threshold_and_ovm.py` |
| L§V  Basis comparison, paired statistics | `verification/v2_t3_recheck.py`, `basis_parity_sweep.py`, `parity_followup.py` |
| L§V  Negative result: no usable within-family predictor of the undecided probability | `verification/cond_predictor_sweep.py` |

Added for the resubmission:

| claim | script |
|---|---|
| **L§III, S-II**  The closed-form example: exact rational rules, the undecided interval `U`, `P = 0.0552`, controls K1–K3, and the dependence of the maximum on `min J_mn` | `verification/v12_analytic_cycle.py` |
| **S-IV**  Moderate regimes `L = 6, 10`; contaminated-Gaussian and Student families; exact versus estimated moments; calibration of `r_mn` against the exact pairwise posterior; override accuracy; the margin sum as a stand-alone rule | `verification/v13_heavy_tails.py` |
| **L§V, S-IV**  Why the gain is a skewness effect: the 2 × 2 design (shape × tail) over `s = 2…5`, lognormal and generalized-Gaussian families, and the impulsive dose-response | `verification/v14_why_no_gain.py` |
| **L§II-A, S-IV**  Shift sensitivity of the fractional basis (with the power basis as the control), per-configuration rows for the bootstrap, and where the information sits in the tail | `verification/v15_open_checks.py` |
| **L§V, S-IV**  Low-error regimes `L = 16, 25, 40` (Bayes error 0.23, 0.15, 0.08) | `verification/v16_low_error.py` |

The added scripts build on the earlier ones rather than duplicating them: `v13` imports
the scenario stream and the coupling from `v8_table1.py`, `v14` extends `v13`'s noise
model, and `v15` and `v16` reuse `v14`'s configuration generator. Run them from the
repository root so those imports resolve.

`verification/p2_intransitivity.py` is the shared core: the scenario model
(shifted standardized gamma), the bases, whitening, and the pairwise normal system
`(C^(m) + C^(n)) K = μ^(n) − μ^(m)`.

## Reproduce

```bash
python3 -m pip install -r requirements.txt
cd <repo root>
python3 verification/v8_table1.py          # L Table I  (s = 3; pass 2 or 4 for the other degrees)
python3 verification/v3_gap_vs_s_M.py      # L Fig. 1 data
python3 verification/v10_N_observations.py # S Fig. S1 data
python3 verification/v12_analytic_cycle.py # S-II, the closed-form example
python3 verification/v13_heavy_tails.py    # S-IV, moderate regimes and further families
python3 verification/v14_why_no_gain.py A  # S-IV, the 2 x 2 design (then B, C)
python3 verification/v15_open_checks.py S  # shift study (then R, T)
python3 verification/v16_low_error.py      # S-IV, low-error regimes
python3 make_figs.py                       # renders everything in figures/
```

Scripts are run from the repository root and write their JSON next to themselves in
`verification/`. All of them are seeded — `SEED = 20260817` for the original set,
`20260917`–`20260921` for the added ones, with the shared configuration generator
`numpy.random.default_rng(2026)` — so a rerun reproduces the published numbers up to
the Monte Carlo error stated in the letter.

A caution when rerunning: each script overwrites its own `results_*.json`, and the
argument that limits the number of configurations (`v16_low_error.py 3`, for instance)
writes a short run to the same path. Keep a copy if the published JSON matters to you.

Runtimes on a 16-core laptop: `v8_table1.py` about 6 min, `v3_gap_vs_s_M.py` about
1.5 min, `v10_N_observations.py` about 8 min, `v12_analytic_cycle.py` about 2.5 min,
`v13_heavy_tails.py` about 30 min, `v16_low_error.py` about 5 min, the rest under
2 min each.

## Lean formalization

`Lean/GSA/Part2/MultiAlternativePE.lean` contains the machine-checked part:

- `copeland_of_wins_all` — a Condorcet winner is the unique Copeland maximizer,
  which is the conservativeness half of Proposition 4(a);
- `error_subset_duels` — the error event of any such completion is contained in the
  union of the lost duels of the true hypothesis;
- `PE_duel_bound`, `PE_error_bound`, `PE_error_bound_uniform` — a Chebyshev-type
  M-ary error bound under the probability-error threshold. **These are not used in
  the letter**; they are part of the same module and are listed so that the file's
  contents are not mistaken for the scope of what the letter claims to have checked.

The two claims the letter attributes to Lean are the first two. The file is a module
of the GSA Lean project (Lean 4 v4.26.0, mathlib at the same tag) and is included here
for inspection; building it requires that project's `lakefile.lean` and toolchain.

## Scope

The measurements concern per-observation or small-`N` `M`-ary decisions; the
undecided event decays with `N`, as Fig. S1 shows. The 120 random configurations
are a deliberately hard regime (error rates near 0.6), chosen so that differences
between rules are visible. The resubmission adds five better-separated regimes down
to a Bayes error of 0.08 (`v16_low_error.py`), where the ordering of the completions
that keep the pairwise detectors survives but pooling and the margin sum do not, and
where the undecided event has all but disappeared.

Two limits worth stating plainly. The noise families are gamma, lognormal,
generalized Gaussian, contaminated Gaussian, Student and impulsively contaminated
gamma, all scalar and all standardized to unit variance; `M ≤ 6`. And the fractional
basis is not shift-invariant, so its numbers hold for the level placement used here —
`v15_open_checks.py S` measures how much that matters.

## License

MIT, see `LICENSE`.
