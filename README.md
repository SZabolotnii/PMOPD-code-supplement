# PMOPD Code Supplement

Reproducibility code for the letter

**Pairwise Moment-Optimal Polynomial Detectors Are Incomplete for M-ary Discrimination in Non-Gaussian Noise**

This repository is deliberately separate from the manuscript repository. It contains
only code, computed JSON summaries, rendered figures and the Lean formalization.

## What the letter claims, and which script establishes it

| claim in the letter | script |
|---|---|
| §III  Identity `Λ = (1 + J/2)·Π_s tanh(ℓ/2)`; criterion ceiling `2Δ/(2−Δ)`; behaviour up to `s = 8` with exact moments | `verification/v6_large_s_llr.py` |
| §III  Transitivity of likelihood-ratio duels and the threshold condition `τ₁₂τ₂₃τ₃₁ = 1` | `verification/v6_lr_threshold_check.py` |
| §III  Counting `M·2^{C(M−1,2)}`; undecided ⊊ intransitive from `M = 4` | `verification/incompleteness_general_M.py` |
| §III  Covariance dispersion `0.003` at `s = 1` versus `0.14–0.64` at `s ≥ 2` under a **common** noise law | `verification/s1_risk_common_noise.py` |
| §III  `s = 1` is nearest-mean and never undecided | `verification/s1_linear_baseline.py` |
| §IV, Prop. 4  Copeland ties: undecided ⇒ tie at the top for `M ≤ 4`; conservativeness over `5.8·10⁷` decided events | `verification/v1_copeland_ties.py` |
| **Fig. 1**  Distribution of the undecided probability versus `s`, per `M` | `verification/v3_gap_vs_s_M.py`, `v3_gap_vs_s_M_repro.py` |
| **Fig. 2(a)**  Fraction of configurations above 1 %; intransitivity for comparison | same two scripts |
| **Fig. 2(b)**  Decay with the number of observations `N` per decision | `verification/v10_N_observations.py` |
| **Table I**  Linear, pooled, QDA, Copeland, coupling, Bayes on the same 120 configurations | `verification/v8_table1.py` |
| §V  Robustness to a training set of 500 and 5000 samples per hypothesis | `verification/v9_trainsize_coupling.py` |
| §II  Threshold ablation: mid-mean versus Gaussian equal-error and Gaussian-MAP thresholds | `verification/v11_threshold_and_ovm.py` |
| **Table I**  One-versus-mixture projection baseline | `verification/v11_threshold_and_ovm.py` |
| §V  Basis comparison, paired statistics | `verification/v2_t3_recheck.py`, `basis_parity_sweep.py`, `parity_followup.py` |
| §V  Negative result: no usable within-family predictor of the undecided probability | `verification/cond_predictor_sweep.py` |

`verification/p2_intransitivity.py` is the shared core: the scenario model
(shifted standardized gamma), the bases, whitening, and the pairwise normal system
`(C^(m) + C^(n)) K = μ^(n) − μ^(m)`.

## Reproduce

```bash
python3 -m pip install -r requirements.txt
cd <repo root>
python3 verification/v8_table1.py          # Table I  (s = 3; pass 2 or 4 for the other degrees)
python3 verification/v3_gap_vs_s_M.py      # Fig. 1 and Fig. 2(a) data
python3 verification/v10_N_observations.py # Fig. 2(b) data
python3 make_figs.py                       # renders figures/fig1*.pdf and figures/fig2*.pdf
```

Scripts are run from the repository root and write their JSON next to themselves in
`verification/`. All of them are seeded (`SEED = 20260817`, configuration generator
`numpy.random.default_rng(2026)`), so a rerun reproduces the published numbers up to
the Monte Carlo error stated in the letter.

Runtimes on a 16-core laptop: `v8_table1.py` about 6 min, `v3_gap_vs_s_M.py` about
1.5 min, `v10_N_observations.py` about 8 min, the rest under 2 min each.

## Lean formalization

`Lean/GSA/Part2/MultiAlternativePE.lean` contains the machine-checked part:

- `copeland_of_wins_all` — a Condorcet winner is the unique Copeland maximizer,
  which is the conservativeness half of Proposition 4(a);
- `error_subset_duels` — the error event of any such completion is contained in the
  union of the lost duels of the true hypothesis;
- `PE_duel_bound`, `PE_error_bound`, `PE_error_bound_uniform` — a Chebyshev-type
  M-ary error bound under the probability-error threshold.

The file is a module of the GSA Lean project (Lean 4, mathlib) and is included here
for inspection; building it requires that project's `lakefile.lean` and toolchain.

## Scope

The measurements concern per-observation or small-`N` `M`-ary decisions; the
undecided event decays with `N`, as Fig. 2(b) shows. The random configurations
are a deliberately hard regime (error rates near 0.6), chosen so that differences
between rules are visible; the letter states this and reports a well-separated
control.

## License

MIT, see `LICENSE`.
