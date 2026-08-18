import Mathlib.Probability.Moments.Variance
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.Tactic.Linarith
import GSA.Part2.RobustPE
import GSA.Part2.FAR_ADD

open MeasureTheory ProbabilityTheory
open scoped ENNReal BigOperators

namespace GSA.Part2.MAry

/-!
# Error Probability Bound for Multi-Alternative Decision Rules (PE Criterion)

Probabilistic extension of `GSA.Part2.MAry` (`MultiAlternative.lean`).
That module establishes when pairwise moment-based rules are consistent and when they are not.
Here we bound the probability of error under pairwise aggregation.

The proof chain is structured as follows:

1. `copeland_of_wins_all` -- Combinatorial core. If hypothesis `m` wins
   all `M-1` duels, it is the **unique** Copeland winner. Consequently,
   intransitive cycles cannot corrupt decisions when the true hypothesis is dominant.
2. `error_subset_duels` -- The overall decision error event is a subset of the union
   of the `M-1` events representing individual duel losses.
3. `PE_duel_bound` -- Under the PE threshold `h = E[Λ] + √(Var[Λ]/ε)`, a duel is lost
   with probability at most `ε`. This is the binary `exceed_le_eps` result
   from `FAR_ADD.lean`.
4. `PE_error_bound` -- Union bound: `P(error | H_m) ≤ Σ_{n≠m} ε_n`,
   which simplifies under uniform budget allocation `ε_n = ε/(M-1)` to `ε`
   (`PE_error_bound_uniform`).

The multi-alternative penalty corresponds to the factor `M-1` in step 4: to maintain
total error at `ε`, each individual duel must satisfy an `M-1` times stricter budget,
scaling the threshold by `√(M-1)`.

No pairwise independence assumption is required -- the union bound holds under
arbitrary dependence structures among test statistics `Λ_{mn}`.
-/

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-! ### 1. Combinatorial component: dominant hypothesis wins under Copeland aggregation -/

/-- Total pairwise win count of hypothesis `a` in tournament `beats`. -/
def score (beats : ι → ι → Bool) (a : ι) : ℕ :=
  (Finset.univ.filter fun b => beats a b = true).card

/-- **L5.1.** If `m` wins all pairwise duels, its Copeland score is strictly greater
than the score of any alternative hypothesis.

The proof does not rely on transitivity: any competitor `n` loses at least one
duel (against `m`), so its score cannot attain the maximum `M-1`.
Intransitive cycles can only arise among hypotheses that have already lost to a dominant alternative. -/
theorem copeland_of_wins_all (beats : ι → ι → Bool)
    (hirr : ∀ a, beats a a = false)
    (hasym : ∀ a b, beats a b = true → beats b a = false)
    {m : ι} (hm : ∀ n, n ≠ m → beats m n = true) {n : ι} (hn : n ≠ m) :
    score beats n < score beats m := by
  classical
  have hcard : 2 ≤ Fintype.card ι := Finset.one_lt_card.mpr
    ⟨n, Finset.mem_univ _, m, Finset.mem_univ _, hn⟩
  -- `m` beats every alternative except itself
  have hM : score beats m = Fintype.card ι - 1 := by
    have : (Finset.univ.filter fun b => beats m b = true) = Finset.univ.erase m := by
      ext b
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_erase, and_true]
      constructor
      · intro hb hbm; rw [hbm, hirr] at hb; exact Bool.noConfusion hb
      · intro hb; exact hm b hb
    rw [score, this, Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ]
  -- `n` beats neither itself nor `m`
  have hN : score beats n ≤ Fintype.card ι - 2 := by
    have hsub : (Finset.univ.filter fun b => beats n b = true)
        ⊆ (Finset.univ.erase m).erase n := by
      intro b hb
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hb
      refine Finset.mem_erase.mpr ⟨?_, Finset.mem_erase.mpr ⟨?_, Finset.mem_univ _⟩⟩
      · rintro rfl; rw [hirr] at hb; exact Bool.noConfusion hb
      · rintro rfl; rw [hasym _ _ (hm n hn)] at hb; exact Bool.noConfusion hb
    calc score beats n ≤ ((Finset.univ.erase m).erase n).card := Finset.card_le_card hsub
      _ = Fintype.card ι - 2 := by
          rw [Finset.card_erase_of_mem (Finset.mem_erase.mpr ⟨hn, Finset.mem_univ _⟩),
            Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ]
          omega
  omega

/-! ### 2. Decision error is a subset of the union of lost duels -/

/-- **L5.2.** If the decision rule `choose` selects `m` whenever `m` wins all
pairwise duels, an error occurs only if at least one duel is lost. -/
theorem error_subset_duels (win : ι → Ω → Prop) (choose : Ω → ι) (m : ι)
    (hchoose : ∀ ω, (∀ n ∈ Finset.univ.erase m, win n ω) → choose ω = m) :
    {ω | choose ω ≠ m} ⊆ ⋃ n ∈ Finset.univ.erase m, {ω | ¬ win n ω} := by
  intro ω hω
  by_contra hcon
  simp only [Set.mem_iUnion, Set.mem_setOf_eq, not_exists, not_not] at hcon
  exact hω (hchoose ω fun n hn => hcon n hn)

/-! ### 3. PE threshold guarantees probability bound `ε` per duel -/

/-- **L5.3.** Under the PE threshold `h = E[Λ] + √(Var[Λ]/ε)`, a duel is lost with
probability at most `ε`. -/
theorem PE_duel_bound [IsFiniteMeasure μ] {Λ : Ω → ℝ} (hΛ : MemLp Λ 2 μ)
    {ε : ℝ} (hε : 0 < ε) (hv : 0 < variance Λ μ) :
    μ {ω | GSA.Part2.PE_threshold (μ[Λ]) (variance Λ μ) ε ≤ Λ ω} ≤ ENNReal.ofReal ε :=
  GSA.Part2.exceed_le_eps μ Λ hΛ ε hε hv

/-! ### 4. Union bound for `M` hypotheses -/

/-- **L5 (Main).** The error probability of a multi-alternative rule under `H_m`
is bounded by the sum of error budgets across individual pairwise duels.

Pairwise independence is not assumed -- the union bound holds under arbitrary dependence. -/
theorem PE_error_bound [IsProbabilityMeasure μ] {m : ι}
    (Lam : ι → Ω → ℝ) (hL : ∀ n, MemLp (Lam n) 2 μ)
    (hv : ∀ n, 0 < variance (Lam n) μ)
    (εs : ι → ℝ) (hε : ∀ n, 0 < εs n)
    (choose : Ω → ι)
    (hchoose : ∀ ω, (∀ n ∈ Finset.univ.erase m,
        Lam n ω < GSA.Part2.PE_threshold (μ[Lam n]) (variance (Lam n) μ) (εs n)) →
      choose ω = m) :
    μ {ω | choose ω ≠ m} ≤ ENNReal.ofReal (∑ n ∈ Finset.univ.erase m, εs n) := by
  classical
  set thr : ι → ℝ := fun n =>
    GSA.Part2.PE_threshold (μ[Lam n]) (variance (Lam n) μ) (εs n) with hthr
  have hsub := error_subset_duels (fun n ω => Lam n ω < thr n) choose m hchoose
  have hstep : μ {ω | choose ω ≠ m}
      ≤ ∑ n ∈ Finset.univ.erase m, μ {ω | ¬ Lam n ω < thr n} := by
    refine le_trans (measure_mono hsub) ?_
    exact measure_biUnion_finset_le _ _
  refine le_trans hstep ?_
  have hterm : ∀ n ∈ Finset.univ.erase m,
      μ {ω | ¬ Lam n ω < thr n} ≤ ENNReal.ofReal (εs n) := by
    intro n _
    have : {ω | ¬ Lam n ω < thr n} = {ω | thr n ≤ Lam n ω} := by
      ext ω; simp [not_lt]
    rw [this]
    exact PE_duel_bound (hL n) (hε n) (hv n)
  refine le_trans (Finset.sum_le_sum hterm) ?_
  rw [← ENNReal.ofReal_sum_of_nonneg fun n _ => (hε n).le]

/-- **Corollary.** A uniform budget `ε_n = ε/(M-1)` guarantees total error at most `ε`.
This characterizes the multi-alternative price: each duel is tightened by a factor of `M-1`,
raising the threshold by `√(M-1)`. -/
theorem PE_error_bound_uniform [IsProbabilityMeasure μ] {m : ι}
    (Lam : ι → Ω → ℝ) (hL : ∀ n, MemLp (Lam n) 2 μ)
    (hv : ∀ n, 0 < variance (Lam n) μ)
    {ε : ℝ} (hε : 0 < ε) (hcard : 2 ≤ Fintype.card ι)
    (choose : Ω → ι)
    (hchoose : ∀ ω, (∀ n ∈ Finset.univ.erase m,
        Lam n ω < GSA.Part2.PE_threshold (μ[Lam n]) (variance (Lam n) μ)
          (ε / ((Fintype.card ι : ℝ) - 1))) → choose ω = m) :
    μ {ω | choose ω ≠ m} ≤ ENNReal.ofReal ε := by
  classical
  have hN : (0 : ℝ) < (Fintype.card ι : ℝ) - 1 := by
    have : (2 : ℝ) ≤ (Fintype.card ι : ℝ) := by exact_mod_cast hcard
    linarith
  have hcards : ((Finset.univ.erase m).card : ℝ) = (Fintype.card ι : ℝ) - 1 := by
    rw [Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ]
    have : 1 ≤ Fintype.card ι := by omega
    push_cast [Nat.cast_sub this]
    ring
  refine le_trans (PE_error_bound Lam hL hv (fun _ => ε / ((Fintype.card ι : ℝ) - 1))
    (fun _ => div_pos hε hN) choose hchoose) ?_
  rw [Finset.sum_const, nsmul_eq_mul, hcards, mul_comm, div_mul_cancel₀ _ hN.ne']

end GSA.Part2.MAry
