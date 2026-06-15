# Phase 3 (follow-on) — Blended Product Model (Elo + profile) — Implementation Plan

> **Status:** planning. **Scope: Phase-3 backtest only** — evaluate the blend, lock the weight/combiner.
> The live serve path (`src/predict/`) stays Phase 4 and only *consumes* the weight locked here.
>
> **One-line:** keep the pure bottom-up model as the research artifact (the gate), and **serve** a
> blend of it with Elo as the 2026 product model. The blend is reported but **never counted as a gate
> pass** — it uses Elo, which the non-negotiable principles forbid as a model feature.

---

## 1. Why this exists

The Phase-3 success gate asks: *do player-profile features alone beat team-identity Elo?* They do not,
yet — the enriched bottom-up model reached **RPS 0.1874** (from 0.1923), closing **~60%** of the gap to
**Elo's 0.1802**, but still loses. See [`phase-3-model-backtest.md`](phase-3-model-backtest.md) for the
gate definition.

The residual gap is plausibly **form-shaped**: the bottom-up model is, by design, blind to recent
form / momentum / cohesion — exactly what Elo is made of. Blending adds that missing ingredient back.

**Why a blend should beat both members (the mechanism):**

- **Decorrelated, complementary errors.** Elo encodes results-based, time-evolving signal (form,
  momentum, cohesion); the profile model encodes talent-stock signal (who is in the squad). Each is
  structurally blind to what the other sees, so they miss on *different* matches. Averaging cancels the
  uncorrelated errors and keeps the shared signal.
- **Metric guarantee (RPS).** RPS is a sum of squared errors on the W/D/L CDF, so the Krogh–Vedelsby
  ambiguity decomposition applies exactly: `RPS(blend) = weighted-avg RPS(members) − ambiguity`, with
  `ambiguity ≥ 0`. The blend always beats the *average* member, and beats the *best* member whenever
  disagreement exceeds the skill gap — precisely our case (members close in skill: 0.1874 vs 0.1802,
  but using disjoint information → high disagreement).
- **Calibration bonus.** Pooling shrinks overconfident probabilities toward the centre, which log-loss
  rewards.

**This is a product decision, not a research result.** Folding Elo in violates the "Elo is never a
model feature" principle, so the blend cannot pass the gate. We report the pure model honestly and
serve the blend.

---

## 2. Key implementation finding — this is low-effort

`src/models/evaluate.py::_fold_probabilities` already computes, **per fold**, calibrated W/D/L
probability matrices for every candidate, including:

- `proba["Elo-only"]` — leakage-free Elo (`baselines.compute_elo_features` → calibrated logit), and
- `proba["Ensemble"]` — the profile model (mean of LogReg + HGB profile).

The blend is just a convex combination of two matrices **we already have**:

```
P_blend = w · P_Ensemble + (1 − w) · P_Elo-only
```

No new features, no new training, no schema change. Because the combination is linear and `w` is fixed
per run, computing it post-hoc on the pooled arrays is mathematically identical to per-fold and is
leakage-free.

> **Hard rule:** pool the **outputs**. Never add `elo_diff` to `MODEL_FEATURES` — that would make Elo a
> model feature and break the bottom-up principle (and the gate's meaning).

---

## 3. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | **Phase-3 backtest only** | Smallest self-contained change; serve path is Phase 4. |
| Combiner | **Data-chosen** — global-weight linear pool **vs** regularized logistic stacker; serve the nested-validation winner | "Best performance" without guessing which generalizes. |
| Weight metric | **Nested across-edition** (honest), served weight = global LOEO `w*` | Maximize out-of-sample performance while staying leakage-honest. |
| Gate | Blend **excluded** from `gate_verdict` | Uses Elo → cannot be a bottom-up gate pass. |
| Profile member | `Ensemble` (LogReg+HGB profile) | Most robust profile estimate already in the harness. |

---

## 4. Weight & combiner design (performance-optimal **and** leakage-honest)

With only ~698 matches across a handful of editions, the combiner that *generalizes* best is usually
**not** the most expressive one. A blind `w=0.5` leaves performance on the table; a logistic stacker is
most expressive but can overfit the meta-layer and generalize worse. A **single global weight** fits
just **one scalar** (≈1 DoF over ~698 points → negligible overfit) and is typically the best
out-of-sample bet. So we implement both leakage-free combiners and let validation decide.

**Protocol:**

1. **Sweep** `w ∈ {0.00, 0.05, …, 1.00}` over the pooled held-out predictions (each edition is already
   predicted out-of-sample by a model trained on strictly-prior editions — the existing
   expanding-window backtest). This is the leave-one-edition-out (LOEO) curve. Diagnostic: confirms the
   blend beats Elo across a **broad band** (robustness, not a knife-edge).
2. **Served weight `w*`** = `argmin` pooled RPS over the sweep. Legitimate: one robust scalar set from
   all historical out-of-sample data. This is what locks into `config.blend.weight` for 2026.
3. **Honest reported metric = nested**, *not* the sweep argmin. For each outer edition, pick `w` on the
   *inner* (strictly-prior) editions only, apply to the outer edition, pool, score. This is the
   unbiased estimate of what weight-tuning delivers out-of-sample. The sweep argmin is reported only as
   a slightly-optimistic (~1 DoF) reference. **Fallback:** outer folds with `< 2` inner editions use
   `w = 0.5`.
4. **Combiner bake-off.** Run the same nested protocol for a calibrated, regularized logistic stacker
   (meta-logit on the two models' out-of-fold probabilities). **Serve whichever combiner has the lower
   nested RPS** (tie-break: nested log-loss). If the stacker's nested RPS is worse, it overfit — drop it
   and serve the global-weight pool.

> **Leakage discipline:** the *served* weight (LOEO `w*`) and the *reported* metric (nested) are
> deliberately different. Serving uses all history to set one scalar; reporting never lets `w` see the
> edition it's scored on. Both base-model probability sets are already leakage-free (trained on prior
> editions only).

---

## 5. Changes, file by file

### 5.0 Precondition gate (cheap go/no-go — do first)
Per-match RPS-error correlation between `P["Elo-only"]` and `P["Ensemble"]` on the pooled held-out.
- Low (expected) → proceed.
- High (`> ~0.8`) → errors are not decorrelated; blending can't help. **Stop and report that** instead
  of shipping a no-op.

### 5.1 New `src/models/blend.py` (pure, fixture-testable — mirrors `compute_elo_features` style)
```python
def linear_pool(p_a, p_b, w):            # w·p_a + (1−w)·p_b; asserts a valid prob matrix out
def error_correlation(p_a, p_b, y):      # per-match RPS-error corr — the §5.0 precondition
def blend_weight_sweep(p_a, p_b, y):     # DataFrame[w, rps, log_loss] — the LOEO curve
def select_weight_loeo(p_a, p_b, y):     # global w* = argmin pooled RPS
def nested_blend_eval(folds):            # honest nested RPS/log-loss (<2-inner-edition → w=0.5)
def fit_stacker(P_oof, y):               # calibrated, regularized meta-logit combiner
```

### 5.2 `src/models/evaluate.py` (wiring — ~10–20 lines)
- Add `PRODUCT_MODELS = ("Blend (Elo+profile)",)`, kept **separate** from `PROFILE_MODELS`.
- Compute the served blend from the already-pooled `P` dict (`linear_pool` or `fit_stacker`,
  per `config.blend.combiner`).
- Append it to the summary table tagged `[product]`.
- **`gate_verdict` unchanged** — still operates on `PROFILE_MODELS` only; the blend cannot pass by
  construction.
- Extend `_print_report` with: the product line, the LOEO sweep table, the nested numbers, the combiner
  bake-off result, and the error-correlation value.

### 5.3 `config.yaml`
```yaml
# Product model (Phase 4 serving): blend of the bottom-up profile model with the Elo baseline.
# NOT a gate pass — Elo is never a profile feature. Weight/combiner are locked by the Phase-3 backtest.
blend:
  combiner: linear        # 'linear' (global-weight pool) or 'stacker' — set to the bake-off winner
  weight: 0.5             # served global w* (profile share); replaced by the LOEO argmin after the run
  profile_model: Ensemble # which profile candidate is pooled with Elo
```

### 5.4 `tests/`
- Extend [`test_no_leakage.py`](../tests/test_no_leakage.py): the blend introduces no new leakage (it
  consumes only already-leakage-free fold outputs; the nested weight uses inner editions only).
- New `test_blend.py`: `linear_pool` of two valid prob matrices is valid (rows sum to 1, non-negative);
  the ambiguity property holds (pooled RPS ≤ weighted-avg of the two members' RPS).

### 5.5 Report + memory
- Update the Phase-3 report HTML and the `project-wc2026-phase3` memory: gate still honestly **FAILS**
  (pure model); product blend serves at **nested RPS X** (target `< 0.1802`); record combiner + `w*` +
  error-correlation.

---

## 6. Acceptance criteria

1. **Precondition:** error-correlation is low (errors decorrelated).
2. **Beats Elo where it counts:** **nested** blend RPS `< 0.1802` **and** nested log-loss `<` Elo's —
   at the validated weight, not the swept optimum.
3. **Robust, not knife-edge:** the LOEO sweep shows the blend beating Elo across a broad `w` band.
4. **Served combiner = nested winner;** blend stays calibrated (reliability check; recalibrate only if
   it drifts).
5. **Gate integrity:** `gate_verdict` on the pure profile model is unchanged (still an honest FAIL).

---

## 7. Build sequence (checklist)

1. [ ] §5.0 precondition diagnostic (error-correlation). Stop if high.
2. [ ] `src/models/blend.py` pure helpers + `tests/test_blend.py`.
3. [ ] Wire `Blend (Elo+profile)` into `evaluate.py` (summary + report), gate untouched.
4. [ ] LOEO sweep + nested eval + stacker bake-off; pick combiner & `w*`.
5. [ ] Write `w*`/combiner into `config.yaml`.
6. [ ] Extend `test_no_leakage.py`; run `pytest tests/`.
7. [ ] Re-run `python -m src.models.evaluate`; capture numbers.
8. [ ] Update Phase-3 report HTML + `project-wc2026-phase3` memory.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Errors too correlated → blend is a no-op | §5.0 precondition gate; stop early and report. |
| Weight overfit to held-out | Nested metric is the headline; served `w*` is one scalar (≈1 DoF). |
| Stacker overfits the meta-layer on small N | Bake-off; serve it only if nested RPS actually wins. |
| Blend silently treated as a gate pass | `PRODUCT_MODELS` separate from `PROFILE_MODELS`; gate logic untouched; row tagged `[product]`. |
| FC26 lacks reputation/potential attrs (known Phase-4 caveat) | Affects the profile member, not the blend mechanism; tracked in memory. |

---

## 9. Hand-off to Phase 4 (serve path — not built here)

`src/predict/` is currently empty. When Phase 4 builds it, the served blend **reuses the exact same
code paths** (no train/serve skew): `compute_elo_features` over all matches through 2026 → pre-match
`elo_diff` for the 104 fixtures → profile model via `build_features` →
`linear_pool(P_profile, P_elo, config.blend.weight)` (or the locked stacker) → `predictions` table.
The whole point of locking `w*`/combiner now is that Phase-4 serving consumes them unchanged.
