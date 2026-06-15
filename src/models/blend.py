"""Blended product model — Elo + bottom-up profile (Phase 3 follow-on).

The Phase-3 *gate* asks whether the player-profile model alone beats team-identity Elo. It does
not (yet): the enriched profile model closed ~60% of the gap but still loses. The residual gap is
plausibly **form-shaped** — Elo encodes results-based, time-evolving signal (form, momentum,
cohesion) that the profile model is, by design, blind to. This module **blends the two models'
outputs** to add that missing ingredient back.

**This is a product decision, not a gate pass.** Folding Elo in violates the non-negotiable "Elo is
never a model feature" principle, so the blend can never count as a bottom-up gate pass — it is
reported separately and tagged ``[product]``. The pure model remains the honest research result.

Everything here is **pure** (probability matrices in, numbers/matrices out) and fixture-testable,
mirroring :func:`src.models.baselines.compute_elo_features`. We pool the **outputs** of two
already-leakage-free, already-calibrated candidates (``Ensemble`` profile and ``Elo-only``) that the
across-edition backtest computes per fold — no new features, no new training, no schema change.

Two combiners are offered and chosen by validation (the bake-off in :mod:`src.models.evaluate`):

* **Linear pool** — ``w · p_profile + (1 − w) · p_elo``; fits a single global scalar ``w`` (≈1 DoF,
  negligible overfit on ~700 matches), usually the best out-of-sample bet on small data.
* **Logistic stacker** — a calibrated, regularized meta-logit on the two models' probabilities; more
  expressive but can overfit the meta-layer.

Leakage discipline (see the nested protocol in :func:`nested_blend_eval`): the *served* weight uses
all history to set one scalar, while the *reported* metric never lets the combiner see the edition
it is scored on (it is fit on strictly-prior editions only).
"""

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from src.models.metrics import (
    CLASSES,
    multiclass_log_loss,
    ranked_probability_score,
    rps_vector,
)
from src.models.train import build_logreg

logger = logging.getLogger(__name__)

# Default weight grid for the LOEO sweep: profile share w from all-Elo (0) to all-profile (1).
WEIGHT_STEP = 0.05
# Outer folds with fewer than this many strictly-prior inner editions can't tune a weight honestly.
MIN_INNER_EDITIONS = 2


def _as_prob_matrix(p) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(CLASSES):
        raise ValueError(f"expected an (n, {len(CLASSES)}) probability matrix, got shape {arr.shape}")
    return arr


def linear_pool(p_a: np.ndarray, p_b: np.ndarray, w: float) -> np.ndarray:
    """Convex combination ``w · p_a + (1 − w) · p_b`` (``p_a`` = profile, ``p_b`` = Elo).

    ``w`` is the **profile share** in ``[0, 1]``. A convex mix of two valid probability matrices is
    itself valid (non-negative, rows sum to 1); we assert that to catch malformed inputs early.
    """
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"weight w must be in [0, 1], got {w}")
    a, b = _as_prob_matrix(p_a), _as_prob_matrix(p_b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    out = w * a + (1.0 - w) * b
    assert np.all(out >= -1e-9), "linear pool produced a negative probability"
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6), "linear pool rows do not sum to 1"
    return out


def error_correlation(p_a: np.ndarray, p_b: np.ndarray, y: Sequence[str]) -> float:
    """Pearson correlation of the two models' **per-match RPS errors** — the §5.0 precondition.

    Low correlation means the models miss on *different* matches, so averaging cancels uncorrelated
    error (the mechanism that lets a blend beat both members). High correlation (``> ~0.8``) means
    the errors move together and blending is close to a no-op. Returns 0.0 if either error vector is
    constant (correlation undefined).
    """
    ea = rps_vector(y, _as_prob_matrix(p_a))
    eb = rps_vector(y, _as_prob_matrix(p_b))
    if ea.std() < 1e-12 or eb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ea, eb)[0, 1])


def blend_weight_sweep(
    p_a: np.ndarray, p_b: np.ndarray, y: Sequence[str], step: float = WEIGHT_STEP
) -> pd.DataFrame:
    """LOEO curve: pooled RPS and log-loss of the linear pool across ``w ∈ {0, step, …, 1}``.

    Diagnostic — used to confirm the blend beats Elo across a **broad band** of ``w`` (robustness),
    not at a single knife-edge value.
    """
    ws = np.round(np.arange(0.0, 1.0 + 1e-9, step), 6)
    rows = []
    for w in ws:
        p = linear_pool(p_a, p_b, float(w))
        rows.append(
            {
                "w": float(w),
                "rps": ranked_probability_score(y, p),
                "log_loss": multiclass_log_loss(y, p),
            }
        )
    return pd.DataFrame(rows)


def select_weight_loeo(
    p_a: np.ndarray, p_b: np.ndarray, y: Sequence[str], step: float = WEIGHT_STEP
) -> float:
    """Global served weight ``w*`` = ``argmin`` pooled RPS over the sweep (one robust scalar)."""
    sweep = blend_weight_sweep(p_a, p_b, y, step=step)
    return float(sweep.loc[sweep["rps"].idxmin(), "w"])


def stack_features(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Meta-features for the stacker: the two models' W/D/L probabilities side by side ``(n, 6)``."""
    return np.hstack([_as_prob_matrix(p_a), _as_prob_matrix(p_b)])


def fit_stacker(p_a: np.ndarray, p_b: np.ndarray, y: Sequence[str], C: float = 1.0):
    """Calibrated, regularized logistic meta-combiner on the two models' out-of-fold probabilities.

    Reuses :func:`src.models.train.build_logreg` (impute → scale → multinomial logit, wrapped in
    probability calibration), so the stacker is regularized and calibrated exactly like the base
    candidates. Returns the fitted estimator; pair with :func:`stacker_proba` to predict.
    """
    est = build_logreg(C=C, calibrated=True)
    est.fit(stack_features(p_a, p_b), list(y))
    return est


def stacker_proba(est, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Predict with a fitted stacker, reordered to the ordinal :data:`CLASSES` column order."""
    proba = est.predict_proba(stack_features(p_a, p_b))
    col = {c: i for i, c in enumerate(est.classes_)}
    out = np.zeros((proba.shape[0], len(CLASSES)), dtype=float)
    for j, c in enumerate(CLASSES):
        if c in col:
            out[:, j] = proba[:, col[c]]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


def nested_blend_eval(
    folds: list,
    y_full: Sequence[str],
    combiner: str = "linear",
    step: float = WEIGHT_STEP,
) -> dict:
    """Honest **nested across-edition** estimate of what the blend delivers out-of-sample.

    ``folds`` is a list of per-edition dicts, each holding the held-out predictions for one edition::

        {"edition": str, "order": sortable, "idx": np.ndarray,  # row positions in y_full
         "p_a": (n, 3) profile proba, "p_b": (n, 3) Elo proba, "y": [labels]}

    For each outer edition (chronological), the combiner is fit on the **inner** editions only
    (those with a strictly-earlier ``order``) and applied to the outer edition. The combiner never
    sees the edition it scores, and the base predictions are themselves leakage-free, so the pooled
    result is an unbiased estimate. Outer folds with ``< MIN_INNER_EDITIONS`` inner editions fall
    back to a blind ``w = 0.5`` linear pool. Any per-fold combiner failure also degrades to that
    blind pool (logged), so the run never aborts.

    Returns ``{combiner, rps, log_loss, n, pred (aligned to y_full), folds: [per-fold info]}``.
    """
    folds_sorted = sorted(folds, key=lambda f: f["order"])
    n_total = len(y_full)
    pred = np.full((n_total, len(CLASSES)), np.nan)
    info = []

    for i, outer in enumerate(folds_sorted):
        inner = folds_sorted[:i]
        used = combiner
        w = None
        if len(inner) < MIN_INNER_EDITIONS:
            p = linear_pool(outer["p_a"], outer["p_b"], 0.5)
            used, w = "blind(0.5)", 0.5
        else:
            ia = np.vstack([f["p_a"] for f in inner])
            ib = np.vstack([f["p_b"] for f in inner])
            iy = [lab for f in inner for lab in f["y"]]
            try:
                if combiner == "stacker":
                    est = fit_stacker(ia, ib, iy)
                    p = stacker_proba(est, outer["p_a"], outer["p_b"])
                else:
                    w = select_weight_loeo(ia, ib, iy, step=step)
                    p = linear_pool(outer["p_a"], outer["p_b"], w)
            except Exception as exc:  # never abort the backtest on one fold
                logger.warning("Blend combiner '%s' failed on %s (%s); falling back to w=0.5.",
                               combiner, outer["edition"], exc)
                p = linear_pool(outer["p_a"], outer["p_b"], 0.5)
                used, w = "fallback(0.5)", 0.5
        pred[outer["idx"]] = p
        info.append({"edition": outer["edition"], "n_inner_editions": len(inner),
                     "combiner": used, "w": w, "n": len(outer["idx"])})

    if np.isnan(pred).any():  # every tested row belongs to exactly one fold; guard anyway
        raise AssertionError("nested blend left some rows unpredicted — folds do not cover y_full")

    return {
        "combiner": combiner,
        "rps": ranked_probability_score(y_full, pred),
        "log_loss": multiclass_log_loss(y_full, pred),
        "n": n_total,
        "pred": pred,
        "folds": info,
    }
