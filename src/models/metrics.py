"""Scoring metrics for 3-way W/D/L prediction (Phase 3).

**RPS (Ranked Probability Score)** is the primary metric: unlike log-loss/accuracy it respects
the *ordering* of the outcomes (a team1-win predicted as a draw is a smaller error than predicted
as a team1-loss). Classes are ordinal ``[H, D, A]`` = team1-win → draw → team1-loss. Lower is
better for RPS and log-loss; higher for accuracy.
"""

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ordinal outcome order from team1's (home) perspective: win, draw, loss.
CLASSES = ["H", "D", "A"]


def _one_hot(y_true: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    oh = np.zeros((len(y_true), len(classes)), dtype=float)
    for i, v in enumerate(y_true):
        oh[i, idx[v]] = 1.0
    return oh


def ranked_probability_score(
    y_true: Sequence[str], probs: np.ndarray, classes: Sequence[str] = CLASSES
) -> float:
    """Mean RPS over rows. ``probs`` is ``(n, K)`` in ``classes`` order; ``y_true`` are labels.

    RPS = 1/(K−1) · Σ_{k=1}^{K−1} (CDF_pred(k) − CDF_obs(k))²  (the K-th cumulative term is 0).
    """
    probs = np.asarray(probs, dtype=float)
    obs = _one_hot(y_true, classes)
    cum_p = np.cumsum(probs, axis=1)[:, :-1]
    cum_o = np.cumsum(obs, axis=1)[:, :-1]
    per_row = np.sum((cum_p - cum_o) ** 2, axis=1) / (probs.shape[1] - 1)
    return float(np.mean(per_row))


def rps_vector(
    y_true: Sequence[str], probs: np.ndarray, classes: Sequence[str] = CLASSES
) -> np.ndarray:
    """Per-match RPS (for calibration/inspection)."""
    probs = np.asarray(probs, dtype=float)
    obs = _one_hot(y_true, classes)
    cum_p = np.cumsum(probs, axis=1)[:, :-1]
    cum_o = np.cumsum(obs, axis=1)[:, :-1]
    return np.sum((cum_p - cum_o) ** 2, axis=1) / (probs.shape[1] - 1)


def multiclass_log_loss(
    y_true: Sequence[str], probs: np.ndarray, classes: Sequence[str] = CLASSES, eps: float = 1e-15
) -> float:
    """Mean cross-entropy. Computed directly against the ``classes`` column order so it cannot be
    confused by scikit-learn's assumption that ``y_prob`` columns follow lexicographic label order.
    """
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    obs = _one_hot(y_true, classes)
    return float(-np.mean(np.sum(obs * np.log(probs), axis=1)))


def score_all(
    y_true: Sequence[str], probs: np.ndarray, classes: Sequence[str] = CLASSES
) -> dict:
    """RPS (primary), log-loss, accuracy, and macro precision/recall in one call."""
    probs = np.asarray(probs, dtype=float)
    preds = [classes[i] for i in probs.argmax(axis=1)]
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, preds, labels=list(classes), average="macro", zero_division=0
    )
    return {
        "rps": ranked_probability_score(y_true, probs, classes),
        "log_loss": multiclass_log_loss(y_true, probs, classes),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "n": int(len(y_true)),
    }


def per_class_precision_recall(
    y_true: Sequence[str], probs: np.ndarray, classes: Sequence[str] = CLASSES
) -> dict:
    preds = [classes[i] for i in np.asarray(probs).argmax(axis=1)]
    prec, rec, _, support = precision_recall_fscore_support(
        y_true, preds, labels=list(classes), average=None, zero_division=0
    )
    return {
        c: {"precision": float(prec[i]), "recall": float(rec[i]), "support": int(support[i])}
        for i, c in enumerate(classes)
    }
