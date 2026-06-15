"""Blend helpers (Phase 3 follow-on, product model).

Pure, fixture-backed — no DB. Covers the two properties the design leans on:
* :func:`linear_pool` of two valid probability matrices is itself valid (non-negative, rows sum 1);
* the **ambiguity** property — a convex pool's RPS never exceeds the weighted average of the
  members' RPS (Krogh–Vedelsby / Jensen on the convex RPS), so the blend always beats the *average*
  member. Beating the *best* member additionally needs decorrelated errors (checked in the backtest).
"""

import numpy as np

from src.models.blend import (
    blend_weight_sweep,
    error_correlation,
    fit_stacker,
    linear_pool,
    nested_blend_eval,
    select_weight_loeo,
    stacker_proba,
)
from src.models.metrics import ranked_probability_score


def _two_models():
    # Two distinct, valid W/D/L probability matrices over [H, D, A] and the labels.
    p_a = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.1, 0.2, 0.7]])
    p_b = np.array([[0.4, 0.4, 0.2], [0.3, 0.3, 0.4], [0.2, 0.5, 0.3], [0.3, 0.3, 0.4]])
    y = ["H", "A", "D", "A"]
    return p_a, p_b, y


def test_linear_pool_is_a_valid_probability_matrix():
    p_a, p_b, _ = _two_models()
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = linear_pool(p_a, p_b, w)
        assert np.all(out >= 0.0)
        assert np.allclose(out.sum(axis=1), 1.0)
    # Endpoints recover the members exactly.
    assert np.allclose(linear_pool(p_a, p_b, 1.0), p_a)
    assert np.allclose(linear_pool(p_a, p_b, 0.0), p_b)


def test_linear_pool_rejects_out_of_range_weight():
    p_a, p_b, _ = _two_models()
    for bad in (-0.1, 1.1):
        try:
            linear_pool(p_a, p_b, bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for w={bad}")


def test_ambiguity_pooled_rps_beats_weighted_average_member():
    # Krogh–Vedelsby: RPS(blend) <= w·RPS(a) + (1−w)·RPS(b) for any convex w (RPS is convex).
    p_a, p_b, y = _two_models()
    rps_a = ranked_probability_score(y, p_a)
    rps_b = ranked_probability_score(y, p_b)
    for w in (0.2, 0.5, 0.8):
        blended = ranked_probability_score(y, linear_pool(p_a, p_b, w))
        weighted_avg = w * rps_a + (1.0 - w) * rps_b
        assert blended <= weighted_avg + 1e-12


def test_error_correlation_in_range_and_identical_models_are_perfectly_correlated():
    p_a, p_b, y = _two_models()
    c = error_correlation(p_a, p_b, y)
    assert -1.0 - 1e-9 <= c <= 1.0 + 1e-9
    # A model vs itself has identical per-match errors -> correlation 1.
    assert abs(error_correlation(p_a, p_a, y) - 1.0) < 1e-9


def test_sweep_and_select_weight_pick_the_sweep_argmin():
    p_a, p_b, y = _two_models()
    sweep = blend_weight_sweep(p_a, p_b, y, step=0.1)
    assert {"w", "rps", "log_loss"} <= set(sweep.columns)
    assert sweep["w"].min() == 0.0 and abs(sweep["w"].max() - 1.0) < 1e-9
    w_star = select_weight_loeo(p_a, p_b, y, step=0.1)
    assert abs(w_star - sweep.loc[sweep["rps"].idxmin(), "w"]) < 1e-12
    assert 0.0 <= w_star <= 1.0


def test_nested_blend_eval_covers_all_rows_and_falls_back_early():
    # Three editions; the first two outer folds have < 2 inner editions -> blind w=0.5.
    p_a, p_b, y = _two_models()  # 4 rows
    folds = [
        {"edition": "E1", "order": 1, "idx": np.array([0]), "p_a": p_a[:1], "p_b": p_b[:1], "y": y[:1]},
        {"edition": "E2", "order": 2, "idx": np.array([1]), "p_a": p_a[1:2], "p_b": p_b[1:2], "y": y[1:2]},
        {"edition": "E3", "order": 3, "idx": np.array([2, 3]), "p_a": p_a[2:], "p_b": p_b[2:], "y": y[2:]},
    ]
    res = nested_blend_eval(folds, y, combiner="linear", step=0.1)
    assert res["n"] == 4
    assert not np.isnan(res["pred"]).any()  # every row predicted
    by_ed = {f["edition"]: f for f in res["folds"]}
    assert by_ed["E1"]["combiner"] == "blind(0.5)"
    assert by_ed["E2"]["combiner"] == "blind(0.5)"
    assert by_ed["E3"]["n_inner_editions"] == 2  # E1+E2 are strictly prior


def test_stacker_round_trip_is_a_valid_probability_matrix():
    # Enough rows / class coverage for the calibrated meta-logit to fit.
    rng = np.random.RandomState(0)
    n = 60
    p_a = rng.dirichlet(np.ones(3), size=n)
    p_b = rng.dirichlet(np.ones(3), size=n)
    y = [["H", "D", "A"][i % 3] for i in range(n)]
    est = fit_stacker(p_a, p_b, y)
    out = stacker_proba(est, p_a, p_b)
    assert out.shape == (n, 3)
    assert np.all(out >= 0.0) and np.allclose(out.sum(axis=1), 1.0)
