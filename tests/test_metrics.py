import numpy as np

from src.models.metrics import (
    multiclass_log_loss,
    ranked_probability_score,
    rps_vector,
    score_all,
)

# Class order is ordinal [H, D, A] = team1-win, draw, team1-loss.


def test_rps_perfect_prediction_is_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert ranked_probability_score(["H", "A"], probs) == 0.0


def test_rps_known_value():
    # probs [0.6,0.3,0.1], actual H: CDF_pred=[.6,.9], CDF_obs=[1,1]
    # RPS = ((.6-1)^2 + (.9-1)^2)/2 = (0.16+0.01)/2 = 0.085
    probs = np.array([[0.6, 0.3, 0.1]])
    assert abs(ranked_probability_score(["H"], probs) - 0.085) < 1e-9


def test_rps_uniform():
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
    # actual D: CDF_pred=[.333,.667], CDF_obs=[0,1] -> (.111+.111)/2 = .111
    assert abs(ranked_probability_score(["D"], probs) - 1 / 9) < 1e-9


def test_rps_respects_ordinal_distance():
    # Actual is H. A draw-leaning wrong guess should score better than a loss-leaning one.
    near = np.array([[0.2, 0.7, 0.1]])   # mass on D (adjacent to H)
    far = np.array([[0.2, 0.1, 0.7]])    # mass on A (opposite of H)
    assert ranked_probability_score(["H"], near) < ranked_probability_score(["H"], far)


def test_rps_vector_matches_mean():
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    v = rps_vector(["H", "D"], probs)
    assert abs(v.mean() - ranked_probability_score(["H", "D"], probs)) < 1e-12


def test_log_loss_uses_correct_class_columns():
    # Confident and CORRECT 'H' (columns are [H, D, A]) must give -log(0.8), not -log(0.1).
    # Guards against scikit-learn's lexicographic-column assumption swapping H and A.
    probs = np.array([[0.8, 0.1, 0.1]])
    assert abs(multiclass_log_loss(["H"], probs) - (-np.log(0.8))) < 1e-9
    # A confident wrong prediction must score much worse than a confident right one.
    right = multiclass_log_loss(["H"], np.array([[0.8, 0.1, 0.1]]))
    wrong = multiclass_log_loss(["A"], np.array([[0.8, 0.1, 0.1]]))
    assert wrong > right


def test_log_loss_and_score_all_keys():
    probs = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    ll = multiclass_log_loss(["H", "A", "D"], probs)
    assert ll > 0
    s = score_all(["H", "A", "D"], probs)
    assert set(s) >= {"rps", "log_loss", "accuracy", "precision_macro", "recall_macro", "n"}
    assert s["n"] == 3
    assert 0.0 <= s["accuracy"] <= 1.0
