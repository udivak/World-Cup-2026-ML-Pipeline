import numpy as np
import pandas as pd

from src.models.baselines import _mov_multiplier, attach_elo, compute_elo_features


def test_mov_multiplier_steps():
    assert _mov_multiplier(0) == 1.0
    assert _mov_multiplier(1) == 1.0
    assert _mov_multiplier(2) == 1.5
    assert _mov_multiplier(3) == 1.75
    assert _mov_multiplier(5) == 1.75 + 2 / 8.0  # 4+ grows


def test_compute_elo_features_is_pre_match_and_updates():
    matches = pd.DataFrame(
        [
            {"date": "2018-06-01", "home_team": "A", "away_team": "B",
             "home_score": 1, "away_score": 0, "neutral": False, "result": "H"},
            {"date": "2018-06-05", "home_team": "B", "away_team": "A",
             "home_score": 0, "away_score": 0, "neutral": True, "result": "D"},
        ]
    )
    elo = compute_elo_features(matches, k=32, home_advantage=100, mov=True).sort_values("date")
    first, second = elo.iloc[0], elo.iloc[1]
    # Match 1: both start at 1500, home advantage 100 -> elo_diff exactly 100 (pre-match, no leakage).
    assert abs(first["elo_diff"] - 100.0) < 1e-9
    assert first["home_elo"] == 1500.0 and first["away_elo"] == 1500.0
    # After A beats B, A's rating rises above B's. Match 2 is at a neutral venue (no home adv),
    # so its elo_diff (B − A) must be negative.
    assert second["elo_diff"] < 0


def test_compute_elo_features_neutral_has_no_home_advantage():
    matches = pd.DataFrame(
        [{"date": "2018-06-01", "home_team": "A", "away_team": "B",
          "home_score": np.nan, "away_score": np.nan, "neutral": True, "result": None}]
    )
    elo = compute_elo_features(matches, home_advantage=100)
    assert elo.iloc[0]["elo_diff"] == 0.0  # equal ratings, neutral -> no advantage


def test_attach_elo_left_joins_on_match_key():
    feats = pd.DataFrame(
        [{"date": pd.Timestamp("2018-06-01"), "home_team": "A", "away_team": "B", "result": "H"}]
    )
    elo = pd.DataFrame(
        [{"date": pd.Timestamp("2018-06-01"), "home_team": "A", "away_team": "B",
          "elo_diff": 42.0, "home_elo": 1521.0, "away_elo": 1479.0}]
    )
    merged = attach_elo(feats, elo)
    assert merged.iloc[0]["elo_diff"] == 42.0
