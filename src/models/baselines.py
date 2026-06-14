"""Reference baselines the profile model must beat (Phase 3).

Two bars, **reference only** — neither is ever a feature in the profile model:

* **Elo-only.** A chronological football-Elo is rolled over the *entire* match history (friendlies,
  qualifiers, finals — Elo needs the continuous record), recording each match's **pre-match**
  ``elo_diff = home_elo − away_elo (+ home advantage if not neutral)``. A multinomial logit then
  maps ``elo_diff`` → 3-way probabilities. Elo is the classic team-identity bar.
* **Squad-overall difference.** A logit on the single ``diff_overall_xi`` (team1 − team2 mean XI
  rating) plus the shared home-advantage context — the naive *bottom-up* bar.

The Elo pass uses only results that precede each match, so ``elo_diff`` is leakage-free.
``compute_elo_features`` is pure (matches in, ratings out) and fixture-testable.
"""

import logging

import numpy as np
import pandas as pd

from src.models.train import build_logreg

logger = logging.getLogger(__name__)

ELO_FEATURES = ["elo_diff"]
SQUAD_OVERALL_FEATURES = ["diff_overall_xi", "home_adv"]


def _mov_multiplier(goal_diff: float) -> float:
    """World-Football-Elo margin-of-victory multiplier."""
    g = abs(int(goal_diff))
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    if g == 3:
        return 1.75
    return 1.75 + (g - 3) / 8.0


def compute_elo_features(
    matches: pd.DataFrame,
    k: float = 32.0,
    home_advantage: float = 100.0,
    mov: bool = True,
    base: float = 1500.0,
) -> pd.DataFrame:
    """Chronological Elo over all matches → per-match pre-match ``elo_diff`` (no leakage).

    Returns ``[date, home_team, away_team, elo_diff, home_elo, away_elo]`` for every input match.
    Ratings update only on rows with a known ``result``; rows without scores skip the MoV bonus.
    """
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date"], kind="mergesort").reset_index(drop=True)

    home = df["home_team"].to_numpy()
    away = df["away_team"].to_numpy()
    res = df["result"].to_numpy()
    hs = pd.to_numeric(df.get("home_score"), errors="coerce").to_numpy(dtype=float)
    as_ = pd.to_numeric(df.get("away_score"), errors="coerce").to_numpy(dtype=float)
    neutral = df["neutral"].fillna(False).to_numpy() if "neutral" in df else np.zeros(len(df), bool)

    rating: dict = {}
    elo_diff = np.full(len(df), np.nan)
    home_elo = np.full(len(df), np.nan)
    away_elo = np.full(len(df), np.nan)

    for i in range(len(df)):
        rh = rating.get(home[i], base)
        ra = rating.get(away[i], base)
        adv = 0.0 if bool(neutral[i]) else home_advantage
        ed = rh - ra + adv
        elo_diff[i], home_elo[i], away_elo[i] = ed, rh, ra

        r = res[i]
        if r not in ("H", "D", "A"):
            continue  # future/void fixture: record pre-match Elo, don't update
        exp_h = 1.0 / (1.0 + 10 ** (-ed / 400.0))
        act_h = 1.0 if r == "H" else (0.5 if r == "D" else 0.0)
        mult = 1.0
        if mov and not np.isnan(hs[i]) and not np.isnan(as_[i]):
            mult = _mov_multiplier(hs[i] - as_[i])
        change = k * mult * (act_h - exp_h)
        rating[home[i]] = rh + change
        rating[away[i]] = ra - change

    df["elo_diff"] = elo_diff
    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    return df[["date", "home_team", "away_team", "elo_diff", "home_elo", "away_elo"]]


def attach_elo(features: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """Left-join ``elo_diff`` onto a feature frame by ``(date, home_team, away_team)``."""
    f = features.copy()
    f["date"] = pd.to_datetime(f["date"])
    elo = elo.copy()
    elo["date"] = pd.to_datetime(elo["date"])
    return f.merge(elo, on=["date", "home_team", "away_team"], how="left")


def build_elo_baseline(calibrated: bool = True):
    """Multinomial logit on ``elo_diff`` (reference baseline; calibrated like the contenders)."""
    return build_logreg(C=1.0, calibrated=calibrated)


def build_squad_overall_baseline(calibrated: bool = True):
    """Multinomial logit on ``diff_overall_xi`` + home advantage (naive bottom-up bar)."""
    return build_logreg(C=1.0, calibrated=calibrated)
