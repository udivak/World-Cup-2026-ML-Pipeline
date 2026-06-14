from datetime import date

import pandas as pd

from src.aggregate.team_profile import (
    _is_top5,
    _nearest_prior,
    build_profile_row,
    target_edition,
)


# ---------------------------------------------------- nearest-prior / no leakage
def test_target_edition_release_timing():
    # FIFA N releases ~30 Sep of N-1, so a summer 2018 event maps to FIFA 18.
    assert target_edition(date(2018, 6, 14)) == 2018
    # Euro 2020 was played in June 2021 → FIFA 21.
    assert target_edition(date(2021, 6, 11)) == 2021
    # A late-November 2022 event is after FIFA 23's release → FIFA 23.
    assert target_edition(date(2022, 11, 20)) == 2023
    # Boundary: on/after 30 Sep flips to next edition.
    assert target_edition(date(2017, 9, 30)) == 2018
    assert target_edition(date(2017, 9, 29)) == 2017


def test_nearest_prior_picks_latest_not_after_target():
    rows = [
        {"season_year": 2018, "overall": 80},
        {"season_year": 2020, "overall": 85},
        {"season_year": 2022, "overall": 90},
    ]
    assert _nearest_prior(rows, 2021)["season_year"] == 2020  # never reaches FIFA 22
    assert _nearest_prior(rows, 2023)["season_year"] == 2022
    assert _nearest_prior(rows, 2017) is None
    assert _nearest_prior(None, 2020) is None


def test_is_top5_league_matching():
    assert _is_top5("English Premier League")
    assert _is_top5("Spain Primera Division")
    assert _is_top5("Italian Serie A")
    assert _is_top5("German 1. Bundesliga")
    assert _is_top5("French Ligue 1")
    assert not _is_top5("Saudi Pro League")
    assert not _is_top5(None)


# --------------------------------------------------------- profile aggregation
def _attr_index():
    return {
        1: [
            {"season_year": 2020, "overall": 90, "value": 1_000_000,
             "league": "English Premier League", "age": 28, "positions": "GK",
             "composure": "80"},
            # A *later* edition that must NOT be used for a 2021 snapshot (leakage guard).
            {"season_year": 2022, "overall": 99, "value": 9_000_000,
             "league": "X", "age": 30, "positions": "GK", "composure": "99"},
        ],
        2: [
            {"season_year": 2020, "overall": 85, "value": 500_000,
             "league": "Spain Primera Division", "age": 25, "positions": "CB",
             "composure": "70"},
        ],
    }


def _roster():
    return pd.DataFrame(
        [
            {"player_id": 1, "position": "GK", "dob": "1992-09-02", "caps": 50},
            {"player_id": 2, "position": "DF", "dob": "1996-01-01", "caps": 20},
            # Unmatched roster spelling: kept, but contributes no attributes.
            {"player_id": None, "position": "FW", "dob": "2000-01-01", "caps": None},
        ]
    )


def test_build_profile_row_aggregates_and_respects_no_leakage():
    formation = {"gk": 1, "def": 4, "mid": 3, "att": 3}
    profile = build_profile_row(
        _roster(), _attr_index(), date(2021, 6, 11), formation, substitutes=15
    )
    # Player 1's 2022 edition (overall 99) is released after the snapshot → ignored.
    assert profile["fifa_edition"] == 2020
    assert profile["gk_strength"] == 90
    assert profile["att_strength"] is None  # the only attacker is unmatched (no overall)
    assert profile["def_strength"] == 85
    assert profile["overall_xi"] == 87.5   # mean of 90 and 85 (unmatched FW has no overall)
    assert profile["star_power"] == 87.5
    assert profile["squad_size"] == 3
    assert profile["matched_players"] == 2
    assert profile["total_caps"] == 70
    assert profile["top5_league_share"] == 1.0  # both matched players in a top-5 league
    assert profile["mean_composure"] == 75.0


def test_build_profile_row_aggregates_reputation_and_potential():
    # Enrichment: mean intl-reputation / potential over the XI, and the count of world-class
    # players (international_reputation >= 4) in the squad.
    attr_index = {
        1: [{"season_year": 2018, "overall": 90, "potential": 92, "value": 1_000_000,
             "league": "X", "age": 28, "positions": "ST", "composure": "80", "reputation": "5"}],
        2: [{"season_year": 2018, "overall": 84, "potential": 85, "value": 500_000,
             "league": "Y", "age": 30, "positions": "CB", "composure": "70", "reputation": "4"}],
        3: [{"season_year": 2018, "overall": 78, "potential": 80, "value": 100_000,
             "league": "Z", "age": 24, "positions": "GK", "composure": "65", "reputation": "2"}],
    }
    roster = pd.DataFrame(
        [
            {"player_id": 1, "position": "ST", "dob": "1990-01-01", "caps": 50},
            {"player_id": 2, "position": "CB", "dob": "1988-01-01", "caps": 40},
            {"player_id": 3, "position": "GK", "dob": "1994-01-01", "caps": 10},
        ]
    )
    formation = {"gk": 1, "def": 4, "mid": 3, "att": 3}
    profile = build_profile_row(roster, attr_index, date(2018, 6, 14), formation, substitutes=15)
    assert profile["elite_count"] == 2          # reputations 5 and 4 are >= 4
    assert profile["mean_intl_rep"] == 3.67     # mean(5,4,2)
    assert profile["mean_potential"] == 85.67   # mean(92,85,80)


def test_build_profile_row_handles_empty_attributes():
    formation = {"gk": 1, "def": 4, "mid": 3, "att": 3}
    roster = pd.DataFrame(
        [{"player_id": 99, "position": "GK", "dob": "1990-01-01", "caps": 5}]
    )
    profile = build_profile_row(roster, {}, date(2018, 6, 14), formation, substitutes=15)
    assert profile["matched_players"] == 0
    assert profile["overall_xi"] is None
    assert profile["fifa_edition"] is None
    assert profile["total_caps"] == 5
