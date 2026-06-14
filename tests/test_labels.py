import pandas as pd

from src.features.labels import assign_editions


def _editions():
    return pd.DataFrame(
        [
            {"tournament": "FIFA World Cup", "edition_year": 2018, "snapshot_date": "2018-06-14"},
            {"tournament": "FIFA World Cup", "edition_year": 2022, "snapshot_date": "2022-11-20"},
            {"tournament": "UEFA Euro", "edition_year": 2020, "snapshot_date": "2021-06-11"},
        ]
    )


def test_assign_editions_maps_matches_within_window():
    matches = pd.DataFrame(
        [
            {"date": "2018-06-20", "home_team": "Brazil", "away_team": "Switzerland",
             "tournament": "FIFA World Cup", "result": "D"},
            {"date": "2018-07-15", "home_team": "France", "away_team": "Croatia",
             "tournament": "FIFA World Cup", "result": "H"},  # final, ~31 days in
        ]
    )
    out = assign_editions(matches, _editions())
    assert len(out) == 2
    assert set(out["edition_year"].astype(int)) == {2018}


def test_assign_editions_drops_out_of_window_and_untracked():
    matches = pd.DataFrame(
        [
            {"date": "2018-03-01", "home_team": "X", "away_team": "Y",
             "tournament": "FIFA World Cup", "result": "H"},   # before the window
            {"date": "2018-06-20", "home_team": "A", "away_team": "B",
             "tournament": "Friendly", "result": "H"},          # untracked tournament
            {"date": "2021-06-20", "home_team": "Italy", "away_team": "Wales",
             "tournament": "UEFA Euro", "result": "H"},         # Euro 2020, played 2021
        ]
    )
    out = assign_editions(matches, _editions())
    assert len(out) == 1
    assert out.iloc[0]["tournament"] == "UEFA Euro"
    assert int(out.iloc[0]["edition_year"]) == 2020


def test_assign_editions_picks_the_right_edition_not_a_distant_one():
    matches = pd.DataFrame(
        [
            {"date": "2022-11-25", "home_team": "Argentina", "away_team": "Mexico",
             "tournament": "FIFA World Cup", "result": "H"},
        ]
    )
    out = assign_editions(matches, _editions())
    assert int(out.iloc[0]["edition_year"]) == 2022


def test_assign_editions_empty_inputs():
    empty = pd.DataFrame(columns=["date", "home_team", "away_team", "tournament", "result"])
    assert assign_editions(empty, _editions()).empty
    matches = pd.DataFrame(
        [{"date": "2018-06-20", "home_team": "A", "away_team": "B",
          "tournament": "FIFA World Cup", "result": "H"}]
    )
    assert assign_editions(matches, _editions().iloc[0:0]).empty
