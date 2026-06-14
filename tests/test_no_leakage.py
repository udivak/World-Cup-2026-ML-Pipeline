"""No-leakage enforcement (Phase 3, critical).

Every player attribute / squad snapshot feeding a match must be dated **before** the match.
Two layers are checked:

* pure: :func:`assign_editions` never attaches an edition whose ``snapshot_date`` is in the
  match's future, so the joined profile is always a pre-match input;
* live (DB-gated): every ``team_profiles`` row uses only the nearest-prior FIFA edition (never
  one released after the tournament), a 2018 WC squad uses FIFA 18 (not 19), and every
  materialized ``match_features`` row joins profiles dated on/before the match.
"""

import os

import pandas as pd
import pytest

from src.features.labels import assign_editions

_DB = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set — skipping live DB test"
)


def test_assign_editions_never_attaches_a_future_snapshot():
    editions = pd.DataFrame(
        [
            {"tournament": "FIFA World Cup", "edition_year": 2018, "snapshot_date": "2018-06-14"},
            {"tournament": "FIFA World Cup", "edition_year": 2022, "snapshot_date": "2022-11-20"},
        ]
    )
    matches = pd.DataFrame(
        [{"date": "2018-06-20", "home_team": "A", "away_team": "B",
          "tournament": "FIFA World Cup", "result": "H"}]
    )
    out = assign_editions(matches, editions)
    assert int(out.iloc[0]["edition_year"]) == 2018  # the future 2022 edition is ignored
    snap = pd.to_datetime(out.iloc[0]["snapshot_date"])
    assert snap <= pd.to_datetime(out.iloc[0]["date"])


@_DB
def test_profiles_use_only_nearest_prior_fifa_edition():
    from src.aggregate.team_profile import target_edition
    from src.common.io import read_table

    profiles = read_table("team_profiles")
    profiles = profiles[profiles["fifa_edition"].notna() & profiles["snapshot_date"].notna()]
    assert len(profiles) > 0
    for _, r in profiles.iterrows():
        snap = pd.to_datetime(r["snapshot_date"]).date()
        assert int(r["fifa_edition"]) <= target_edition(snap), (
            f"{r['team']} {r['tournament']} {r['edition_year']} uses FIFA "
            f"{r['fifa_edition']} released after its {snap} snapshot"
        )


@_DB
def test_wc2018_squads_use_fifa18_not_later():
    from src.common.io import read_table

    profiles = read_table("team_profiles")
    wc18 = profiles[(profiles["tournament"] == "FIFA World Cup") & (profiles["edition_year"] == 2018)]
    assert len(wc18) > 0
    assert int(wc18["fifa_edition"].dropna().max()) <= 2018


@_DB
def test_match_features_join_is_strictly_pre_match():
    from src.common.io import read_table

    mf = read_table("match_features")
    if mf.empty:
        pytest.skip("match_features not materialized")
    tp = read_table("team_profiles")[["team", "tournament", "edition_year", "snapshot_date"]]
    mf["date"] = pd.to_datetime(mf["date"])
    for side in ("home_team", "away_team"):
        joined = mf.merge(
            tp.rename(columns={"team": side, "snapshot_date": "snap"}),
            on=["tournament", "edition_year", side], how="left",
        )
        snap = pd.to_datetime(joined["snap"])
        assert (snap <= joined["date"]).all(), f"a {side} profile is dated after its match"
