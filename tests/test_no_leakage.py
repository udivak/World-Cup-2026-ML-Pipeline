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

import numpy as np
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


def test_nested_blend_weight_uses_only_strictly_prior_editions():
    """The product blend introduces no new leakage: the combiner weight for an outer edition is
    chosen on **strictly-prior** inner editions only, never peeking at the edition it scores.

    Construct a worst-case probe. The two inner editions are perfectly predicted by Elo (``p_b``)
    and badly by the profile (``p_a``), so the honest inner sweep picks ``w*=0`` (all-Elo). The
    outer edition is the reverse — profile is perfect, Elo is wrong — so a *leaky* combiner that
    peeked at the outer edition would pick ``w=1`` (all-profile). A leakage-free nested eval must
    apply the inner-derived ``w*=0`` to the outer edition, i.e. its prediction equals the Elo
    probabilities, not the profile ones.
    """
    from src.models.blend import nested_blend_eval

    elo_perfect, profile_perfect = [[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]
    wrong = [[0.0, 0.0, 1.0]]  # confident, wrong (actual is always "H")
    y = ["H", "H", "H"]
    folds = [
        # inner: Elo perfect, profile wrong -> inner sweep favours all-Elo (w*=0)
        {"edition": "E1", "order": 1, "idx": np.array([0]), "p_a": np.array(wrong),
         "p_b": np.array(elo_perfect), "y": ["H"]},
        {"edition": "E2", "order": 2, "idx": np.array([1]), "p_a": np.array(wrong),
         "p_b": np.array(elo_perfect), "y": ["H"]},
        # outer: profile perfect, Elo wrong -> a leaky combiner would pick w=1 here
        {"edition": "E3", "order": 3, "idx": np.array([2]), "p_a": np.array(profile_perfect),
         "p_b": np.array(wrong), "y": ["H"]},
    ]
    res = nested_blend_eval(folds, y, combiner="linear")
    outer_pred = res["pred"][2]
    # Used the inner-derived all-Elo weight on the outer edition (== Elo proba, the wrong one here).
    assert np.allclose(outer_pred, wrong[0]), "nested blend weight leaked the outer edition"
    # And explicitly NOT the outer-optimal profile prediction.
    assert not np.allclose(outer_pred, profile_perfect[0])
    e3 = next(f for f in res["folds"] if f["edition"] == "E3")
    assert e3["n_inner_editions"] == 2 and e3["w"] == 0.0


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
