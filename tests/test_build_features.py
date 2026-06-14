import pandas as pd

from src.features.build_features import (
    FEATURE_COLUMNS,
    PROFILE_DIFF_COLS,
    assemble_features,
    infer_confederation_map,
)


def _profiles():
    def mk(team, **vals):
        row = {"team": team, "tournament": "FIFA World Cup", "edition_year": 2018,
               "matched_players": 20}
        row.update({c: 0.0 for c in PROFILE_DIFF_COLS})
        row.update(vals)
        return row

    return pd.DataFrame(
        [
            mk("Brazil", overall_xi=82.0, gk_strength=84.0, att_strength=86.0),
            mk("Switzerland", overall_xi=78.0, gk_strength=80.0, att_strength=79.0),
        ]
    )


def _labeled(neutral=True, away="Switzerland"):
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2018-06-17"), "tournament": "FIFA World Cup",
             "edition_year": 2018, "home_team": "Brazil", "away_team": away,
             "neutral": neutral, "result": "D"},
        ]
    )


def test_assemble_features_computes_diffs_and_context():
    out = assemble_features(_labeled(), _profiles(), conf_map={})
    row = out.iloc[0]
    assert row["diff_overall_xi"] == 4.0     # 82 − 78
    assert row["diff_gk_strength"] == 4.0
    assert row["diff_att_strength"] == 7.0   # 86 − 79
    assert row["home_adv"] == 0              # neutral venue
    assert row["cross_conf"] == 0            # no confederation map
    assert set(FEATURE_COLUMNS).issubset(out.columns)
    assert int(row["min_matched"]) == 20


def test_home_adv_set_when_not_neutral():
    out = assemble_features(_labeled(neutral=False), _profiles(), {})
    assert out.iloc[0]["home_adv"] == 1


def test_cross_conf_flag_uses_conf_map():
    out = assemble_features(
        _labeled(), _profiles(), conf_map={"Brazil": "CONMEBOL", "Switzerland": "UEFA"}
    )
    assert out.iloc[0]["cross_conf"] == 1


def test_missing_profile_drops_the_match():
    out = assemble_features(_labeled(away="Nowhere"), _profiles(), {})
    assert len(out) == 0


def test_nan_diff_when_unit_strength_missing():
    profiles = _profiles()
    profiles.loc[profiles["team"] == "Brazil", "gk_strength"] = None  # no matched GK
    out = assemble_features(_labeled(), profiles, {})
    assert pd.isna(out.iloc[0]["diff_gk_strength"])


def test_infer_confederation_map_excludes_world_cup():
    profiles = pd.DataFrame(
        [
            {"team": "Brazil", "tournament": "Copa América", "edition_year": 2019},
            {"team": "Brazil", "tournament": "FIFA World Cup", "edition_year": 2018},  # excluded
            {"team": "Spain", "tournament": "UEFA Euro", "edition_year": 2020},
            {"team": "Egypt", "tournament": "African Cup of Nations", "edition_year": 2019},
        ]
    )
    m = infer_confederation_map(profiles)
    assert m["Brazil"] == "CONMEBOL"
    assert m["Spain"] == "UEFA"
    assert m["Egypt"] == "CAF"
