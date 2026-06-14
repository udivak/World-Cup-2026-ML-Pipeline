"""Per-match features from team profiles (Phase 3).

For each supervised tournament match we join **both** teams' nearest-prior ``team_profiles`` for
that edition and take **team1 − team2 differences** (team1 = home) over every profile column,
then add pre-match context (home advantage, cross-confederation). The realized ``result`` is the
label, never a feature.

This is the **single feature code path** used for both training (Phase 3) and live 2026 scoring
(Phase 4): :func:`assemble_features` is pure (matches + profiles + a confederation map in,
features out), and :func:`build_feature_frame` / :func:`write_match_features` wire it to the DB.

No leakage: the joined profile is the edition's pre-tournament squad (a legitimate pre-match
input) and was itself built from FIFA data released before the tournament — see
:mod:`src.aggregate.team_profile` and ``tests/test_no_leakage.py``.

Run: ``python -m src.features.build_features``  (build + write ``match_features`` + summary).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.features.labels import load_labeled_matches

logger = logging.getLogger(__name__)

# team_profile numeric columns we difference (home − away). Null-valued columns
# (total_wc_apps / total_continental_apps / mean_work_rate) are excluded — no data yet.
PROFILE_DIFF_COLS = [
    "gk_strength", "def_strength", "mid_strength", "att_strength", "overall_xi",
    "star_power", "depth", "total_caps", "mean_age", "total_value", "avg_value",
    "top5_league_share", "mean_composure",
]

# The full feature vector the models consume (single source of truth for train + serve).
FEATURE_COLUMNS = [f"diff_{c}" for c in PROFILE_DIFF_COLS] + ["home_adv", "cross_conf"]

# Identity + label columns carried alongside the features.
ID_COLUMNS = ["date", "tournament", "edition_year", "home_team", "away_team", "neutral", "result"]

# The naive bottom-up baseline's lone strength feature (Phase 3 baselines).
OVERALL_DIFF_COLUMN = "diff_overall_xi"


# A team's confederation is inferred from which continental cup it plays in (the World Cup is
# the only cross-confederation event, so it is excluded from the inference).
CONTINENTAL_CONFED = {
    "UEFA Euro": "UEFA",
    "Copa América": "CONMEBOL",
    "African Cup of Nations": "CAF",
    "AFC Asian Cup": "AFC",
    "Gold Cup": "CONCACAF",
    "Oceania Nations Cup": "OFC",
}


def infer_confederation_map(profiles: pd.DataFrame) -> dict:
    """``{team: confederation}`` from continental-cup membership in ``team_profiles``.

    A team's confederation is the one whose continental cup it appears in most often (guest
    invitees — e.g. an AFC side invited to the Gold Cup — resolve to their home confederation
    by majority; ties break deterministically by confederation name).
    """
    if profiles.empty:
        return {}
    cont = profiles[profiles["tournament"].isin(CONTINENTAL_CONFED)].copy()
    if cont.empty:
        return {}
    cont["conf"] = cont["tournament"].map(CONTINENTAL_CONFED)
    counts = cont.groupby(["team", "conf"]).size().reset_index(name="n")
    counts.sort_values(["team", "n", "conf"], ascending=[True, False, True], inplace=True)
    top = counts.drop_duplicates("team", keep="first")
    return dict(zip(top["team"], top["conf"]))


def load_confederation_map(profiles: Optional[pd.DataFrame] = None) -> dict:
    """``{team: confederation}`` for the cross-conf flag.

    Prefers an explicit ``team_aliases.confederation`` mapping; falls back to inferring it from
    continental-cup membership when that column is unpopulated (current state of the data).
    """
    from src.common.io import read_table

    aliases = read_table("team_aliases")
    if not aliases.empty and "confederation" in aliases and aliases["confederation"].notna().any():
        conf = aliases.dropna(subset=["confederation"])
        return dict(zip(conf["canonical"], conf["confederation"]))
    return infer_confederation_map(read_table("team_profiles") if profiles is None else profiles)


def assemble_features(
    labeled: pd.DataFrame,
    profiles: pd.DataFrame,
    conf_map: Optional[dict] = None,
) -> pd.DataFrame:
    """Build the feature frame from labeled matches + team profiles — pure (no DB).

    ``labeled`` rows carry match identity, ``edition_year`` and ``result`` (see
    :func:`src.features.labels.load_labeled_matches`). ``profiles`` is ``team_profiles``.
    Returns one row per match with :data:`ID_COLUMNS` + :data:`FEATURE_COLUMNS` + ``min_matched``.
    Matches missing a profile on either side are dropped (logged).
    """
    conf_map = conf_map or {}
    if labeled.empty:
        return pd.DataFrame(columns=ID_COLUMNS + FEATURE_COLUMNS + ["min_matched"])

    prof = profiles.copy()
    for c in PROFILE_DIFF_COLS + ["matched_players"]:
        prof[c] = pd.to_numeric(prof.get(c), errors="coerce")
    keep = ["team", "tournament", "edition_year"] + PROFILE_DIFF_COLS + ["matched_players"]
    prof = prof[keep]
    prof["edition_year"] = prof["edition_year"].astype("Int64")

    df = labeled.copy()
    df["edition_year"] = df["edition_year"].astype("Int64")
    df = df.merge(
        prof.rename(columns={"team": "home_team"}),
        on=["tournament", "edition_year", "home_team"], how="left",
    )
    df = df.merge(
        prof.rename(columns={"team": "away_team"}),
        on=["tournament", "edition_year", "away_team"], how="left", suffixes=("_h", "_a"),
    )

    # Drop matches without a profile on either side (matched_players is NULL when no join).
    before = len(df)
    df = df[df["matched_players_h"].notna() & df["matched_players_a"].notna()].copy()
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d/%d matches missing a team profile on one side.", dropped, before)

    for c in PROFILE_DIFF_COLS:
        df[f"diff_{c}"] = df[f"{c}_h"] - df[f"{c}_a"]
    df["min_matched"] = df[["matched_players_h", "matched_players_a"]].min(axis=1)

    # Context. home_adv = 1 when team1 (home) is not at a neutral venue — so host nations
    # (which the source marks neutral=False) keep their advantage and true-neutral games get 0.
    neutral = df["neutral"].astype("boolean").fillna(False)
    df["home_adv"] = (~neutral).astype(int)
    home_conf = df["home_team"].map(conf_map)
    away_conf = df["away_team"].map(conf_map)
    df["cross_conf"] = (
        home_conf.notna() & away_conf.notna() & (home_conf != away_conf)
    ).astype(int)

    out = df[ID_COLUMNS + FEATURE_COLUMNS + ["min_matched"]].copy()
    out["min_matched"] = out["min_matched"].astype("Int64")
    return out.reset_index(drop=True)


def build_feature_frame(window_days: Optional[int] = None) -> pd.DataFrame:
    """Load labels + profiles from the DB and assemble the feature frame (single code path)."""
    from src.common.io import read_table
    from src.features.labels import TOURNAMENT_WINDOW_DAYS

    labeled = load_labeled_matches(window_days=window_days or TOURNAMENT_WINDOW_DAYS)
    profiles = read_table("team_profiles")
    conf_map = load_confederation_map(profiles)
    feats = assemble_features(labeled, profiles, conf_map)
    logger.info("Assembled %d match-feature rows.", len(feats))
    return feats


def _py(value):
    """Coerce numpy/pandas scalars to DB-friendly Python types (NaN/NaT/NA → None)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def write_match_features(feats: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build (if needed) and upsert the feature frame into ``wc2026.match_features``."""
    from src.common.io import bulk_upsert

    feats = build_feature_frame() if feats is None else feats
    if feats.empty:
        logger.warning("No features to write.")
        return feats

    cols = ID_COLUMNS + FEATURE_COLUMNS + ["min_matched"]
    payload = [{c: _py(r[c]) for c in cols} for r in feats.to_dict("records")]
    bulk_upsert(
        "match_features",
        payload,
        conflict_cols=["date", "home_team", "away_team"],
        update_cols=[c for c in cols if c not in ("date", "home_team", "away_team")],
    )
    logger.info("Upserted %d match_features rows.", len(payload))
    return feats


def _summary(feats: pd.DataFrame) -> None:
    print("\n================ Phase 3 match-features summary ================")
    if feats.empty:
        print("No features built (are team_profiles / matches populated?).")
        print("===============================================================")
        return
    print(f"Feature rows           : {len(feats)}")
    print(f"Feature columns ({len(FEATURE_COLUMNS)}) : {FEATURE_COLUMNS}")
    print(f"home_adv = 1 (non-neutral) : {int(feats['home_adv'].sum())} matches")
    print(f"cross_conf = 1            : {int(feats['cross_conf'].sum())} matches")
    nan_share = feats[FEATURE_COLUMNS].isna().mean().sort_values(ascending=False)
    print("Top feature NaN shares (sparse units for low-coverage squads):")
    for c, v in nan_share.head(5).items():
        print(f"  {c:22s}: {v:.1%}")
    print("Sample (largest |diff_overall_xi|):")
    s = feats.reindex(feats["diff_overall_xi"].abs().sort_values(ascending=False).index).iloc[0]
    print(
        f"  {s['date'].date()} {s['home_team']} vs {s['away_team']} "
        f"({s['tournament']} {int(s['edition_year'])}) result={s['result']} "
        f"diff_overall_xi={s['diff_overall_xi']:.1f} home_adv={s['home_adv']}"
    )
    print("===============================================================")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    feats = write_match_features()
    _summary(feats)
