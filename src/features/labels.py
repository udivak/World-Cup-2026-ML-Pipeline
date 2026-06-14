"""Supervised label subset: tournament-finals matches tagged with their edition (Phase 3).

The supervised layer learns W/D/L on **tournament matches only** (the spec keeps historical
matches as *labels*, never as a team-strength signal). A match is in-scope iff it belongs to a
tournament edition for which we built a ``team_profiles`` row — i.e. one of the 16 completed
editions (FIFA WC / Euro / Copa / AFCON / Asian Cup / Gold Cup, 2018→2025). 2026 fixtures carry
no ``result`` yet, so they are excluded from labels (they are the Phase-4 prediction target).

Each match is mapped to its edition by **tournament name + date window**: a match belongs to
edition *E* of a tournament iff its date falls in ``[snapshot_date, snapshot_date + WINDOW]``.
``snapshot_date`` is the real play date (so AFCON-2021, played Jan-2022, is matched correctly),
and editions of the same tournament are spaced > 1 year apart, so the window assignment is
unambiguous. The mapping is leakage-free: the squad/profile is a legitimate *pre-match* input,
and the edition's profile uses only FIFA data released before ``snapshot_date``.

Run: ``python -m src.features.labels``  (prints the label-subset summary).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# A tournament's matches fall within this many days of its start (longest finals ~32 days;
# 60 leaves margin while staying well under the >1-year spacing between same-tournament editions).
TOURNAMENT_WINDOW_DAYS = 60

# Stable match identity within the finals (no surrogate key needed for the join).
MATCH_KEY = ["date", "home_team", "away_team"]


def assign_editions(
    matches: pd.DataFrame,
    editions: pd.DataFrame,
    window_days: int = TOURNAMENT_WINDOW_DAYS,
) -> pd.DataFrame:
    """Tag each match with its ``(edition_year, snapshot_date)`` — pure (no DB).

    ``editions`` has columns ``tournament, edition_year, snapshot_date``. A match joins the
    edition of the same tournament whose ``snapshot_date`` is the latest one on/before the match
    date and within ``window_days``. Matches outside every window (friendlies, qualifiers, or
    editions we have no profiles for) are dropped.
    """
    if matches.empty or editions.empty:
        return matches.iloc[0:0].assign(edition_year=pd.Series(dtype="Int64"))

    m = matches.copy()
    m["_date"] = pd.to_datetime(m["date"]).dt.normalize()
    e = editions[["tournament", "edition_year", "snapshot_date"]].copy()
    e["_snap"] = pd.to_datetime(e["snapshot_date"]).dt.normalize()

    merged = m.merge(e, on="tournament", how="inner")
    window = pd.to_timedelta(window_days, unit="D")
    in_window = (merged["_date"] >= merged["_snap"]) & (merged["_date"] <= merged["_snap"] + window)
    merged = merged[in_window].copy()
    if merged.empty:
        return matches.iloc[0:0].assign(edition_year=pd.Series(dtype="Int64"))

    # If a match somehow falls in two windows, keep the closest (latest) snapshot.
    merged.sort_values("_snap", inplace=True)
    merged = merged.drop_duplicates(subset=MATCH_KEY, keep="last")

    merged["edition_year"] = merged["edition_year"].astype("Int64")
    return merged.drop(columns=["_date", "_snap"])


def load_tournament_editions() -> pd.DataFrame:
    """Distinct ``(tournament, edition_year, snapshot_date)`` we have profiles for (DB)."""
    from src.common.io import read_table

    profiles = read_table("team_profiles")
    if profiles.empty:
        return profiles.assign(snapshot_date=pd.Series(dtype="datetime64[ns]"))
    editions = (
        profiles[["tournament", "edition_year", "snapshot_date"]]
        .drop_duplicates()
        .sort_values(["snapshot_date", "tournament"])
        .reset_index(drop=True)
    )
    return editions


def load_labeled_matches(window_days: int = TOURNAMENT_WINDOW_DAYS) -> pd.DataFrame:
    """The supervised label subset: in-scope tournament matches with a known ``result`` (DB).

    Returns one row per finals match with identity + ``edition_year`` + ``snapshot_date`` +
    the ``result`` label. Matches with no result (future 2026 fixtures) are excluded.
    """
    from src.common.io import read_table

    matches = read_table("matches")
    editions = load_tournament_editions()
    labeled = assign_editions(matches, editions, window_days=window_days)
    labeled = labeled[labeled["result"].isin(["H", "D", "A"])].copy()
    labeled["date"] = pd.to_datetime(labeled["date"])
    labeled.sort_values(["date"] + MATCH_KEY[1:], inplace=True)
    return labeled.reset_index(drop=True)


def _summary(labeled: pd.DataFrame) -> None:
    print("\n================ Phase 3 label-subset summary ================")
    if labeled.empty:
        print("No labeled matches (is wc2026.team_profiles populated?).")
        print("=============================================================")
        return
    print(f"Labeled tournament matches : {len(labeled)}")
    dist = labeled["result"].value_counts()
    total = len(labeled)
    print(
        "Result distribution        : "
        + ", ".join(f"{k}={int(v)} ({v / total:.1%})" for k, v in dist.items())
    )
    print("By edition (chronological):")
    by_ed = (
        labeled.groupby(["snapshot_date", "tournament", "edition_year"])
        .size()
        .reset_index(name="n")
        .sort_values("snapshot_date")
    )
    for _, r in by_ed.iterrows():
        print(f"  {str(r['snapshot_date'])[:10]}  {r['tournament']} {int(r['edition_year'])}: {int(r['n'])}")
    print("=============================================================")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    labeled = load_labeled_matches()
    _summary(labeled)
