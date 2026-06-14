"""Bottom-up team profiles from squads (Phase 2).

For each (team, tournament edition) we take the submitted roster, join every player to their
**nearest-PRIOR** FIFA attribute snapshot (the most recent edition released *before* the
tournament start — never a later one, so no leakage), assemble the best XI + depth via
:mod:`src.aggregate.squad_assembly`, and aggregate into one interpretable ``team_profiles``
row: positional unit strengths, star power, depth, experience, age, market value, and
top-5-league share.

FIFA edition N is released in autumn of year N-1, so a June-2018 tournament uses FIFA 18,
never FIFA 19. ``fifa_edition`` records the principal edition actually used. Coverage is
bounded by where rosters and FIFA editions overlap (currently FIFA 18/20/22).

Run: ``python -m src.aggregate.team_profile``  (prints a sanity summary).
"""

import json
import logging
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.aggregate.squad_assembly import UNITS, build_squad
from src.common.config import load_config

logger = logging.getLogger(__name__)

# Substring keywords identifying a top-5 European league in FIFA's league_name strings
# (English Premier League / Spain Primera Division / Italian Serie A / German 1. Bundesliga
# / French Ligue 1).
_TOP5_KEYWORDS = ("premier league", "primera division", "serie a", "bundesliga", "ligue 1")


def _to_date(value) -> Optional[date]:
    if value is None or value != value:  # noqa: PLR0124
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def target_edition(snapshot_date: date) -> int:
    """FIFA season_year that is the latest edition *released* on/before ``snapshot_date``.

    FIFA N releases ~30 Sep of year N-1. So any date qualifies edition ``year``; a date on
    or after 30 Sep additionally qualifies edition ``year+1``.
    """
    if (snapshot_date.month, snapshot_date.day) >= (9, 30):
        return snapshot_date.year + 1
    return snapshot_date.year


def _is_top5(league: object) -> bool:
    if not league or league != league:  # noqa: PLR0124
        return False
    low = str(league).lower()
    return any(k in low for k in _TOP5_KEYWORDS)


def _age_at(dob, snapshot_date: Optional[date]) -> Optional[float]:
    d = _to_date(dob)
    if d is None or snapshot_date is None:
        return None
    return (snapshot_date - d).days / 365.25


def _mean(values: list) -> Optional[float]:
    arr = np.array([v for v in values if v is not None and v == v], dtype=float)  # noqa: PLR0124
    if arr.size == 0:
        return None
    return round(float(arr.mean()), 2)


def _round(value) -> Optional[float]:
    if value is None or value != value:  # noqa: PLR0124
        return None
    return round(float(value), 2)


# ------------------------------------------------------------------ attribute index
def _load_attribute_index() -> tuple[dict[int, list], list[int]]:
    """Return ``{player_id: [rows sorted by season_year]}`` and the available editions.

    Each row is a dict with overall/value/league/age/positions/composure for a (player,
    season). FIFA only — the profile join is FIFA-based; FM degrades to nulls.
    """
    from sqlalchemy import text

    from src.common.db import get_engine

    sql = text(
        "SELECT player_id, season_year, overall, potential, value, league, age, positions,"
        " attrs->>'mentality_composure' AS composure,"
        " attrs->>'international_reputation' AS reputation"
        " FROM wc2026.player_attributes WHERE source = 'fifa'"
    )
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    index: dict[int, list] = {}
    for rec in df.to_dict("records"):
        index.setdefault(int(rec["player_id"]), []).append(rec)
    for rows in index.values():
        rows.sort(key=lambda r: r["season_year"])
    editions = sorted(int(y) for y in df["season_year"].dropna().unique())
    return index, editions


def _nearest_prior(rows: Optional[list], target: int) -> Optional[dict]:
    if not rows:
        return None
    best = None
    for r in rows:
        if r["season_year"] <= target:
            best = r
        else:
            break
    return best


# ------------------------------------------------------------------ profile build
def build_profile_row(
    roster: pd.DataFrame,
    attr_index: dict[int, list],
    snapshot_date: Optional[date],
    formation: dict,
    substitutes: int,
) -> dict:
    """Aggregate one team's roster into a profile row (pure given the attribute index)."""
    target = target_edition(snapshot_date) if snapshot_date else 9999

    squad_players: list[dict] = []
    values, ages, composures, top5_flags, editions_used = [], [], [], [], []
    matched = 0
    for r in roster.to_dict("records"):
        pid = r.get("player_id")
        pid = int(pid) if pid is not None and pid == pid else None  # noqa: PLR0124
        rec = _nearest_prior(attr_index.get(pid), target) if pid is not None else None
        overall = rec["overall"] if rec else None
        position = (rec["positions"] if rec and rec.get("positions") else r.get("position"))
        squad_players.append({
            "player_id": pid, "overall": overall, "position": position,
            "reputation": _to_float(rec.get("reputation")) if rec else None,
            "potential": _to_float(rec.get("potential")) if rec else None,
        })
        if rec is not None:
            matched += 1
            editions_used.append(int(rec["season_year"]))
            values.append(rec.get("value"))
            composures.append(_to_float(rec.get("composure")))
            top5_flags.append(_is_top5(rec.get("league")))
        # Age from roster DOB (anchored to the tournament), falling back to FIFA age.
        age = _age_at(r.get("dob"), snapshot_date)
        if age is None and rec is not None:
            age = _to_float(rec.get("age"))
        ages.append(age)

    squad = build_squad(squad_players, formation=formation, substitutes=substitutes)
    xi_overalls = [p["overall"] for p in squad["xi"]]
    depth_overalls = [p["overall"] for p in squad["depth"]]
    all_overalls = sorted(
        (p["overall"] for p in squad_players if p["overall"] is not None),
        reverse=True,
    )

    unit_strength = {
        u: _mean([p["overall"] for p in squad["by_unit"][u]]) for u in UNITS
    }
    # Top-end-talent signals (orthogonal to the compressed `overall`): the XI's mean global
    # stature / ceiling, and the count of genuine world-class players (reputation >= 4) in the squad.
    mean_intl_rep = _mean([p.get("reputation") for p in squad["xi"]])
    mean_potential = _mean([p.get("potential") for p in squad["xi"]])
    elite_count = sum(
        1 for p in squad_players
        if p.get("reputation") is not None and p["reputation"] >= 4
    )
    total_value = _round(np.nansum([v for v in values if v is not None])) if values else None
    total_caps = _opt_int(np.nansum(
        [c for c in roster["caps"].tolist() if c is not None and c == c]  # noqa: PLR0124
    )) if "caps" in roster else None

    return {
        "snapshot_date": snapshot_date.isoformat() if snapshot_date else None,
        "fifa_edition": max(editions_used) if editions_used else None,
        "squad_size": len(squad_players),
        "matched_players": matched,
        "gk_strength": unit_strength["GK"],
        "def_strength": unit_strength["DEF"],
        "mid_strength": unit_strength["MID"],
        "att_strength": unit_strength["ATT"],
        "overall_xi": _mean(xi_overalls),
        "star_power": _mean(all_overalls[:3]),
        "depth": _mean(depth_overalls),
        "total_caps": total_caps,
        "total_wc_apps": None,        # not in the Wikipedia squad tables
        "total_continental_apps": None,
        "mean_age": _mean(ages),
        "total_value": total_value,
        "avg_value": _round(total_value / matched) if total_value and matched else None,
        "top5_league_share": _round(np.mean(top5_flags)) if top5_flags else None,
        "mean_composure": _mean(composures),
        "mean_intl_rep": mean_intl_rep,
        "elite_count": elite_count,
        "mean_potential": mean_potential,
        "mean_work_rate": None,       # FM-only; null until FM data is ingested
        "attrs": {
            "editions_used": sorted(set(editions_used)),
            "xi_size": len(squad["xi"]),
            "depth_size": len(squad["depth"]),
        },
    }


def build_team_profiles(write: bool = True) -> pd.DataFrame:
    """Build one profile per (team, tournament, edition) from ``rosters`` + attributes."""
    from src.common.io import read_table

    cfg = load_config()
    snap_dates = {
        (s.tournament, s.edition_year): s.start_date for s in cfg.rosters.sources
    }

    rosters = read_table("rosters")
    if rosters.empty:
        logger.warning("rosters is empty — run rosters_loader first. Nothing to build.")
        return pd.DataFrame()

    attr_index, editions = _load_attribute_index()
    logger.info("Loaded attributes for %d players (editions %s).", len(attr_index), editions)

    rows = []
    for (team, tournament, edition_year), grp in rosters.groupby(
        ["team", "tournament", "edition_year"]
    ):
        snap = _to_date(snap_dates.get((tournament, int(edition_year))))
        profile = build_profile_row(
            grp, attr_index, snap, cfg.squad.formation, cfg.squad.substitutes
        )
        profile.update(
            {"team": team, "tournament": tournament, "edition_year": int(edition_year)}
        )
        rows.append(profile)

    profiles = pd.DataFrame(rows)
    logger.info("Built %d team profiles.", len(profiles))
    if write and not profiles.empty:
        _write_profiles(rows)
    return profiles


def _write_profiles(rows: list[dict]) -> None:
    from src.common.io import bulk_upsert

    metric_cols = [
        "snapshot_date", "fifa_edition", "squad_size", "matched_players",
        "gk_strength", "def_strength", "mid_strength", "att_strength", "overall_xi",
        "star_power", "depth", "total_caps", "total_wc_apps", "total_continental_apps",
        "mean_age", "total_value", "avg_value", "top5_league_share",
        "mean_composure", "mean_intl_rep", "elite_count", "mean_potential",
        "mean_work_rate", "attrs",
    ]
    payload = [
        {
            "team": r["team"],
            "tournament": r["tournament"],
            "edition_year": r["edition_year"],
            **{c: r[c] for c in metric_cols},
        }
        for r in rows
    ]
    bulk_upsert(
        "team_profiles",
        payload,
        conflict_cols=["team", "tournament", "edition_year"],
        update_cols=metric_cols,
    )
    logger.info("Upserted %d team_profiles rows.", len(payload))


def _to_float(value) -> Optional[float]:
    if value is None or value != value:  # noqa: PLR0124
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value) -> Optional[int]:
    if value is None or value != value:  # noqa: PLR0124
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary(profiles: pd.DataFrame) -> None:
    print("\n================ Phase 2 team-profile summary ================")
    if profiles.empty:
        print("No profiles built (is wc2026.rosters populated?).")
        print("=============================================================")
        return
    print(f"Profiles built : {len(profiles)}")
    print("Coverage by tournament/edition:")
    cov = profiles.groupby(["tournament", "edition_year"]).agg(
        teams=("team", "nunique"),
        avg_matched=("matched_players", "mean"),
        avg_squad=("squad_size", "mean"),
    )
    for (t, y), row in cov.iterrows():
        print(
            f"  {t} {y}: {int(row['teams'])} teams, "
            f"avg matched {row['avg_matched']:.1f}/{row['avg_squad']:.1f}"
        )
    # Strongest sample profile (highest XI) for eyeballing.
    sample = profiles.sort_values("overall_xi", ascending=False, na_position="last").iloc[0]
    print("\nSample profile (highest overall_xi):")
    for k in (
        "team", "tournament", "edition_year", "fifa_edition", "overall_xi",
        "gk_strength", "def_strength", "mid_strength", "att_strength",
        "star_power", "depth", "mean_age", "total_caps", "top5_league_share",
    ):
        print(f"  {k:18s}: {sample[k]}")
    print("=============================================================")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    profiles = build_team_profiles(write=True)
    _summary(profiles)
