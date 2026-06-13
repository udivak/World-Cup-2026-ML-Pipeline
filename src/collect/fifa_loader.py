"""FIFA / EA FC yearly player-attribute ingestion (Phase 1).

Reads sofifa-schema "complete player dataset" CSVs from ``data/raw/fifa/`` (one file per
edition, e.g. ``players_22.csv``), types the common columns, and preserves the full
~100-attribute long tail in an ``attrs`` JSONB payload ("ingest wide, select narrow").

Identities are seeded here (FIFA has the richest birthdate/nationality coverage): one
canonical ``players`` row per distinct ``(normalized_name, birthdate, nationality)``, reused
across editions, with FIFA spellings recorded as ``player_aliases``. One
``player_attributes`` row is written per (player, season_year). All writes are idempotent.

Run: ``python -m src.collect.fifa_loader``
"""

import logging
import re
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.players import PlayerCanonicalizer, normalize_name

logger = logging.getLogger(__name__)

# Columns lifted into typed table columns. Everything else (minus the URL/id columns
# below) is preserved verbatim in attrs JSONB.
_RENAME = {
    "player_positions": "positions",
    "value_eur": "value",
    "nationality_name": "nationality",
    "club_name": "club",
    "league_name": "league",
}
# Image/URL columns are not player attributes — excluded from attrs to keep payloads lean.
_DROP_FROM_ATTRS = {
    "player_url",
    "player_face_url",
    "club_logo_url",
    "club_flag_url",
    "nation_logo_url",
    "nation_flag_url",
    "sofifa_id",  # kept as a typed external id instead
}


def season_from_filename(path: str | Path) -> int:
    """``players_22.csv`` → 2022, ``fc24.csv`` → 2024. Two-digit year → 2000+yy."""
    stem = Path(path).stem
    match = re.search(r"(\d{2,4})", stem)
    if not match:
        raise ValueError(f"Cannot infer season year from filename: {path!r}")
    digits = match.group(1)
    year = int(digits)
    if year < 100:
        year += 2000
    return year


def _primary_position(positions: object) -> object:
    if positions is None or positions != positions:  # noqa: PLR0124  NaN check
        return None
    first = str(positions).split(",")[0].strip()
    return first or None


def parse_fifa_csv(path: str | Path) -> pd.DataFrame:
    """Parse one sofifa-schema edition CSV into normalized player-attribute rows.

    Pure (no DB / no canonicalization). Returns a DataFrame with the typed columns plus an
    ``attrs`` dict column holding the full attribute long tail. ``season_year`` is derived
    from the filename.
    """
    season = season_from_filename(path)
    raw = pd.read_csv(path, low_memory=False)

    df = pd.DataFrame(index=raw.index)
    df["source"] = "fifa"
    df["season_year"] = season
    df["sofifa_id"] = raw.get("sofifa_id")
    df["name"] = raw["long_name"]
    df["short_name"] = raw.get("short_name")
    df["birthdate"] = raw.get("dob")
    df["positions"] = raw.get("player_positions")
    df["primary_position"] = df["positions"].map(_primary_position)
    df["nationality"] = raw.get("nationality_name")
    df["club"] = raw.get("club_name")
    df["league"] = raw.get("league_name")
    df["overall"] = raw.get("overall")
    df["potential"] = raw.get("potential")
    df["value"] = raw.get("value_eur")
    df["age"] = raw.get("age")
    df["normalized_name"] = df["name"].map(normalize_name)

    attr_cols = [c for c in raw.columns if c not in _DROP_FROM_ATTRS]

    def _pack(row: pd.Series) -> dict:
        return {
            k: (None if (v is None or (isinstance(v, float) and v != v)) else v)
            for k, v in row.items()
        }

    df["attrs"] = [
        _pack(row) for _, row in raw[attr_cols].rename(columns=_RENAME).iterrows()
    ]
    return df


def load_fifa(write: bool = True) -> pd.DataFrame:
    """Ingest every ``data/raw/fifa/*.csv`` edition; seed players; upsert attributes.

    Returns the combined parsed frame (with assigned ``player_id``). When ``write`` is
    False, runs the full parse + canonicalization in memory without touching the DB.
    """
    cfg = load_config()
    fifa_dir = Path(cfg.raw_data_dir) / cfg.player_data.fifa.raw_subdir
    files = sorted(fifa_dir.glob("*.csv"))
    if not files:
        logger.warning("No FIFA CSVs found in %s — nothing to ingest.", fifa_dir)
        return pd.DataFrame()

    frames = []
    for f in files:
        logger.info("Parsing %s …", f.name)
        frame = parse_fifa_csv(f)
        logger.info("  %s → %d rows (season %d)", f.name, len(frame), frame["season_year"].iloc[0])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    # Seed canonical identities from the richest-coverage feed.
    canon = PlayerCanonicalizer(players=[])
    combined["player_id"] = [
        canon.add(name, bd, nat)
        for name, bd, nat in zip(
            combined["name"], combined["birthdate"], combined["nationality"]
        )
    ]

    if not write:
        return combined

    _write_to_db(combined, canon)
    return combined


def _write_to_db(combined: pd.DataFrame, canon: PlayerCanonicalizer) -> None:
    from src.common.db import ensure_schema, get_engine
    from src.common.io import bulk_upsert

    ensure_schema(get_engine())

    # Latest known primary position per in-memory player id (vectorized; one pass).
    ordered = combined.sort_values("season_year")
    latest_pos = (
        ordered.dropna(subset=["primary_position"])
        .groupby("player_id")["primary_position"]
        .last()
        .to_dict()
    )

    # 1. Upsert canonical players (DB assigns real ids; identity key dedupes).
    players_payload = [
        {
            "canonical_name": str(rec["canonical_name"]),
            "normalized_name": rec["normalized_name"],
            "birthdate": rec["birthdate"],
            "nationality": rec["nationality"],
            "primary_position": latest_pos.get(rec["player_id"]),
        }
        for rec in canon.players()
    ]
    bulk_upsert(
        "players",
        players_payload,
        conflict_cols=["normalized_name", "birthdate", "nationality"],
        update_cols=["canonical_name", "primary_position"],
    )

    # 2. Resolve our in-memory ids to the DB's player_id via the identity key.
    id_map = _db_id_map(canon)

    # 3. Aliases (long + short FIFA spellings).
    alias_rows: list[dict] = []
    seen_aliases: set[tuple] = set()
    for _, row in combined.iterrows():
        pid = id_map.get(row["player_id"])
        if pid is None:
            continue
        for alias in (row["name"], row["short_name"]):
            if alias is None or (isinstance(alias, float) and alias != alias):
                continue
            key = (str(alias), "fifa", pid)
            if key in seen_aliases:
                continue
            seen_aliases.add(key)
            alias_rows.append({"alias": str(alias), "source": "fifa", "player_id": pid})
    bulk_upsert(
        "player_aliases", alias_rows, conflict_cols=["alias", "source", "player_id"]
    )

    # 4. Attributes — one row per (player, season). Distinct sofifa entries can collapse
    # onto the same canonical id within an edition (duplicate listings, or two people
    # sharing name+birthdate+nationality); keep the highest-rated so ON CONFLICT DO UPDATE
    # never targets the same row twice.
    attr_src = (
        combined.sort_values("overall", ascending=False, na_position="last")
        .drop_duplicates(subset=["player_id", "season_year"], keep="first")
    )
    attr_rows = []
    for _, row in attr_src.iterrows():
        pid = id_map.get(row["player_id"])
        if pid is None:
            continue
        attr_rows.append(
            {
                "player_id": pid,
                "source": "fifa",
                "season_year": int(row["season_year"]),
                "overall": _int_or_none(row["overall"]),
                "potential": _int_or_none(row["potential"]),
                "positions": row["positions"],
                "club": row["club"],
                "league": row["league"],
                "nationality": row["nationality"],
                "value": _num_or_none(row["value"]),
                "age": _int_or_none(row["age"]),
                "attrs": row["attrs"],
            }
        )
    bulk_upsert(
        "player_attributes",
        attr_rows,
        conflict_cols=["player_id", "source", "season_year"],
        update_cols=[
            "overall", "potential", "positions", "club", "league",
            "nationality", "value", "age", "attrs",
        ],
    )
    logger.info(
        "Upserted %d players, %d aliases, %d attribute rows.",
        len(players_payload), len(alias_rows), len(attr_rows),
    )


def _db_id_map(canon: PlayerCanonicalizer) -> dict[int, int]:
    """Map in-memory player ids → DB player_id by re-reading the identity key."""
    from sqlalchemy import text

    from src.common.db import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT player_id, normalized_name, birthdate, nationality"
                " FROM wc2026.players"
            )
        ).all()
    db_by_key = {
        (n, d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else (str(d)[:10] if d else None), nat): pid
        for pid, n, d, nat in rows
    }
    id_map: dict[int, int] = {}
    for rec in canon.players():
        key = (rec["normalized_name"], rec["birthdate"], rec["nationality"])
        db_id = db_by_key.get(key)
        if db_id is not None:
            id_map[rec["player_id"]] = db_id
    return id_map


def _int_or_none(value: object) -> object:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return int(value)


def _num_or_none(value: object) -> object:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return float(value)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_fifa(write=True)
    if df.empty:
        print("No FIFA data ingested.")
    else:
        n_players = df["player_id"].nunique()
        print(f"Editions ingested : {sorted(df['season_year'].unique())}")
        print(f"Attribute rows    : {len(df)}")
        print(f"Distinct players  : {n_players}")
        rows_per = df.groupby("season_year").size()
        print("Rows per season:")
        for season, n in rows_per.items():
            print(f"  {season}: {n}")
