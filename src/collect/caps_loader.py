"""International-experience ingestion: caps + tournament appearances (Phase 1).

Reads caps snapshots from ``data/raw/caps/`` into ``caps_snapshots(player_id, as_of_date,
caps, wc_apps, continental_apps)``. Experience degrades gracefully: a player with no caps
row simply has none — downstream team aggregates tolerate nulls (never an error).

Expected CSV columns (case-insensitive): ``name``, ``as_of_date``, ``caps`` and optionally
``birthdate``/``nationality`` (to disambiguate), ``wc_apps``, ``continental_apps``.

Run: ``python -m src.collect.caps_loader``  (no-op with a clear log if no caps CSVs present).
"""

import logging
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.players import PlayerCanonicalizer, normalize_date

logger = logging.getLogger(__name__)


def _pick(raw: pd.DataFrame, *names: str) -> pd.Series:
    lower = {str(c).strip().lower(): c for c in raw.columns}
    for n in names:
        if n in lower:
            return raw[lower[n]]
    return pd.Series([None] * len(raw))


def parse_caps_csv(path: str | Path) -> pd.DataFrame:
    """Parse one caps CSV into normalized rows. Pure (no DB / no canonicalization)."""
    raw = pd.read_csv(path)
    out = pd.DataFrame()
    out["name"] = _pick(raw, "name", "player", "full name")
    out["birthdate"] = _pick(raw, "birthdate", "dob", "born", "date of birth")
    out["nationality"] = _pick(raw, "nationality", "nation", "nat")
    out["as_of_date"] = _pick(raw, "as_of_date", "date", "as_of").map(normalize_date)
    out["caps"] = _pick(raw, "caps", "appearances")
    out["wc_apps"] = _pick(raw, "wc_apps", "world_cup_apps", "wc")
    out["continental_apps"] = _pick(raw, "continental_apps", "continental", "cont_apps")
    return out


def load_caps(write: bool = True) -> pd.DataFrame:
    """Ingest every ``data/raw/caps/*.csv``; match to players; upsert caps_snapshots.

    Skips cleanly (empty frame) when no caps CSVs are present — experience is optional.
    """
    cfg = load_config()
    caps_dir = Path(cfg.raw_data_dir) / cfg.player_data.caps.raw_subdir
    files = sorted(caps_dir.glob("*.csv"))
    if not files:
        logger.info(
            "No caps CSVs in %s — skipping (experience is optional / nullable).", caps_dir
        )
        return pd.DataFrame()

    combined = pd.concat([parse_caps_csv(f) for f in files], ignore_index=True)
    canon = PlayerCanonicalizer()
    combined["player_id"] = [
        canon.canonicalize(name, bd, nat, source="caps")
        for name, bd, nat in zip(
            combined["name"], combined["birthdate"], combined["nationality"]
        )
    ]
    matched = combined["player_id"].notna().sum()
    logger.info("Caps matched %d/%d names.", matched, len(combined))
    canon.write_review(cfg.raw_data_dir)

    if write:
        _write_caps(combined)
    return combined


def _write_caps(combined: pd.DataFrame) -> None:
    from src.common.io import bulk_upsert

    rows = []
    for _, row in combined[combined["player_id"].notna()].iterrows():
        rows.append(
            {
                "player_id": int(row["player_id"]),
                "as_of_date": row["as_of_date"],
                "caps": _int_or_none(row["caps"]),
                "wc_apps": _int_or_none(row["wc_apps"]),
                "continental_apps": _int_or_none(row["continental_apps"]),
            }
        )
    if rows:
        bulk_upsert(
            "caps_snapshots",
            rows,
            conflict_cols=["player_id", "as_of_date"],
            update_cols=["caps", "wc_apps", "continental_apps"],
        )
    logger.info("Upserted %d caps snapshots.", len(rows))


def _int_or_none(value: object) -> object:
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_caps(write=True)
    if df.empty:
        print("No caps data ingested (drop caps CSVs into data/raw/caps/ to enable).")
    else:
        matched = df["player_id"].notna().sum()
        print(f"Caps rows parsed: {len(df)}")
        print(f"Matched players : {matched}")
