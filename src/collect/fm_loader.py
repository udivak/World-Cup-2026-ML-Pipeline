"""Football Manager attribute ingestion (Phase 1).

FM exports (technical / mental / physical attributes on a 0–20 scale) land in
``data/raw/fm/`` as CSVs. FM is *additive*: FIFA is the backbone, FM merges onto the same
canonical players via the name canonicalizer and degrades gracefully to null when a player
has no FM row. Unmatched FM names are surfaced to the review CSV, never silently dropped.

Run: ``python -m src.collect.fm_loader``  (no-op with a clear log if no FM CSVs present).
"""

import logging
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.players import PlayerCanonicalizer

logger = logging.getLogger(__name__)

# Meta columns describe identity, not ability — everything else is an attribute.
_META_COLS = {
    "name", "player", "full name",
    "age", "born", "dob", "date of birth", "birthdate",
    "club", "team", "nat", "nation", "nationality",
    "position", "positions", "pos",
}


def _col(row: pd.Series, *names: str) -> object:
    for n in names:
        for col in row.index:
            if str(col).strip().lower() == n:
                return row[col]
    return None


def parse_fm_csv(path: str | Path, season_year: int | None = None) -> pd.DataFrame:
    """Parse one FM export into normalized rows: name/birthdate/nationality + attrs dict.

    Pure (no DB / no canonicalization). Any column not in :data:`_META_COLS` is treated as
    a 0–20 attribute and packed into ``attrs``.
    """
    raw = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in raw.columns}
    attr_cols = [orig for low, orig in lower.items() if low not in _META_COLS]

    out = pd.DataFrame()
    out["source"] = ["fm"] * len(raw)
    out["season_year"] = season_year
    out["name"] = [_col(r, "name", "player", "full name") for _, r in raw.iterrows()]
    out["birthdate"] = [
        _col(r, "born", "dob", "date of birth", "birthdate") for _, r in raw.iterrows()
    ]
    out["nationality"] = [
        _col(r, "nat", "nation", "nationality") for _, r in raw.iterrows()
    ]
    out["attrs"] = [
        {c: (None if (v is None or (isinstance(v, float) and v != v)) else v)
         for c, v in raw.loc[i, attr_cols].items()}
        for i in range(len(raw))
    ]
    return out


def load_fm(write: bool = True) -> pd.DataFrame:
    """Ingest every ``data/raw/fm/*.csv``; match to canonical players; upsert attributes.

    Returns the parsed frame with a ``player_id`` column (None where unmatched). Skips
    cleanly (empty frame) when no FM CSVs are present — FM is optional.
    """
    cfg = load_config()
    fm_dir = Path(cfg.raw_data_dir) / cfg.player_data.fm.raw_subdir
    files = sorted(fm_dir.glob("*.csv"))
    if not files:
        logger.info(
            "No FM CSVs in %s — skipping (FIFA is the backbone; FM is additive).", fm_dir
        )
        return pd.DataFrame()

    frames = [parse_fm_csv(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)

    canon = PlayerCanonicalizer()  # loads canonical players seeded by FIFA
    combined["player_id"] = [
        canon.canonicalize(name, bd, nat, source="fm")
        for name, bd, nat in zip(
            combined["name"], combined["birthdate"], combined["nationality"]
        )
    ]
    matched = combined["player_id"].notna().sum()
    logger.info(
        "FM matched %d/%d names (%.1f%% unmatched).",
        matched, len(combined),
        (len(combined) - matched) / len(combined) * 100 if len(combined) else 0.0,
    )
    canon.write_review(cfg.raw_data_dir)

    if write:
        _write_fm(combined)
    return combined


def _write_fm(combined: pd.DataFrame) -> None:
    from src.common.io import bulk_upsert

    matched = combined[combined["player_id"].notna()]
    alias_rows, attr_rows = [], []
    seen = set()
    for _, row in matched.iterrows():
        pid = int(row["player_id"])
        alias = str(row["name"])
        if (alias, pid) not in seen:
            seen.add((alias, pid))
            alias_rows.append({"alias": alias, "source": "fm", "player_id": pid})
        attr_rows.append(
            {
                "player_id": pid,
                "source": "fm",
                "season_year": int(row["season_year"]) if pd.notna(row["season_year"]) else 0,
                "attrs": row["attrs"],
            }
        )
    if alias_rows:
        bulk_upsert(
            "player_aliases", alias_rows, conflict_cols=["alias", "source", "player_id"]
        )
    if attr_rows:
        bulk_upsert(
            "player_attributes",
            attr_rows,
            conflict_cols=["player_id", "source", "season_year"],
            update_cols=["attrs"],
        )
    logger.info("Upserted %d FM aliases, %d FM attribute rows.", len(alias_rows), len(attr_rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_fm(write=True)
    if df.empty:
        print("No FM data ingested (drop FM exports into data/raw/fm/ to enable).")
    else:
        matched = df["player_id"].notna().sum()
        print(f"FM rows parsed : {len(df)}")
        print(f"Matched players: {matched} ({matched / len(df) * 100:.1f}%)")
