"""Normalize non-canonical FIFA / EA-FC source CSVs into the sofifa ``players_NN.csv`` schema.

FIFA's public-dataset schema drifts between editions, so two real-world variants need
adapting before :mod:`src.collect.fifa_loader` can ingest them:

- **multiversion** — stefanoleone "FC24+" exports (``male_players.csv``) carry one row per
  in-season patch (``fifa_version`` / ``fifa_update``). Keep the latest update per player.
- **fc-ratings** — EA-FC ratings exports (e.g. FC26 ``ea_fc26_players.csv``) use different
  column names. Remap to the sofifa schema: ``commonName`` (or ``firstName`` + ``lastName``)
  → ``long_name``; ``birthdate`` → ``dob``; ``position`` (+ ``alternatePositions``) →
  ``player_positions``; ``overallRating`` → ``overall``; ``nationality``/``team``/
  ``leagueName`` → ``nationality_name``/``club_name``/``league_name``; ``composure`` →
  ``mentality_composure``. The full attribute long tail is preserved for ``attrs`` JSONB.

Output keeps the canonical column names and is named ``players_<yy>.csv`` so fifa_loader
infers the edition from the filename.

Run: ``python -m src.collect.fifa_normalize <in.csv> <out.csv> --format {multiversion,fc-ratings}``
"""

import argparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def dedupe_multiversion(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest in-season patch per player; alias ``player_id`` → ``sofifa_id``."""
    sort_col = next((c for c in ("update_as_of", "fifa_update") if c in df.columns), None)
    key = next((c for c in ("player_id", "sofifa_id") if c in df.columns), None)
    if sort_col and key:
        df = df.sort_values(sort_col).drop_duplicates(subset=[key], keep="last")
    if "player_id" in df.columns and "sofifa_id" not in df.columns:
        df = df.rename(columns={"player_id": "sofifa_id"})
    return df.reset_index(drop=True)


def _combined_positions(df: pd.DataFrame) -> list[str]:
    pos = df["position"] if "position" in df.columns else pd.Series([None] * len(df))
    alt = df["alternatePositions"] if "alternatePositions" in df.columns else pd.Series([None] * len(df))
    out = []
    for p, a in zip(pos, alt):
        tokens: list[str] = []
        if p is not None and p == p:  # noqa: PLR0124
            tokens.append(str(p).strip())
        if a is not None and a == a:  # noqa: PLR0124
            tokens += [t.strip() for t in str(a).split(",")]
        seen, ordered = set(), []
        for t in tokens:
            if t and t not in seen:
                seen.add(t)
                ordered.append(t)
        out.append(", ".join(ordered))
    return out


def normalize_fc_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Remap an EA-FC ratings export to the sofifa schema (keeps the full long tail)."""
    out = df.copy()

    first = df["firstName"].fillna("") if "firstName" in df.columns else ""
    last = df["lastName"].fillna("") if "lastName" in df.columns else ""
    full = (first + " " + last).str.strip() if hasattr(first, "str") else last
    if "commonName" in df.columns:
        out["long_name"] = df["commonName"].where(df["commonName"].notna(), full)
        out["short_name"] = df["commonName"].where(df["commonName"].notna(), last if hasattr(last, "str") else full)
    else:
        out["long_name"] = full
        out["short_name"] = full

    if "birthdate" in df.columns:
        # EA exports dates as "M/D/YYYY H:MM:SS AM"; parse the date part explicitly, then
        # fall back to generic inference for any value that doesn't match.
        date_part = df["birthdate"].astype(str).str.split().str[0]
        parsed = pd.to_datetime(date_part, format="%m/%d/%Y", errors="coerce")
        missing = parsed.isna() & df["birthdate"].notna()
        if missing.any():
            parsed.loc[missing] = pd.to_datetime(df["birthdate"][missing], errors="coerce")
        dob = parsed.dt.strftime("%Y-%m-%d")
        out["dob"] = dob.where(dob.notna() & dob.ne("NaT"))

    out["player_positions"] = _combined_positions(df)

    rename = {
        "overallRating": "overall",
        "nationality": "nationality_name",
        "team": "club_name",
        "leagueName": "league_name",
        "composure": "mentality_composure",
        "id": "sofifa_id",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    return out


def normalize(in_path: str, out_path: str, fmt: str) -> pd.DataFrame:
    df = pd.read_csv(in_path, low_memory=False)
    n0 = len(df)
    if fmt == "multiversion":
        df = dedupe_multiversion(df)
    elif fmt == "fc-ratings":
        df = normalize_fc_ratings(df)
    else:
        raise ValueError(f"Unknown format: {fmt!r}")
    df.to_csv(out_path, index=False)
    logger.info("Normalized %s (%d → %d rows) → %s", in_path, n0, len(df), out_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--format", choices=["multiversion", "fc-ratings"], required=True)
    args = ap.parse_args()
    normalize(args.input, args.output, args.format)
