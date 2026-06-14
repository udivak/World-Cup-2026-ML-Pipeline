import logging
from pathlib import Path

import pandas as pd
import requests

from src.common.config import load_config

logger = logging.getLogger(__name__)

_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]


def _csv_path() -> Path:
    cfg = load_config()
    return Path(cfg.raw_data_dir) / "results.csv"


def _null_scores_log() -> Path:
    cfg = load_config()
    return Path(cfg.raw_data_dir) / "null_scores.log"


def _download_csv(dest: Path) -> None:
    logger.info("Downloading results.csv from GitHub…")
    resp = requests.get(_RESULTS_URL, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    logger.info("Saved to %s", dest)


def _derive_result(row: pd.Series) -> str:
    if row["home_score"] > row["away_score"]:
        return "H"
    elif row["home_score"] == row["away_score"]:
        return "D"
    else:
        return "A"


def load_raw_matches() -> pd.DataFrame:
    dest = _csv_path()
    if not dest.exists():
        _download_csv(dest)
    else:
        logger.info("Using cached %s", dest)

    df = pd.read_csv(dest, parse_dates=["date"])
    df = df[_COLUMNS].copy()

    null_mask = df["home_score"].isna() | df["away_score"].isna()
    if null_mask.any():
        null_rows = df[null_mask]
        log_path = _null_scores_log()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        null_rows.to_csv(log_path, index=False)
        logger.warning(
            "%d rows with null scores logged to %s", len(null_rows), log_path
        )

    df = df[~null_mask].copy()
    df["result"] = df.apply(_derive_result, axis=1)

    return df


def load_matches() -> None:
    from src.common.teams import Canonicalizer
    from src.common.io import write_table

    df = load_raw_matches()
    canon = Canonicalizer()

    unmapped: list[str] = []

    def _canon_team(name: str) -> str:
        result = canon.canonicalize(name)
        if result == name and not canon.has_alias(name):
            unmapped.append(name)
        return result

    df["home_team"] = df["home_team"].map(_canon_team)
    df["away_team"] = df["away_team"].map(_canon_team)

    cfg = load_config()
    unmapped_log = Path(cfg.raw_data_dir) / "unmapped_teams.log"
    unique_unmapped = sorted(set(unmapped))
    if unique_unmapped:
        unmapped_log.write_text("\n".join(unique_unmapped) + "\n")
        total_rows = len(df)
        unmapped_rows = sum(
            1 for h, a in zip(df["home_team"], df["away_team"])
            if h in unique_unmapped or a in unique_unmapped
        )
        pct = unmapped_rows / total_rows * 100
        logger.warning(
            "%d unmapped team names affecting %.1f%% of rows — see %s",
            len(unique_unmapped), pct, unmapped_log,
        )

    write_table(df, "matches")
    logger.info("Loaded %d rows into wc2026.matches", len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_raw_matches()
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    dist = df["result"].value_counts(normalize=True).mul(100).round(1)
    print("Result distribution (%):")
    for label, pct in dist.items():
        print(f"  {label}: {pct}%")
    # Canonicalize team names and write to wc2026.matches (this is the actual load).
    load_matches()
