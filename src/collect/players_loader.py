"""Phase 1 orchestrator: ingest FIFA → FM → caps, then print a sanity summary.

This is the Phase 1 "definition of done" entry point. It runs the three source loaders in
order (FIFA seeds canonical identities; FM and caps merge onto them and skip cleanly when
absent) and reports player counts, rows per source, season coverage, and the unmatched
rate so coverage can be eyeballed against the <2% target.

Run: ``python -m src.collect.players_loader``
"""

import logging

from src.collect.caps_loader import load_caps
from src.collect.fifa_loader import load_fifa
from src.collect.fm_loader import load_fm

logger = logging.getLogger(__name__)


def run(write: bool = True) -> None:
    logger.info("=== FIFA ingestion ===")
    load_fifa(write=write)
    logger.info("=== FM ingestion ===")
    load_fm(write=write)
    logger.info("=== Caps ingestion ===")
    load_caps(write=write)


def _summary() -> None:
    from sqlalchemy import text

    from src.common.db import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        players = conn.execute(text("SELECT COUNT(*) FROM wc2026.players")).scalar()
        by_source = conn.execute(
            text(
                "SELECT source, COUNT(*) AS n, COUNT(DISTINCT player_id) AS players"
                " FROM wc2026.player_attributes GROUP BY source ORDER BY source"
            )
        ).all()
        seasons = conn.execute(
            text(
                "SELECT source, MIN(season_year), MAX(season_year),"
                " COUNT(DISTINCT season_year)"
                " FROM wc2026.player_attributes GROUP BY source ORDER BY source"
            )
        ).all()
        aliases = conn.execute(
            text("SELECT source, COUNT(*) FROM wc2026.player_aliases GROUP BY source")
        ).all()
        caps = conn.execute(text("SELECT COUNT(*) FROM wc2026.caps_snapshots")).scalar()

    print("\n================ Phase 1 sanity summary ================")
    print(f"Canonical players       : {players}")
    print("Attribute rows by source:")
    for source, n, pl in by_source:
        print(f"  {source:5s}: {n:>7} rows across {pl} players")
    print("Season coverage:")
    for source, lo, hi, n in seasons:
        print(f"  {source:5s}: {lo}–{hi} ({n} editions)")
    print("Aliases by source:")
    for source, n in aliases:
        print(f"  {source:5s}: {n}")
    print(f"Caps snapshots          : {caps}")
    print("=======================================================")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(write=True)
    _summary()
