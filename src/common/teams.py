import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Canonicalizer:
    """Looks up team-name aliases against wc2026.team_aliases.

    Falls back to the input name if the alias is not found, logging a warning.
    """

    def __init__(self, alias_map: Optional[dict[str, str]] = None) -> None:
        if alias_map is not None:
            self._map = alias_map
        else:
            self._map = self._load_from_db()

    def _load_from_db(self) -> dict[str, str]:
        from src.common.db import get_engine
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT alias, canonical FROM wc2026.team_aliases")
            ).fetchall()
        return {alias: canonical for alias, canonical in rows}

    def has_alias(self, name: str) -> bool:
        return name in self._map

    def canonicalize(self, name: str) -> str:
        canonical = self._map.get(name)
        if canonical is None:
            logger.warning("Unmapped team name: %r", name)
            return name
        return canonical

    @classmethod
    def seed_from_csv(cls, csv_path: str) -> None:
        """Seed team_aliases with identity mappings from raw results CSV."""
        from src.common.db import get_engine
        from sqlalchemy import text

        df = pd.read_csv(csv_path)
        names = sorted(
            set(df["home_team"].dropna()) | set(df["away_team"].dropna())
        )
        engine = get_engine()
        with engine.connect() as conn:
            for name in names:
                conn.execute(
                    text(
                        "INSERT INTO wc2026.team_aliases (alias, canonical)"
                        " VALUES (:alias, :canonical)"
                        " ON CONFLICT (alias) DO NOTHING"
                    ),
                    {"alias": name, "canonical": name},
                )
            conn.commit()
        logger.info("Seeded %d team aliases", len(names))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.common.db import get_engine
    from sqlalchemy import text

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT COUNT(*) AS total,"
                " COUNT(*) FILTER (WHERE alias != canonical) AS resolved"
                " FROM wc2026.team_aliases"
            )
        ).fetchone()

    total, resolved = rows
    unresolved = total - resolved
    print(f"Total aliases : {total}")
    print(f"Resolved      : {resolved}")
    print(f"Unresolved    : {unresolved}")

    canon = Canonicalizer()
    test_names = ["United States", "Korea Republic", "China PR"]
    for name in test_names:
        print(f"  {name!r} → {canon.canonicalize(name)!r}")
