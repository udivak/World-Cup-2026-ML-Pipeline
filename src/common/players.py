"""Player identity canonicalization across FIFA / FM / roster sources.

Player names collide (many "Rodrigo"s, many "J. Silva"s), so a name alone cannot
identify a player. We key on the composite ``(normalized_name, birthdate, nationality)``
— birthdate + nationality disambiguate same-named players. ``players`` holds canonical
identities; ``player_aliases`` maps each source spelling to a ``player_id``.

The canonicalizer is a pure in-memory structure (no DB required) so it is fully
fixture-testable. Loaders drive it: FIFA *seeds* identities via :meth:`add` (the richest
nationality/birthdate coverage), then FM / roster spellings are *matched* against the
seeded set via :meth:`canonicalize`. Unmatched / ambiguous names are surfaced to a review
log + CSV — never silently dropped.
"""

import csv
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


def normalize_name(name: Any) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace.

    ``"Müller"`` → ``"muller"``, ``"L. Messi"`` → ``"l messi"``,
    ``"N'Golo Kanté"`` → ``"n golo kante"``. Deterministic and idempotent.
    """
    if name is None:
        return ""
    # pandas/NumPy NaN floats are not equal to themselves.
    if isinstance(name, float) and name != name:  # noqa: PLR0124
        return ""
    text = str(name)
    decomposed = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = no_accents.lower()
    # Drop intra-name apostrophes/periods entirely so "N'Golo"/"Ngolo" and "L."/"L"
    # converge; turn remaining punctuation (hyphens, slashes, commas) into spaces.
    deticked = re.sub(r"[’'.]", "", lowered)
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", deticked)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_date(value: Any) -> Optional[str]:
    """Coerce a date-like value to an ISO ``YYYY-MM-DD`` string, or ``None``."""
    if value is None:
        return None
    # pandas NaT / NaN are not equal to themselves.
    if value != value:  # noqa: PLR0124
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    return text[:10]


def _norm_nat(value: Any) -> Optional[str]:
    if value is None or value != value:  # noqa: PLR0124
        return None
    text = str(value).strip()
    return text or None


class PlayerCanonicalizer:
    """Resolve source player names to stable ``player_id`` integers.

    Construct with an explicit ``players`` iterable for fixture-backed use, or omit it
    to load ``wc2026.players`` from the DB. Each player record needs ``player_id`` and
    ``canonical_name``; ``birthdate`` and ``nationality`` are optional but strongly
    recommended (they are what make same-named players distinguishable).
    """

    def __init__(self, players: Optional[Iterable[dict]] = None) -> None:
        self._full: dict[tuple[str, Optional[str], Optional[str]], int] = {}
        self._by_name: dict[str, set[int]] = {}
        self._by_birthdate: dict[str, set[int]] = {}
        self._records: dict[int, dict] = {}
        self._next_id = 1
        self.unmatched: list[dict] = []

        if players is None:
            players = self._load_from_db()
        for p in players:
            self._register(
                int(p["player_id"]),
                p["canonical_name"],
                p.get("birthdate"),
                p.get("nationality"),
            )

    # ------------------------------------------------------------------ loading
    def _load_from_db(self) -> list[dict]:
        from sqlalchemy import text

        from src.common.db import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT player_id, canonical_name, birthdate, nationality"
                    " FROM wc2026.players"
                )
            ).mappings().all()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ internals
    def _register(
        self,
        pid: int,
        name: Any,
        birthdate: Any,
        nationality: Any,
    ) -> None:
        norm = normalize_name(name)
        bd = normalize_date(birthdate)
        nat = _norm_nat(nationality)
        self._full[(norm, bd, nat)] = pid
        self._by_name.setdefault(norm, set()).add(pid)
        if bd is not None:
            self._by_birthdate.setdefault(bd, set()).add(pid)
        self._records[pid] = {
            "player_id": pid,
            "canonical_name": name,
            "normalized_name": norm,
            "birthdate": bd,
            "nationality": nat,
        }
        if pid >= self._next_id:
            self._next_id = pid + 1

    def _token_birthdate_match(
        self, norm: str, bd: str, nat: Optional[str], require_nat: bool = False
    ) -> Optional[int]:
        """Unique same-birthdate player whose name tokens nest with the query's, or None.

        With ``require_nat`` the candidate's nationality must equal ``nat`` (used when
        merging same-player spelling variants during seeding, where nationality is reliable).
        """
        q_tokens = set(norm.split())
        if not q_tokens:
            return None
        matches = set()
        for pid in self._by_birthdate.get(bd, set()):
            if require_nat and self._records[pid]["nationality"] != nat:
                continue
            cand_tokens = set(self._records[pid]["normalized_name"].split())
            if q_tokens <= cand_tokens or cand_tokens <= q_tokens:
                matches.add(pid)
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) > 1 and nat is not None and not require_nat:
            nat_matches = {
                pid for pid in matches if self._records[pid]["nationality"] == nat
            }
            if len(nat_matches) == 1:
                return next(iter(nat_matches))
        return None

    # ------------------------------------------------------------------ public API
    def add(self, name: Any, birthdate: Any = None, nationality: Any = None) -> int:
        """Idempotently insert a player; return its ``player_id``.

        Keyed on ``(normalized_name, birthdate, nationality)`` — calling twice with the
        same key returns the same id. Used to *seed* identities from the FIFA feed.
        """
        norm = normalize_name(name)
        bd = normalize_date(birthdate)
        nat = _norm_nat(nationality)
        key = (norm, bd, nat)
        existing = self._full.get(key)
        if existing is not None:
            return existing

        # Merge spelling variants of the SAME player across editions: editions disagree on
        # name form (FIFA's legal "Lionel Andrés Messi" vs FC's common "Lionel Messi"), so a
        # same-(birthdate, nationality) record with nesting name tokens is the same person —
        # collapse onto it so all editions share one identity (else historical attribute
        # snapshots fragment away from the common-name id). Conservative: unique match only.
        if bd is not None and nat is not None:
            merged = self._token_birthdate_match(norm, bd, nat, require_nat=True)
            if merged is not None:
                self._full[key] = merged
                self._by_name.setdefault(norm, set()).add(merged)
                return merged

        pid = self._next_id
        self._register(pid, name, bd, nat)
        return pid

    def canonicalize(
        self,
        name: Any,
        birthdate: Any = None,
        nationality: Any = None,
        source: str = "fm",
    ) -> Optional[int]:
        """Match a source spelling to an existing ``player_id`` (read-only).

        Resolution order: exact composite key → ``(name, nationality)`` if unique →
        ``name`` if unique. Returns ``None`` for unmatched or ambiguous names and records
        them in :attr:`unmatched` for review. Never creates a player.
        """
        norm = normalize_name(name)
        bd = normalize_date(birthdate)
        nat = _norm_nat(nationality)

        exact = self._full.get((norm, bd, nat))
        if exact is not None:
            return exact

        # Birthdate is a strong disambiguator. Roster spellings often carry a DOB but a
        # nationality string that differs from FIFA's (e.g. "South Korea" vs "Korea
        # Republic"), so the exact triple misses. If (name, birthdate) is unique, accept
        # it regardless of the nationality string.
        if bd is not None:
            bd_matches = {
                pid
                for pid in self._by_name.get(norm, set())
                if self._records[pid]["birthdate"] == bd
            }
            if len(bd_matches) == 1:
                return next(iter(bd_matches))

            # Common names drop or add the middle names FIFA's legal long_name carries
            # ("Lionel Messi" vs "Lionel Andrés Messi"; "Richarlison" vs "Richarlison de
            # Andrade"). Among players sharing this exact birthdate, accept the one whose
            # name tokens are a superset/subset of the query's — if unique (birthdate makes
            # this high-precision). Prefer a nationality match to break ties.
            token_match = self._token_birthdate_match(norm, bd, nat)
            if token_match is not None:
                return token_match

        candidates = self._by_name.get(norm, set())
        if nat is not None and candidates:
            nat_matches = {
                pid for pid in candidates if self._records[pid]["nationality"] == nat
            }
            if len(nat_matches) == 1:
                return next(iter(nat_matches))
            if len(nat_matches) > 1:
                candidates = nat_matches

        if len(candidates) == 1:
            return next(iter(candidates))

        reason = "ambiguous" if len(candidates) > 1 else "unmatched"
        self.unmatched.append(
            {
                "name": name,
                "birthdate": bd,
                "nationality": nat,
                "source": source,
                "reason": reason,
            }
        )
        logger.warning(
            "%s player name (source=%s): %r [%s, %s]", reason, source, name, bd, nat
        )
        return None

    # ------------------------------------------------------------------ accessors
    def record(self, pid: int) -> dict:
        return self._records[pid]

    def players(self) -> list[dict]:
        """All canonical player records, ordered by id."""
        return [self._records[pid] for pid in sorted(self._records)]

    @property
    def unmatched_rate(self) -> float:
        attempts = len(self.unmatched) + len(self._records)
        return len(self.unmatched) / attempts if attempts else 0.0

    def write_review(self, raw_data_dir: str | Path) -> Optional[Path]:
        """Persist unmatched/ambiguous names to a review CSV + log. Returns CSV path."""
        if not self.unmatched:
            return None
        raw_dir = Path(raw_data_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        log_path = raw_dir / "unmatched_players.log"
        names = sorted({str(u["name"]) for u in self.unmatched})
        log_path.write_text("\n".join(names) + "\n")

        csv_path = raw_dir / "player_review.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["name", "birthdate", "nationality", "source", "reason"]
            )
            writer.writeheader()
            writer.writerows(self.unmatched)
        logger.warning(
            "%d unmatched/ambiguous player names → %s", len(self.unmatched), csv_path
        )
        return csv_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from sqlalchemy import text

    from src.common.db import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM wc2026.players")).scalar()
        per_source = conn.execute(
            text(
                "SELECT source, COUNT(*) FROM wc2026.player_aliases GROUP BY source"
            )
        ).all()
    print(f"Canonical players: {total}")
    for source, n in per_source:
        print(f"  aliases [{source}]: {n}")
