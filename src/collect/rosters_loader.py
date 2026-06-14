"""National-team roster ingestion from Wikipedia squad pages (Phase 2).

Each tournament edition has a Wikipedia "<event> squads" article with one wikitable per
national team (columns: No., Pos., Player, Date of birth, Caps, Club). We fetch that article
(cached to ``data/raw/rosters/``), parse every squad table, associate it with its team via
the nearest preceding heading, and resolve players + teams through the Phase-1
``PlayerCanonicalizer`` and ``teams.Canonicalizer``.

Output: ``wc2026.rosters`` — the links between players and teams. The squad tables also carry
DOB and caps, so we record roster ``player_aliases`` and seed ``caps_snapshots`` (as-of the
tournament start) as a side effect — both legitimate pre-tournament inputs (no leakage).

Degrades gracefully: a missing/unparseable page logs a warning and contributes nothing;
unmatched player spellings keep their raw name (``player_id`` NULL) and surface to a review
CSV — never silently dropped.

Run: ``python -m src.collect.rosters_loader``  (``--refresh`` to bypass the HTML cache).
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd
import requests

from src.common.config import load_config
from src.common.players import PlayerCanonicalizer, normalize_date

logger = logging.getLogger(__name__)

_WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/html/"
# Wikipedia asks for a descriptive User-Agent; anonymous default UAs get 403s.
_HEADERS = {
    "User-Agent": "WC2026-Pipeline/0.1 (https://github.com/udivak/World-Cup-2026-ML-Pipeline)"
}


def _slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")


def fetch_wiki_html(
    page_title: str, cache_dir: Path, refresh: bool = False
) -> Optional[str]:
    """Return the rendered HTML of a Wikipedia article, caching it under ``cache_dir``.

    Reads the cache unless ``refresh``; on a miss fetches the Parsoid REST HTML. Returns
    ``None`` (and logs) on any HTTP error so the caller can skip that source cleanly.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_slug(page_title)}.html"
    if cache_path.exists() and cache_path.stat().st_size > 0 and not refresh:
        logger.info("Using cached %s", cache_path.name)
        return cache_path.read_text(encoding="utf-8")

    url = _WIKI_REST + quote(page_title, safe="")
    logger.info("Fetching %s", url)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=60)
    except requests.RequestException as exc:
        logger.warning("Fetch failed for %r: %s", page_title, exc)
        return None
    if resp.status_code != 200:
        logger.warning("HTTP %d for %r — skipping.", resp.status_code, page_title)
        return None
    cache_path.write_text(resp.text, encoding="utf-8")
    return resp.text


# ------------------------------------------------------------------ HTML parsing
def _norm_header(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)  # footnote markers
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _heading_text(el) -> str:
    for junk in el.select("span.mw-editsection"):
        junk.decompose()
    text = el.get_text(" ", strip=True)
    text = re.sub(r"\[.*?\]", "", text)  # footnote markers
    return re.sub(r"\s+", " ", text).strip()


def _header_index(rows) -> Optional[dict[str, int]]:
    """Find the squad-table header row and map logical columns → cell index.

    Returns ``None`` if no row identifies the table as a squad table (needs a Player column
    and a Pos column).
    """
    for tr in rows:
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            cells = tr.find_all(["th", "td"])
        headers = [_norm_header(c.get_text(" ", strip=True)) for c in cells]
        if not any("player" in h for h in headers):
            continue
        idx: dict[str, int] = {}
        for i, h in enumerate(headers):
            if "player" in h and "player" not in idx:
                idx["player"] = i
            elif h.startswith("pos") and "pos" not in idx:
                idx["pos"] = i
            elif h in {"no", "no ", "number", ""} and "no" not in idx and i == 0:
                idx["no"] = i
            elif "date of birth" in h or h == "dob":
                idx["dob"] = i
            elif "caps" in h and "caps" not in idx:
                idx["caps"] = i
            elif "club" in h and "club" not in idx:
                idx["club"] = i
        if "player" in idx and "pos" in idx:
            return idx
    return None


def _cell(cells, idx: dict, key: str):
    i = idx.get(key)
    if i is None or i >= len(cells):
        return None
    return cells[i]


def _int_in(text: object) -> Optional[int]:
    if text is None:
        return None
    m = re.search(r"\d+", str(text))
    return int(m.group()) if m else None


def _clean_player(cell) -> str:
    text = cell.get_text(" ", strip=True)
    text = re.sub(r"\(.*?\)", "", text)  # (captain), (c)
    text = re.sub(r"\[.*?\]", "", text)  # footnotes
    return re.sub(r"\s+", " ", text).strip()


def parse_squads_html(html: str, tournament: str, edition_year: int) -> pd.DataFrame:
    """Parse a "<event> squads" article into roster rows (pure: no DB / no canonicalize).

    Columns: ``tournament, edition_year, team, player_name, shirt_no, position, dob, caps,
    club``. ``team`` is the raw heading text (canonicalized by the loader).
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []
    current_team: Optional[str] = None

    for el in soup.find_all(["h2", "h3", "h4", "table"]):
        if el.name != "table":
            heading = _heading_text(el)
            if heading:
                current_team = heading
            continue
        if current_team is None:
            continue
        trs = el.find_all("tr")
        idx = _header_index(trs)
        if idx is None:
            continue  # not a squad table (e.g. a group-standings wikitable)
        for tr in trs:
            cells = tr.find_all(["th", "td"])
            player_cell = _cell(cells, idx, "player")
            if player_cell is None:
                continue
            name = _clean_player(player_cell)
            # Skip the header row and any blank rows.
            if not name or _norm_header(name) in {"player", ""}:
                continue
            pos_cell = _cell(cells, idx, "pos")
            position = None
            if pos_cell is not None:
                letters = "".join(
                    ch for ch in pos_cell.get_text(strip=True).upper() if ch.isalpha()
                )
                position = letters or None
            dob = None
            dob_cell = _cell(cells, idx, "dob")
            if dob_cell is not None:
                bday = dob_cell.select_one("span.bday")
                if bday is not None:
                    dob = normalize_date(bday.get_text(strip=True))
            club_cell = _cell(cells, idx, "club")
            club = club_cell.get_text(" ", strip=True) if club_cell is not None else None
            if club:
                club = re.sub(r"\[.*?\]", "", club).strip() or None
            out.append(
                {
                    "tournament": tournament,
                    "edition_year": edition_year,
                    "team": current_team,
                    "player_name": name,
                    "shirt_no": _int_in(
                        _cell(cells, idx, "no").get_text() if _cell(cells, idx, "no") else None
                    ),
                    "position": position,
                    "dob": dob,
                    "caps": _int_in(
                        _cell(cells, idx, "caps").get_text() if _cell(cells, idx, "caps") else None
                    ),
                    "club": club,
                }
            )
    return pd.DataFrame(out)


# ------------------------------------------------------------------ load + write
def load_rosters(write: bool = True, refresh: bool = False) -> pd.DataFrame:
    """Fetch + parse every configured roster source; canonicalize; upsert ``rosters``.

    Returns the combined parsed frame with a resolved ``player_id`` column (NaN where
    unmatched). Sources that fail to fetch/parse contribute nothing (graceful degradation).
    """
    cfg = load_config()
    rosters_dir = Path(cfg.raw_data_dir) / cfg.rosters.raw_subdir
    if not cfg.rosters.sources:
        logger.info("No roster sources configured — nothing to ingest.")
        return pd.DataFrame()

    frames = []
    start_dates: dict[tuple[str, int], str] = {}
    for src in cfg.rosters.sources:
        start_dates[(src.tournament, src.edition_year)] = src.start_date
        html = fetch_wiki_html(src.wiki_page, rosters_dir, refresh=refresh)
        if not html:
            continue
        frame = parse_squads_html(html, src.tournament, src.edition_year)
        logger.info(
            "%s %d: parsed %d players across %d teams.",
            src.tournament, src.edition_year, len(frame),
            frame["team"].nunique() if not frame.empty else 0,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        logger.warning("No roster data parsed from any source.")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    # Canonicalize teams and players.
    team_canon = _team_canonicalizer()
    combined["team"] = combined["team"].map(
        lambda t: team_canon.canonicalize(t) if team_canon else t
    )

    canon = PlayerCanonicalizer()
    combined["player_id"] = [
        canon.canonicalize(name, dob, nat, source="roster")
        for name, dob, nat in zip(
            combined["player_name"], combined["dob"], combined["team"]
        )
    ]
    matched = combined["player_id"].notna().sum()
    rate = matched / len(combined) if len(combined) else 0.0
    logger.info(
        "Matched %d/%d roster players (%.1f%% unmatched).",
        matched, len(combined), 100 * (1 - rate),
    )
    canon.write_review(rosters_dir)

    if write:
        _write_rosters(combined, start_dates)
    return combined


def _team_canonicalizer():
    try:
        from src.common.teams import Canonicalizer

        return Canonicalizer()
    except Exception as exc:  # no DB / no aliases → keep raw heading names
        logger.warning("Team canonicalizer unavailable (%s); using raw team names.", exc)
        return None


def _write_rosters(
    combined: pd.DataFrame, start_dates: dict[tuple[str, int], str]
) -> None:
    from src.common.io import bulk_upsert

    # Deduplicate on the natural key before upsert so ON CONFLICT never targets a row twice.
    deduped = combined.drop_duplicates(
        subset=["tournament", "edition_year", "team", "player_name"], keep="first"
    )

    roster_rows, alias_rows, caps_rows = [], [], []
    seen_alias: set[tuple] = set()
    seen_caps: set[tuple] = set()
    for _, row in deduped.iterrows():
        pid = row["player_id"]
        pid = int(pid) if pid is not None and pid == pid else None  # noqa: PLR0124
        roster_rows.append(
            {
                "tournament": row["tournament"],
                "edition_year": int(row["edition_year"]),
                "team": row["team"],
                "player_id": pid,
                "player_name": row["player_name"],
                "shirt_no": _opt_int(row["shirt_no"]),
                "position": row["position"],
                "dob": row["dob"],
                "caps": _opt_int(row["caps"]),
                "club": row["club"],
            }
        )
        if pid is not None:
            akey = (row["player_name"], "roster", pid)
            if akey not in seen_alias:
                seen_alias.add(akey)
                alias_rows.append(
                    {"alias": row["player_name"], "source": "roster", "player_id": pid}
                )
            caps = _opt_int(row["caps"])
            as_of = start_dates.get((row["tournament"], int(row["edition_year"])))
            ckey = (pid, as_of)
            if caps is not None and as_of is not None and ckey not in seen_caps:
                seen_caps.add(ckey)
                caps_rows.append(
                    {"player_id": pid, "as_of_date": as_of, "caps": caps}
                )

    bulk_upsert(
        "rosters",
        roster_rows,
        conflict_cols=["tournament", "edition_year", "team", "player_name"],
        update_cols=["player_id", "shirt_no", "position", "dob", "caps", "club"],
    )
    bulk_upsert(
        "player_aliases",
        alias_rows,
        conflict_cols=["alias", "source", "player_id"],
    )
    if caps_rows:
        bulk_upsert(
            "caps_snapshots",
            caps_rows,
            conflict_cols=["player_id", "as_of_date"],
            update_cols=["caps"],
        )
    logger.info(
        "Upserted %d roster rows, %d roster aliases, %d caps snapshots.",
        len(roster_rows), len(alias_rows), len(caps_rows),
    )


def _opt_int(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and value != value):  # noqa: PLR0124
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    refresh = "--refresh" in sys.argv
    df = load_rosters(write=True, refresh=refresh)
    if df.empty:
        print("No roster data ingested.")
    else:
        matched = df["player_id"].notna().sum()
        print(f"\nRoster rows parsed : {len(df)}")
        print(f"Teams              : {df['team'].nunique()}")
        print(f"Matched players    : {matched} ({100 * matched / len(df):.1f}%)")
        print("By edition:")
        for (t, y), n in df.groupby(["tournament", "edition_year"]).size().items():
            print(f"  {t} {y}: {n}")
