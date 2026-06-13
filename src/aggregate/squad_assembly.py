"""Squad assembly: turn a ≤26-man roster into a best XI + depth (Phase 2).

The 2026 World Cup squad unit is **26 players = best XI (11) + 15 substitutes**. Given a
roster and each player's nearest-prior attributes, we map every position to one of four
units {GK, DEF, MID, ATT}, fill the configured formation (default 1-4-3-3) with the
highest-overall players per unit, and treat the remainder as depth.

This module owns the single position→unit mapping table reused across Phase 2 (FIFA/FM
attribute positions *and* roster-listed positions resolve through the same map). The
contract is clean: players without a known unit or without an overall sink to depth — they
never raise. ``build_squad`` is pure (no DB), so it is fully fixture-testable.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Single source of truth: every FIFA/FM/roster position code → one of four units.
# Keys are upper-cased and stripped of non-letters before lookup, so "1GK"/"gk"/"GK" and
# FIFA's comma lists ("RW, ST, CF" → first token "RW") all resolve.
_POSITION_UNIT: dict[str, str] = {
    # goalkeeper
    "GK": "GK",
    # defenders
    "DF": "DEF", "CB": "DEF", "LCB": "DEF", "RCB": "DEF",
    "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF", "WB": "DEF",
    # midfielders
    "MF": "MID", "DM": "MID", "CDM": "MID", "LDM": "MID", "RDM": "MID",
    "CM": "MID", "LCM": "MID", "RCM": "MID", "LM": "MID", "RM": "MID",
    "AM": "MID", "CAM": "MID", "LAM": "MID", "RAM": "MID",
    # attackers
    "FW": "ATT", "ST": "ATT", "CF": "ATT", "SS": "ATT",
    "LW": "ATT", "RW": "ATT", "LF": "ATT", "RF": "ATT",
    "LS": "ATT", "RS": "ATT", "LWF": "ATT", "RWF": "ATT",
    # full-word spellings (older roster pages)
    "GOALKEEPER": "GK", "DEFENDER": "DEF", "MIDFIELDER": "MID", "FORWARD": "ATT",
}

UNITS = ("GK", "DEF", "MID", "ATT")


def position_to_unit(position: object) -> Optional[str]:
    """Map a position code (any source) to a unit in {GK, DEF, MID, ATT}, or ``None``.

    Accepts FIFA comma-lists ("RW, ST, CF" → first token), roster codes ("DF", "1GK"),
    and FM codes. Returns ``None`` for blanks / unrecognized codes (the caller routes such
    players to depth rather than failing).
    """
    if position is None or (isinstance(position, float) and position != position):  # noqa: PLR0124
        return None
    first = str(position).split(",")[0].strip()
    key = "".join(ch for ch in first.upper() if ch.isalpha())
    return _POSITION_UNIT.get(key)


def _overall_key(player: dict) -> float:
    """Sort key: higher overall first, missing overall last."""
    ov = player.get("overall")
    if ov is None or (isinstance(ov, float) and ov != ov):  # noqa: PLR0124
        return float("-inf")
    return float(ov)


def build_squad(
    players: list[dict],
    formation: Optional[dict[str, int]] = None,
    substitutes: int = 15,
) -> dict:
    """Select the best XI for ``formation`` and split the rest into depth.

    ``players`` is a list of dicts with at least ``overall`` and a position. Each player's
    unit is taken from ``unit`` if present, else derived from ``position`` (or
    ``primary_position``) via :func:`position_to_unit`. Within each unit, the highest-overall
    players fill the formation's slots; everyone left over is depth (capped at
    ``substitutes``, highest-overall first).

    Returns ``{"xi": [...], "depth": [...], "by_unit": {unit: [xi players]}}``. Players with
    no resolvable unit, or no overall, drift to depth — never an error. Underfilled units
    simply yield a short XI.
    """
    formation = formation or {"gk": 1, "def": 4, "mid": 3, "att": 3}
    want = {u.upper(): n for u, n in formation.items()}

    # Annotate each player with a resolved unit (don't mutate caller's dicts).
    annotated = []
    for p in players:
        unit = p.get("unit") or position_to_unit(
            p.get("position") or p.get("primary_position") or p.get("positions")
        )
        annotated.append({**p, "unit": unit})

    by_unit: dict[str, list[dict]] = {u: [] for u in UNITS}
    unassigned: list[dict] = []
    for p in sorted(annotated, key=_overall_key, reverse=True):
        if p["unit"] in by_unit:
            by_unit[p["unit"]].append(p)
        else:
            unassigned.append(p)

    xi: list[dict] = []
    xi_by_unit: dict[str, list[dict]] = {u: [] for u in UNITS}
    leftover: list[dict] = list(unassigned)
    for unit in UNITS:
        n = want.get(unit, 0)
        pool = by_unit[unit]  # already overall-sorted desc
        xi_by_unit[unit] = pool[:n]
        xi.extend(pool[:n])
        leftover.extend(pool[n:])

    leftover.sort(key=_overall_key, reverse=True)
    depth = leftover[:substitutes]

    return {"xi": xi, "depth": depth, "by_unit": xi_by_unit}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    demo = [
        {"player_id": 1, "overall": 90, "position": "GK"},
        {"player_id": 2, "overall": 88, "position": "CB"},
        {"player_id": 3, "overall": 85, "position": "RB"},
        {"player_id": 4, "overall": 84, "position": "LB"},
        {"player_id": 5, "overall": 83, "position": "CB"},
        {"player_id": 6, "overall": 80, "position": "CB"},  # 5th defender → depth
        {"player_id": 7, "overall": 89, "position": "CM"},
        {"player_id": 8, "overall": 87, "position": "CDM"},
        {"player_id": 9, "overall": 86, "position": "CAM"},
        {"player_id": 10, "overall": 91, "position": "RW, ST"},
        {"player_id": 11, "overall": 90, "position": "ST"},
        {"player_id": 12, "overall": 88, "position": "LW"},
    ]
    squad = build_squad(demo)
    print(f"XI ({len(squad['xi'])}):", [p['player_id'] for p in squad['xi']])
    print("Depth:", [p['player_id'] for p in squad['depth']])
    for unit, members in squad["by_unit"].items():
        print(f"  {unit}: {[p['player_id'] for p in members]}")
