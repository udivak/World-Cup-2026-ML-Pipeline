from src.aggregate.squad_assembly import build_squad, position_to_unit


# ------------------------------------------------------------- position → unit
def test_position_to_unit_handles_all_sources():
    assert position_to_unit("GK") == "GK"
    assert position_to_unit("DF") == "DEF"
    assert position_to_unit("CB") == "DEF"
    assert position_to_unit("MF") == "MID"
    assert position_to_unit("CDM") == "MID"
    assert position_to_unit("FW") == "ATT"
    # FIFA comma list → first token wins.
    assert position_to_unit("RW, ST, CF") == "ATT"
    # Roster sort-key prefix and full-word spellings still resolve.
    assert position_to_unit("1GK") == "GK"
    assert position_to_unit("Goalkeeper") == "GK"


def test_position_to_unit_unknown_is_none():
    assert position_to_unit(None) is None
    assert position_to_unit("") is None
    assert position_to_unit("XYZ") is None
    assert position_to_unit(float("nan")) is None


# --------------------------------------------------------------- best XI / depth
def _squad_fixture():
    return [
        {"player_id": 1, "overall": 90, "position": "GK"},
        {"player_id": 2, "overall": 70, "position": "GK"},   # backup keeper → depth
        {"player_id": 3, "overall": 88, "position": "CB"},
        {"player_id": 4, "overall": 86, "position": "RB"},
        {"player_id": 5, "overall": 85, "position": "LB"},
        {"player_id": 6, "overall": 84, "position": "CB"},
        {"player_id": 7, "overall": 80, "position": "CB"},   # 5th defender → depth
        {"player_id": 8, "overall": 89, "position": "CM"},
        {"player_id": 9, "overall": 87, "position": "CDM"},
        {"player_id": 10, "overall": 86, "position": "CAM"},
        {"player_id": 11, "overall": 75, "position": "CM"},  # 4th mid → depth
        {"player_id": 12, "overall": 91, "position": "ST"},
        {"player_id": 13, "overall": 90, "position": "LW"},
        {"player_id": 14, "overall": 88, "position": "RW"},
        {"player_id": 15, "overall": 70, "position": "ST"},  # 4th att → depth
    ]


def test_best_xi_fills_formation_by_overall():
    squad = build_squad(_squad_fixture())
    xi_ids = {p["player_id"] for p in squad["xi"]}
    assert len(squad["xi"]) == 11
    assert xi_ids == {1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14}
    assert [p["player_id"] for p in squad["by_unit"]["GK"]] == [1]
    assert len(squad["by_unit"]["DEF"]) == 4
    assert len(squad["by_unit"]["MID"]) == 3
    assert len(squad["by_unit"]["ATT"]) == 3


def test_depth_is_the_remainder_sorted_by_overall():
    squad = build_squad(_squad_fixture())
    depth_ids = [p["player_id"] for p in squad["depth"]]
    # Five players left over: 5th defender, backup keeper, 4th mid, 4th att.
    assert set(depth_ids) == {2, 7, 11, 15}
    # Highest-overall first.
    overalls = [p["overall"] for p in squad["depth"]]
    assert overalls == sorted(overalls, reverse=True)


def test_substitutes_cap_limits_depth():
    squad = build_squad(_squad_fixture(), substitutes=2)
    assert len(squad["depth"]) == 2
    # The two strongest leftovers (the 5th defender 80, the 4th mid 75).
    assert [p["player_id"] for p in squad["depth"]] == [7, 11]


def test_missing_attributes_do_not_raise_and_sink_to_depth():
    players = [
        {"player_id": 1, "overall": 90, "position": "GK"},
        {"player_id": 2, "overall": None, "position": "CB"},   # no overall
        {"player_id": 3, "overall": 80, "position": None},     # no position/unit
    ]
    squad = build_squad(players)
    # GK fills its slot; the no-position player has no unit → depth.
    assert {p["player_id"] for p in squad["by_unit"]["GK"]} == {1}
    assert 3 in {p["player_id"] for p in squad["depth"]}
    # The no-overall defender still fills the (otherwise empty) DEF slot — no error.
    assert 2 in {p["player_id"] for p in squad["xi"]}
