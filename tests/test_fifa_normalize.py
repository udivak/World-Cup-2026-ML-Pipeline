import pandas as pd

from src.collect.fifa_normalize import dedupe_multiversion, normalize_fc_ratings


def test_normalize_fc_ratings_maps_to_sofifa_schema():
    raw = pd.DataFrame(
        [
            {"id": 1, "firstName": "Mohamed", "lastName": "Salah", "commonName": None,
             "birthdate": "6/15/1992 12:00:00 AM", "overallRating": 91, "position": "RM",
             "alternatePositions": "RW", "nationality": "Egypt", "team": "Liverpool",
             "leagueName": "Premier League", "composure": 88},
            {"id": 2, "firstName": "Kylian", "lastName": "Mbappé", "commonName": "Mbappé",
             "birthdate": "12/20/1998 12:00:00 AM", "overallRating": 91, "position": "ST",
             "alternatePositions": "LW", "nationality": "France", "team": "Real Madrid",
             "leagueName": "La Liga", "composure": 90},
        ]
    )
    out = normalize_fc_ratings(raw)
    salah = out.iloc[0]
    # commonName missing → built from firstName + lastName.
    assert salah["long_name"] == "Mohamed Salah"
    assert salah["dob"] == "1992-06-15"            # M/D/YYYY → ISO
    assert salah["player_positions"] == "RM, RW"   # primary + alternates
    assert salah["overall"] == 91
    assert salah["nationality_name"] == "Egypt"
    assert salah["club_name"] == "Liverpool"
    assert salah["league_name"] == "Premier League"
    assert salah["mentality_composure"] == 88      # composure → mentality_composure
    # commonName present → used as name.
    assert out.iloc[1]["long_name"] == "Mbappé"


def test_dedupe_multiversion_keeps_latest_patch_per_player():
    raw = pd.DataFrame(
        [
            {"player_id": 10, "update_as_of": "2023-09-22", "overall": 80, "long_name": "A"},
            {"player_id": 10, "update_as_of": "2024-05-01", "overall": 83, "long_name": "A"},
            {"player_id": 20, "update_as_of": "2023-10-01", "overall": 75, "long_name": "B"},
        ]
    )
    out = dedupe_multiversion(raw)
    assert len(out) == 2
    # player_id aliased to sofifa_id; latest update kept (overall 83 for player 10).
    assert "sofifa_id" in out.columns
    a = out[out["sofifa_id"] == 10].iloc[0]
    assert a["overall"] == 83
