import pandas as pd
import pytest

from src.collect.fifa_loader import parse_fifa_csv, season_from_filename

# Tiny sofifa-schema fixture: identity + a few typed cols + attribute long tail + a URL col.
_FIXTURE_CSV = """\
sofifa_id,player_url,short_name,long_name,player_positions,overall,potential,value_eur,age,dob,nationality_name,club_name,league_name,preferred_foot,pace,shooting,player_face_url
158023,https://x/messi,L. Messi,Lionel Andrés Messi,"RW, ST, CF",93,93,78000000,34,1987-06-24,Argentina,Paris Saint-Germain,French Ligue 1,Left,85,92,https://img/messi
20801,https://x/cr7,Cristiano Ronaldo,Cristiano Ronaldo dos Santos,"ST, LW",91,91,45000000,36,1985-02-05,Portugal,Manchester United,English Premier League,Right,87,94,https://img/cr7
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "players_22.csv"
    p.write_text(_FIXTURE_CSV)
    return p


def test_season_from_filename():
    assert season_from_filename("players_22.csv") == 2022
    assert season_from_filename("/a/b/players_18.csv") == 2018
    assert season_from_filename("fc24.csv") == 2024
    assert season_from_filename("players_2026.csv") == 2026


def test_parse_typed_columns(csv_path):
    df = parse_fifa_csv(csv_path)
    assert len(df) == 2
    assert (df["source"] == "fifa").all()
    assert (df["season_year"] == 2022).all()
    messi = df[df["name"] == "Lionel Andrés Messi"].iloc[0]
    assert messi["overall"] == 93
    assert messi["nationality"] == "Argentina"
    assert messi["primary_position"] == "RW"
    assert messi["club"] == "Paris Saint-Germain"
    assert messi["normalized_name"] == "lionel andres messi"


def test_attrs_jsonb_packs_full_long_tail(csv_path):
    df = parse_fifa_csv(csv_path)
    attrs = df.iloc[0]["attrs"]
    assert isinstance(attrs, dict)
    # Attribute long tail is preserved...
    assert attrs["pace"] == 85
    assert attrs["shooting"] == 92
    # ...renamed typed keys are present in attrs too...
    assert attrs["nationality"] == "Argentina"
    # ...but URL/id columns are excluded to keep payloads lean.
    assert "player_face_url" not in attrs
    assert "player_url" not in attrs
    assert "sofifa_id" not in attrs


def test_short_name_preserved(csv_path):
    df = parse_fifa_csv(csv_path)
    assert set(df["short_name"]) == {"L. Messi", "Cristiano Ronaldo"}
