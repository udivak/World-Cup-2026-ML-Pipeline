import pandas as pd
import pytest

from src.collect.fm_loader import parse_fm_csv

# FM export: identity meta columns + 0–20 attribute columns.
_FM_CSV = """\
Name,Age,Club,Nationality,Born,Position,Finishing,Dribbling,Passing,Tackling
Lionel Messi,34,PSG,Argentina,1987-06-24,AM (R),18,20,19,8
Virgil van Dijk,30,Liverpool,Netherlands,1991-07-08,D (C),9,11,13,19
"""


@pytest.fixture
def fm_path(tmp_path):
    p = tmp_path / "fm_export.csv"
    p.write_text(_FM_CSV)
    return p


def test_parse_identity_columns(fm_path):
    df = parse_fm_csv(fm_path, season_year=2022)
    assert len(df) == 2
    assert (df["source"] == "fm").all()
    messi = df[df["name"] == "Lionel Messi"].iloc[0]
    assert messi["nationality"] == "Argentina"
    assert messi["birthdate"] == "1987-06-24"


def test_attribute_columns_go_to_attrs(fm_path):
    df = parse_fm_csv(fm_path, season_year=2022)
    attrs = df[df["name"] == "Virgil van Dijk"].iloc[0]["attrs"]
    assert attrs["Tackling"] == 19
    assert attrs["Finishing"] == 9
    # Meta columns must NOT leak into attrs.
    assert "Club" not in attrs
    assert "Nationality" not in attrs
    assert "Name" not in attrs
