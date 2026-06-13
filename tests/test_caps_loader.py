import pytest

from src.collect.caps_loader import parse_caps_csv

_CAPS_CSV = """\
name,nationality,as_of_date,caps,wc_apps,continental_apps
Lionel Messi,Argentina,2022-06-01,162,19,34
Harry Kane,England,2022-06-01,73,11,5
"""


@pytest.fixture
def caps_path(tmp_path):
    p = tmp_path / "caps.csv"
    p.write_text(_CAPS_CSV)
    return p


def test_parse_caps_columns(caps_path):
    df = parse_caps_csv(caps_path)
    assert len(df) == 2
    messi = df[df["name"] == "Lionel Messi"].iloc[0]
    assert messi["caps"] == 162
    assert messi["wc_apps"] == 19
    assert messi["as_of_date"] == "2022-06-01"
    assert messi["nationality"] == "Argentina"


def test_missing_optional_columns_are_none(tmp_path):
    p = tmp_path / "caps_min.csv"
    p.write_text("name,as_of_date,caps\nKylian Mbappe,2022-06-01,59\n")
    df = parse_caps_csv(p)
    row = df.iloc[0]
    assert row["caps"] == 59
    assert row["wc_apps"] is None
    assert row["continental_apps"] is None
