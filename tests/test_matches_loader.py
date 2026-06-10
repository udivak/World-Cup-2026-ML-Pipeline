import io
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.collect.matches_loader import load_raw_matches, _derive_result


_FIXTURE_CSV = """\
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
1990-01-01,Brazil,Argentina,3,1,Friendly,Rio de Janeiro,Brazil,FALSE
1990-01-02,Germany,France,1,1,Friendly,Berlin,Germany,FALSE
1990-01-03,Spain,Italy,0,2,Friendly,Madrid,Spain,FALSE
1990-01-04,USA,Mexico,2,0,CONCACAF Gold Cup,Los Angeles,USA,FALSE
1990-01-05,England,Portugal,1,2,Friendly,London,England,TRUE
"""


@pytest.fixture
def raw_df():
    df = pd.read_csv(io.StringIO(_FIXTURE_CSV), parse_dates=["date"])
    return df


def test_result_derivation():
    row_h = pd.Series({"home_score": 3, "away_score": 1})
    row_d = pd.Series({"home_score": 1, "away_score": 1})
    row_a = pd.Series({"home_score": 0, "away_score": 2})
    assert _derive_result(row_h) == "H"
    assert _derive_result(row_d) == "D"
    assert _derive_result(row_a) == "A"


def test_result_column_present(raw_df, tmp_path):
    csv_path = tmp_path / "results.csv"
    raw_df.to_csv(csv_path, index=False)

    with patch("src.collect.matches_loader._csv_path", return_value=csv_path), \
         patch("src.collect.matches_loader._null_scores_log", return_value=tmp_path / "null_scores.log"):
        df = load_raw_matches()

    assert "result" in df.columns
    assert set(df["result"]).issubset({"H", "D", "A"})


def test_no_null_results(raw_df, tmp_path):
    csv_path = tmp_path / "results.csv"
    raw_df.to_csv(csv_path, index=False)

    with patch("src.collect.matches_loader._csv_path", return_value=csv_path), \
         patch("src.collect.matches_loader._null_scores_log", return_value=tmp_path / "null_scores.log"):
        df = load_raw_matches()

    assert df["result"].isna().sum() == 0


def test_null_score_rows_excluded(tmp_path):
    csv_with_null = _FIXTURE_CSV + "1990-01-06,Chile,Peru,,1,Friendly,Santiago,Chile,FALSE\n"
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(csv_with_null)
    null_log = tmp_path / "null_scores.log"

    with patch("src.collect.matches_loader._csv_path", return_value=csv_path), \
         patch("src.collect.matches_loader._null_scores_log", return_value=null_log):
        df = load_raw_matches()

    assert len(df) == 5
    assert null_log.exists()


def test_correct_result_distribution(raw_df, tmp_path):
    csv_path = tmp_path / "results.csv"
    raw_df.to_csv(csv_path, index=False)

    with patch("src.collect.matches_loader._csv_path", return_value=csv_path), \
         patch("src.collect.matches_loader._null_scores_log", return_value=tmp_path / "null_scores.log"):
        df = load_raw_matches()

    counts = df["result"].value_counts()
    assert counts["H"] == 2
    assert counts["D"] == 1
    assert counts["A"] == 2
