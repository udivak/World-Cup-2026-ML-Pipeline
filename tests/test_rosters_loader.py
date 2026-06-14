import pytest

pytest.importorskip("bs4")

from src.collect.rosters_loader import _slug, parse_squads_html

# Minimal two-team fixture mirroring a Wikipedia "<event> squads" article: a team heading,
# a non-squad table (group standings) that must be ignored, then squad wikitables with the
# standard No./Pos./Player/DOB/Caps/Club columns and a span.bday DOB.
_FIXTURE_HTML = """
<h2>Group A</h2>
<table class="wikitable">
  <tr><th>Team</th><th>Pld</th><th>Pts</th></tr>
  <tr><td>Argentina</td><td>3</td><td>9</td></tr>
</table>
<h3>Argentina</h3>
<p>Head coach: Lionel Scaloni</p>
<table class="wikitable">
  <tr><th>No.</th><th>Pos.</th><th>Player</th><th>Date of birth (age)</th><th>Caps</th><th>Club</th></tr>
  <tr>
    <td>1</td><td>GK</td>
    <td><a href="/wiki/Emiliano_Martinez">Emiliano Martínez</a></td>
    <td><span class="bday">1992-09-02</span> (aged&nbsp;30)</td>
    <td>17</td><td><a href="/wiki/Aston_Villa">Aston Villa</a></td>
  </tr>
  <tr>
    <td>10</td><td>FW</td>
    <td><a href="/wiki/Lionel_Messi">Lionel Messi</a> (captain)</td>
    <td><span class="bday">1987-06-24</span> (aged&nbsp;35)</td>
    <td>165</td><td>Paris Saint-Germain</td>
  </tr>
</table>
<h3>Brazil</h3>
<table class="wikitable">
  <tr><th>No.</th><th>Pos.</th><th>Player</th><th>Date of birth (age)</th><th>Caps</th><th>Club</th></tr>
  <tr>
    <td>9</td><td>FW</td><td>Richarlison</td>
    <td><span class="bday">1997-05-10</span></td><td>40</td><td>Tottenham Hotspur</td>
  </tr>
</table>
"""


def test_slug():
    assert _slug("2022 FIFA World Cup squads") == "2022_FIFA_World_Cup_squads"


def test_parse_extracts_players_per_team():
    df = parse_squads_html(_FIXTURE_HTML, "FIFA World Cup", 2022)
    assert len(df) == 3
    assert set(df["team"]) == {"Argentina", "Brazil"}
    assert (df["tournament"] == "FIFA World Cup").all()
    assert (df["edition_year"] == 2022).all()


def test_parse_player_fields_and_annotation_stripping():
    df = parse_squads_html(_FIXTURE_HTML, "FIFA World Cup", 2022)
    messi = df[df["player_name"] == "Lionel Messi"].iloc[0]
    # "(captain)" annotation stripped from the name.
    assert messi["player_name"] == "Lionel Messi"
    assert messi["shirt_no"] == 10
    assert messi["position"] == "FW"
    assert messi["dob"] == "1987-06-24"   # parsed from span.bday
    assert messi["caps"] == 165
    assert messi["club"] == "Paris Saint-Germain"


def test_parse_ignores_non_squad_tables():
    df = parse_squads_html(_FIXTURE_HTML, "FIFA World Cup", 2022)
    # The group-standings table (Team/Pld/Pts) under "Group A" must not produce rows.
    assert "Argentina" not in set(df["player_name"])
    assert (df["dob"].notna()).all()
