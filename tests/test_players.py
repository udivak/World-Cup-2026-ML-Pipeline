import pytest

from src.common.players import PlayerCanonicalizer, normalize_date, normalize_name


# ----------------------------------------------------------------- normalization
def test_normalize_strips_accents_case_punctuation():
    assert normalize_name("Müller") == "muller"
    assert normalize_name("L. Messi") == "l messi"
    # Apostrophes are dropped (not spaced) so spelling variants converge.
    assert normalize_name("N'Golo Kanté") == "ngolo kante"
    assert normalize_name("  Cristiano   Ronaldo  ") == "cristiano ronaldo"


def test_normalize_is_idempotent():
    once = normalize_name("João Félix")
    assert normalize_name(once) == once


def test_normalize_handles_none_and_nan():
    assert normalize_name(None) == ""
    assert normalize_name(float("nan")) == ""  # str(nan) -> "nan"; only digits/letters kept


def test_normalize_date_variants():
    assert normalize_date("1987-06-24") == "1987-06-24"
    assert normalize_date(None) is None
    assert normalize_date(float("nan")) is None
    assert normalize_date("1990-01-01 00:00:00") == "1990-01-01"


# ------------------------------------------------------------------- seeding/add
@pytest.fixture
def canon():
    return PlayerCanonicalizer(players=[])


def test_add_is_idempotent_on_composite_key(canon):
    a = canon.add("Lionel Messi", "1987-06-24", "Argentina")
    b = canon.add("Lionel Messi", "1987-06-24", "Argentina")
    assert a == b
    assert len(canon.players()) == 1


def test_add_distinguishes_same_name_by_birthdate_and_nationality(canon):
    # Two distinct real-world "Danilo"s, different nationality/birthdate.
    danilo_br = canon.add("Danilo", "1991-07-15", "Brazil")
    danilo_pt = canon.add("Danilo", "2001-09-29", "Portugal")
    assert danilo_br != danilo_pt
    assert len(canon.players()) == 2


# ------------------------------------------------------------------ canonicalize
def test_canonicalize_exact_composite(canon):
    pid = canon.add("Kevin De Bruyne", "1991-06-28", "Belgium")
    assert canon.canonicalize("Kevin De Bruyne", "1991-06-28", "Belgium") == pid


def test_canonicalize_matches_accent_and_case_variants(canon):
    pid = canon.add("N'Golo Kanté", "1991-03-29", "France")
    # Roster spelling without accents / different punctuation still resolves.
    assert canon.canonicalize("Ngolo Kante", "1991-03-29", "France") == pid


def test_canonicalize_by_name_and_nationality_when_birthdate_missing(canon):
    br = canon.add("Danilo", "1991-07-15", "Brazil")
    canon.add("Danilo", "2001-09-29", "Portugal")
    # No birthdate, but nationality uniquely selects the Brazilian Danilo.
    assert canon.canonicalize("Danilo", None, "Brazil") == br


def test_canonicalize_unique_name_without_disambiguators(canon):
    pid = canon.add("Erling Haaland", "2000-07-21", "Norway")
    assert canon.canonicalize("Erling Haaland") == pid


def test_canonicalize_ambiguous_returns_none_and_records(canon):
    canon.add("Danilo", "1991-07-15", "Brazil")
    canon.add("Danilo", "1995-04-10", "Brazil")  # same name + nationality
    assert canon.canonicalize("Danilo", None, "Brazil") is None
    assert canon.unmatched and canon.unmatched[-1]["reason"] == "ambiguous"


def test_canonicalize_unmatched_returns_none_and_records(canon):
    canon.add("Erling Haaland", "2000-07-21", "Norway")
    assert canon.canonicalize("Unknown Player", "1999-01-01", "Narnia") is None
    assert canon.unmatched[-1]["reason"] == "unmatched"


def test_canonicalize_is_stable(canon):
    pid = canon.add("Luka Modrić", "1985-09-09", "Croatia")
    first = canon.canonicalize("Luka Modrić", "1985-09-09", "Croatia")
    second = canon.canonicalize("Luka Modrić", "1985-09-09", "Croatia")
    assert first == second == pid


def test_write_review_emits_csv_and_log(canon, tmp_path):
    canon.add("Erling Haaland", "2000-07-21", "Norway")
    canon.canonicalize("Ghost", "1900-01-01", "Nowhere")
    csv_path = canon.write_review(tmp_path)
    assert csv_path.exists()
    assert (tmp_path / "unmatched_players.log").exists()
    assert "Ghost" in csv_path.read_text()


def test_canonicalize_token_subset_with_birthdate(canon):
    # FIFA carries the full legal name; the roster uses the common name. Same birthdate +
    # nested name tokens → match (the main lever for roster recall).
    pid = canon.add("Lionel Andrés Messi", "1987-06-24", "Argentina")
    assert canon.canonicalize("Lionel Messi", "1987-06-24", "Argentina") == pid
    # Single-token common name (Richarlison) still resolves via birthdate.
    rich = canon.add("Richarlison de Andrade", "1997-05-10", "Brazil")
    assert canon.canonicalize("Richarlison", "1997-05-10", "Brazil") == rich


def test_token_subset_requires_matching_birthdate(canon):
    canon.add("Lionel Andrés Messi", "1987-06-24", "Argentina")
    # Right name tokens but a different birthdate → not the same person.
    assert canon.canonicalize("Lionel Messi", "1990-01-01", "Argentina") is None


def test_token_subset_ambiguous_returns_none(canon):
    # Two same-birthdate players whose tokens both nest with "Silva" → ambiguous, no guess.
    canon.add("Silva Santos", "1995-01-01", "Brazil")
    canon.add("Silva Costa", "1995-01-01", "Brazil")
    assert canon.canonicalize("Silva", "1995-01-01", "Brazil") is None


def test_seed_from_existing_records_reuses_ids():
    seed = [
        {"player_id": 7, "canonical_name": "Bukayo Saka", "birthdate": "2001-09-05", "nationality": "England"},
    ]
    canon = PlayerCanonicalizer(players=seed)
    assert canon.canonicalize("Bukayo Saka", "2001-09-05", "England") == 7
    # New additions must not collide with the seeded id.
    new_id = canon.add("Phil Foden", "2000-05-28", "England")
    assert new_id != 7
