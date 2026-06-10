import pytest
from src.common.teams import Canonicalizer


@pytest.fixture
def alias_map():
    return {
        "United States": "USA",
        "Korea Republic": "South Korea",
        "China PR": "China",
        "USA": "USA",
        "South Korea": "South Korea",
        "China": "China",
        "Brazil": "Brazil",
    }


@pytest.fixture
def canon(alias_map):
    return Canonicalizer(alias_map=alias_map)


def test_known_alias(canon):
    assert canon.canonicalize("United States") == "USA"
    assert canon.canonicalize("Korea Republic") == "South Korea"
    assert canon.canonicalize("China PR") == "China"


def test_already_canonical(canon):
    assert canon.canonicalize("Brazil") == "Brazil"


def test_unknown_returns_unchanged(canon):
    assert canon.canonicalize("Atlantis") == "Atlantis"


def test_idempotent(canon, alias_map):
    for alias in alias_map:
        first = canon.canonicalize(alias)
        second = canon.canonicalize(first)
        assert first == second, f"Not idempotent for {alias!r}: {first!r} → {second!r}"


def test_has_alias(canon):
    assert canon.has_alias("United States")
    assert not canon.has_alias("Atlantis")
