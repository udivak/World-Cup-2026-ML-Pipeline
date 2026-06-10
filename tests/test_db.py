import os
import pytest


def test_get_engine_import():
    from src.common.db import get_engine
    assert callable(get_engine)


def test_ensure_schema_import():
    from src.common.db import ensure_schema
    assert callable(ensure_schema)


@pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping live DB test",
)
def test_ensure_schema_live():
    from src.common.db import get_engine, ensure_schema
    engine = get_engine()
    ensure_schema(engine)
    ensure_schema(engine)
