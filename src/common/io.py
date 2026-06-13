import json
from typing import Any, Optional, Sequence

import pandas as pd
from psycopg2.extras import Json, execute_values

from src.common.db import get_engine

SCHEMA = "wc2026"


def read_table(name: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(f"SELECT * FROM {SCHEMA}.{name}", conn)


def write_table(df: pd.DataFrame, name: str, if_exists: str = "replace") -> None:
    engine = get_engine()
    df.to_sql(name, engine, schema=SCHEMA, if_exists=if_exists, index=False)


def _adapt(value: Any) -> Any:
    """Adapt Python values for psycopg2: dicts → JSONB, NaN → NULL."""
    if isinstance(value, (dict, list)):
        return Json(value)
    if isinstance(value, str):
        return value
    # pandas NaN / NaT are not equal to themselves.
    if value is not None and value != value:  # noqa: PLR0124
        return None
    return value


def bulk_upsert(
    table: str,
    rows: Sequence[dict],
    conflict_cols: Sequence[str],
    update_cols: Optional[Sequence[str]] = None,
    returning: Optional[str] = None,
    page_size: int = 1000,
) -> list[tuple]:
    """Batched ``INSERT ... ON CONFLICT`` upsert into ``wc2026.<table>``.

    Idempotent: re-running with the same rows is a no-op (``DO NOTHING``) or refresh
    (``DO UPDATE``). Preserves the migration-defined schema (typed columns, JSONB, FKs) —
    unlike ``write_table('replace')`` which drops and recreates with pandas-inferred types.

    ``rows`` is a list of dicts sharing the same keys. dict-valued fields are sent as JSONB.
    Returns rows from the ``RETURNING`` clause when provided, else ``[]``.
    """
    if not rows:
        return []
    cols = list(rows[0].keys())
    col_list = ", ".join(cols)
    conflict = ", ".join(conflict_cols)
    if update_cols:
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        action = f"DO UPDATE SET {set_clause}"
    else:
        action = "DO NOTHING"
    sql = (
        f"INSERT INTO {SCHEMA}.{table} ({col_list}) VALUES %s "
        f"ON CONFLICT ({conflict}) {action}"
    )
    if returning:
        sql += f" RETURNING {returning}"

    template = "(" + ", ".join(["%s"] * len(cols)) + ")"
    values = [[_adapt(r.get(c)) for c in cols] for r in rows]

    conn = get_engine().raw_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, values, template=template, page_size=page_size)
            out = cur.fetchall() if returning else []
        conn.commit()
        return out
    finally:
        conn.close()
