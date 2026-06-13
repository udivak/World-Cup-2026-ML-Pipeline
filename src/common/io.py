import pandas as pd
from sqlalchemy import text
from src.common.db import get_engine


SCHEMA = "wc2026"


def read_table(name: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(f"SELECT * FROM {SCHEMA}.{name}", conn)


def write_table(df: pd.DataFrame, name: str, if_exists: str = "replace") -> None:
    engine = get_engine()
    df.to_sql(name, engine, schema=SCHEMA, if_exists=if_exists, index=False)
