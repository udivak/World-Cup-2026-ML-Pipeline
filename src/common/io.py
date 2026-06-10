import pandas as pd
from sqlalchemy import text
from src.common.db import get_engine


SCHEMA = "wc2026"


def read_table(name: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql_table(name, engine, schema=SCHEMA)


def write_table(df: pd.DataFrame, name: str, if_exists: str = "replace") -> None:
    engine = get_engine()
    df.to_sql(name, engine, schema=SCHEMA, if_exists=if_exists, index=False)
