"""
Fetch Connected Vehicle data from Postgres database.
"""
# -- Imports --
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import pandas as pd

# -- Function Definitions --


def get_engine() -> Engine:
    """
    Makes a SQLAlchemy engine connected to a database.

    Returns:
        sqlalchemy.engine.Engine: SQL engine.

    Raises:
        ValueError: DATABASE_URL not found in .env.
    """
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not found. Did you set up your .env?")
    return create_engine(url)


_SQL = text("""
    SELECT *
    FROM vehicle.v_txn_for_dashboard
""")


def load_training_frame(engine: Engine = None) -> pd.DataFrame:
    """
    Loads Connected Vehicle data into a DataFrame.

    Args:
        engine (Engine, optional):
            Existing SQLAlchemy Engine connected to the Postgres database. 
            If None, a new engine will be created with `get_engine()`.

    Returns:
        pd.DataFrame: DataFrame with joined transaction + merchant data, 
        including engineered columns (hour, day of week, geo coords).
    """
    eng = engine or get_engine()
    return pd.read_sql(_SQL, eng)
