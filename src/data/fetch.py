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
    SELECT t.txn_id, t.amount, t.channel, t.is_fraud, t.txn_ts, t.latitude AS t_lat, t.longitude AS t_lon,
           m.category, m.latitude as m_lat, m.longitude as m_lon, 
           EXTRACT (HOUR FROM t.txn_ts) AS hour, EXTRACT (DOW FROM t.txn_ts) AS dow
    FROM vehicle.transactions t
    JOIN vehicle.merchants m USING (merchant_id)
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
