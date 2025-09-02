"""
Feature engineering utilities for the Connected Vehicle data.

Currently adds:
- log_amount
- geo_delta (transaction ↔ merchant distance proxy)

Defines NUM_COLS, CAT_COLS, and DROP_COLS for downstream preprocessing.
"""
# -- Imports --
import numpy as np
import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic engineered features for fraud detection.

    Args:
        df (pd.DataFrame): Input DataFrame with transaction-level columns
            including `amount`, `t_lat`, `t_lon`, `m_lat`, `m_lon`.

    Returns:
        pd.DataFrame: Copy of input DataFrame with new columns:
            - log_amount: log(1 + amount)
            - geo_delta: crude distance proxy between transaction and merchant
    """
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])
    dlat = df["t_lat"] - df["m_lat"]
    dlon = df["t_lon"] - df["m_lon"]
    df["geo_delta"] = np.sqrt(
        dlat * dlat + dlon * dlon)  # crude distance proxy
    return df


#  What we feed to the preprocessor:
NUM_COLS = ["log_amount", "hour", "dow", "geo_delta"]
CAT_COLS = ["channel", "category"]

# Columns we don't model directly:
DROP_COLS = ["is_fraud", "txn_id", "txn_ts"]
