"""
Connected Vehicle Dashboard (Streamlit).

Reads aggregated/safe transaction features from Postgres (stable view),
shows KPIs, a daily spend histogram, and a table of labeled frauds.

Requirements:
- .streamlit/secrets.toml with:
    DATABASE_URL = "postgresql+psycopg2://user:pass@host:5432/db"
- View: vehicle.v_txn_for_dashboard (columns used below)
- Packages: streamlit, pandas, sqlalchemy, altair, psycopg2-binary (or psycopg)
"""

# -- Imports --
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import altair as alt

# -- Function Definitions --


@st.cache_resource
def get_engine():
    """
    Create (and cache) a single SQLAlchemy engine for the Streamlit process.

    Returns:
        sqlalchemy.engine.Engine: Pooled engine created from secrets.DATABASE_URL.
    """
    url = st.secrets["DATABASE_URL"]  # set in secrets
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(ttl=60)  # re-query at most once per minute
def fetch_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    """
    Execute a SQL query and return a DataFrame. Cached briefly for responsiveness.

    Args:
        sql (str): Parameterized SQL query string (use :name placeholders).
        params (dict | None): Dictionary of parameter values.

    Returns:
        pd.DataFrame: Query results.
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


@st.cache_data(ttl=300)
def fetch_distincts():
    """
    Get distinct category/channel values for sidebar filters.

    Returns:
        tuple[list[str], list[str]]: (sorted categories, sorted channels)
    """
    sql = "SELECT DISTINCT category, channel FROM vehicle.v_txn_for_dashboard"
    df = fetch_df(sql)
    cats = sorted(df["category"].dropna().unique().tolist())
    chans = sorted(df["channel"].dropna().unique().tolist())
    return cats, chans


BASE_SQL = """
SELECT
  txn_id, txn_ts, vehicle_id, merchant_id, merchant_name,
  category, channel, amount,
  t_lat, t_lon, m_lat, m_lon,
  hour, dow, log_amount, geo_delta,
  is_fraud
FROM vehicle.v_txn_for_dashboard
WHERE 1=1
  -- optional filters:
  {category_clause}
  {channel_clause}
ORDER BY txn_ts DESC
LIMIT :limit
"""


def load_transactions(category: str | None, channel: str | None, limit: int = 2000):
    """
    Load filtered transactions from the stable dashboard view.

    Args:
        category (str | None): Merchant category filter; None to disable.
        channel (str | None): Channel filter; None to disable.
        limit (int): Row cap to keep app snappy.

    Returns:
        pd.DataFrame: Filtered transactions.
    """
    category_clause = "AND category = :category" if category else ""
    channel_clause = "AND channel  = :channel" if channel else ""
    sql = BASE_SQL.format(category_clause=category_clause,
                          channel_clause=channel_clause)
    params = {"limit": limit}
    if category:
        params["category"] = category
    if channel:
        params["channel"] = channel
    return fetch_df(sql, params)


# -- Dashboard --
st.set_page_config(page_title="Connected Vehicle Dashboard", layout="wide")
st.title("Connected Vehicle — Fraud Overview")

# Sidebar filters
cats, chans = fetch_distincts()
with st.sidebar:
    st.header("Filters")
    category = st.selectbox("Merchant category", ["(all)"] + cats, index=0)
    channel = st.selectbox("Channel", ["(all)"] + chans, index=0)
    th = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    limit = st.number_input("Row limit", 100, 10000, 6000, 100)

# Normalize "(all)" -> None for SQL
sel_category = None if category == "(all)" else category
sel_channel = None if channel == "(all)" else channel

# Data fetch + guard
df = load_transactions(sel_category, sel_channel, limit=limit)
if df.empty:
    st.warning("No transactions found for the selected filters.")
    st.stop()

# Top-level KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Fraud rate (label)", f"{100*df['is_fraud'].mean():.2f}%")
col3.metric("Categories", df["category"].nunique())
col4.metric("Channels", df["channel"].nunique())

# -- Daily spend (bar chart) --
df["_date"] = pd.to_datetime(df["txn_ts"]).dt.date
daily = (
    df.groupby("_date", as_index=False)["amount"]
    .sum()
    .rename(columns={"_date": "date", "amount": "amount_spent"})
)

# Fraudulent transactions (labeled)
st.subheader("Amount Spent per Day")
if daily.empty:
    st.info("No data for the selected filters.")
else:
    chart = (
        alt.Chart(daily)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("amount_spent:Q", title="Total Amount"),
            tooltip=["date:T", "amount_spent:Q"]
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# -- Fraudulent Transactions (List) --
st.subheader("Fraudulent Transactions (labeled)")
fraud_cols = ["txn_ts", "txn_id", "merchant_name",
              "category", "channel", "amount", "is_fraud"]
fraud_df = (
    df.loc[df["is_fraud"] == True, fraud_cols]
      .sort_values("txn_ts", ascending=False)
)

if fraud_df.empty:
    st.info("No labeled fraudulent transactions for the current filters.")
else:
    st.dataframe(fraud_df.head(500), use_container_width=True)
