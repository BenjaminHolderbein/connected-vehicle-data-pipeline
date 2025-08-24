#  --Imports--
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
import pandas as pd
import datetime

#  --Connect to Database-- 
#  Get Database URL from .env
load_dotenv()
database_url = os.getenv("DATABASE_URL")
print(f"Database URL: {database_url}")

#  Connect to Database
engine = create_engine(database_url)

#  Test connection
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print(result.scalar())

# --Truncate before reload--
with engine.begin() as conn:
    conn.execute(text("""
        TRUNCATE vehicle.transactions,
                 vehicle.merchants,
                 vehicle.vehicles
        RESTART IDENTITY CASCADE
                 """))


#  --Load Data--
#  Load CSVs
merchants = pd.read_csv("data/raw/merchants.csv")
transactions = pd.read_csv("data/raw/transactions.csv")
vehicles = pd.read_csv("data/raw/vehicles.csv")

#  Check Content
print(merchants.head(3))
print(transactions.head(3))
print(vehicles.head(3))

# --Transform Data--
vehicles.rename(columns={"year": "model_year"}, inplace=True)
vehicles["model_year"] = pd.to_numeric(vehicles["model_year"])

merchants.rename(columns={"name": "merchant_name"}, inplace=True)

transactions["txn_ts"] = pd.to_datetime(transactions["txn_ts"])
transactions["amount"] = pd.to_numeric(transactions["amount"])
transactions["is_fraud"] = transactions["is_fraud"].fillna(False).astype(bool)

print(merchants.dtypes)
print(vehicles.dtypes)
print(transactions.dtypes)

# --Add Data to Database--
with engine.begin() as conn:
    vehicles.to_sql("vehicles", conn, schema="vehicle",
                    if_exists="append", index=False)
    merchants.to_sql("merchants", conn, schema="vehicle",
                     if_exists="append", index=False)
    transactions.to_sql("transactions", conn, schema="vehicle",
                        if_exists="append", index=False)
    
# --Testing--
tests = [
    ("Vehicle count:", "SELECT COUNT(*) FROM vehicle.vehicles"),
    ("Merchant count:", "SELECT COUNT(*) FROM vehicle.merchants"),
    ("Transaction count:", "SELECT COUNT(*) FROM vehicle.transactions"),
    ("Sample vehicles:", "SELECT * FROM vehicle.vehicles LIMIT 3")
]
with engine.connect() as conn:
    for label, sql in tests:
        result = conn.execute(text(sql))
        rows = result.fetchall()
        print(f"\n{label}")
        for row in rows:
            print(row)