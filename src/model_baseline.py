"""
Connected Vehicle Data Pipeline - Baseline Fraud Prediction Model
----------------------------------------
Purpose
- Connects to a postgres database, gets minimal training data.
- Engineer simple features (log(amount), hour/dow, geo distance).
- Train a Logistic Regression baseline and report metrics.

Inputs
- DATABASE_URL in .env (points to your Postgres DB).
- Tables: vehicle.transactions, vehicle.merchants (created/loaded earlier).

Outputs
- Printed metrics: ROC AUC, PR AUC, and a classification report @ threshold.

Reproducibility
- Fixed random_state for train/test split and model to make runs repeatable.

Usage
- python src/model_baseline.py
"""

# -- Imports --
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

# -- Connect to Database --
load_dotenv()
database_url = os.getenv("DATABASE_URL")
engine = create_engine(database_url)
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print(result.scalar())

sql = text("""
    SELECT t.txn_id, t.amount, t.channel, t.is_fraud, t.txn_ts, t.latitude AS t_lat, t.longitude AS t_lon,
           m.category, m.latitude as m_lat, m.longitude as m_lon, 
           EXTRACT (HOUR FROM t.txn_ts) AS hour, EXTRACT (DOW FROM t.txn_ts) AS dow
    FROM vehicle.transactions t
    JOIN vehicle.merchants m USING (merchant_id)
""")

df = pd.read_sql(sql, engine)

# -- Feature Engineering --
df["log_amount"] = np.log1p(df["amount"])

# Distance proxy 
def find_geo_delta(lat_1, lon_1, lat_2, lon_2):
    lat_diff = lat_1 - lat_2
    lon_diff = lon_1 - lon_2
    distance = np.sqrt(np.square(lat_diff) + np.square(lon_diff))
    return distance


df["geo_delta"] = find_geo_delta(
    df["t_lat"], df["t_lon"], df["m_lat"], df["m_lon"])
print(df.head())

# -- Train/Val Split --
y = df["is_fraud"]
X = df.drop(columns=["is_fraud", "txn_id", "txn_ts"])
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
    X, y, test_size=0.25, random_state=123, stratify=y)
print("Fraud rate train/test:", y_train.mean(), y_val.mean())

# -- Preprocessing --
num_cols = ["log_amount", "hour", "dow", "geo_delta"]
cat_cols = ["channel", "category"]

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=123)
pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

# -- Fit and Evaluate --
pipe.fit(X_train, y_train)
print("Pipeline fitted!")

proba = pipe.predict_proba(X_val)[:, 1]
roc = roc_auc_score(y_val, proba)
pr = average_precision_score(y_val, proba)

print(f"ROC AUC: {roc:.3f}")
print(f"PR  AUC: {pr:.3f}")

pred = (proba >= 0.5).astype(int)
print("\nClassification report @0.5 threshold")
print(classification_report(y_val, pred, digits=3))
