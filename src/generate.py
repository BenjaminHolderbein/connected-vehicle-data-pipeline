"""
Synthetic data generator for the Connected Vehicle Data Pipeline.

Creates three CSVs under data/raw/ :
- vehicles.csv
- merchants.csv
- transactions.csv

Design choices:
- Amount distributions vary by merchant category (Fuel, Parking, Maintenance, ...).
- Geography: Bay-Area-ish lat/lon for merchants; transactions jitter near merchants.
- Fraud injection: very high amounts, far-away coordinates, suspicious channel/category mixes.
- Deterministic seeds for reproducibility.

Run:
  python src/generate_data.py
"""
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

N_VEH = 300
N_MERCH = 120
N_TXN = 6000

makes = ['Toyota','Honda','Ford','Chevy','BMW','Mercedes','Hyundai','Kia','Tesla']
models = ['Sedan','SUV','Truck','Hatch','EV']
years = list(range(2005,2025))

categories = ['Fuel','Parking','Maintenance','Tolls','CarWash','Food','Groceries']
channels = ['in_app','card_present','web']

def rand_coord():
    """Random (lat, lon) in a Bay-Area-ish bounding box."""
    lat = np.random.uniform(37.3, 38.2)
    lon = np.random.uniform(-122.55, -121.7)
    return float(lat), float(lon)

# Vehicles
vehicles = pd.DataFrame({
    'vehicle_id': [f'V{idx:04d}' for idx in range(N_VEH)],
    'make': np.random.choice(makes, N_VEH),
    'model': np.random.choice(models, N_VEH),
    'year': np.random.choice(years, N_VEH)
})

# Merchants
coords = [rand_coord() for _ in range(N_MERCH)]
merchants = pd.DataFrame({
    'merchant_id': [f'M{idx:04d}' for idx in range(N_MERCH)],
    'name': [f'Merchant_{idx:04d}' for idx in range(N_MERCH)],
    'category': np.random.choice(categories, N_MERCH, p=[0.35,0.15,0.10,0.10,0.05,0.15,0.10]),
    'latitude': [c[0] for c in coords],
    'longitude': [c[1] for c in coords],
})

# Transactions
start = datetime.now() - timedelta(days=60)
rows = []
for i in range(N_TXN):
    vrow = vehicles.sample(1).iloc[0]
    mrow = merchants.sample(1).iloc[0]

    vid = vrow['vehicle_id']
    mid = mrow['merchant_id']
    cat = mrow['category']

    # time moves forward with exponential gaps
    start = start + timedelta(minutes=int(np.random.exponential(45)))
    ts = start

    # category-dependent amount baseline
    amt_base = {'Fuel':55,'Parking':18,'Maintenance':250,'Tolls':6,'CarWash':14,'Food':22,'Groceries':80}[cat]
    amount = max(1, np.random.normal(amt_base, amt_base*0.35))

    # jitter near merchant
    lat = float(mrow['latitude'] + np.random.normal(0, 0.01))
    lon = float(mrow['longitude'] + np.random.normal(0, 0.01))

    channel = np.random.choice(channels, p=[0.5,0.3,0.2])

    # simple fraud rules
    is_fraud = False
    if amount > amt_base * 3.0:
        is_fraud = True
    if np.random.rand() < 0.03:        # distance anomaly
        lat += np.random.uniform(1.0, 2.0)
        lon += np.random.uniform(1.0, 2.0)
        is_fraud = True
    if channel == 'web' and cat in ['Fuel','CarWash'] and np.random.rand() < 0.3:
        is_fraud = True

    rows.append((f'T{i:06d}', vid, mid, ts, round(float(amount),2), lat, lon, channel, is_fraud))

transactions = pd.DataFrame(rows, columns=[
    'txn_id','vehicle_id','merchant_id','txn_ts','amount','latitude','longitude','channel','is_fraud'
])

vehicles.to_csv(RAW/'vehicles.csv', index=False)
merchants.to_csv(RAW/'merchants.csv', index=False)
transactions.to_csv(RAW/'transactions.csv', index=False)

print('Wrote: data/raw/vehicles.csv, merchants.csv, transactions.csv')