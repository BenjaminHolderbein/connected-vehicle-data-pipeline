# Connected Vehicle Data Pipeline

## Overview
This project demonstrates a simplified connected vehicle data pipeline.  
It simulates synthetic transaction data, stores it in a Postgres database, and trains a baseline fraud detection model.

The pipeline is **batch-oriented** for now but is designed to be extended to support streaming or real-time dashboards.

---

## Project Structure
```
.
├── app/                     # (placeholder for future app code, e.g. Streamlit)
├── data/
│   └── raw/                 # synthetic CSVs (vehicles.csv, merchants.csv, transactions.csv)
├── sql/
│   └── create_schema.sql    # schema definition for Postgres
├── src/
│   ├── generate.py          # generates synthetic data
│   ├── load_to_postgres.py  # loads CSVs into Postgres
│   └── model_baseline.py    # baseline fraud detection model
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Components
- **Data Generation** (`src/generate_data.py`)  
  Creates synthetic CSVs for vehicles, merchants, and transactions.

- **Schema Definition** (`sql/create_schema.sql`)  
  Defines Postgres schema (`vehicle`) with three tables:
  - `vehicles` — vehicle metadata  
  - `merchants` — merchant metadata  
  - `transactions` — individual transactions linked to vehicles and merchants  

- **Data Loading** (`src/load_to_postgres.py`)  
  Loads the CSVs into Postgres.  
  - Truncates tables before reload (dev-only).  
  - Applies column renames & type coercions.  
  - Runs basic sanity checks (row counts, sample rows).

- **Baseline Model** (`src/model_baseline.py`)  
  - Pulls minimal training data with SQL joins.  
  - Engineers simple features (`log(amount)`, `hour`, `dow`, `geo distance`).  
  - Trains a logistic regression with balanced class weights.  
  - Prints ROC AUC, PR AUC, and classification report.  

---

## Prerequisites
- Python 3.11+ (with `venv` or `conda`)  
- Postgres running locally (or via Docker)  
- Environment file `.env` containing:
```dotenv
DATABASE_URL=postgresql://user:password@localhost:5432/vehicledb
```
- Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage
1.  **Create Schema**
```bash
psql -d vehicledb -f sql/create_schema.sql
```
2.  **Generate Data**
```bash
python src/generate_data.py
```
3.  **Load Data**
```bash
python src/load_to_postgres.py
```
4.  **Train Baseline Model**
```bash
python src/model_baseline.py
```

## Next Steps
- Add feature engineering for richer transaction/vehicle history.
- Experiment with tree-based models (Random Forest, XGBoost).
- Explore incremental / streaming ingestion for near real-time detection.
- Build a Streamlit dashboard to visualize fraud patterns interactively.

## Notes
- **All data is synthetic** and generated locally.
- This project is for learning/demo purposes only.
