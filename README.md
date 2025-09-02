# Connected Vehicle Data Pipeline

## Overview
This project demonstrates a simplified connected vehicle data pipeline.
It simulates synthetic transaction data, stores it in a Postgres database, and trains baseline fraud detection models (Logistic Regression: scikit-learn & from scratch).

The pipeline is **batch-oriented** for now but is designed to be extended to support streaming or real-time dashboards.

---

## Project Structure
```
.
├── app
│   └── app.py
├── data
│   └── raw
│       ├── merchants.csv
│       ├── transactions.csv
│       └── vehicles.csv
├── models
├── README.md
├── requirements.txt
├── sql
│   ├── create_schema.sql
│   └── create_views.sql
└── src
    ├── data
    │   └── fetch.py
    ├── eval
    │   └── metrics.py
    ├── features
    │   └── build.py
    ├── generate.py
    ├── load_to_postgres.py
    ├── models
    │   ├── __init__.py
    │   ├── logreg_scratch.py
    │   ├── logreg_sklearn.py
    │   └── preprocess.py
    └── train.py
```

---

## Components
- **Schema Definition** (`sql/create_schema.sql`)  
  Creates the Postgres `vehicle` schema with three tables:
  - `vehicles` — vehicle metadata  
  - `merchants` — merchant metadata  
  - `transactions` — individual transactions linked to vehicles and merchants  

- **Views** (`sql/create_views.sql`)
  Defines helper views like v_txn_for_dashboard used by both modeling and dashboard.

- **Data Generation** (`src/generate.py`)  
  Produces synthetic CSVs: `data/raw/vehicles.csv`, `merchants.csv`, `transactions.csv`.

- **Data Loading** (`src/load_to_postgres.py`)  
  Loads CSVs into Postgres.  
  - Truncates tables before reload (dev-only).  
  - Applies column renames & type coercions.  
  - Prints quick sanity checks (row counts, samples).

- **Data Access (SQL <-> DataFrame)** (`src/data/fetch.py`)  
  Central place to connect (via `.env` `DATABASE_URL`) and fetch the joined training frame.

- **Feature Engineering** (`src/features/build.py`)  
  Adds simple features used by all models:  
  - `log_amount`, `hour`, `dow`, `geo_delta` (crude distance proxy)  
  Exposes lists for preprocessing: `NUM_COLS`, `CAT_COLS`, `DROP_COLS`.

- **Preprocessing** (`src/models/preprocess.py`)  
  Builds a `ColumnTransformer` that standardizes numeric features and one-hot encodes categoricals
  (dense output to support both sklearn and scratch models).

- **Model Wrappers** (`src/models/`)  
  Uniform “`Model`” API (`fit`, `predict_proba`) so models are swappable:
  - `logreg_sklearn.py` — scikit-learn Logistic Regression
  - `logreg_scratch.py` — NumPy Logistic Regression (from scratch)
  - *(placeholders for)* `rf_sklearn.py`, `xgb_classifier.py`

- **Training Entry Point** (`src/train.py`)  
  One CLI to train/evaluate any registered model:
  ```bash
  python -m src.train --model logreg --test-size 0.25 --threshold 0.5 --save models/logreg.pkl
  ```
  Prints ROC AUC, PR AUC, and a classification report; optionally saves the fitted pipeline.

- **Evaluation Utilities** (`src/eval/metrics.py`)
  Computes ROC AUC, PR AUC, and thresholded classification report for consistent comparisons.

- **Saved Artifacts (models/)** (`models/`)
  Stores serialized pipelines (e.g., `models/logreg.pkl`). 
  Binaries are gitignored; keep a .gitkeep to show the folder.

- **Dashboard App** (`app/app.py`) 
  Streamlit dashboard for interactive fraud monitoring (filters, charts, fraud table).

---

## Prerequisites
- Python 3.11+ (with `venv` or `conda`)  
- Postgres running locally (or via Docker)  
- Environment file `.env` containing:
```dotenv
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/vehicledb
```
- Streamlit secrets file `.streamlit/secrets.toml` containing:
```toml
DATABASE_URL = "postgresql+psycopg2://user:password@localhost:5432/vehicledb"
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
python src/generate.py
```
3.  **Load Data**
```bash
python src/load_to_postgres.py
```
4.  **Train Baseline Model**
```bash
# Train a baseline Logistic Regression (scikit-learn)
python -m src.train --model logreg --test-size 0.25 --threshold 0.5 --save models/logreg.pkl

# Or use the from-scratch Logistic Regression
python -m src.train --model logreg_scratch --test-size 0.25 --threshold 0.5 --save models/scratch_logreg.pkl
```
5.  **Streamlit app**
```bash
streamlit run app/app.py
```

---

## Next Steps
- **Modeling**
  - Add Random Forest and XGBoost baselines.
  - Standardize evaluation metrics and comparisons across models.

- **System Extensions**
  - Save and version trained models (`models/` folder, joblib/npz).
  - Simulate real-time ingestion:
    - Generator appends new transactions.
    - Scorer service loads saved model, scores unscored txns, writes to `scores` table.
  - Prepare for containerization with Docker.

- **Presentation**
  - Extend Streamlit dashboard:
    - Show recent scored transactions.
    - Plot live fraud metrics (ROC/PR, precision/recall at threshold).
    - Add threshold slider for what-if analysis.
  - Deploy the dashboard (Streamlit Cloud, Render, or similar) to share via a public link.

---

## Notes
- **All data is synthetic** and generated locally.
- This project is for learning/demo purposes only.
