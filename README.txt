# Connected Vehicle Data Pipeline (Postgres → Python → Warehouse Snapshot)

## Overview
End-to-end demo:
- **Postgres** for live/OLTP transaction data
- **Python .py** scripts for load + features + modeling
- **Warehouse snapshots** (partitioned files) for analytics

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env