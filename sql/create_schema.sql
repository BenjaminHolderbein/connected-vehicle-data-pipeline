DROP SCHEMA IF EXISTS vehicle CASCADE;
CREATE SCHEMA vehicle;

CREATE TABLE vehicle.vehicles (
    vehicle_id VARCHAR(5) PRIMARY KEY,
    make TEXT NOT NULL,
    model TEXT NOT NULL,
    model_year INT NOT NULL 
        CHECK (model_year BETWEEN 1900 AND EXTRACT(YEAR FROM CURRENT_DATE) + 1)
);

CREATE TABLE vehicle.merchants (
    merchant_id VARCHAR(5) PRIMARY KEY,
    merchant_name TEXT NOT NULL,
    category TEXT NOT NULL
        CHECK (category IN ('CarWash', 'Food', 'Fuel', 'Groceries', 'Maintenance', 'Parking', 'Tolls')),
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL
);

CREATE TABLE vehicle.transactions (
    txn_id VARCHAR(7) PRIMARY KEY,
    vehicle_id VARCHAR(5) NOT NULL
        REFERENCES vehicle.vehicles(vehicle_id),
    merchant_id VARCHAR(5) NOT NULL
        REFERENCES vehicle.merchants(merchant_id),
    txn_ts TIMESTAMP NOT NULL,
    amount NUMERIC(10,2) NOT NULL
        CHECK (amount > 0),
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    channel TEXT NOT NULL
        CHECK (channel IN ('card_present', 'in_app', 'web')),
    is_fraud BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_transactions_vehicle_ts
    ON vehicle.transactions (vehicle_id, txn_ts);