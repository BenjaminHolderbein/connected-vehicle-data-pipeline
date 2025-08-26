-- Connected Vehicle Data Pipeline
-- Defines core tables (vehicles, merchants, transactions) for storing
-- synthetic connected-vehicle transaction data. Includes primary/foreign
-- key, constraints, and helpful indexes.
-- NOTE: All data loaded into this schema is synthetic (generated).

DROP SCHEMA IF EXISTS vehicle CASCADE;
CREATE SCHEMA vehicle;

-- Vehicles:
-- Columns:
--  vehicle_id: synthetic primary key (e.g., V00001)
--  make/model/year: descriptive attributes with validation
CREATE TABLE vehicle.vehicles (
    vehicle_id VARCHAR(5) PRIMARY KEY,
    make TEXT NOT NULL,
    model TEXT NOT NULL,
    model_year INT NOT NULL 
        CHECK (model_year BETWEEN 1900 AND EXTRACT(YEAR FROM CURRENT_DATE) + 1)
);

-- Merchants:
-- Columns:
--  merchant_id: synthetic primary key (e.g., M00001)
--  merchant_name/category/latitude/longitude: descriptive attributes with validation
CREATE TABLE vehicle.merchants (
    merchant_id VARCHAR(5) PRIMARY KEY,
    merchant_name TEXT NOT NULL,
    category TEXT NOT NULL
        CHECK (category IN ('CarWash', 'Food', 'Fuel', 'Groceries', 'Maintenance', 'Parking', 'Tolls')),
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL
);

-- Transactions:
-- Each transaction links a vehicle to a merchant at a point in time.
-- Foreign keys enforce referential integrity.
-- Columns:
--  txn_id: synthetic primary key (e.g., T000001)
--  vehicle_id/merchant_id/txn_ts/amount: descriptive attributes with validation
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

-- Helpful index for common queries:
-- e.g., retrieving transactions for a specific vehicle in time order.
CREATE INDEX IF NOT EXISTS idx_transactions_vehicle_ts
    ON vehicle.transactions (vehicle_id, txn_ts);