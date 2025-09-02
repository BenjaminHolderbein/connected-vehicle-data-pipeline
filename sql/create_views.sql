-- Stable “interface” for apps & analysis
-- Joins transactions+merchants and exposes only the columns UIs/models depend on.
CREATE OR REPLACE VIEW vehicle.v_txn_for_dashboard AS
SELECT
    t.txn_id,
    t.txn_ts,
    t.vehicle_id,
    t.merchant_id,
    m.merchant_name,
    m.category,
    t.channel,
    t.amount,
    t.latitude  AS t_lat,
    t.longitude AS t_lon,
    m.latitude  AS m_lat,
    m.longitude AS m_lon,
    -- handy engineered features (stable names)
    EXTRACT(HOUR FROM t.txn_ts)::int AS hour,
    EXTRACT(DOW  FROM t.txn_ts)::int AS dow,
    LN(t.amount + 1.0)                AS log_amount,
    SQRT(POWER(t.latitude  - m.latitude , 2)
       + POWER(t.longitude - m.longitude, 2)) AS geo_delta,
    -- label
    t.is_fraud
FROM vehicle.transactions t
JOIN vehicle.merchants m USING (merchant_id);
