-- stg_subscriptions: clean subscription records
SELECT
    subscription_id,
    customer_id,
    plan                                        AS plan_name,
    billing_cycle,
    ROUND(CAST(mrr AS DOUBLE), 2)              AS mrr,
    ROUND(CAST(mrr AS DOUBLE) * 12, 2)        AS arr,
    CAST(start_date AS DATE)                   AS start_date,
    CAST(end_date   AS DATE)                   AS end_date,
    status,
    -- duration
    DATE_DIFF('day',
        CAST(start_date AS DATE),
        COALESCE(CAST(end_date AS DATE), CURRENT_DATE)
    )                                           AS subscription_days,
    DATE_DIFF('month',
        CAST(start_date AS DATE),
        COALESCE(CAST(end_date AS DATE), CURRENT_DATE)
    )                                           AS subscription_months,
    -- flags
    CASE WHEN end_date IS NULL THEN TRUE ELSE FALSE END AS is_active,
    CASE WHEN status = 'churned'  THEN TRUE ELSE FALSE END AS is_churned,
    CASE WHEN status = 'upgraded' THEN TRUE ELSE FALSE END AS is_upgraded
FROM raw_subscriptions
WHERE subscription_id IS NOT NULL
  AND customer_id     IS NOT NULL
