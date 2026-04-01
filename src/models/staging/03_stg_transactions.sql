-- stg_transactions: clean and enrich transaction records
SELECT
    transaction_id,
    customer_id,
    subscription_id,
    CAST(transaction_date AS DATE)              AS transaction_date,
    DATE_TRUNC('month', CAST(transaction_date AS DATE)) AS transaction_month,
    DATE_TRUNC('year',  CAST(transaction_date AS DATE)) AS transaction_year,
    ROUND(CAST(amount AS DOUBLE), 2)           AS amount,
    currency,
    transaction_type,
    status,
    -- revenue recognition
    CASE
        WHEN status = 'success'  THEN ROUND(CAST(amount AS DOUBLE), 2)
        ELSE 0.0
    END                                         AS recognized_revenue,
    -- type flags
    CASE WHEN transaction_type IN ('monthly_subscription','annual_subscription')
         THEN TRUE ELSE FALSE END               AS is_subscription,
    CASE WHEN transaction_type = 'churn'        THEN TRUE ELSE FALSE END AS is_churn,
    CASE WHEN transaction_type = 'upgrade'      THEN TRUE ELSE FALSE END AS is_upgrade,
    CASE WHEN status = 'failed'                 THEN TRUE ELSE FALSE END AS is_failed,
    CASE WHEN status = 'refunded'               THEN TRUE ELSE FALSE END AS is_refunded
FROM raw_transactions
WHERE transaction_id IS NOT NULL
