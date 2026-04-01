-- mart_cohort_analysis
-- Monthly cohort retention matrix.
-- cohort_month = first subscription month for the customer
-- period_number = months since cohort month (0 = acquisition month)
WITH
cohort_base AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', signup_date) AS cohort_month
    FROM stg_customers
),

-- All months a customer was "active" (had an active subscription)
customer_active_months AS (
    SELECT DISTINCT
        s.customer_id,
        DATE_TRUNC('month', CAST(d.generate_series AS DATE)) AS active_month
    FROM stg_subscriptions s
    JOIN (
        SELECT generate_series AS generate_series
        FROM generate_series(
            DATE '2022-01-01',
            DATE '2025-03-31',
            INTERVAL '1 month'
        )
    ) d
        ON CAST(d.generate_series AS DATE) >= s.start_date
        AND CAST(d.generate_series AS DATE) <  COALESCE(s.end_date, DATE '2099-12-31')
),

cohort_activity AS (
    SELECT
        cb.cohort_month,
        cam.active_month,
        DATE_DIFF('month', cb.cohort_month, cam.active_month) AS period_number,
        COUNT(DISTINCT cam.customer_id) AS active_customers
    FROM cohort_base cb
    JOIN customer_active_months cam USING (customer_id)
    GROUP BY 1, 2, 3
),

cohort_sizes AS (
    SELECT
        cohort_month,
        MAX(active_customers) FILTER (WHERE period_number = 0) AS cohort_size
    FROM cohort_activity
    GROUP BY cohort_month
)

SELECT
    ca.cohort_month,
    cs.cohort_size,
    ca.period_number,
    ca.active_customers,
    ROUND(
        100.0 * ca.active_customers / NULLIF(cs.cohort_size, 0),
        1
    ) AS retention_rate,
    -- churned from prior period
    LAG(ca.active_customers) OVER (
        PARTITION BY ca.cohort_month ORDER BY ca.period_number
    ) - ca.active_customers AS churned_this_period
FROM cohort_activity ca
JOIN cohort_sizes     cs USING (cohort_month)
WHERE ca.cohort_month >= DATE '2022-01-01'
  AND ca.period_number >= 0
  AND ca.period_number <= 24
ORDER BY ca.cohort_month, ca.period_number
