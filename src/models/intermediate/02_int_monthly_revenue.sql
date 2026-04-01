-- int_monthly_revenue
-- Monthly MRR waterfall: new, expansion, contraction, churn, net
-- Foundation for revenue forecasting and trend analysis.
WITH
calendar AS (
    -- Generate a spine of months from first transaction to SIM_END
    SELECT DATE_TRUNC('month', transaction_month) AS month
    FROM stg_transactions
    GROUP BY 1
),

-- Active MRR per customer per month: join each customer to
-- all months where their subscription was live
customer_monthly AS (
    SELECT
        DATE_TRUNC('month', CAST(d.month AS DATE))  AS month,
        s.customer_id,
        s.plan_name,
        s.mrr
    FROM calendar d
    JOIN stg_subscriptions s
        ON  CAST(d.month AS DATE) >= s.start_date
        AND CAST(d.month AS DATE) <  COALESCE(s.end_date, DATE '2099-12-31')
    WHERE s.mrr > 0
),

-- Build monthly snapshots with prior-month MRR for waterfall
monthly_with_lag AS (
    SELECT
        month,
        customer_id,
        plan_name,
        mrr,
        LAG(mrr)   OVER (PARTITION BY customer_id ORDER BY month) AS prev_mrr,
        LAG(month) OVER (PARTITION BY customer_id ORDER BY month) AS prev_month
    FROM customer_monthly
),

-- Classify each customer-month into MRR movement category
mrr_movements AS (
    SELECT
        month,
        customer_id,
        plan_name,
        mrr,
        prev_mrr,
        CASE
            WHEN prev_mrr IS NULL                    THEN 'new'
            WHEN mrr > prev_mrr                      THEN 'expansion'
            WHEN mrr < prev_mrr                      THEN 'contraction'
            ELSE                                          'retained'
        END AS mrr_type
    FROM monthly_with_lag
),

-- Customers who churned: present last month, absent this month
churned_customers AS (
    SELECT
        DATE_TRUNC('month', CAST(end_date AS DATE)) AS month,
        customer_id,
        plan_name,
        -mrr                                         AS churn_mrr
    FROM stg_subscriptions
    WHERE end_date IS NOT NULL
      AND status   = 'churned'
),

-- Aggregate into monthly revenue waterfall
monthly_agg AS (
    SELECT
        month,
        COUNT(DISTINCT customer_id)                          AS active_customers,
        SUM(mrr)                                             AS total_mrr,
        SUM(mrr) FILTER (WHERE mrr_type = 'new')            AS new_mrr,
        SUM(mrr) FILTER (WHERE mrr_type = 'expansion')
         - SUM(prev_mrr) FILTER (WHERE mrr_type = 'expansion')
                                                             AS expansion_mrr,
        SUM(prev_mrr) FILTER (WHERE mrr_type = 'contraction')
         - SUM(mrr)     FILTER (WHERE mrr_type = 'contraction')
                                                             AS contraction_mrr,
        SUM(mrr) FILTER (WHERE mrr_type = 'retained')       AS retained_mrr,
        COUNT(*) FILTER (WHERE mrr_type = 'new')            AS new_customers,
        COUNT(*) FILTER (WHERE mrr_type = 'expansion')      AS expanded_customers,
        COUNT(*) FILTER (WHERE mrr_type = 'contraction')    AS contracted_customers,
        SUM(mrr) / NULLIF(COUNT(DISTINCT customer_id), 0)   AS arpu
    FROM mrr_movements
    GROUP BY month
),

churn_agg AS (
    SELECT
        month,
        COUNT(DISTINCT customer_id) AS churned_customers,
        SUM(churn_mrr)              AS churn_mrr  -- negative value
    FROM churned_customers
    GROUP BY month
)

SELECT
    m.month,
    COALESCE(m.active_customers,      0)    AS active_customers,
    COALESCE(m.total_mrr,             0)    AS total_mrr,
    ROUND(COALESCE(m.total_mrr,0) * 12, 2) AS total_arr,
    COALESCE(m.new_mrr,               0)    AS new_mrr,
    COALESCE(m.expansion_mrr,         0)    AS expansion_mrr,
    COALESCE(m.contraction_mrr,       0)    AS contraction_mrr,
    COALESCE(m.retained_mrr,          0)    AS retained_mrr,
    COALESCE(c.churn_mrr,             0)    AS churn_mrr,
    COALESCE(m.new_mrr, 0)
     + COALESCE(m.expansion_mrr, 0)
     - COALESCE(m.contraction_mrr, 0)
     + COALESCE(c.churn_mrr, 0)            AS net_new_mrr,
    COALESCE(m.new_customers,         0)    AS new_customers,
    COALESCE(c.churned_customers,     0)    AS churned_customers,
    COALESCE(m.expanded_customers,    0)    AS expanded_customers,
    COALESCE(m.contracted_customers,  0)    AS contracted_customers,
    ROUND(COALESCE(m.arpu, 0), 2)           AS arpu,
    -- growth rate MoM
    ROUND(
        100.0 * (m.total_mrr - LAG(m.total_mrr) OVER (ORDER BY m.month))
        / NULLIF(LAG(m.total_mrr) OVER (ORDER BY m.month), 0),
        2
    ) AS mrr_growth_pct,
    -- gross churn rate
    ROUND(
        100.0 * ABS(COALESCE(c.churn_mrr, 0))
        / NULLIF(LAG(m.total_mrr) OVER (ORDER BY m.month), 0),
        2
    ) AS gross_churn_rate_pct

FROM monthly_agg m
LEFT JOIN churn_agg c USING (month)
ORDER BY month
