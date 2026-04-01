-- mart_revenue_summary
-- Pre-aggregated revenue KPIs used by the executive dashboard.
-- Includes MRR waterfall, plan-level breakdown, and segment slices.
WITH
monthly AS (SELECT * FROM int_monthly_revenue),

plan_monthly AS (
    SELECT
        DATE_TRUNC('month', CAST(d.generate_series AS DATE)) AS month,
        s.plan_name,
        COUNT(DISTINCT s.customer_id)                        AS customers,
        SUM(s.mrr)                                           AS mrr
    FROM stg_subscriptions s
    JOIN (
        SELECT generate_series
        FROM generate_series(
            DATE '2022-01-01',
            DATE '2025-03-31',
            INTERVAL '1 month'
        )
    ) d
        ON CAST(d.generate_series AS DATE) >= s.start_date
        AND CAST(d.generate_series AS DATE) <  COALESCE(s.end_date, DATE '2099-12-31')
    GROUP BY 1, 2
),

plan_pivot AS (
    SELECT
        month,
        SUM(mrr) FILTER (WHERE plan_name = 'Starter')      AS starter_mrr,
        SUM(mrr) FILTER (WHERE plan_name = 'Growth')        AS growth_mrr,
        SUM(mrr) FILTER (WHERE plan_name = 'Professional')  AS professional_mrr,
        SUM(mrr) FILTER (WHERE plan_name = 'Enterprise')    AS enterprise_mrr,
        SUM(customers) FILTER (WHERE plan_name = 'Starter')     AS starter_customers,
        SUM(customers) FILTER (WHERE plan_name = 'Growth')       AS growth_customers,
        SUM(customers) FILTER (WHERE plan_name = 'Professional') AS professional_customers,
        SUM(customers) FILTER (WHERE plan_name = 'Enterprise')   AS enterprise_customers
    FROM plan_monthly
    GROUP BY month
),

-- Quick-ratio: (new + expansion) / (contraction + churn)
quick_ratio AS (
    SELECT
        month,
        ROUND(
            (new_mrr + expansion_mrr) /
            NULLIF(ABS(churn_mrr) + contraction_mrr, 0),
            2
        ) AS revenue_quick_ratio
    FROM monthly
)

SELECT
    m.*,
    pp.starter_mrr,
    pp.growth_mrr,
    pp.professional_mrr,
    pp.enterprise_mrr,
    pp.starter_customers,
    pp.growth_customers,
    pp.professional_customers,
    pp.enterprise_customers,
    qr.revenue_quick_ratio,

    -- Running totals
    SUM(m.total_mrr) OVER (ORDER BY m.month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)   AS cumulative_mrr,
    -- 3-month rolling average MRR
    ROUND(AVG(m.total_mrr) OVER (
        ORDER BY m.month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2)                                                   AS mrr_3mo_avg

FROM monthly m
LEFT JOIN plan_pivot pp USING (month)
LEFT JOIN quick_ratio qr USING (month)
ORDER BY m.month
