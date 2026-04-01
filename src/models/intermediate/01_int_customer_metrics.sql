-- int_customer_metrics
-- One row per customer with all lifetime and recent-activity metrics
-- used by RFM scoring, churn prediction, and the 360 mart.
WITH
customer_base AS (
    SELECT * FROM stg_customers
),

-- Latest active subscription per customer
latest_sub AS (
    SELECT
        customer_id,
        plan_name,
        billing_cycle,
        mrr,
        arr,
        is_active,
        start_date  AS sub_start_date,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id ORDER BY start_date DESC
        ) AS rn
    FROM stg_subscriptions
),

active_sub AS (
    SELECT * FROM latest_sub WHERE rn = 1
),

-- Revenue aggregates
rev_agg AS (
    SELECT
        customer_id,
        COUNT(*)                                             AS total_transactions,
        COUNT(*) FILTER (WHERE is_subscription)             AS subscription_payments,
        COUNT(*) FILTER (WHERE is_failed)                   AS failed_payments,
        COUNT(*) FILTER (WHERE is_refunded)                 AS refunded_payments,
        SUM(recognized_revenue)                             AS total_revenue,
        SUM(recognized_revenue) FILTER (WHERE is_subscription) AS subscription_revenue,
        MIN(transaction_date)                               AS first_payment_date,
        MAX(transaction_date) FILTER (WHERE is_subscription AND status='success')
                                                            AS last_payment_date,
        AVG(recognized_revenue) FILTER (WHERE is_subscription AND status='success')
                                                            AS avg_transaction_value,
    FROM stg_transactions
    GROUP BY customer_id
),

-- Product engagement aggregates (last 90 days)
recent_cutoff AS (
    SELECT (MAX(event_date) - INTERVAL '90 days') AS cutoff FROM stg_product_events
),

engagement AS (
    SELECT
        e.customer_id,
        COUNT(*)                                        AS total_events_90d,
        COUNT(DISTINCT e.event_date)                    AS active_days_90d,
        COUNT(*) FILTER (WHERE e.event_type = 'login') AS logins_90d,
        COUNT(*) FILTER (WHERE e.event_type = 'support_ticket') AS support_tickets_90d,
        SUM(e.engagement_weight)                        AS engagement_score_90d,
        MAX(e.event_date)                               AS last_active_date
    FROM stg_product_events e
    CROSS JOIN recent_cutoff rc
    WHERE e.event_date >= rc.cutoff
    GROUP BY e.customer_id
),

-- All-time engagement
engagement_all AS (
    SELECT
        customer_id,
        COUNT(*)                                        AS total_events_all,
        COUNT(DISTINCT DATE_TRUNC('month', event_date)) AS active_months,
        SUM(engagement_weight)                          AS engagement_score_all,
        MAX(event_date)                                 AS last_event_date
    FROM stg_product_events
    GROUP BY customer_id
)

SELECT
    c.customer_id,
    c.company_name,
    c.industry,
    c.country,
    c.company_size_segment,
    c.signup_date,
    c.signup_cohort,
    c.churned_at,
    c.is_churned,
    c.days_as_customer,
    c.months_as_customer,
    c.csm,
    c.acquisition_channel,
    c.health_score,
    c.nps_score,
    c.seats,
    c.employee_count,

    -- subscription
    s.plan_name,
    s.billing_cycle,
    s.mrr,
    s.arr,
    s.is_active         AS sub_is_active,
    s.sub_start_date,

    -- revenue
    COALESCE(r.total_transactions, 0)          AS total_transactions,
    COALESCE(r.failed_payments, 0)             AS failed_payments,
    COALESCE(r.refunded_payments, 0)           AS refunded_payments,
    COALESCE(r.total_revenue, 0)               AS ltv,
    COALESCE(r.subscription_revenue, 0)        AS subscription_revenue,
    r.first_payment_date,
    r.last_payment_date,
    COALESCE(r.avg_transaction_value, 0)       AS avg_transaction_value,

    -- recent engagement
    COALESCE(e.total_events_90d, 0)            AS total_events_90d,
    COALESCE(e.active_days_90d, 0)             AS active_days_90d,
    COALESCE(e.logins_90d, 0)                  AS logins_90d,
    COALESCE(e.support_tickets_90d, 0)         AS support_tickets_90d,
    COALESCE(e.engagement_score_90d, 0)        AS engagement_score_90d,
    e.last_active_date,

    -- all-time engagement
    COALESCE(ea.total_events_all, 0)           AS total_events_all,
    COALESCE(ea.active_months, 0)              AS active_months,
    COALESCE(ea.engagement_score_all, 0)       AS engagement_score_all,
    ea.last_event_date,

    -- derived signals
    DATE_DIFF('day', ea.last_event_date, CURRENT_DATE)  AS days_since_last_event,
    DATE_DIFF('day', r.last_payment_date, CURRENT_DATE) AS days_since_last_payment,
    CASE
        WHEN COALESCE(r.failed_payments, 0) >= 2  THEN TRUE
        ELSE FALSE
    END AS has_payment_issues,
    ROUND(
        COALESCE(r.total_revenue, 0) /
        NULLIF(c.months_as_customer, 0), 2
    ) AS avg_monthly_revenue

FROM customer_base c
LEFT JOIN active_sub       s  ON c.customer_id = s.customer_id
LEFT JOIN rev_agg          r  ON c.customer_id = r.customer_id
LEFT JOIN engagement       e  ON c.customer_id = e.customer_id
LEFT JOIN engagement_all   ea ON c.customer_id = ea.customer_id
