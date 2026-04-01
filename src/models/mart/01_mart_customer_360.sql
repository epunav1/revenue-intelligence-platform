-- mart_customer_360
-- Single source of truth for customer-level analytics.
-- Includes RFM scores, health tiers, and key account metrics.
WITH
base AS (SELECT * FROM int_customer_metrics),

-- RFM quintile scoring (1=worst, 5=best)
rfm_scores AS (
    SELECT
        customer_id,
        -- Recency: days since last event (lower = better)
        NTILE(5) OVER (ORDER BY days_since_last_event DESC)  AS r_score,
        -- Frequency: total transactions
        NTILE(5) OVER (ORDER BY total_transactions ASC)      AS f_score,
        -- Monetary: lifetime value
        NTILE(5) OVER (ORDER BY ltv ASC)                     AS m_score
    FROM base
),

rfm_combined AS (
    SELECT
        r.customer_id,
        r.r_score,
        r.f_score,
        r.m_score,
        ROUND((r.r_score + r.f_score + r.m_score) / 3.0, 2) AS rfm_score,
        CASE
            WHEN r.r_score >= 4 AND r.f_score >= 4 AND r.m_score >= 4
                THEN 'Champions'
            WHEN r.r_score >= 3 AND r.f_score >= 3 AND r.m_score >= 3
                THEN 'Loyal Customers'
            WHEN r.r_score >= 3 AND r.f_score <= 2
                THEN 'Potential Loyalists'
            WHEN r.r_score >= 4 AND r.f_score <= 2 AND r.m_score <= 2
                THEN 'New Customers'
            WHEN r.r_score <= 2 AND r.f_score >= 3 AND r.m_score >= 3
                THEN 'At Risk'
            WHEN r.r_score <= 2 AND r.f_score <= 2 AND r.m_score >= 3
                THEN 'Cant Lose Them'
            WHEN r.r_score <= 2 AND r.f_score >= 2
                THEN 'Hibernating'
            ELSE 'Lost'
        END AS rfm_segment
    FROM rfm_scores r
)

SELECT
    b.*,
    rc.r_score,
    rc.f_score,
    rc.m_score,
    rc.rfm_score,
    rc.rfm_segment,

    -- Health tier based on composite signals
    CASE
        WHEN b.health_score >= 80
         AND b.engagement_score_90d >= 50
         AND b.failed_payments = 0
            THEN 'Healthy'
        WHEN b.health_score >= 60
         AND b.engagement_score_90d >= 20
            THEN 'Neutral'
        WHEN b.health_score < 60
          OR b.days_since_last_event > 30
          OR b.failed_payments >= 2
            THEN 'At Risk'
        ELSE 'Neutral'
    END AS health_tier,

    -- LTV tier
    CASE
        WHEN b.ltv >= 50000 THEN 'Platinum'
        WHEN b.ltv >= 20000 THEN 'Gold'
        WHEN b.ltv >= 8000  THEN 'Silver'
        ELSE                     'Bronze'
    END AS ltv_tier,

    -- Engagement velocity (events per active month)
    CASE
        WHEN b.active_months > 0
        THEN ROUND(b.total_events_all::DOUBLE / b.active_months, 1)
        ELSE 0
    END AS events_per_month

FROM base b
LEFT JOIN rfm_combined rc USING (customer_id)
