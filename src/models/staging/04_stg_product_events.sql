-- stg_product_events: standardise product usage events
SELECT
    event_id,
    customer_id,
    LOWER(TRIM(event_type))                     AS event_type,
    CAST(event_date AS DATE)                    AS event_date,
    DATE_TRUNC('month', CAST(event_date AS DATE)) AS event_month,
    DATE_TRUNC('week',  CAST(event_date AS DATE)) AS event_week,
    -- categorise event depth
    CASE
        WHEN event_type IN ('login','dashboard_view','billing_viewed')
             THEN 'shallow'
        WHEN event_type IN ('report_generated','export_csv','share_report')
             THEN 'medium'
        WHEN event_type IN ('api_call','integration_added','feature_flag_used')
             THEN 'deep'
        WHEN event_type = 'support_ticket'
             THEN 'support'
        ELSE 'other'
    END                                         AS event_depth,
    -- engagement score weight per event
    CASE
        WHEN event_type = 'integration_added'   THEN 10
        WHEN event_type = 'api_call'            THEN 7
        WHEN event_type = 'feature_flag_used'   THEN 6
        WHEN event_type = 'report_generated'    THEN 5
        WHEN event_type = 'share_report'        THEN 5
        WHEN event_type = 'export_csv'          THEN 4
        WHEN event_type = 'dashboard_view'      THEN 3
        WHEN event_type = 'billing_viewed'      THEN 2
        WHEN event_type = 'login'               THEN 1
        WHEN event_type = 'support_ticket'      THEN -2  -- friction signal
        ELSE 1
    END                                         AS engagement_weight
FROM raw_product_events
WHERE event_id      IS NOT NULL
  AND customer_id   IS NOT NULL
  AND event_date    IS NOT NULL
