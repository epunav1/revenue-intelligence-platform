-- stg_customers: clean and standardise the raw customer table
SELECT
    customer_id,
    TRIM(company_name)                          AS company_name,
    UPPER(TRIM(industry))                       AS industry,
    UPPER(TRIM(country))                        AS country,
    plan                                         AS plan_name,
    billing_cycle,
    CAST(seats AS INTEGER)                      AS seats,
    CAST(employee_count AS INTEGER)             AS employee_count,
    CAST(signup_date AS DATE)                   AS signup_date,
    CAST(churned_at AS DATE)                    AS churned_at,
    csm,
    acquisition_channel,
    ROUND(CAST(health_score AS DOUBLE), 1)      AS health_score,
    CAST(nps_score AS INTEGER)                  AS nps_score,
    CAST(is_churned AS BOOLEAN)                 AS is_churned,
    -- derived fields
    DATE_DIFF('day',  CAST(signup_date AS DATE), CURRENT_DATE)  AS days_as_customer,
    DATE_DIFF('month', CAST(signup_date AS DATE), CURRENT_DATE) AS months_as_customer,
    DATE_TRUNC('month', CAST(signup_date AS DATE))              AS signup_cohort,
    CASE
        WHEN employee_count BETWEEN 1   AND 10  THEN 'Micro (1-10)'
        WHEN employee_count BETWEEN 11  AND 50  THEN 'Small (11-50)'
        WHEN employee_count BETWEEN 51  AND 200 THEN 'Mid-Market (51-200)'
        WHEN employee_count BETWEEN 201 AND 1000 THEN 'Enterprise (201-1000)'
        ELSE 'Large Enterprise (1000+)'
    END                                         AS company_size_segment
FROM raw_customers
WHERE customer_id IS NOT NULL
