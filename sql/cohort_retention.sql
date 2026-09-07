-- ============================================================
-- Weekly Cohort Retention Analysis
-- Techniques: DATE_TRUNC, DENSE_RANK, self-join via CTEs,
--             pivoting retention into a matrix format
-- ============================================================

-- ── Query 1: Retention Matrix (long format) ─────────────────

WITH user_cohorts AS (
    -- Assign each user to their acquisition week (first event)
    SELECT
        user_id,
        DATE_TRUNC('week', MIN(event_timestamp))::DATE   AS cohort_week
    FROM   events
    GROUP  BY user_id
),

user_purchase_weeks AS (
    -- Every week a user made a purchase (their "activity" weeks)
    SELECT DISTINCT
        user_id,
        DATE_TRUNC('week', event_timestamp)::DATE        AS activity_week
    FROM   events
    WHERE  event_type = 'purchase'
),

cohort_activity AS (
    -- Compute which "week number" each purchase falls into
    -- relative to the user's acquisition cohort week
    SELECT
        uc.cohort_week,
        upw.activity_week,
        (
          DATE_PART('day', upw.activity_week::TIMESTAMP
                          - uc.cohort_week::TIMESTAMP) / 7
        )::INTEGER                                       AS week_number,
        COUNT(DISTINCT upw.user_id)                      AS active_users
    FROM   user_cohorts           uc
    JOIN   user_purchase_weeks    upw USING (user_id)
    GROUP  BY uc.cohort_week, upw.activity_week, week_number
),

cohort_sizes AS (
    -- How many users belong to each cohort
    SELECT
        cohort_week,
        COUNT(*)                                         AS cohort_size,
        -- Rank cohorts newest → oldest (useful for UI ordering)
        DENSE_RANK() OVER (ORDER BY cohort_week DESC)    AS cohort_rank
    FROM   user_cohorts
    GROUP  BY cohort_week
)

SELECT
    ca.cohort_week::TEXT                                         AS cohort_week,
    cs.cohort_size,
    cs.cohort_rank,
    ca.week_number,
    ca.active_users,
    ROUND(100.0 * ca.active_users / cs.cohort_size, 1)          AS retention_pct
FROM   cohort_activity  ca
JOIN   cohort_sizes     cs USING (cohort_week)
WHERE  ca.week_number BETWEEN 0 AND 8   -- focus on first 8 weeks
ORDER  BY ca.cohort_week, ca.week_number;


-- ============================================================
-- Query 2: Revenue Cohort — total revenue contributed
--          by each cohort over their lifetime
-- Techniques: SUM partitioned window, running total
-- ============================================================

WITH user_cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('week', MIN(event_timestamp))::DATE   AS cohort_week
    FROM   events
    GROUP  BY user_id
),

cohort_revenue AS (
    SELECT
        uc.cohort_week,
        DATE_TRUNC('week', e.event_timestamp)::DATE      AS purchase_week,
        (
          DATE_PART('day',
                    DATE_TRUNC('week', e.event_timestamp)::TIMESTAMP
                    - uc.cohort_week::TIMESTAMP) / 7
        )::INTEGER                                       AS week_number,
        COUNT(DISTINCT e.user_id)                        AS buying_users,
        ROUND(SUM(e.revenue)::NUMERIC, 2)                AS weekly_revenue
    FROM   events        e
    JOIN   user_cohorts  uc USING (user_id)
    WHERE  e.event_type = 'purchase'
    GROUP  BY uc.cohort_week, purchase_week, week_number
)

SELECT
    cohort_week::TEXT,
    week_number,
    buying_users,
    weekly_revenue,
    -- Running cumulative revenue per cohort using SUM window
    ROUND(
      SUM(weekly_revenue) OVER (
          PARTITION BY cohort_week
          ORDER BY week_number
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      )::NUMERIC, 2)                                     AS cumulative_revenue
FROM   cohort_revenue
WHERE  week_number BETWEEN 0 AND 8
ORDER  BY cohort_week, week_number;
