-- ============================================================
-- Conversion Funnel Analysis
-- Techniques: Multi-level CTEs, LAG(), FIRST_VALUE(), RANK()
-- ============================================================

-- ── Query 1: Core Funnel Drop-off Metrics ───────────────────

WITH funnel_stages AS (
    -- Step 1: Aggregate unique users at each funnel stage
    SELECT
        event_type,
        COUNT(DISTINCT user_id)                         AS unique_users,
        CASE event_type
            WHEN 'page_view'   THEN 1
            WHEN 'add_to_cart' THEN 2
            WHEN 'checkout'    THEN 3
            WHEN 'purchase'    THEN 4
        END                                             AS stage_order
    FROM   events
    WHERE  event_type IN ('page_view', 'add_to_cart', 'checkout', 'purchase')
    GROUP  BY event_type
),

funnel_with_context AS (
    -- Step 2: Attach previous-stage and top-of-funnel counts via window functions
    SELECT
        stage_order,
        event_type                                      AS stage_name,
        unique_users,
        -- LAG: pull the user count from the immediately preceding stage
        LAG(unique_users)   OVER (ORDER BY stage_order) AS prev_stage_users,
        -- FIRST_VALUE: always carries the top-of-funnel (page_view) count
        FIRST_VALUE(unique_users) OVER (ORDER BY stage_order
                                        ROWS BETWEEN UNBOUNDED PRECEDING
                                                 AND UNBOUNDED FOLLOWING) AS top_of_funnel
    FROM   funnel_stages
),

funnel_metrics AS (
    -- Step 3: Derive conversion and drop-off rates
    SELECT
        stage_order,
        stage_name,
        unique_users,
        ROUND(100.0 * unique_users          / top_of_funnel,                        2) AS overall_conv_pct,
        COALESCE(
          ROUND(100.0 * unique_users        / NULLIF(prev_stage_users, 0),           2),
          100.00)                                                                       AS step_conv_pct,
        COALESCE(
          ROUND(100.0 * (prev_stage_users - unique_users)
                       / NULLIF(prev_stage_users, 0),                                2),
          0.00)                                                                         AS step_dropoff_pct
    FROM   funnel_with_context
)

SELECT
    stage_order,
    stage_name,
    unique_users,
    overall_conv_pct,
    step_conv_pct,
    step_dropoff_pct
FROM   funnel_metrics
ORDER  BY stage_order;


-- ============================================================
-- Query 2: Top 20 Products by Revenue (with category ranking)
-- Techniques: RANK() OVER, CTE chaining, JOIN
-- ============================================================

WITH product_stats AS (
    SELECT
        product_id,
        category,
        COUNT(DISTINCT session_id)                                       AS sessions,
        COUNT(*)                                                         AS purchases,
        ROUND(SUM(revenue)::NUMERIC,  2)                                 AS total_revenue,
        ROUND(AVG(revenue)::NUMERIC,  2)                                 AS avg_order_value,
        RANK() OVER (ORDER BY SUM(revenue) DESC)                         AS revenue_rank
    FROM   events
    WHERE  event_type = 'purchase'
    GROUP  BY product_id, category
),

category_rank AS (
    SELECT
        category,
        ROUND(SUM(total_revenue)::NUMERIC, 2)                            AS category_revenue,
        RANK() OVER (ORDER BY SUM(total_revenue) DESC)                   AS cat_rank
    FROM   product_stats
    GROUP  BY category
)

SELECT
    ps.revenue_rank,
    ps.product_id,
    ps.category,
    cr.cat_rank        AS category_rank,
    ps.sessions,
    ps.purchases,
    ps.total_revenue,
    ps.avg_order_value
FROM   product_stats  ps
JOIN   category_rank  cr USING (category)
WHERE  ps.revenue_rank <= 20
ORDER  BY ps.revenue_rank;


-- ============================================================
-- Query 3: Device & Country Segmented Funnel
-- Techniques: Conditional aggregation, multi-dimensional grouping
-- ============================================================

WITH device_funnel AS (
    SELECT
        device_type,
        COUNT(DISTINCT CASE WHEN event_type = 'page_view'   THEN user_id END) AS visitors,
        COUNT(DISTINCT CASE WHEN event_type = 'add_to_cart' THEN user_id END) AS cart_users,
        COUNT(DISTINCT CASE WHEN event_type = 'checkout'    THEN user_id END) AS checkout_users,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase'    THEN user_id END) AS buyers,
        ROUND(SUM(CASE WHEN event_type = 'purchase' THEN revenue END)::NUMERIC, 2) AS total_revenue
    FROM   events
    GROUP  BY device_type
)

SELECT
    device_type,
    visitors,
    cart_users,
    checkout_users,
    buyers,
    total_revenue,
    ROUND(100.0 * buyers / NULLIF(visitors, 0), 2)      AS overall_conv_pct,
    ROUND(total_revenue  / NULLIF(buyers, 0),   2)      AS avg_order_value
FROM   device_funnel
ORDER  BY overall_conv_pct DESC;
