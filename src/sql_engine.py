"""
sql_engine.py
-------------
Thin wrapper around SQLAlchemy that executes parameterised queries
and raw SQL files against the PostgreSQL database.
"""

import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


class PostgreSQLEngine:
    """Manages a SQLAlchemy connection pool to PostgreSQL."""

    def __init__(self, db_url: str | None = None):
        url = db_url or os.getenv("DATABASE_URL", "")
        if not url:
            raise ValueError(
                "DATABASE_URL is not set. "
                "Copy .env.example → .env and fill in your credentials."
            )
        self.engine = create_engine(url, future=True, pool_pre_ping=True)

    # ── Public helpers ───────────────────────────────────────────────────────

    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        """Execute a SELECT statement and return a DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)

    def execute(self, sql: str) -> None:
        """Execute a non-SELECT statement (DDL / DML)."""
        with self.engine.begin() as conn:
            conn.execute(text(sql))

    def scalar(self, sql: str) -> object:
        """Return a single scalar value."""
        with self.engine.connect() as conn:
            return conn.execute(text(sql)).scalar()

    # ── Named query shortcuts ────────────────────────────────────────────────

    def get_funnel_metrics(self) -> pd.DataFrame:
        sql = """
        WITH funnel_stages AS (
            SELECT
                event_type,
                COUNT(DISTINCT user_id) AS unique_users,
                CASE event_type
                    WHEN 'page_view'   THEN 1
                    WHEN 'add_to_cart' THEN 2
                    WHEN 'checkout'    THEN 3
                    WHEN 'purchase'    THEN 4
                END AS stage_order
            FROM events
            WHERE event_type IN ('page_view','add_to_cart','checkout','purchase')
            GROUP BY event_type
        ),
        funnel_with_context AS (
            SELECT
                stage_order,
                event_type AS stage_name,
                unique_users,
                LAG(unique_users) OVER (ORDER BY stage_order)       AS prev_stage_users,
                FIRST_VALUE(unique_users) OVER (
                    ORDER BY stage_order
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                )                                                    AS top_of_funnel
            FROM funnel_stages
        )
        SELECT
            stage_order,
            stage_name,
            unique_users,
            ROUND(100.0 * unique_users / top_of_funnel, 2)           AS overall_conv_pct,
            COALESCE(ROUND(100.0 * unique_users
                           / NULLIF(prev_stage_users,0), 2), 100.00) AS step_conv_pct,
            COALESCE(ROUND(100.0 * (prev_stage_users - unique_users)
                           / NULLIF(prev_stage_users,0), 2),   0.00) AS step_dropoff_pct
        FROM funnel_with_context
        ORDER BY stage_order
        """
        return self.query(sql)

    def get_device_breakdown(self) -> pd.DataFrame:
        sql = """
        SELECT
            device_type,
            COUNT(DISTINCT CASE WHEN event_type='page_view'   THEN user_id END) AS visitors,
            COUNT(DISTINCT CASE WHEN event_type='add_to_cart' THEN user_id END) AS cart_users,
            COUNT(DISTINCT CASE WHEN event_type='checkout'    THEN user_id END) AS checkout_users,
            COUNT(DISTINCT CASE WHEN event_type='purchase'    THEN user_id END) AS buyers,
            ROUND(SUM(CASE WHEN event_type='purchase' THEN revenue END)::NUMERIC,2) AS total_revenue,
            ROUND(100.0 * COUNT(DISTINCT CASE WHEN event_type='purchase' THEN user_id END)
                  / NULLIF(COUNT(DISTINCT CASE WHEN event_type='page_view' THEN user_id END),0),
                  2) AS overall_conv_pct
        FROM events
        GROUP BY device_type
        ORDER BY overall_conv_pct DESC
        """
        return self.query(sql)

    def get_category_revenue(self) -> pd.DataFrame:
        sql = """
        SELECT
            category,
            COUNT(DISTINCT user_id)                            AS buyers,
            COUNT(*)                                           AS purchases,
            ROUND(SUM(revenue)::NUMERIC, 2)                   AS total_revenue,
            ROUND(AVG(revenue)::NUMERIC, 2)                   AS avg_order_value
        FROM events
        WHERE event_type = 'purchase'
        GROUP BY category
        ORDER BY total_revenue DESC
        """
        return self.query(sql)

    def get_cohort_retention(self) -> pd.DataFrame:
        sql = """
        WITH user_cohorts AS (
            SELECT user_id,
                   DATE_TRUNC('week', MIN(event_timestamp))::DATE AS cohort_week
            FROM   events GROUP BY user_id
        ),
        user_purchase_weeks AS (
            SELECT DISTINCT user_id,
                   DATE_TRUNC('week', event_timestamp)::DATE AS activity_week
            FROM   events WHERE event_type = 'purchase'
        ),
        cohort_activity AS (
            SELECT uc.cohort_week,
                   (DATE_PART('day', upw.activity_week::TIMESTAMP
                              - uc.cohort_week::TIMESTAMP)/7)::INTEGER AS week_number,
                   COUNT(DISTINCT upw.user_id) AS active_users
            FROM   user_cohorts uc
            JOIN   user_purchase_weeks upw USING (user_id)
            GROUP  BY uc.cohort_week, week_number
        ),
        cohort_sizes AS (
            SELECT cohort_week, COUNT(*) AS cohort_size
            FROM   user_cohorts GROUP BY cohort_week
        )
        SELECT ca.cohort_week::TEXT,
               cs.cohort_size,
               ca.week_number,
               ca.active_users,
               ROUND(100.0 * ca.active_users / cs.cohort_size, 1) AS retention_pct
        FROM   cohort_activity ca
        JOIN   cohort_sizes cs USING (cohort_week)
        WHERE  ca.week_number BETWEEN 0 AND 8
        ORDER  BY ca.cohort_week, ca.week_number
        """
        return self.query(sql)

    def get_ab_test_data(self) -> pd.DataFrame:
        return self.query("SELECT * FROM ab_test")

    def ping(self) -> bool:
        """Return True if the database is reachable."""
        try:
            self.scalar("SELECT 1")
            return True
        except Exception:
            return False
