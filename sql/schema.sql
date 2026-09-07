-- ============================================================
-- E-Commerce Analytics · PostgreSQL Schema
-- ============================================================

-- Drop existing tables (safe for re-runs)
DROP TABLE IF EXISTS ab_test CASCADE;
DROP TABLE IF EXISTS events  CASCADE;

-- ── Events table (raw event log) ────────────────────────────
CREATE TABLE events (
    event_id        BIGINT          PRIMARY KEY,
    user_id         VARCHAR(50)     NOT NULL,
    session_id      VARCHAR(50)     NOT NULL,
    event_type      VARCHAR(50)     NOT NULL,   -- page_view | add_to_cart | checkout | purchase
    event_timestamp TIMESTAMPTZ     NOT NULL,
    product_id      VARCHAR(50),
    category        VARCHAR(50),
    revenue         NUMERIC(10, 2),             -- populated only on purchase events
    device_type     VARCHAR(20),                -- desktop | mobile | tablet
    country         VARCHAR(50)
);

CREATE INDEX idx_events_user_id     ON events (user_id);
CREATE INDEX idx_events_event_type  ON events (event_type);
CREATE INDEX idx_events_timestamp   ON events (event_timestamp);
CREATE INDEX idx_events_type_ts     ON events (event_type, event_timestamp);

-- ── A/B Test table ──────────────────────────────────────────
CREATE TABLE ab_test (
    user_id              VARCHAR(50)   PRIMARY KEY,
    group_name           VARCHAR(10)   NOT NULL,        -- control | treatment
    converted            SMALLINT      NOT NULL DEFAULT 0,
    order_value          NUMERIC(10, 2),                -- NULL if not converted
    session_duration_sec INTEGER,
    assigned_at          TIMESTAMPTZ   NOT NULL
);

CREATE INDEX idx_ab_group ON ab_test (group_name);

-- ── Useful comments for interviewers ───────────────────────
COMMENT ON TABLE events  IS 'Raw event log capturing every user interaction step in the funnel';
COMMENT ON TABLE ab_test IS 'Checkout redesign A/B experiment: control (baseline) vs treatment (new CTA + dynamic pricing)';
COMMENT ON COLUMN events.event_type   IS 'Funnel stage: page_view → add_to_cart → checkout → purchase';
COMMENT ON COLUMN ab_test.group_name  IS 'Experiment arm: control = old checkout, treatment = redesigned checkout';
COMMENT ON COLUMN ab_test.order_value IS 'Revenue generated; NULL means user did not convert';
