"""
db_setup.py
-----------
Creates the PostgreSQL schema and bulk-loads the generated CSV files.
Run ONCE after data_generator.py.

Usage:
    python src/db_setup.py

Requires DATABASE_URL in .env (see .env.example).
"""

import os
import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent   # project root


def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        sys.exit(
            "\n❌  DATABASE_URL is not set.\n"
            "   Copy .env.example → .env and fill in your credentials.\n"
        )
    return create_engine(db_url, future=True)


def run_schema(engine) -> None:
    print("\n[1/3] Creating schema …")
    schema_path = ROOT / "sql" / "schema.sql"
    raw = schema_path.read_text(encoding="utf-8")
    raw = "\n".join(
        line for line in raw.splitlines()
        if not line.strip().startswith("--")
    )

    with engine.begin() as conn:
        conn.exec_driver_sql(raw)

    print("      ✓  Schema ready (events + ab_test tables created)")


def load_events(engine) -> None:
    csv_path = ROOT / "data" / "raw" / "events.csv"
    if not csv_path.exists():
        sys.exit(
            "\n❌  data/raw/events.csv not found.\n"
            "   Run  python src/data_generator.py  first.\n"
        )

    print("\n[2/3] Loading events.csv into PostgreSQL …")
    df = pd.read_csv(csv_path)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    df.to_sql(
        "events",
        engine,
        if_exists="append",     # schema already created above
        index=False,
        chunksize=2_000,
        method="multi",
    )
    print(f"      ✓  {len(df):,} rows loaded")


def load_ab_test(engine) -> None:
    csv_path = ROOT / "data" / "raw" / "ab_test_raw.csv"
    if not csv_path.exists():
        sys.exit(
            "\n❌  data/raw/ab_test_raw.csv not found.\n"
            "   Run  python src/data_generator.py  first.\n"
        )

    print("\n[3/3] Loading ab_test_raw.csv into PostgreSQL …")
    df = pd.read_csv(csv_path)
    df["assigned_at"]  = pd.to_datetime(df["assigned_at"], utc=True)
    df["order_value"]  = pd.to_numeric(df["order_value"], errors="coerce")
    df["converted"]    = df["converted"].astype(int)

    df.to_sql(
        "ab_test",
        engine,
        if_exists="append",
        index=False,
        chunksize=2_000,
        method="multi",
    )
    print(f"      ✓  {len(df):,} rows loaded")


def verify(engine) -> None:
    with engine.connect() as conn:
        n_events = conn.execute(text("SELECT COUNT(*) FROM events")).scalar()
        n_ab     = conn.execute(text("SELECT COUNT(*) FROM ab_test")).scalar()
        n_users  = conn.execute(
            text("SELECT COUNT(DISTINCT user_id) FROM events")
        ).scalar()

    print(f"\n  Verification")
    print(f"  ─────────────────────────────────────")
    print(f"  events rows   : {n_events:>10,}")
    print(f"  ab_test rows  : {n_ab:>10,}")
    print(f"  unique users  : {n_users:>10,}")
    print(f"\n  ✅  Database is ready. Launch the app with:")
    print(f"      streamlit run app.py\n")


if __name__ == "__main__":
    print("=" * 55)
    print("  E-Commerce Analytics — Database Setup")
    print("=" * 55)

    eng = get_engine()
    run_schema(eng)
    load_events(eng)
    load_ab_test(eng)
    verify(eng)
