"""
data_generator.py
-----------------
Generates realistic synthetic e-commerce event logs and A/B test data.
Run this FIRST before db_setup.py.

Usage:
    python src/data_generator.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Constants ────────────────────────────────────────────────────────────────
N_USERS         = 5_000
DAYS            = 90                        # 3-month observation window
START_DATE      = datetime(2024, 1, 1)
N_AB_USERS      = 5_000
AB_START        = datetime(2024, 3, 1)      # experiment launched after baseline period

PRODUCTS        = [f"PROD_{i:04d}" for i in range(1, 101)]
CATEGORIES      = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Beauty"]
DEVICES         = ["desktop", "mobile", "tablet"]
COUNTRIES       = ["US", "UK", "CA", "AU", "IN", "DE", "FR"]

DEVICE_W        = [0.45, 0.40, 0.15]
COUNTRY_W       = [0.50, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05]

# Funnel drop-off probabilities (realistic e-commerce benchmarks)
P_ADD_TO_CART   = 0.40   # of page_view users
P_CHECKOUT      = 0.60   # of add_to_cart users
P_PURCHASE      = 0.65   # of checkout users  →  overall ~15.6%


def generate_events(n_users: int = N_USERS, days: int = DAYS) -> pd.DataFrame:
    """
    Simulate raw event logs for an e-commerce platform.
    Each user may have 1–5 sessions, each progressing through the funnel
    with realistic drop-off probabilities.

    Returns a DataFrame matching the `events` PostgreSQL table schema.
    """
    np.random.seed(SEED)
    rows = []
    event_id = 1

    for u_idx in range(n_users):
        user_id     = f"USER_{u_idx:05d}"
        device      = np.random.choice(DEVICES, p=DEVICE_W)
        country     = np.random.choice(COUNTRIES, p=COUNTRY_W)
        join_day    = np.random.randint(0, max(1, days - 7))
        n_sessions  = np.random.randint(1, 6)

        for s_idx in range(n_sessions):
            session_id  = f"SESS_{u_idx:05d}_{s_idx}"
            session_day = join_day + np.random.randint(0, max(1, days - join_day))
            session_ts  = START_DATE + timedelta(
                days    = session_day,
                hours   = int(np.random.randint(0, 24)),
                minutes = int(np.random.randint(0, 60)),
            )
            product     = np.random.choice(PRODUCTS)
            category    = np.random.choice(CATEGORIES)

            def make_event(etype, ts_offset_min=(0, 0), revenue=None):
                nonlocal event_id
                offset = np.random.randint(*ts_offset_min) if ts_offset_min[1] else 0
                row = dict(
                    event_id        = event_id,
                    user_id         = user_id,
                    session_id      = session_id,
                    event_type      = etype,
                    event_timestamp = session_ts + timedelta(minutes=int(offset)),
                    product_id      = product,
                    category        = category,
                    revenue         = revenue,
                    device_type     = device,
                    country         = country,
                )
                event_id += 1
                return row

            # Stage 1 – always fires
            rows.append(make_event("page_view", (0, 1)))

            # Stage 2 – 40% proceed
            if np.random.random() < P_ADD_TO_CART:
                rows.append(make_event("add_to_cart", (1, 10)))

                # Stage 3 – 60% of those who added to cart
                if np.random.random() < P_CHECKOUT:
                    rows.append(make_event("checkout", (10, 20)))

                    # Stage 4 – 65% of those who checked out
                    if np.random.random() < P_PURCHASE:
                        rev = float(np.clip(
                            np.random.lognormal(mean=4.0, sigma=0.8), 10, 500
                        ))
                        rows.append(make_event("purchase", (20, 30), revenue=round(rev, 2)))

    df = pd.DataFrame(rows)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    return df


def generate_ab_test(n_users: int = N_AB_USERS) -> pd.DataFrame:
    """
    Simulate the checkout redesign A/B experiment.

    Control  (n = 2500): old checkout – 12.0% conversion, AOV ~$82
    Treatment(n = 2500): new checkout – 14.5% conversion, AOV ~$91  (+11% uplift)

    Returns a DataFrame matching the `ab_test` PostgreSQL table schema.
    """
    np.random.seed(SEED)
    rows = []
    half = n_users // 2

    for i in range(n_users):
        user_id    = f"AB_USER_{i:05d}"
        group      = "control" if i < half else "treatment"
        conv_prob  = 0.120 if group == "control" else 0.145
        aov_mean   = 4.40  if group == "control" else 4.51   # log-scale means
        converted  = int(np.random.random() < conv_prob)

        if converted:
            order_value = round(float(np.clip(
                np.random.lognormal(mean=aov_mean, sigma=0.65), 15, 450
            )), 2)
        else:
            order_value = None

        session_dur = int(np.clip(
            np.random.lognormal(mean=5.5, sigma=0.8), 30, 3600
        ))
        assigned_at = AB_START + timedelta(
            days    = int(np.random.randint(0, 30)),
            hours   = int(np.random.randint(0, 24)),
            minutes = int(np.random.randint(0, 60)),
        )

        rows.append(dict(
            user_id              = user_id,
            group_name           = group,
            converted            = converted,
            order_value          = order_value,
            session_duration_sec = session_dur,
            assigned_at          = assigned_at,
        ))

    df = pd.DataFrame(rows)
    df["assigned_at"] = pd.to_datetime(df["assigned_at"])
    return df


if __name__ == "__main__":
    os.makedirs("data/raw",       exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("=" * 55)
    print("  E-Commerce Analytics — Data Generator")
    print("=" * 55)

    print("\n[1/2] Generating event logs …")
    ev = generate_events()
    ev.to_csv("data/raw/events.csv", index=False)
    print(f"      ✓  {len(ev):,} events  |  {ev['user_id'].nunique():,} users  |  "
          f"{ev['session_id'].nunique():,} sessions")
    print(f"      ✓  Purchases : {(ev['event_type']=='purchase').sum():,}  "
          f"({(ev['event_type']=='purchase').sum()/ev['session_id'].nunique()*100:.1f}% of sessions)")

    print("\n[2/2] Generating A/B test data …")
    ab = generate_ab_test()
    ab.to_csv("data/raw/ab_test_raw.csv", index=False)
    ctrl = ab[ab["group_name"] == "control"]
    trt  = ab[ab["group_name"] == "treatment"]
    print(f"      ✓  {len(ab):,} users  |  Control: {len(ctrl):,}  |  Treatment: {len(trt):,}")
    print(f"      ✓  Control conv.   : {ctrl['converted'].mean()*100:.2f}%")
    print(f"      ✓  Treatment conv. : {trt['converted'].mean()*100:.2f}%")

    print("\n  Data written to  data/raw/")
    print("  Next step: python src/db_setup.py\n")
