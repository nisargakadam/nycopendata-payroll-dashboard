"""
data_pipeline.py
────────────────────────────────────────────────────────────────────────────
Downloads NYC Citywide Payroll Data from the Socrata OpenData API and loads
it into a local DuckDB database for fast SQL-based analysis.

Dataset: https://data.cityofnewyork.us/resource/k397-673e.json
Coverage: ~6.78M rows across all fiscal years

Run:
    python src/data_pipeline.py
"""

import os
import time
import requests
import pandas as pd
import duckdb
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE     = "https://data.cityofnewyork.us/resource/k397-673e.json"
APP_TOKEN    = os.getenv("SOCRATA_APP_TOKEN", "")   # Optional: register free token to raise rate limit
DB_PATH      = "data/payroll.duckdb"
BATCH_SIZE   = 50_000          # rows per API request
MAX_ROWS     = None            # Set to e.g. 500_000 to limit for dev; None = full dataset

DTYPE_MAP = {
    "fiscal_year":           "Int64",
    "payroll_number":        "Int64",
    "base_salary":           "float64",
    "regular_hours":         "float64",
    "regular_gross_paid":    "float64",
    "ot_hours":              "float64",
    "total_ot_paid":         "float64",
    "total_other_pay":       "float64",
}


def fetch_batch(offset: int, limit: int) -> list[dict]:
    """Fetch a single batch of rows from Socrata API."""
    params = {
        "$limit":  limit,
        "$offset": offset,
        "$order":  "fiscal_year ASC",
    }
    headers = {}
    if APP_TOKEN:
        headers["X-App-Token"] = APP_TOKEN

    resp = requests.get(API_BASE, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_all() -> pd.DataFrame:
    """Page through the full dataset and return a single DataFrame."""
    all_rows = []
    offset   = 0
    print("📡 Fetching data from NYC OpenData…")

    while True:
        batch = fetch_batch(offset, BATCH_SIZE)
        if not batch:
            break

        all_rows.extend(batch)
        offset += len(batch)
        print(f"   ↳ Fetched {offset:,} rows so far…", end="\r")

        if MAX_ROWS and offset >= MAX_ROWS:
            print(f"\n⚠️  MAX_ROWS cap hit ({MAX_ROWS:,}). Truncating.")
            all_rows = all_rows[:MAX_ROWS]
            break

        if len(batch) < BATCH_SIZE:
            break          # last page

        time.sleep(0.2)    # polite pause

    print(f"\n✅ Total rows fetched: {len(all_rows):,}")
    return pd.DataFrame(all_rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dtypes, derived columns, and basic cleaning."""
    # ── Standardise column names ──────────────────────────────────────────────
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # ── Cast numerics ─────────────────────────────────────────────────────────
    for col, dtype in DTYPE_MAP.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    # ── Parse date ────────────────────────────────────────────────────────────
    if "agency_start_date" in df.columns:
        df["agency_start_date"] = pd.to_datetime(df["agency_start_date"], errors="coerce")

    # ── Derived features ──────────────────────────────────────────────────────
    df["total_compensation"] = (
        df.get("regular_gross_paid", 0).fillna(0)
        + df.get("total_ot_paid", 0).fillna(0)
        + df.get("total_other_pay", 0).fillna(0)
    )

    df["ot_ratio"] = np.where(
        df["regular_hours"].fillna(0) > 0,
        df["ot_hours"].fillna(0) / df["regular_hours"],
        0.0,
    )

    df["ot_pay_rate"] = np.where(
        df["ot_hours"].fillna(0) > 0,
        df["total_ot_paid"].fillna(0) / df["ot_hours"],
        0.0,
    )

    # Agency tenure in years (at time of record)
    if "agency_start_date" in df.columns:
        df["tenure_years"] = (
            pd.to_datetime(df["fiscal_year"].astype(str) + "-06-30", errors="coerce")
            - df["agency_start_date"]
        ).dt.days / 365.25
        df["tenure_years"] = df["tenure_years"].clip(lower=0)

    # ── Standardise text fields ───────────────────────────────────────────────
    for col in ["agency_name", "title_description", "work_location_borough",
                "leave_status_as_of_june_30", "pay_basis"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()

    # ── Drop rows with no salary data ─────────────────────────────────────────
    df = df.dropna(subset=["regular_gross_paid"])
    df = df[df["regular_gross_paid"] > 0]

    return df.reset_index(drop=True)


def load_to_duckdb(df: pd.DataFrame, db_path: str) -> None:
    """Persist the clean DataFrame into DuckDB."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS payroll")
    con.execute("CREATE TABLE payroll AS SELECT * FROM df")

    # ── Useful views ──────────────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE VIEW agency_annual AS
        SELECT
            fiscal_year,
            agency_name,
            COUNT(*)                          AS headcount,
            ROUND(AVG(base_salary), 2)        AS avg_base_salary,
            ROUND(AVG(regular_gross_paid), 2) AS avg_gross,
            ROUND(SUM(regular_gross_paid), 2) AS total_gross,
            ROUND(SUM(total_ot_paid), 2)      AS total_ot,
            ROUND(AVG(ot_ratio), 4)           AS avg_ot_ratio,
            ROUND(SUM(total_compensation), 2) AS total_spend
        FROM payroll
        GROUP BY fiscal_year, agency_name
    """)

    con.execute("""
        CREATE OR REPLACE VIEW sbs_detail AS
        SELECT *
        FROM payroll
        WHERE agency_name LIKE '%SMALL BUSINESS%'
    """)

    row_count = con.execute("SELECT COUNT(*) FROM payroll").fetchone()[0]
    con.close()

    print(f"✅ DuckDB loaded → {db_path}")
    print(f"   ↳ {row_count:,} rows in `payroll` table")
    print(f"   ↳ Views created: `agency_annual`, `sbs_detail`")


def sample_queries(db_path: str) -> None:
    """Print a few sanity-check query results."""
    con = duckdb.connect(db_path, read_only=True)

    print("\n── Fiscal years in dataset ──────────────────────────────────────────")
    print(con.execute("SELECT DISTINCT fiscal_year FROM payroll ORDER BY 1").df().T.to_string())

    print("\n── Top 10 agencies by total spend (latest FY) ───────────────────────")
    print(con.execute("""
        SELECT agency_name, headcount, total_spend
        FROM agency_annual
        WHERE fiscal_year = (SELECT MAX(fiscal_year) FROM agency_annual)
        ORDER BY total_spend DESC
        LIMIT 10
    """).df().to_string(index=False))

    print("\n── SBS snapshot ──────────────────────────────────────────────────────")
    print(con.execute("""
        SELECT fiscal_year, headcount, avg_gross, total_spend, avg_ot_ratio
        FROM agency_annual
        WHERE agency_name LIKE '%SMALL BUSINESS%'
        ORDER BY fiscal_year
    """).df().to_string(index=False))

    con.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw   = fetch_all()
    df_clean = clean(df_raw)
    load_to_duckdb(df_clean, DB_PATH)
    sample_queries(DB_PATH)
    print("\n🎉 Pipeline complete. Run `streamlit run src/app.py` to launch the dashboard.")
