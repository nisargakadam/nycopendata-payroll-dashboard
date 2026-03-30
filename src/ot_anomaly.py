"""
ot_anomaly.py
────────────────────────────────────────────────────────────────────────────
Overtime anomaly detection using Isolation Forest.

Identifies employees or agencies with statistically unusual OT patterns —
surfacing potential overtime abuse, data entry errors, or staffing issues.

Features used:
  - ot_hours        : raw overtime hours
  - total_ot_paid   : raw OT dollar amount
  - ot_ratio        : OT hours / regular hours
  - ot_pay_rate     : $ per OT hour
  - total_comp_share: OT as share of total compensation

Outputs:
  - Flagged anomaly DataFrame
  - Agency-level anomaly rate summary
  - Plotly figures for dashboard

Run:
    python src/ot_anomaly.py
"""

import duckdb
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DB_PATH         = "data/payroll.duckdb"
MODEL_PATH      = "models/ot_isolation_forest.joblib"
CONTAMINATION   = 0.05   # Flag top 5% as anomalous

OT_FEATURES = [
    "ot_hours",
    "total_ot_paid",
    "ot_ratio",
    "ot_pay_rate",
]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_ot_data(db_path: str = DB_PATH, fiscal_year: int = None) -> pd.DataFrame:
    """Load employees who worked OT hours."""
    con = duckdb.connect(db_path, read_only=True)
    fy_clause = f"AND fiscal_year = {fiscal_year}" if fiscal_year else ""
    df = con.execute(f"""
        SELECT
            fiscal_year,
            agency_name,
            last_name,
            first_name,
            title_description,
            work_location_borough,
            regular_hours,
            regular_gross_paid,
            ot_hours,
            total_ot_paid,
            ot_ratio,
            ot_pay_rate,
            total_ot_paid / NULLIF(total_ot_paid + regular_gross_paid, 0) AS ot_income_share
        FROM payroll
        WHERE ot_hours > 0
          AND total_ot_paid > 0
          {fy_clause}
    """).df()
    con.close()
    return df


# ── Model ─────────────────────────────────────────────────────────────────────

def train_isolation_forest(df: pd.DataFrame) -> tuple:
    """Fit Isolation Forest on OT features. Returns model, scaler, anomaly df."""
    X = df[OT_FEATURES].copy().fillna(0)

    # Winsorise extreme outliers before scaling (don't let one outlier dominate)
    for col in OT_FEATURES:
        p99 = X[col].quantile(0.99)
        X[col] = X[col].clip(upper=p99)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    df = df.copy()
    df["anomaly_score"] = model.score_samples(X_scaled)    # more negative = more anomalous
    df["is_anomaly"]    = model.predict(X_scaled) == -1    # True = anomalous

    return model, scaler, df


def save_ot_model(model, scaler, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)
    print(f"✅ OT model saved → {path}")


def load_ot_model(path: str = MODEL_PATH) -> tuple:
    payload = joblib.load(path)
    return payload["model"], payload["scaler"]


def score_new(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """Score a new DataFrame using a saved model."""
    X = df[OT_FEATURES].copy().fillna(0)
    X_scaled = scaler.transform(X)
    df = df.copy()
    df["anomaly_score"] = model.score_samples(X_scaled)
    df["is_anomaly"]    = model.predict(X_scaled) == -1
    return df


# ── Agency Anomaly Rate ───────────────────────────────────────────────────────

def agency_anomaly_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise anomaly rates by agency.
    Agencies with high anomaly rates may warrant deeper review.
    """
    return (
        df.groupby("agency_name")
        .agg(
            ot_employees=("is_anomaly", "count"),
            anomalies=("is_anomaly", "sum"),
            anomaly_rate=("is_anomaly", "mean"),
            avg_ot_hours=("ot_hours", "mean"),
            avg_ot_paid=("total_ot_paid", "mean"),
            avg_anomaly_score=("anomaly_score", "mean"),
        )
        .reset_index()
        .sort_values("anomaly_rate", ascending=False)
    )


# ── Plotly Figures ────────────────────────────────────────────────────────────

def fig_anomaly_scatter(df: pd.DataFrame, agency_filter: str = None) -> go.Figure:
    """Scatter of OT hours vs OT paid, colored by anomaly status."""
    subset = df.copy()
    if agency_filter:
        subset = subset[subset["agency_name"].str.contains(agency_filter, na=False)]

    subset["label"] = subset["is_anomaly"].map({True: "⚠️ Anomaly", False: "Normal"})

    fig = px.scatter(
        subset,
        x="ot_hours",
        y="total_ot_paid",
        color="label",
        color_discrete_map={"⚠️ Anomaly": "#EF553B", "Normal": "#636EFA"},
        opacity=0.6,
        size_max=8,
        title=f"OT Hours vs OT Pay {'— ' + agency_filter if agency_filter else '(All Agencies)'}",
        labels={"ot_hours": "OT Hours", "total_ot_paid": "Total OT Paid ($)"},
        hover_data=["agency_name", "title_description", "ot_ratio"],
        template="plotly_dark",
    )
    return fig


def fig_anomaly_rate_by_agency(agency_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Bar chart of anomaly rate by agency."""
    top = agency_df[agency_df["ot_employees"] >= 10].head(top_n)   # min 10 OT workers
    fig = px.bar(
        top,
        x="anomaly_rate",
        y="agency_name",
        orientation="h",
        color="anomaly_rate",
        color_continuous_scale="Oranges",
        title=f"Top {top_n} Agencies by OT Anomaly Rate",
        labels={"anomaly_rate": "Anomaly Rate", "agency_name": "Agency"},
        template="plotly_dark",
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


def fig_anomaly_score_distribution(df: pd.DataFrame, agency_filter: str = None) -> go.Figure:
    """Histogram of anomaly scores (more negative = more anomalous)."""
    subset = df.copy()
    if agency_filter:
        subset = subset[subset["agency_name"].str.contains(agency_filter, na=False)]

    fig = px.histogram(
        subset,
        x="anomaly_score",
        color="is_anomaly",
        color_discrete_map={True: "#EF553B", False: "#636EFA"},
        nbins=50,
        title="Anomaly Score Distribution",
        labels={"anomaly_score": "Isolation Forest Score (lower = more anomalous)", "is_anomaly": "Anomaly"},
        template="plotly_dark",
    )
    return fig


def fig_ot_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of average OT hours by agency and fiscal year."""
    pivot = (
        df.groupby(["fiscal_year", "agency_name"])["ot_hours"]
        .mean()
        .reset_index()
        .pivot(index="agency_name", columns="fiscal_year", values="ot_hours")
    )
    # Keep only agencies with data in multiple years and top by OT
    pivot = pivot.dropna(thresh=3)
    pivot = pivot.loc[pivot.mean(axis=1).nlargest(25).index]

    fig = px.imshow(
        pivot,
        color_continuous_scale="YlOrRd",
        title="Average OT Hours by Agency & Fiscal Year",
        labels={"color": "Avg OT Hours"},
        aspect="auto",
        template="plotly_dark",
    )
    return fig


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading OT data…")
    df = load_ot_data()
    print(f"Employees with OT: {len(df):,}")

    print("Training Isolation Forest…")
    model, scaler, df_scored = train_isolation_forest(df)
    save_ot_model(model, scaler)

    print(f"\nAnomalies detected: {df_scored['is_anomaly'].sum():,} "
          f"({df_scored['is_anomaly'].mean()*100:.1f}%)")

    print("\n── Agency Anomaly Rates (Top 10) ────────────────────────────────────")
    agg = agency_anomaly_rate(df_scored)
    print(agg.head(10)[["agency_name", "ot_employees", "anomaly_rate", "avg_ot_hours"]].to_string(index=False))

    print("\n── Top Anomalous Employees ───────────────────────────────────────────")
    top_anomalies = (
        df_scored[df_scored["is_anomaly"]]
        .nsmallest(10, "anomaly_score")
        [["agency_name", "title_description", "ot_hours", "total_ot_paid", "anomaly_score"]]
    )
    print(top_anomalies.to_string(index=False))
