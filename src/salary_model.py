"""
salary_model.py
────────────────────────────────────────────────────────────────────────────
Random Forest salary prediction model.

Trains on: agency, title, borough, pay_basis, tenure, fiscal year
Predicts:  regular_gross_paid

Equity flags employees where actual pay diverges from model prediction
by more than FLAG_THRESHOLD, surfacing potential pay inequities.

Outputs:
  - Trained model saved to models/salary_rf.joblib
  - Feature importance plot
  - Flagged employee DataFrame (over/underpaid outliers)

Run:
    python src/salary_model.py
"""

import os
import duckdb
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

DB_PATH        = "data/payroll.duckdb"
MODEL_PATH     = "models/salary_rf.joblib"
FLAG_THRESHOLD = 0.20    # Flag if |actual - predicted| / predicted > 20%

FEATURES = [
    "agency_name",
    "title_description",
    "work_location_borough",
    "pay_basis",
    "fiscal_year",
    "tenure_years",
]
TARGET = "regular_gross_paid"


# ── Data Prep ─────────────────────────────────────────────────────────────────

def load_and_prep(db_path: str = DB_PATH) -> tuple[pd.DataFrame, dict]:
    """Load data from DuckDB and encode categorical features."""
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"""
        SELECT {', '.join(FEATURES)}, {TARGET}
        FROM payroll
        WHERE pay_basis = 'PER ANNUM'
          AND regular_gross_paid BETWEEN 10000 AND 500000
          AND tenure_years IS NOT NULL
          AND tenure_years >= 0
    """).df()
    con.close()

    print(f"Training set: {len(df):,} rows")

    encoders = {}
    for col in ["agency_name", "title_description", "work_location_borough", "pay_basis"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].fillna("UNKNOWN").astype(str))
            encoders[col] = le

    return df, encoders


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric feature matrix for model training."""
    return df[[
        "agency_name_enc",
        "title_description_enc",
        "work_location_borough_enc",
        "pay_basis_enc",
        "fiscal_year",
        "tenure_years",
    ]].fillna(0)


# ── Training ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, encoders: dict) -> tuple:
    """Train Random Forest and evaluate on held-out test set."""
    X = feature_matrix(df)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest… (this may take a few minutes on the full dataset)")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test.clip(lower=1))) * 100

    metrics = {"MAE": round(mae, 2), "R²": round(r2, 4), "MAPE": round(mape, 2)}
    print(f"  MAE : ${mae:,.2f}")
    print(f"  R²  : {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return model, metrics, X_test, y_test, y_pred


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model, encoders: dict, metrics: dict, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "encoders": encoders, "metrics": metrics}, path)
    print(f"✅ Model saved → {path}")


def load_model(path: str = MODEL_PATH) -> tuple:
    payload  = joblib.load(path)
    return payload["model"], payload["encoders"], payload["metrics"]


# ── Equity Flagging ───────────────────────────────────────────────────────────

def flag_equity_outliers(
    df: pd.DataFrame,
    model,
    encoders: dict,
    agency_filter: str = None,
    threshold: float = FLAG_THRESHOLD,
) -> pd.DataFrame:
    """
    Predict expected salary for each employee and flag those where
    actual pay differs from predicted by more than `threshold`.

    Returns a DataFrame with equity_flag column:
      'UNDERPAID'  → actual < predicted * (1 - threshold)
      'OVERPAID'   → actual > predicted * (1 + threshold)
      'FAIR'       → within threshold
    """
    subset = df.copy()
    if agency_filter:
        subset = subset[subset["agency_name"].str.contains(agency_filter, na=False, case=False)]

    if subset.empty:
        print(f"⚠️  flag_equity_outliers: 0 rows matched agency_filter='{agency_filter}'")
        print(f"   Available agency names (sample): {sorted(df['agency_name'].dropna().unique())[:15]}")
        return subset  # return empty df gracefully

    for col in ["agency_name", "title_description", "work_location_borough", "pay_basis"]:
        le = encoders[col]
        known = set(le.classes_)
        subset[col + "_enc"] = subset[col].fillna("UNKNOWN").apply(
            lambda x: le.transform([x])[0] if x in known else -1
        )

    X = feature_matrix(subset)
    subset["predicted_salary"] = model.predict(X)
    subset["salary_gap"]       = subset[TARGET] - subset["predicted_salary"]
    subset["gap_pct"]          = subset["salary_gap"] / subset["predicted_salary"].clip(lower=1)

    subset["equity_flag"] = "FAIR"
    subset.loc[subset["gap_pct"] < -threshold, "equity_flag"] = "UNDERPAID"
    subset.loc[subset["gap_pct"] >  threshold, "equity_flag"] = "OVERPAID"

    return subset.sort_values("gap_pct")


# ── Plotly Figures ────────────────────────────────────────────────────────────

def fig_feature_importance(model, feature_names: list[str]) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    fi = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    fig = px.bar(
        fi, x="importance", y="feature", orientation="h",
        title="Random Forest Feature Importances",
        labels={"importance": "Importance", "feature": "Feature"},
        color="importance", color_continuous_scale="Blues",
        template="plotly_dark",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def fig_actual_vs_predicted(y_test, y_pred, sample_n: int = 3000) -> go.Figure:
    """Scatter of actual vs predicted salary (sampled for speed)."""
    idx = np.random.choice(len(y_test), min(sample_n, len(y_test)), replace=False)
    yt  = np.array(y_test)[idx]
    yp  = np.array(y_pred)[idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yt, y=yp, mode="markers",
        marker=dict(size=3, opacity=0.4, color="#636EFA"),
        name="Employees",
    ))
    max_val = max(yt.max(), yp.max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(color="red", dash="dash"), name="Perfect Fit",
    ))
    fig.update_layout(
        title="Actual vs. Predicted Salary",
        xaxis_title="Actual Gross ($)",
        yaxis_title="Predicted Gross ($)",
        template="plotly_dark",
    )
    return fig


def fig_equity_flags(flagged_df: pd.DataFrame, agency_name: str = "SBS") -> go.Figure:
    """Histogram of salary gaps, colored by equity flag."""
    fig = px.histogram(
        flagged_df,
        x="gap_pct",
        color="equity_flag",
        nbins=60,
        color_discrete_map={"UNDERPAID": "#EF553B", "FAIR": "#00CC96", "OVERPAID": "#636EFA"},
        title=f"Salary Gap Distribution — {agency_name}",
        labels={"gap_pct": "Gap % (Actual - Predicted) / Predicted", "count": "Employees"},
        template="plotly_dark",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Fair")
    fig.update_xaxes(tickformat=".0%")
    return fig


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, encoders = load_and_prep()
    model, metrics, X_test, y_test, y_pred = train(df, encoders)
    save_model(model, encoders, metrics)

    # ── SBS equity flags ───────────────────────────────────────────────────────
    print("\n── SBS Equity Flags ──────────────────────────────────────────────────")
    SBS_AGENCY = "DEPARTMENT OF BUSINESS SERV."
    flagged = flag_equity_outliers(df, model, encoders, agency_filter=SBS_AGENCY)

    if flagged.empty:
        print(f"⚠️  No rows for '{SBS_AGENCY}'. Run data_pipeline.py to check agency names.")
    else:
        summary = flagged["equity_flag"].value_counts()
        print(summary.to_string())
        print(f"\nFlagged {(flagged['equity_flag'] != 'FAIR').sum()} employees for review")
        print("\nTop underpaid employees:")
        print(
            flagged[flagged["equity_flag"] == "UNDERPAID"]
            [["title_description", "regular_gross_paid", "predicted_salary", "gap_pct"]]
            .head(10)
            .to_string(index=False)
        )

    # ── Sample of full dataset for reference (capped to avoid hang) ────────────
    print("\n── Full Dataset Equity Summary (10k sample) ──────────────────────────")
    sample = df.sample(n=min(10_000, len(df)), random_state=42)
    flagged_sample = flag_equity_outliers(sample, model, encoders, agency_filter=None)
    print(flagged_sample["equity_flag"].value_counts().to_string())
    pct = (flagged_sample["equity_flag"] != "FAIR").mean() * 100
    print(f"\nEstimated flagged citywide: ~{pct:.1f}% of annual employees")
