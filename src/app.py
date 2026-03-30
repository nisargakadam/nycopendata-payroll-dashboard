"""
app.py — NYC Payroll Intelligence Dashboard
────────────────────────────────────────────────────────────────────────────
Main Streamlit application. Requires:
  - data/payroll.duckdb          (run data_pipeline.py first)
  - models/salary_rf.joblib      (run salary_model.py first)
  - models/ot_isolation_forest.joblib  (run ot_anomaly.py first)
  - models/budget_lstm.pt        (run budget_forecast.py first)

Launch:
    streamlit run src/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # makes src/ importable

import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Payroll Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH   = "data/payroll.duckdb"
RF_PATH   = "models/salary_rf.joblib"
IF_PATH   = "models/ot_isolation_forest.joblib"
LSTM_PATH = "models/budget_lstm.pt"

DEFAULT_AGENCY = "DEPARTMENT OF FINANCE"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
    }
    .metric-val   { font-size: 2rem; font-weight: 700; color: #cba6f7; }
    .metric-label { font-size: 0.85rem; color: #a6adc8; margin-top: 4px; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #cdd6f4;
        border-bottom: 2px solid #313244;
        padding-bottom: 6px;
        margin: 20px 0 12px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Data Helpers ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to DuckDB...")
def get_con():
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found at `{DB_PATH}`. Please run `python src/data_pipeline.py` first.")
        st.stop()
    return duckdb.connect(DB_PATH, read_only=True)


@st.cache_data(show_spinner="Loading payroll data...")
def load_payroll(fiscal_year=None):
    con = get_con()
    fy_clause = f"WHERE fiscal_year = {fiscal_year}" if fiscal_year else ""
    return con.execute(f"SELECT * FROM payroll {fy_clause}").df()


@st.cache_data(show_spinner=False)
def load_agency_annual():
    con = get_con()
    return con.execute("SELECT * FROM agency_annual ORDER BY fiscal_year, agency_name").df()


@st.cache_data(show_spinner=False)
def get_fiscal_years():
    con = get_con()
    return sorted(
        con.execute("SELECT DISTINCT fiscal_year FROM payroll ORDER BY 1")
        .df()["fiscal_year"].tolist()
    )


@st.cache_data(show_spinner=False)
def get_agencies():
    con = get_con()
    return sorted(
        con.execute("SELECT DISTINCT agency_name FROM payroll ORDER BY 1")
        .df()["agency_name"].tolist()
    )


@st.cache_resource(show_spinner="Loading ML models...")
def load_rf_model():
    if not os.path.exists(RF_PATH):
        return None, None, None
    payload = joblib.load(RF_PATH)
    return payload["model"], payload["encoders"], payload["metrics"]


@st.cache_resource(show_spinner=False)
def load_if_model():
    if not os.path.exists(IF_PATH):
        return None, None
    payload = joblib.load(IF_PATH)
    return payload["model"], payload["scaler"]


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(fiscal_years, agencies):
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/NYC_Seal.svg/120px-NYC_Seal.svg.png",
        width=80,
    )
    st.sidebar.title("NYC Payroll Intelligence")
    st.sidebar.markdown("*Pay equity · OT anomalies · Budget forecasting*")
    st.sidebar.divider()

    selected_fy = st.sidebar.selectbox(
        "Fiscal Year", options=fiscal_years, index=len(fiscal_years) - 1
    )

    default_idx = next(
        (i + 1 for i, a in enumerate(agencies) if DEFAULT_AGENCY in a), 0
    )
    selected_agency = st.sidebar.selectbox(
        "Agency Focus", options=["All Agencies"] + agencies, index=default_idx
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Dataset:** [NYC OpenData Citywide Payroll](https://data.cityofnewyork.us/resource/k397-673e.json)"
    )
    st.sidebar.markdown("**Built by:** Nisarga | Data Science Portfolio")

    return selected_fy, selected_agency


# ── Tab 1: Overview ───────────────────────────────────────────────────────────

def tab_overview(df_fy: pd.DataFrame, df_all: pd.DataFrame, agency_annual: pd.DataFrame, fy: int):
    st.markdown(f"<div class='section-header'>Citywide Payroll Overview — FY {fy}</div>", unsafe_allow_html=True)

    total_headcount = len(df_fy)
    total_spend     = df_fy["total_compensation"].sum()
    avg_salary      = df_fy["regular_gross_paid"].mean()
    total_ot        = df_fy["total_ot_paid"].sum()
    pct_ot_workers  = (df_fy["ot_hours"] > 0).mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in zip(
        [c1, c2, c3, c4, c5],
        [f"{total_headcount:,}", f"${total_spend/1e9:.2f}B", f"${avg_salary:,.0f}",
         f"${total_ot/1e6:.1f}M", f"{pct_ot_workers:.1f}%"],
        ["Total Employees", "Total Payroll Spend", "Avg Gross Salary",
         "Total OT Paid", "% Employees w/ OT"],
    ):
        col.markdown(
            f"<div class='metric-card'><div class='metric-val'>{val}</div>"
            f"<div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='section-header'>Top 15 Agencies by Total Spend</div>", unsafe_allow_html=True)
        top_agencies = (
            df_fy.groupby("agency_name")["total_compensation"].sum()
            .nlargest(15).reset_index()
        )
        fig = px.bar(
            top_agencies, x="total_compensation", y="agency_name",
            orientation="h", color="total_compensation",
            color_continuous_scale="Blues",
            labels={"total_compensation": "Total Spend ($)", "agency_name": "Agency"},
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis={"categoryorder": "total ascending"}, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-header'>Citywide Payroll Spend Over Time</div>", unsafe_allow_html=True)
        spend_yoy = agency_annual.groupby("fiscal_year")["total_spend"].sum().reset_index()
        fig2 = px.line(
            spend_yoy, x="fiscal_year", y="total_spend", markers=True,
            labels={"total_spend": "Total Spend ($)", "fiscal_year": "Fiscal Year"},
            template="plotly_dark",
        )
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>Salary Distribution by Pay Basis</div>", unsafe_allow_html=True)
    fig3 = px.histogram(
        df_fy[df_fy["regular_gross_paid"] < 300_000],
        x="regular_gross_paid", color="pay_basis",
        nbins=80, barmode="overlay", opacity=0.7,
        labels={"regular_gross_paid": "Regular Gross Paid ($)", "pay_basis": "Pay Basis"},
        template="plotly_dark",
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Tab 2: Agency Deep Dive ───────────────────────────────────────────────────

def tab_agency(df_all: pd.DataFrame, agency_annual: pd.DataFrame, selected_agency: str):
    agency = selected_agency if selected_agency != "All Agencies" else DEFAULT_AGENCY
    st.markdown(f"<div class='section-header'>Agency Deep Dive — {agency}</div>", unsafe_allow_html=True)

    safe = agency.replace("(", "").replace(")", "")
    agency_df = df_all[df_all["agency_name"].str.contains(safe, na=False, regex=False)]
    agency_aa = agency_annual[agency_annual["agency_name"].str.contains(safe, na=False, regex=False)]

    if agency_df.empty:
        st.warning(f"No records found for: {agency}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>Headcount Over Time</div>", unsafe_allow_html=True)
        fig = px.bar(agency_aa, x="fiscal_year", y="headcount",
                     template="plotly_dark", color_discrete_sequence=["#cba6f7"],
                     labels={"headcount": "Employees", "fiscal_year": "FY"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Average Salary Over Time</div>", unsafe_allow_html=True)
        fig2 = px.line(agency_aa, x="fiscal_year", y="avg_gross", markers=True,
                       template="plotly_dark", color_discrete_sequence=["#89dceb"],
                       labels={"avg_gross": "Avg Gross ($)", "fiscal_year": "FY"})
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-header'>OT Ratio Over Time</div>", unsafe_allow_html=True)
        fig3 = px.line(agency_aa, x="fiscal_year", y="avg_ot_ratio", markers=True,
                       template="plotly_dark", color_discrete_sequence=["#fab387"],
                       labels={"avg_ot_ratio": "Avg OT Ratio (OT hrs / Regular hrs)", "fiscal_year": "FY"})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-header'>Total Payroll Spend Over Time</div>", unsafe_allow_html=True)
        fig4 = px.area(agency_aa, x="fiscal_year", y="total_spend",
                       template="plotly_dark", color_discrete_sequence=["#a6e3a1"],
                       labels={"total_spend": "Total Spend ($)", "fiscal_year": "FY"})
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class='section-header'>Job Title Distribution (Latest FY)</div>", unsafe_allow_html=True)
    latest_fy  = agency_df["fiscal_year"].max()
    title_dist = (
        agency_df[agency_df["fiscal_year"] == latest_fy]
        .groupby("title_description")
        .agg(count=("regular_gross_paid", "count"), avg_salary=("regular_gross_paid", "mean"))
        .nlargest(20, "count").reset_index()
    )
    fig5 = px.bar(
        title_dist, x="count", y="title_description", orientation="h",
        color="avg_salary", color_continuous_scale="Viridis",
        labels={"count": "Employees", "title_description": "Job Title", "avg_salary": "Avg Salary ($)"},
        template="plotly_dark",
    )
    fig5.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig5, use_container_width=True)

    with st.expander("View Employee-Level Data (Latest FY)"):
        show_cols = ["last_name", "first_name", "title_description", "work_location_borough",
                     "base_salary", "regular_gross_paid", "ot_hours", "total_ot_paid",
                     "leave_status_as_of_june_30"]
        st.dataframe(
            agency_df[agency_df["fiscal_year"] == latest_fy][show_cols]
            .sort_values("regular_gross_paid", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )


# ── Tab 3: Pay Equity ─────────────────────────────────────────────────────────

def tab_pay_equity(df_fy: pd.DataFrame, fy: int):
    from scipy import stats

    st.markdown(f"<div class='section-header'>Pay Equity Analysis — FY {fy}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>Average Salary by Borough</div>", unsafe_allow_html=True)
        borough_df = (
            df_fy[df_fy["pay_basis"] == "PER ANNUM"]
            .groupby("work_location_borough")["regular_gross_paid"]
            .agg(mean="mean", median="median", count="count")
            .reset_index()
        )
        fig = px.bar(
            borough_df, x="work_location_borough", y="mean",
            color="mean", color_continuous_scale="Blues",
            labels={"mean": "Mean Salary ($)", "work_location_borough": "Borough"},
            template="plotly_dark",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Salary Box Plot by Borough</div>", unsafe_allow_html=True)
        filtered = df_fy[
            (df_fy["pay_basis"] == "PER ANNUM") &
            (df_fy["regular_gross_paid"] < 300_000) &
            df_fy["work_location_borough"].isin(
                ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]
            )
        ]
        fig2 = px.box(
            filtered, x="work_location_borough", y="regular_gross_paid",
            color="work_location_borough",
            labels={"regular_gross_paid": "Gross Paid ($)", "work_location_borough": "Borough"},
            template="plotly_dark",
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>Top 25 Agencies — Salary Spread</div>", unsafe_allow_html=True)
    top_agencies = (
        df_fy.groupby("agency_name")["regular_gross_paid"]
        .count().nlargest(25).index.tolist()
    )
    filtered_top = df_fy[
        df_fy["agency_name"].isin(top_agencies) & (df_fy["regular_gross_paid"] < 300_000)
    ]
    fig3 = px.box(
        filtered_top, x="agency_name", y="regular_gross_paid",
        color="agency_name",
        labels={"regular_gross_paid": "Gross Paid ($)", "agency_name": "Agency"},
        template="plotly_dark",
    )
    fig3.update_layout(showlegend=False, xaxis_tickangle=-40, height=500)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>ANOVA: Are Agency Salaries Significantly Different?</div>", unsafe_allow_html=True)
    groups = [
        grp["regular_gross_paid"].dropna().values
        for _, grp in filtered_top.groupby("agency_name")
        if len(grp) > 30
    ]
    if groups:
        f_stat, p_val = stats.f_oneway(*groups)
        sig = "Yes — statistically significant" if p_val < 0.05 else "Not significant"
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("F-Statistic", f"{f_stat:,.2f}")
        col_b.metric("p-value", f"{p_val:.2e}")
        col_c.metric("Significant (alpha = 0.05)?", sig)
    else:
        st.warning("Not enough data for ANOVA.")


# ── Tab 4: OT Anomaly ─────────────────────────────────────────────────────────

def tab_ot_anomaly(df_fy: pd.DataFrame, fy: int):
    from ot_anomaly import (
        train_isolation_forest, agency_anomaly_rate,
        fig_anomaly_scatter, fig_anomaly_rate_by_agency,
    )

    st.markdown(f"<div class='section-header'>Overtime Anomaly Detection — FY {fy}</div>", unsafe_allow_html=True)

    ot_df = df_fy[(df_fy["ot_hours"] > 0) & (df_fy["total_ot_paid"] > 0)].copy()
    if ot_df.empty:
        st.warning("No OT data found for this fiscal year.")
        return

    OT_FEATURES = ["ot_hours", "total_ot_paid", "ot_ratio", "ot_pay_rate"]
    if_model, scaler = load_if_model()

    if if_model is not None:
        X = ot_df[OT_FEATURES].fillna(0).copy()
        for col in OT_FEATURES:
            X[col] = X[col].clip(upper=X[col].quantile(0.99))
        X_sc = scaler.transform(X)
        ot_df["anomaly_score"] = if_model.score_samples(X_sc)
        ot_df["is_anomaly"]    = if_model.predict(X_sc) == -1
    else:
        st.info("OT model not found — training on the fly (this may take a moment)...")
        _, _, ot_df = train_isolation_forest(ot_df)

    n_anomalies  = ot_df["is_anomaly"].sum()
    anomaly_rate = ot_df["is_anomaly"].mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total OT Workers",  f"{len(ot_df):,}")
    col2.metric("Anomalies Flagged", f"{n_anomalies:,}")
    col3.metric("Anomaly Rate",      f"{anomaly_rate:.1f}%")

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(fig_anomaly_scatter(ot_df), use_container_width=True)
    with col_r:
        agency_agg = agency_anomaly_rate(ot_df)
        st.plotly_chart(fig_anomaly_rate_by_agency(agency_agg), use_container_width=True)

    st.markdown("<div class='section-header'>Top Flagged Employees</div>", unsafe_allow_html=True)
    top_anom = (
        ot_df[ot_df["is_anomaly"]]
        .nsmallest(50, "anomaly_score")
        [["agency_name", "title_description", "ot_hours", "total_ot_paid",
          "ot_ratio", "ot_pay_rate", "anomaly_score"]]
        .reset_index(drop=True)
    )
    st.dataframe(top_anom.style.format({
        "total_ot_paid": "${:,.0f}",
        "ot_ratio":      "{:.2%}",
        "ot_pay_rate":   "${:.2f}",
        "anomaly_score": "{:.4f}",
    }), use_container_width=True)


# ── Tab 5: Budget Forecast ────────────────────────────────────────────────────

def tab_forecast(df_all: pd.DataFrame, selected_agency: str):
    from budget_forecast import (
        build_sequences, forecast_agencies,
        fig_forecast_bar, fig_agency_trend_with_forecast,
        load_model as load_lstm,
    )

    st.markdown("<div class='section-header'>LSTM Budget Forecasting</div>", unsafe_allow_html=True)

    if not os.path.exists(LSTM_PATH):
        st.warning("LSTM model not found. Please run `python src/budget_forecast.py` to train it first.")
        return

    try:
        model = load_lstm(LSTM_PATH)
    except Exception as e:
        st.error(f"Could not load LSTM: {e}")
        return

    df_long = (
        df_all.groupby(["fiscal_year", "agency_name"])
        .apply(lambda x: (
            x["regular_gross_paid"].sum()
            + x["total_ot_paid"].sum()
            + x["total_other_pay"].sum()
        ))
        .reset_index(name="total_spend")
    )

    X, y, meta, pivot, mins, maxes = build_sequences(df_long)
    forecast_df = forecast_agencies(model, pivot, mins, maxes)

    fin_row = forecast_df[forecast_df["agency_name"].str.contains("FINANCE", na=False)]
    if not fin_row.empty:
        fin_spend  = fin_row["forecast_spend"].values[0]
        fin_actual = fin_row["last_year_spend"].values[0]
        fin_chg    = fin_row["yoy_change_pct"].values[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Dept of Finance — Last FY Spend",    f"${fin_actual/1e6:.2f}M")
        col2.metric("Dept of Finance — Forecast Spend",   f"${fin_spend/1e6:.2f}M")
        col3.metric("Projected YoY Change", f"{fin_chg:+.1f}%", delta=f"{fin_chg:+.1f}%")

    st.plotly_chart(fig_forecast_bar(forecast_df, top_n=20), use_container_width=True)

    st.markdown("<div class='section-header'>Agency Trend and Forecast</div>", unsafe_allow_html=True)
    available   = forecast_df["agency_name"].tolist()
    default_idx = next((i for i, a in enumerate(available) if "FINANCE" in a), 0)
    pick = st.selectbox("Select Agency", options=available, index=default_idx)
    fig  = fig_agency_trend_with_forecast(pick, pivot, forecast_df)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full Forecast Table"):
        st.dataframe(
            forecast_df[["agency_name", "last_year_spend", "forecast_spend", "yoy_change_pct"]]
            .style.format({
                "last_year_spend": "${:,.0f}",
                "forecast_spend":  "${:,.0f}",
                "yoy_change_pct":  "{:+.1f}%",
            }),
            use_container_width=True,
        )


# ── Tab 6: Salary Model ───────────────────────────────────────────────────────

def tab_salary_model(df_all: pd.DataFrame, selected_agency: str):
    from salary_model import flag_equity_outliers, fig_equity_flags

    st.markdown("<div class='section-header'>Salary Prediction and Equity Flagging</div>", unsafe_allow_html=True)

    rf_model, encoders, metrics = load_rf_model()
    if rf_model is None:
        st.warning("Salary model not found. Please run `python src/salary_model.py` first.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Model R2",       f"{metrics.get('R²', 'N/A')}")
    col2.metric("Mean Abs Error", f"${metrics.get('MAE', 0):,.0f}")
    col3.metric("MAPE",           f"{metrics.get('MAPE', 'N/A')}%")

    agency_filter = None if selected_agency == "All Agencies" else selected_agency
    df_annual     = df_all[df_all["pay_basis"] == "PER ANNUM"].copy()

    try:
        flagged = flag_equity_outliers(df_annual, rf_model, encoders, agency_filter=agency_filter)

        if flagged.empty:
            st.warning(f"No annual-basis employees found for: {agency_filter}")
            return

        summary = flagged["equity_flag"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Fair",      int(summary.get("FAIR",      0)))
        c2.metric("Underpaid", int(summary.get("UNDERPAID", 0)))
        c3.metric("Overpaid",  int(summary.get("OVERPAID",  0)))

        label = agency_filter or "All Agencies"
        st.plotly_chart(fig_equity_flags(flagged, label), use_container_width=True)

        with st.expander("View Flagged Employees"):
            display = (
                flagged[flagged["equity_flag"] != "FAIR"][[
                    "agency_name", "title_description", "work_location_borough",
                    "regular_gross_paid", "predicted_salary", "gap_pct", "equity_flag",
                ]]
                .sort_values("gap_pct")
                .reset_index(drop=True)
            )
            st.dataframe(display.style.format({
                "regular_gross_paid": "${:,.0f}",
                "predicted_salary":   "${:,.0f}",
                "gap_pct":            "{:+.1%}",
            }), use_container_width=True)

    except Exception as e:
        st.error(f"Error running salary model: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    fiscal_years  = get_fiscal_years()
    agencies      = get_agencies()
    selected_fy, selected_agency = render_sidebar(fiscal_years, agencies)

    df_all        = load_payroll()
    df_fy         = df_all[df_all["fiscal_year"] == selected_fy]
    agency_annual = load_agency_annual()

    st.title("NYC Payroll Intelligence Dashboard")
    st.caption(
        f"Citywide payroll analysis across {len(df_all):,} records | "
        f"Agency: **{selected_agency}** | FY **{selected_fy}**"
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Agency Deep Dive",
        "Pay Equity",
        "OT Anomalies",
        "Budget Forecast",
        "Salary Model",
    ])

    with tab1:
        tab_overview(df_fy, df_all, agency_annual, selected_fy)
    with tab2:
        tab_agency(df_all, agency_annual, selected_agency)
    with tab3:
        tab_pay_equity(df_fy, selected_fy)
    with tab4:
        tab_ot_anomaly(df_fy, selected_fy)
    with tab5:
        tab_forecast(df_all, selected_agency)
    with tab6:
        tab_salary_model(df_all, selected_agency)


if __name__ == "__main__":
    main()
