"""
pay_equity.py
────────────────────────────────────────────────────────────────────────────
Statistical pay equity analysis across NYC agencies, boroughs, and titles.

Outputs:
  - ANOVA + Tukey HSD results (agency-level salary variance)
  - OT burden analysis (which agencies rely most on overtime)
  - SBS-specific breakdown
  - Plotly figures (embedded in Streamlit dashboard)

Run standalone:
    python src/pay_equity.py
"""

import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

DB_PATH = "data/payroll.duckdb"


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_payroll(db_path: str = DB_PATH, fiscal_year: int = None) -> pd.DataFrame:
    """Load payroll data from DuckDB, optionally filtered to one fiscal year."""
    con = duckdb.connect(db_path, read_only=True)
    query = "SELECT * FROM payroll"
    if fiscal_year:
        query += f" WHERE fiscal_year = {fiscal_year}"
    df = con.execute(query).df()
    con.close()
    return df


def get_agency_annual(db_path: str = DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("SELECT * FROM agency_annual ORDER BY fiscal_year, agency_name").df()
    con.close()
    return df


# ── Agency-Level ANOVA ────────────────────────────────────────────────────────

def anova_agency_salary(df: pd.DataFrame, top_n: int = 20) -> dict:
    """
    One-way ANOVA: does mean salary differ significantly across agencies?
    Focuses on top N agencies by headcount for interpretability.
    """
    latest_fy = df["fiscal_year"].max()
    df_fy = df[df["fiscal_year"] == latest_fy].copy()

    # Top N agencies by headcount
    top_agencies = (
        df_fy.groupby("agency_name")["regular_gross_paid"]
        .count()
        .nlargest(top_n)
        .index.tolist()
    )
    df_top = df_fy[df_fy["agency_name"].isin(top_agencies)]

    groups = [
        grp["regular_gross_paid"].dropna().values
        for _, grp in df_top.groupby("agency_name")
    ]

    f_stat, p_value = stats.f_oneway(*groups)

    result = {
        "fiscal_year": latest_fy,
        "top_n_agencies": top_n,
        "f_statistic": round(f_stat, 4),
        "p_value": p_value,
        "significant": p_value < 0.05,
    }

    # Tukey HSD post-hoc
    tukey = pairwise_tukeyhsd(
        endog=df_top["regular_gross_paid"].dropna(),
        groups=df_top.loc[df_top["regular_gross_paid"].notna(), "agency_name"],
        alpha=0.05,
    )
    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0],
    )
    result["tukey_df"] = tukey_df

    return result


# ── Borough Pay Gap ───────────────────────────────────────────────────────────

def borough_pay_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Mean salary by borough, annual basis only (cleanest comparison)."""
    return (
        df[df["pay_basis"] == "PER ANNUM"]
        .groupby(["fiscal_year", "work_location_borough"])
        .agg(
            avg_salary=("regular_gross_paid", "mean"),
            median_salary=("regular_gross_paid", "median"),
            headcount=("regular_gross_paid", "count"),
        )
        .reset_index()
        .sort_values(["fiscal_year", "work_location_borough"])
    )


# ── OT Burden Analysis ────────────────────────────────────────────────────────

def ot_burden(df: pd.DataFrame, fiscal_year: int = None) -> pd.DataFrame:
    """
    OT burden by agency: agencies where overtime is a large share of total pay.
    High OT burden often signals understaffing.
    """
    fy = fiscal_year or df["fiscal_year"].max()
    df_fy = df[df["fiscal_year"] == fy].copy()

    agg = (
        df_fy.groupby("agency_name")
        .agg(
            headcount=("regular_gross_paid", "count"),
            total_gross=("regular_gross_paid", "sum"),
            total_ot=("total_ot_paid", "sum"),
            avg_ot_ratio=("ot_ratio", "mean"),
            pct_employees_with_ot=(
                "ot_hours",
                lambda x: (x > 0).mean() * 100,
            ),
        )
        .reset_index()
    )
    agg["ot_share_of_total"] = agg["total_ot"] / (agg["total_gross"] + agg["total_ot"])
    return agg.sort_values("ot_share_of_total", ascending=False)


# ── SBS Deep Dive ─────────────────────────────────────────────────────────────

def sbs_analysis(df: pd.DataFrame) -> dict:
    """Focused breakdown of SBS payroll data across all fiscal years."""
    sbs = df[df["agency_name"].str.contains("SMALL BUSINESS", na=False)].copy()

    if sbs.empty:
        return {"error": "No SBS records found. Check agency name in dataset."}

    # Year-over-year trends
    yoy = (
        sbs.groupby("fiscal_year")
        .agg(
            headcount=("regular_gross_paid", "count"),
            avg_salary=("regular_gross_paid", "mean"),
            median_salary=("regular_gross_paid", "median"),
            total_spend=("total_compensation", "sum"),
            total_ot=("total_ot_paid", "sum"),
            avg_ot_ratio=("ot_ratio", "mean"),
        )
        .reset_index()
    )
    yoy["spend_growth_pct"] = yoy["total_spend"].pct_change() * 100

    # Title distribution (latest FY)
    latest_fy = sbs["fiscal_year"].max()
    title_dist = (
        sbs[sbs["fiscal_year"] == latest_fy]
        .groupby("title_description")
        .agg(
            count=("regular_gross_paid", "count"),
            avg_salary=("regular_gross_paid", "mean"),
            avg_ot=("total_ot_paid", "mean"),
        )
        .sort_values("count", ascending=False)
        .head(20)
        .reset_index()
    )

    # Leave status breakdown
    leave_dist = (
        sbs[sbs["fiscal_year"] == latest_fy]["leave_status_as_of_june_30"]
        .value_counts(normalize=True)
        .mul(100)
        .reset_index()
    )
    leave_dist.columns = ["status", "pct"]

    return {
        "yoy_trends": yoy,
        "title_distribution": title_dist,
        "leave_distribution": leave_dist,
        "latest_fiscal_year": latest_fy,
        "total_employees_latest": int(sbs[sbs["fiscal_year"] == latest_fy].shape[0]),
    }


# ── Plotly Figures ────────────────────────────────────────────────────────────

def fig_salary_distribution_by_agency(df: pd.DataFrame, agencies: list[str]) -> go.Figure:
    """Box plot of salary distributions for selected agencies."""
    fy = df["fiscal_year"].max()
    subset = df[(df["fiscal_year"] == fy) & (df["agency_name"].isin(agencies))]
    fig = px.box(
        subset,
        x="agency_name",
        y="regular_gross_paid",
        color="agency_name",
        title=f"Salary Distribution by Agency (FY {fy})",
        labels={"regular_gross_paid": "Regular Gross Paid ($)", "agency_name": "Agency"},
        template="plotly_dark",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-35)
    return fig


def fig_ot_burden_bar(ot_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of OT share by agency."""
    top = ot_df.nlargest(top_n, "ot_share_of_total")
    fig = px.bar(
        top,
        x="ot_share_of_total",
        y="agency_name",
        orientation="h",
        color="ot_share_of_total",
        color_continuous_scale="Reds",
        title=f"Top {top_n} Agencies by OT Share of Total Pay",
        labels={"ot_share_of_total": "OT Share", "agency_name": "Agency"},
        template="plotly_dark",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    fig.update_xaxes(tickformat=".0%")
    return fig


def fig_borough_pay_trend(borough_df: pd.DataFrame) -> go.Figure:
    """Line chart of average salary by borough over time."""
    fig = px.line(
        borough_df,
        x="fiscal_year",
        y="avg_salary",
        color="work_location_borough",
        markers=True,
        title="Average Salary by Borough Over Time",
        labels={"avg_salary": "Avg Salary ($)", "fiscal_year": "Fiscal Year",
                "work_location_borough": "Borough"},
        template="plotly_dark",
    )
    return fig


def fig_sbs_yoy(yoy_df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    """SBS headcount and spend over time."""
    fig_hc = px.bar(
        yoy_df, x="fiscal_year", y="headcount",
        title="SBS Headcount by Fiscal Year",
        labels={"headcount": "Employees", "fiscal_year": "FY"},
        template="plotly_dark", color_discrete_sequence=["#00C9A7"],
    )
    fig_spend = px.line(
        yoy_df, x="fiscal_year", y="total_spend", markers=True,
        title="SBS Total Payroll Spend by Fiscal Year",
        labels={"total_spend": "Total Spend ($)", "fiscal_year": "FY"},
        template="plotly_dark",
    )
    return fig_hc, fig_spend


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data…")
    df = load_payroll()

    print("\n── ANOVA: Agency Salary Differences ─────────────────────────────────")
    anova = anova_agency_salary(df)
    print(f"F-statistic: {anova['f_statistic']} | p-value: {anova['p_value']:.2e}")
    print(f"Statistically significant: {anova['significant']}")

    print("\n── OT Burden (Top 10) ───────────────────────────────────────────────")
    ot = ot_burden(df)
    print(ot[["agency_name", "headcount", "ot_share_of_total", "avg_ot_ratio"]].head(10).to_string(index=False))

    print("\n── SBS Analysis ──────────────────────────────────────────────────────")
    sbs = sbs_analysis(df)
    if "error" not in sbs:
        print(sbs["yoy_trends"].to_string(index=False))
    else:
        print(sbs["error"])
