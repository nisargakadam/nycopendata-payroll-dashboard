# NYC Payroll Intelligence Dashboard

> End-to-end data science project analyzing 6.5M+ NYC municipal payroll records across all city agencies and fiscal years (2014–2025).  
> Built with DuckDB · scikit-learn · PyTorch · Streamlit

---

## Overview

This project surfaces actionable insights from NYC's Citywide Payroll Data (NYC OpenData) through statistical analysis, machine learning, and an interactive dashboard. A dedicated focus is placed on the **Department of Finance**, examining headcount trends, salary distributions, overtime patterns, and budget projections.

**Key capabilities:**
- Pay equity analysis across agencies, boroughs, and job titles using ANOVA and statistical testing
- Salary prediction via Random Forest regression to flag potential under/overpayment
- Overtime anomaly detection using Isolation Forest unsupervised ML
- Budget forecasting with a 2-layer PyTorch LSTM trained on 10+ years of agency spend

---

## Dashboard Preview

### Citywide KPI Overview
![Payroll KPI Dashboard](visualizations/payroll_kpi_dashboard.png)
*533,418 employees · $34.56B total payroll · $2.93B overtime paid across all NYC agencies in FY 2025*

---

### Agency Deep Dive — Department of Finance
![Headcount, Average Salary, OT Ratio, and Payroll Spend Over Time](visualizations/headcount_averagesalary_otratio__payroll_over_time.png)
*Year-over-year trends in headcount, average salary, overtime ratio, and total spend for the Dept of Finance (2014–2025)*

![Job Title Distribution (Latest FY)](visualizations/job_title_distribution_latest_FY.png)
*Top 20 job titles by headcount, color-coded by average salary. City Tax Auditor is the most common role at ~320 employees.*

---

### Pay Equity Analysis
![Salary by Borough and Top 25 Agencies](visualizations/salary_by_borough__top_25_agencies.png)
*One-way ANOVA (F=30,809, p≈0) confirms salaries differ significantly across agencies. Box plots reveal wide intra-agency variance.*

![Salary Distribution by Pay Basis](visualizations/salary_distribution_by_pay_basis.png)
*Bimodal distribution in annual salaries with concentrations around $50K and $110K, consistent with civil service step structures.*

---

### Overtime Anomaly Detection
![Overtime Anomaly Detection — FY 2025](visualizations/anomaly_detection_overtime.png)
*Isolation Forest flags 12,373 employees (9.1% of OT workers) as anomalous. Dept of Citywide Admin Svcs has the highest anomaly rate at ~30%.*

---

### LSTM Budget Forecasting
![LSTM Budget Forecasting](visualizations/LSTM_Budget_Forecasting.png)
*Two-layer PyTorch LSTM trained on per-agency payroll sequences (2014–2025). Dept of Finance forecast: $169.54M (-2.1% YoY).*

![Department of Finance Payroll Spend Forecast](visualizations/dept_of_finance_payroll_spend_forecast_.png)
*Historical trend and LSTM forecast for the Dept of Finance, showing spend growth from $130M (2014) to $173M (2024) with a projected modest decline.*

---

### Salary Prediction and Equity Flagging
![Salary Prediction and Equity Flagging](visualizations/salary_prediction_and_equity_flagging.png)
*Random Forest model (R²=0.716, MAE=$10,577) predicts expected salary per employee. Of Dept of Finance annual employees: 4,479 flagged as potentially underpaid, 3,088 as overpaid.*

![Top Agencies by Total Spend and Citywide Payroll Over Time](visualizations/total_spend_by_agency_and_citywide_payroll_overtime.png)
*Dept of Ed Pedagogical and NYPD dominate total spend. Citywide payroll grew from $22B (2014) to $35B (2024).*

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Storage | DuckDB — in-process SQL on 6.5M rows |
| Data Wrangling | pandas, numpy |
| Statistical Analysis | scipy (ANOVA), statsmodels (Tukey HSD) |
| ML — Salary Model | scikit-learn Random Forest Regressor |
| ML — OT Anomalies | scikit-learn Isolation Forest |
| ML — Budget Forecast | PyTorch 2-layer LSTM |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Data Source | [NYC OpenData Citywide Payroll API](https://data.cityofnewyork.us/resource/k397-673e.json) |

---

## Project Structure

```
nyc_payroll_intelligence/
│
├── src/
│   ├── data_pipeline.py     # NYC OpenData API → DuckDB ingestion
│   ├── pay_equity.py        # Statistical pay equity analysis
│   ├── salary_model.py      # Random Forest salary prediction + flagging
│   ├── ot_anomaly.py        # Isolation Forest OT anomaly detection
│   ├── budget_forecast.py   # PyTorch LSTM budget forecasting
│   └── app.py               # Streamlit dashboard (main entry point)
│
├── visualizations/          # Dashboard screenshots
├── data/                    # Auto-populated by data_pipeline.py (gitignored)
├── models/                  # Saved model artifacts (gitignored)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/nisargakadam/nycopendata-payroll-dashboard
cd nyc_payroll_intelligence
pip install -r requirements.txt

# 2. Download data (~6.5M rows from NYC OpenData — takes ~10 min)
python src/data_pipeline.py

# 3. Train models
python src/salary_model.py
python src/ot_anomaly.py
python src/budget_forecast.py

# 4. Launch dashboard
streamlit run src/app.py
```

> **Note:** `data/` and `models/` are excluded from git due to file size. Running the scripts above will regenerate them locally.

---

## Methodology

### Pay Equity Analysis
- One-way ANOVA across agency groups to test for significant salary variance
- Post-hoc Tukey HSD for pairwise agency comparisons
- OT ratio (OT hours / regular hours) as a proxy for staffing pressure

### Salary Model
- Features: agency, title, borough, pay basis, years at agency, fiscal year
- Target: `regular_gross_paid`
- Flags employees where `|actual - predicted| / predicted > 20%` as equity outliers

### OT Anomaly Detection
- Isolation Forest on `[ot_hours, total_ot_paid, ot_ratio, ot_pay_rate]`
- contamination=0.05 (flags top 5% as anomalous)

### Budget Forecasting (LSTM)
- Input: agency-level total payroll spend per fiscal year
- Sequence length: 3 years
- Output: next fiscal year projected spend
- Architecture: 2-layer LSTM → Linear output

---

## Author

**Nisarga Kadam** | [LinkedIn](https://www.linkedin.com/in/nisarga-kadam/) | [Portfolio](https://nisargakadam.github.io)
