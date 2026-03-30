#  NYC Payroll Intelligence Dashboard

> **Pay equity analysis, OT anomaly detection, and budget forecasting for NYC municipal agencies**  
> Built with DuckDB · scikit-learn · PyTorch · Streamlit | Dataset: 6.78M records (NYC OpenData)

---

##  Project Overview

This end-to-end data science project analyzes NYC's Citywide Payroll Data to surface:
- **Pay equity disparities** across agencies, boroughs, and job titles
- **Overtime anomalies** using unsupervised ML (Isolation Forest)
- **Salary modeling** with Random Forest to detect under/overpayment
- **Budget forecasting** via LSTM time-series models to project future agency spend

A dedicated focus is placed on **NYC Department of Small Business Services (SBS)** — examining headcount trends, salary distributions, overtime patterns, and budget projections.

---

##  Project Structure

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
├── data/                    # Auto-populated by data_pipeline.py
├── models/                  # Saved trained model artifacts
├── requirements.txt
└── README.md
```

---

##  Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/nyc-payroll-intelligence
cd nyc_payroll_intelligence
pip install -r requirements.txt

# 2. Download and load data (fetches from NYC OpenData API)
python src/data_pipeline.py

# 3. Train models
python src/salary_model.py
python src/ot_anomaly.py
python src/budget_forecast.py

# 4. Launch dashboard
streamlit run src/app.py
```

---

##  Tech Stack

| Layer | Tools |
|---|---|
| Data Storage | DuckDB (in-process SQL, handles 6.7M rows locally) |
| Data Wrangling | pandas, numpy |
| Statistical Analysis | scipy, statsmodels |
| ML Models | scikit-learn (Random Forest, Isolation Forest) |
| Deep Learning | PyTorch (LSTM) |
| Visualization | Plotly, Seaborn |
| Dashboard | Streamlit |
| Data Source | [NYC OpenData Citywide Payroll][(https://data.cityofnewyork.us/City-Government/Citywide-Payroll-Data-Fiscal-Year-/k397-673e/about_data)]|

---

## 📊 Key Findings (SBS Spotlight)

*Populated after running full pipeline — see dashboard for live insights.*

---

## 🔍 Methodology

### Pay Equity Analysis
- One-way ANOVA across agency groups to test for significant salary variance
- Post-hoc Tukey HSD for pairwise agency comparisons
- OT ratio (OT hours / regular hours) as a proxy for staffing pressure

### Salary Model
- Features: agency, title, borough, pay_basis, years at agency, fiscal year
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

## 👤 Author

**Nisarga** | [LinkedIn](https://www.linkedin.com/in/nisarga-kadam/)| [Portfolio](https://nisargakadam.github.io)
