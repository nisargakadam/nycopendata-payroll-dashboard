"""
budget_forecast.py
────────────────────────────────────────────────────────────────────────────
PyTorch LSTM model for forecasting agency-level payroll budget spend.

Architecture: 2-layer LSTM → Linear output
Input:        Sequence of total payroll spend over N fiscal years per agency
Output:       Predicted spend for the next fiscal year

Outputs:
  - Trained model saved to models/budget_lstm.pt
  - Per-agency forecast DataFrame
  - Plotly forecast visualization

Run:
    python src/budget_forecast.py
"""

import os
import duckdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
import plotly.express as px

DB_PATH    = "data/payroll.duckdb"
MODEL_PATH = "models/budget_lstm.pt"

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQ_LEN    = 3       # Use 3 years of history to predict next year
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT    = 0.2
LR         = 1e-3
EPOCHS     = 150
BATCH_SIZE = 32
MIN_YEARS  = 5       # Require at least 5 fiscal years of data per agency


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_agency_spend(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Aggregate total payroll spend per agency per fiscal year.
    Returns a wide pivot DataFrame: rows = agencies, cols = fiscal years.
    """
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("""
        SELECT
            fiscal_year,
            agency_name,
            SUM(regular_gross_paid + COALESCE(total_ot_paid, 0) + COALESCE(total_other_pay, 0))
                AS total_spend
        FROM payroll
        GROUP BY fiscal_year, agency_name
        ORDER BY agency_name, fiscal_year
    """).df()
    con.close()
    return df


def build_sequences(df_long: pd.DataFrame, seq_len: int = SEQ_LEN) -> tuple:
    """
    Build supervised learning sequences from the long-format spend DataFrame.
    Returns X (input sequences), y (targets), and metadata.
    """
    pivot = df_long.pivot(index="agency_name", columns="fiscal_year", values="total_spend")

    # Keep only agencies with enough history
    pivot = pivot.dropna(thresh=MIN_YEARS)
    pivot = pivot.fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)

    # Normalise per agency (min-max within each row)
    mins  = pivot.min(axis=1)
    maxes = pivot.max(axis=1)
    pivot_norm = pivot.sub(mins, axis=0).div((maxes - mins).replace(0, 1), axis=0)

    fiscal_years = sorted(pivot.columns.tolist())
    X_list, y_list, meta = [], [], []

    for agency in pivot_norm.index:
        series = pivot_norm.loc[agency, fiscal_years].values.astype(np.float32)
        for i in range(len(series) - seq_len):
            X_list.append(series[i: i + seq_len])
            y_list.append(series[i + seq_len])
            meta.append({
                "agency_name": agency,
                "input_years": fiscal_years[i: i + seq_len],
                "target_year": fiscal_years[i + seq_len],
            })

    X = np.array(X_list, dtype=np.float32)          # (N, seq_len)
    y = np.array(y_list, dtype=np.float32)           # (N,)

    return X, y, meta, pivot, mins, maxes


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpendDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).unsqueeze(-1)   # (N, seq_len, 1)
        self.y = torch.from_numpy(y).unsqueeze(-1)   # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class BudgetLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # Use last timestep output


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray) -> tuple[BudgetLSTM, list[float]]:
    """Train the LSTM model. Returns model and training loss history."""
    dataset = SpendDataset(X, y)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BudgetLSTM().to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    losses = []
    print(f"Training LSTM on {device}…")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            epoch_loss += loss.item() * len(xb)

        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}  |  Loss: {avg_loss:.6f}")

    return model.to("cpu"), losses


def save_model(model: BudgetLSTM, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ LSTM saved → {path}")


def load_model(path: str = MODEL_PATH) -> BudgetLSTM:
    model = BudgetLSTM()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ── Forecasting ───────────────────────────────────────────────────────────────

def forecast_agencies(
    model: BudgetLSTM,
    pivot: pd.DataFrame,
    mins: pd.Series,
    maxes: pd.Series,
    seq_len: int = SEQ_LEN,
    agencies: list[str] = None,
) -> pd.DataFrame:
    """
    For each agency, use the last `seq_len` years as input and forecast
    the next fiscal year's spend. De-normalises back to raw dollar values.
    """
    model.eval()
    fiscal_years = sorted(pivot.columns.tolist())
    last_fy      = max(fiscal_years)
    forecast_fy  = last_fy + 1

    target_agencies = agencies or pivot.index.tolist()
    results = []

    with torch.no_grad():
        for agency in target_agencies:
            if agency not in pivot.index:
                continue

            series_raw  = pivot.loc[agency, fiscal_years].values.astype(np.float32)
            mn, mx      = float(mins[agency]), float(maxes[agency])
            series_norm = (series_raw - mn) / max(mx - mn, 1.0)

            # Use last seq_len values as input
            input_seq = torch.tensor(
                series_norm[-seq_len:], dtype=torch.float32
            ).unsqueeze(0).unsqueeze(-1)   # (1, seq_len, 1)

            pred_norm = model(input_seq).item()
            pred_raw  = pred_norm * (mx - mn) + mn   # de-normalise

            # Historical context
            historical = {str(fy): float(pivot.loc[agency, fy]) for fy in fiscal_years}
            results.append({
                "agency_name":    agency,
                "forecast_year":  forecast_fy,
                "forecast_spend": round(pred_raw, 2),
                "last_year_spend": float(pivot.loc[agency, last_fy]),
                "yoy_change_pct": (pred_raw / max(float(pivot.loc[agency, last_fy]), 1) - 1) * 100,
                **historical,
            })

    return pd.DataFrame(results).sort_values("forecast_spend", ascending=False)


# ── Plotly Figures ────────────────────────────────────────────────────────────

def fig_forecast_bar(forecast_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Bar chart comparing last year spend vs forecast for top agencies."""
    top = forecast_df.head(top_n).copy()
    top["agency_short"] = top["agency_name"].str[:35]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top["agency_short"],
        x=top["last_year_spend"],
        name="Last FY Actual",
        orientation="h",
        marker_color="#636EFA",
    ))
    fig.add_trace(go.Bar(
        y=top["agency_short"],
        x=top["forecast_spend"],
        name="Forecast",
        orientation="h",
        marker_color="#00CC96",
    ))
    fig.update_layout(
        barmode="group",
        title=f"Payroll Spend: Last FY vs Forecast (Top {top_n} Agencies)",
        xaxis_title="Total Payroll Spend ($)",
        template="plotly_dark",
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


def fig_agency_trend_with_forecast(
    agency: str,
    pivot: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> go.Figure:
    """Line chart of historical spend + forecast point for a single agency."""
    if agency not in pivot.index:
        return go.Figure()

    fiscal_years = sorted([c for c in pivot.columns if isinstance(c, (int, float))])
    historical   = [float(pivot.loc[agency, fy]) for fy in fiscal_years]

    forecast_row = forecast_df[forecast_df["agency_name"] == agency]
    if forecast_row.empty:
        return go.Figure()

    forecast_fy    = int(forecast_row["forecast_year"].values[0])
    forecast_spend = float(forecast_row["forecast_spend"].values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fiscal_years, y=historical,
        mode="lines+markers", name="Historical",
        line=dict(color="#636EFA", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[fiscal_years[-1], forecast_fy],
        y=[historical[-1], forecast_spend],
        mode="lines+markers", name="Forecast",
        line=dict(color="#00CC96", width=2, dash="dot"),
        marker=dict(size=10, symbol="star"),
    ))
    fig.update_layout(
        title=f"{agency[:50]} — Payroll Spend Trend & Forecast",
        xaxis_title="Fiscal Year",
        yaxis_title="Total Payroll Spend ($)",
        template="plotly_dark",
    )
    return fig


def fig_training_loss(losses: list[float]) -> go.Figure:
    """Plot training loss curve."""
    fig = px.line(
        x=list(range(1, len(losses) + 1)),
        y=losses,
        labels={"x": "Epoch", "y": "MSE Loss"},
        title="LSTM Training Loss",
        template="plotly_dark",
    )
    return fig


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_long = load_agency_spend()
    print(f"Agencies: {df_long['agency_name'].nunique()}, "
          f"Fiscal Years: {sorted(df_long['fiscal_year'].unique())}")

    X, y, meta, pivot, mins, maxes = build_sequences(df_long)
    print(f"Training sequences: {len(X):,}")

    model, losses = train_model(X, y)
    save_model(model)

    print("\nGenerating forecasts…")
    forecast_df = forecast_agencies(model, pivot, mins, maxes)

    print("\n── Top 10 Forecast Spend (Next FY) ──────────────────────────────────")
    print(forecast_df[["agency_name", "last_year_spend", "forecast_spend", "yoy_change_pct"]]
          .head(10).to_string(index=False))

    print("\n── SBS Forecast ──────────────────────────────────────────────────────")
    sbs_forecast = forecast_df[forecast_df["agency_name"].str.contains("SMALL BUSINESS", na=False)]
    if not sbs_forecast.empty:
        print(sbs_forecast[["agency_name", "last_year_spend", "forecast_spend", "yoy_change_pct"]]
              .to_string(index=False))
    else:
        print("No SBS records found in forecast output.")
