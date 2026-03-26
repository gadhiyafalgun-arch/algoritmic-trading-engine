# Algorithmic Trading Engine

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3D%20Interactive-3d3d3d?logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-f7931e?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189fdd)

> A full end-to-end algorithmic trading engine with backtesting, risk management, machine learning signal generation, and interactive 3D visualization.

## Live Demo

**[View Interactive 3D Chart with Risk Slider →](https://gadhiyafalgun-arch.github.io/algorithmic-trading-engine/)**

Drag the ⚡ Risk Level slider to see how different risk settings affect portfolio performance:
- **Low risk (0.25×)** — smaller positions, steadier equity curve
- **Medium risk (1.0×)** — default, balanced risk/return
- **High risk (3.0×)** — larger positions, amplified gains and drawdowns

---

## What It Does

Runs a complete trading pipeline on 6 stocks (AAPL, GOOGL, MSFT, AMZN, TSLA, SPY) from 2020–2024:

| Phase | Description |
|-------|-------------|
| **Data Pipeline** | Fetches and cleans daily OHLCV data via yfinance |
| **Indicators** | SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, VWAP |
| **Strategies** | SMA Crossover, RSI, MACD, Bollinger Bands, Combined multi-indicator |
| **Backtesting** | Realistic simulation with 0.1% commission, 0.05% slippage, stop-loss/take-profit |
| **Risk Management** | VaR, Sharpe/Sortino/Calmar ratios, max drawdown, position sizing |
| **Machine Learning** | XGBoost + Random Forest with 100+ features, walk-forward validation |
| **Visualization** | 13 interactive 3D Plotly charts |

---

## Charts

| Chart | Description |
|-------|-------------|
| [Risk Bar (3D)](docs/charts/AAPL_combined_signal_backtest_risk_bar_3d.html) | Portfolio backtest with interactive risk level slider |
| [Equity Comparison (3D)](docs/charts/AAPL_equity_comparison_3d.html) | Side-by-side strategy equity curves |
| [ML Signal (3D)](docs/charts/AAPL_ml_signal_3d.html) | Machine learning buy/sell signals |
| [MACD (3D)](docs/charts/AAPL_macd_3d.html) | MACD momentum analysis |
| [Portfolio Backtest](docs/charts/PORTFOLIO_combined%20+%20risk_mgmt_backtest.html) | Multi-stock portfolio results |

---

## Run Locally

```bash
# Clone
git clone https://github.com/gadhiyafalgun-arch/algorithmic-trading-engine.git
cd algorithmic-trading-engine

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

Charts are saved to `docs/charts/` and open automatically in your browser.

---

## Tech Stack

- **Data**: `yfinance`, `pandas`, `numpy`
- **Indicators**: `pandas-ta`
- **ML**: `scikit-learn`, `xgboost`
- **Visualization**: `plotly`
- **Config**: `config.yaml` — adjust symbols, dates, capital, risk parameters
