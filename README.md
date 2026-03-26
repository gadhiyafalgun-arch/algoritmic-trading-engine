<div align="center">

# ⚡ Algorithmic Trading Engine

### Quantitative Strategy Backtester · 3D Interactive Analytics · ML-Ready

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-ff4b4b?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3D%20Interactive-3d3d3d?logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-f7931e?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189fdd)

<br/>

[🚀 **Launch Live Engine**](https://algorithmic-trading-engine-falgun-gadhiya.streamlit.app/) &nbsp;&nbsp;|&nbsp;&nbsp; [📊 **Interactive 3D Charts**](https://gadhiyafalgun-arch.github.io/algorithmic-trading-engine/) &nbsp;&nbsp;|&nbsp;&nbsp; [💻 **Source Code**](https://github.com/gadhiyafalgun-arch/algorithmic-trading-engine)

<br/>

</div>

---

## 📸 Preview

<div align="center">

<a href="https://algorithmic-trading-engine-falgun-gadhiya.streamlit.app/">
  <img src="./screenshots/landing-page.png" alt="Trading Engine Dashboard" width="100%" />
</a>

<p><i>👆 Click the image to launch the live engine</i></p>

</div>

---

## 🧠 What It Does

A **full end-to-end algorithmic trading engine** that runs a complete pipeline on **8 stocks** (AAPL, GOOGL, MSFT, AMZN, TSLA, SPY & more) from **2020–2024**:



| Phase | Description |
|:------|:------------|
| 📥 **Data Pipeline** | Fetches and cleans daily OHLCV data via `yfinance` |
| 📐 **Indicators** | SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, VWAP |
| 🎯 **Strategies** | SMA Crossover, RSI, MACD, Bollinger Bands, Combined Multi-Signal |
| 📈 **Backtesting** | Realistic simulation — 0.1% commission, 0.05% slippage, stop-loss/take-profit |
| ⚖️ **Risk Management** | VaR, Sharpe/Sortino/Calmar ratios, max drawdown, position sizing |
| 🤖 **Machine Learning** | XGBoost + Random Forest with 100+ features, walk-forward validation |
| 📊 **Visualization** | 13 interactive 3D Plotly charts + Streamlit dashboard |

---

## 🎮 Live Engine Features

> **[🚀 Launch the Live Engine →](https://algorithmic-trading-engine-falgun-gadhiya.streamlit.app/)**

| Feature | What You Can Do |
|:--------|:----------------|
| 🔀 **Stock Selector** | Switch between multiple listed stocks |
| 📅 **Date Range** | Customize backtest period (2020–2024) |
| 🎯 **Strategy Picker** | Choose from 5 strategies including Combined Multi-Signal |
| 💰 **Capital & Costs** | Adjust initial capital, commission, and slippage |
| ⚡ **Risk Slider** | Drag to see Low → Medium → High risk impact in real time |
| 📊 **Performance Grade** | Auto-graded performance summary with key metrics |

---

## 📊 Interactive 3D Charts

> **[🌐 View Static 3D Charts →](https://gadhiyafalgun-arch.github.io/algorithmic-trading-engine/)**

Drag the ⚡ **Risk Level** slider to explore:
- 🟢 **Low risk (0.25×)** — Smaller positions, steadier equity curve
- 🟡 **Medium risk (1.0×)** — Default, balanced risk/return
- 🔴 **High risk (3.0×)** — Larger positions, amplified gains & drawdowns

| Chart | Description |
|:------|:------------|
| [📊 Risk Bar (3D)](docs/charts/AAPL_combined_signal_backtest_risk_bar_3d.html) | Portfolio backtest with interactive risk slider |
| [📈 Equity Comparison](docs/charts/AAPL_equity_comparison_3d.html) | Side-by-side strategy equity curves |
| [🤖 ML Signal](docs/charts/AAPL_ml_signal_3d.html) | Machine learning buy/sell signals |
| [📉 MACD](docs/charts/AAPL_macd_3d.html) | MACD momentum analysis |
| [💼 Portfolio Backtest](docs/charts/PORTFOLIO_combined%20+%20risk_mgmt_backtest.html) | Multi-stock portfolio results |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/gadhiyafalgun-arch/algorithmic-trading-engine.git
cd algorithmic-trading-engine

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generates charts)
python main.py

# Or launch the Streamlit app locally
streamlit run app.py
