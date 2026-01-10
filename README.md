# 📊 Macro Regime & Factor Rotation Lab

A Python tool that identifies macroeconomic regimes (Expansion, Slowdown, Recession) from public indicators and rotates an ETF portfolio accordingly.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This project implements a **macro regime-based factor rotation strategy** that:
- Detects economic regimes using FRED macroeconomic data (CPI, Unemployment, NBER Recession indicator)
- Dynamically allocates across ETFs (SPY, TLT, GLD, XLK) based on the current regime
- Backtests the strategy with realistic transaction costs
- Generates performance reports with visualizations

## 📈 Performance Results

| Metric | Strategy | SPY | 60/40 |
|--------|----------|-----|-------|
| **CAGR** | **13.6%** | 9.5% | 6.0% |
| **Sharpe Ratio** | **0.95** | 0.64 | 0.60 |
| **Max Drawdown** | **-26.7%** | -52.2% | -31.3% |

*Backtest period: 2005-2025*

## 🔧 Installation
```bash
# Clone the repository
git clone https://github.com/Marco1-x/Macro-regime-Lab-.git
cd Macro-regime-Lab-

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 Quick Start

### CLI Commands

The project provides three main commands:
```bash
# 1. Detect macro regimes from FRED data
python3 -m src.cli detect-regimes

# 2. Run backtest with transaction costs
python3 -m src.cli backtest

# 3. Generate Markdown report
python3 -m src.cli report
```

### Interactive Dashboard
```bash
streamlit run src/dashboard.py
```

## 📁 Project Structure
```
macro-factor-lab/
├── src/
│   ├── cli.py              # Typer CLI (detect-regimes, backtest, report)
│   ├── dashboard.py        # Streamlit interactive dashboard
│   ├── data_fetcher.py     # Yahoo Finance data downloader
│   ├── models.py           # Regime detection models (HMM, RF, Ensemble)
│   ├── backtest.py         # Backtesting engine with transaction costs
│   ├── stress_testing.py   # VaR and stress testing
│   ├── walk_forward.py     # Walk-forward analysis
│   └── visualization.py    # Plotting utilities
├── data/
│   ├── fred/               # FRED macroeconomic data (offline)
│   │   ├── CPIAUCSL.csv    # Consumer Price Index
│   │   ├── UNRATE.csv      # Unemployment Rate
│   │   └── USREC.csv       # NBER Recession Indicator
│   └── etf_prices.csv      # Historical ETF prices
├── output/
│   ├── regimes.csv         # Detected regimes
│   ├── returns.csv         # Strategy returns
│   ├── metrics.json        # Performance metrics
│   ├── wealth_curve.png    # Wealth curve chart
│   ├── drawdown.png        # Drawdown chart
│   └── REPORT.md           # Generated report
├── docs/
│   ├── API.md              # API documentation
│   └── USER_GUIDE.md       # User guide
├── requirements.txt
├── INSTALLATION.md
└── README.md
```

## 📊 Methodology

### Regime Detection

Regimes are defined using a transparent heuristic:

| Regime | Definition |
|--------|------------|
| **Recession** | USREC = 1 (NBER official recession) |
| **Slowdown** | CPI YoY > rolling median AND ΔUNRATE > 0 |
| **Expansion** | Otherwise |

### Portfolio Allocation

| Regime | SPY | TLT | GLD | XLK |
|--------|-----|-----|-----|-----|
| Expansion | 60% | 0% | 0% | 40% |
| Slowdown | 40% | 40% | 20% | 0% |
| Recession | 0% | 70% | 30% | 0% |

### Backtest Parameters

- **Rebalancing**: Monthly
- **Transaction costs**: 5 bps per unit of turnover
- **Benchmarks**: SPY buy-and-hold, 60/40 portfolio

## 📉 Regime Distribution (1947-2025)

| Regime | Months | Percentage |
|--------|--------|------------|
| Expansion | 705 | 76.5% |
| Recession | 123 | 13.3% |
| Slowdown | 94 | 10.2% |

## 🛠️ Technologies

- **Python 3.9+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Streamlit** - Interactive dashboard
- **Typer** - CLI framework
- **Plotly / Matplotlib** - Visualizations
- **scikit-learn** - Machine learning
- **hmmlearn** - Hidden Markov Models

## 📚 Documentation

- [Installation Guide](INSTALLATION.md)
- [API Reference](src/API.md)
- [User Guide](src/user_guide.md)

## ⚠️ Limitations

1. **NBER dating lag**: Official recession dates are announced with delay
2. **Threshold sensitivity**: Rolling median period affects regime detection
3. **Look-ahead bias**: Strategy uses only information available at decision time
4. **Transaction costs**: Real costs may vary with market conditions

## 🔮 Possible Improvements

- Hidden Markov Models for data-driven regime detection
- Additional indicators (yield curve slope, credit spreads, PMI)
- Dynamic weight optimization within regimes
- Risk parity position sizing

## 👤 Author

**Marc Aurel AMOUSSOU**

University of Lausanne - MSc in Finance

## 📄 License


---

*Project developed for the "Introduction to Data Science and Advanced Programming" course, Fall 2025*
