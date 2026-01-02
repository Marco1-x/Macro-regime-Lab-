# User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Regime Detection](#regime-detection)
5. [Backtesting](#backtesting)
6. [Stress Testing](#stress-testing)
7. [Walk-Forward Analysis](#walk-forward-analysis)
8. [Dashboard](#dashboard)
9. [Best Practices](#best-practices)

---

## Introduction

Macro Regime Lab is a quantitative finance toolkit for:

- **Regime Detection**: Identify market regimes using ML (HMM, Random Forest, Ensemble)
- **Factor Rotation**: Dynamically allocate across assets based on detected regimes
- **Backtesting**: Test strategies with realistic transaction costs and slippage
- **Risk Analysis**: VaR, CVaR, stress testing, and scenario analysis
- **Walk-Forward Analysis**: Validate strategies with rolling out-of-sample testing

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/macro-factor-lab.git
cd macro-factor-lab

# Install dependencies
pip3 install -r requirements.txt

# Verify installation
python3 -c "from src.models import EnsembleRegimeDetector; print('OK')"
```

### Requirements

- Python 3.9+
- pandas, numpy, scipy
- scikit-learn, hmmlearn
- yfinance, fredapi
- matplotlib, plotly
- streamlit

---

## Quick Start

### 1. Load Data

```python
import pandas as pd
import numpy as np
from src.data_loader import DataLoader

# Using FRED API
loader = DataLoader(fred_api_key='YOUR_KEY')
data = loader.load_all_data(
    tickers=['SPY', 'TLT', 'GLD', 'XLK'],
    start_date='2010-01-01',
    end_date='2023-12-31'
)

returns = data['returns']
macro = data['macro']
```

### 2. Build Features

```python
from src.features import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.build_macro_features(macro, data['vix'])
```

### 3. Detect Regimes

```python
from src.models import EnsembleRegimeDetector, EnsembleConfig

config = EnsembleConfig(use_hmm=True, use_rf=True, use_gbm=True)
detector = EnsembleRegimeDetector(n_regimes=3, config=config)

X = features.values
detector.fit(X, heuristic_labels)
regimes = detector.predict(X)
```

### 4. Run Backtest

```python
from src.backtest import BacktestEngine

weights = {
    'expansion': pd.Series({'SPY': 0.6, 'XLK': 0.3, 'TLT': 0.1, 'GLD': 0.0}),
    'slowdown': pd.Series({'SPY': 0.3, 'XLK': 0.2, 'TLT': 0.3, 'GLD': 0.2}),
    'contraction': pd.Series({'SPY': 0.0, 'XLK': 0.0, 'TLT': 0.7, 'GLD': 0.3})
}

engine = BacktestEngine()
result = engine.run_backtest(returns, regimes, weights, vix)

print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Return: {result.total_return*100:.1f}%")
```

---

## Regime Detection

### Heuristic Regimes

Simple rule-based classification:

```python
def assign_heuristic_regimes(features):
    """
    Rules:
    - Contraction: Yield curve inverted (slope <= 0)
    - Slowdown: High inflation + rising unemployment
    - Expansion: Default
    """
    regime = pd.Series('expansion', index=features.index)
    
    # Contraction: inverted yield curve
    regime[features['slope'] <= 0] = 'contraction'
    
    # Slowdown: inflation above median, unemployment rising
    infl_median = features['CPI_YoY'].rolling(60).median()
    slowdown_mask = (features['CPI_YoY'] > infl_median) & (features['dUNRATE'] > 0)
    regime[slowdown_mask & (regime != 'contraction')] = 'slowdown'
    
    return regime
```

### ML Regimes

#### HMM (Unsupervised)

```python
from src.models import HMMRegimeDetector

hmm = HMMRegimeDetector(n_states=3)
hmm.fit(X_train)
regimes = hmm.predict(X_all)
```

#### Ensemble (Semi-supervised)

```python
from src.models import EnsembleRegimeDetector, EnsembleConfig

config = EnsembleConfig(
    use_hmm=True,      # Unsupervised
    use_rf=True,       # Supervised
    use_gbm=True,      # Supervised
    voting_method='soft'
)

ensemble = EnsembleRegimeDetector(n_regimes=3, config=config)
ensemble.fit(X, heuristic_labels)

# Get predictions with confidence
predictions = ensemble.predict(X)
confidence = ensemble.get_confidence(X)
agreement = ensemble.get_model_agreement(X)
```

---

## Backtesting

### Basic Backtest

```python
from src.backtest import quick_backtest

result = quick_backtest(returns, regimes, weights)
print(result.sharpe_ratio)
```

### With Dynamic Slippage

```python
from src.backtest import BacktestEngine, SlippageConfig, TransactionCostConfig

slippage_config = SlippageConfig(
    base_slippage_bps=2.0,
    vix_threshold_low=15,
    vix_threshold_high=30,
    vix_multiplier_high=3.0
)

cost_config = TransactionCostConfig(
    commission_bps=1.0,
    spread_bps=2.0,
    slippage_config=slippage_config
)

engine = BacktestEngine(cost_config=cost_config)
result = engine.run_backtest(returns, regimes, weights, vix)
```

### Compare Strategies

```python
# Compare with/without slippage
comparison = engine.compare_with_without_slippage(
    returns, regimes, weights, vix
)

print(f"Impact: {comparison['slippage_impact']*100:.2f}%")
```

---

## Stress Testing

### VaR Analysis

```python
from src.stress_testing import StressTester

tester = StressTester(strategy_returns)

# Single confidence level
var_95 = tester.calculate_var(0.95)
print(f"VaR 95%: {var_95.var_historical*100:.2f}%")
print(f"CVaR 95%: {var_95.cvar*100:.2f}%")

# Multiple levels
var_results = tester.calculate_var_multiple_levels()
```

### Historical Crisis Replay

```python
# Replay specific crisis
covid_impact = tester.replay_historical_crisis('covid_2020')

# Replay all crises
all_crises = tester.replay_all_crises()
for crisis, result in all_crises.items():
    print(f"{crisis}: {result['portfolio_return']*100:.1f}%")
```

### Hypothetical Scenarios

```python
# Custom scenario
result = tester.run_hypothetical_scenario(
    name='stagflation',
    shocks={'SPY': -0.15, 'TLT': -0.10, 'GLD': 0.20}
)

# Predefined scenarios
scenarios = tester.run_predefined_scenarios()
```

---

## Walk-Forward Analysis

Validate strategy with rolling out-of-sample testing:

```python
from src.walk_forward import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer(
    train_window=36,  # 3 years training
    test_window=12    # 1 year testing
)

results = analyzer.run_walk_forward(
    returns=returns,
    features=features,
    weights_by_regime=weights
)

print(f"Overall Sharpe: {results.overall_sharpe:.2f}")
print(f"Total Return: {results.overall_return*100:.1f}%")

# Per-window analysis
for i, window in enumerate(results.window_results):
    print(f"Window {i+1}: {window['test_return']*100:.1f}%")
```

---

## Dashboard

Launch the interactive Streamlit dashboard:

```bash
python3 -m streamlit run src/dashboard.py
```

### Features

1. **Dashboard Tab**: Key metrics, wealth curves, drawdown
2. **Performance Tab**: Monthly heatmap, rolling Sharpe, regime performance
3. **Regimes Tab**: Distribution, weights, timeline
4. **Risk Tab**: VaR/CVaR, VIX distribution
5. **Export Tab**: Download CSV files

### Configuration

- Adjust period length
- Select assets
- Configure regime weights
- Toggle dynamic slippage
- Set transaction costs

---

## Best Practices

### 1. Data Quality

- Use at least 10 years of data for regime detection
- Handle missing values before analysis
- Align all data to same frequency (monthly recommended)

### 2. Regime Detection

- Start with heuristic regimes as baseline
- Use ensemble methods for robustness
- Check model agreement score (>0.7 is good)
- Validate with walk-forward analysis

### 3. Backtesting

- Always include transaction costs
- Use dynamic slippage in volatile periods
- Test multiple time periods
- Compare against buy-and-hold benchmark

### 4. Risk Management

- Run stress tests before deployment
- Monitor VaR and CVaR regularly
- Set drawdown limits
- Diversify across regimes

### 5. Validation

- Use walk-forward analysis (not just in-sample)
- Check for regime persistence
- Validate on out-of-sample data
- Be skeptical of very high Sharpe ratios

---

## Troubleshooting

### Common Issues

**Import errors:**
```bash
pip3 install -r requirements.txt
```

**FRED API errors:**
```python
# Get free API key at https://fred.stlouisfed.org/docs/api/api_key.html
loader = DataLoader(fred_api_key='your_key_here')
```

**Streamlit not found:**
```bash
python3 -m streamlit run src/dashboard.py
```

**Memory issues with large datasets:**
```python
# Use chunked processing
analyzer = WalkForwardAnalyzer(train_window=24, test_window=6)
```