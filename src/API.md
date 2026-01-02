# API Reference

## Models Module (`src/models.py`)

### HMMRegimeDetector

Gaussian Hidden Markov Model for regime detection.

```python
from src.models import HMMRegimeDetector

detector = HMMRegimeDetector(n_states=3)
detector.fit(X)  # X: numpy array of features
predictions = detector.predict(X)
```

**Parameters:**
- `n_states` (int): Number of hidden states/regimes. Default: 3

**Methods:**
- `fit(X)`: Train the HMM on feature matrix X
- `predict(X)`: Predict regime states for feature matrix X

---

### RandomForestRegimeDetector

Supervised Random Forest classifier for regime detection.

```python
from src.models import RandomForestRegimeDetector

detector = RandomForestRegimeDetector(n_estimators=100)
detector.fit(X, y)  # X: features, y: regime labels
predictions = detector.predict(X)
```

**Parameters:**
- `n_estimators` (int): Number of trees. Default: 100

---

### EnsembleRegimeDetector

Multi-model ensemble combining HMM, RF, GBM, SVM, KNN, Logistic Regression.

```python
from src.models import EnsembleRegimeDetector, EnsembleConfig

config = EnsembleConfig(
    use_hmm=True,
    use_rf=True,
    use_gbm=True,
    use_svm=False,
    use_knn=False,
    use_logistic=False,
    voting_method='soft',  # 'soft' or 'hard'
    weights=None  # Optional dict of model weights
)

ensemble = EnsembleRegimeDetector(n_regimes=3, config=config)
ensemble.fit(X, y)
predictions = ensemble.predict(X)
probabilities = ensemble.predict_proba(X)
agreement = ensemble.get_model_agreement(X)
```

**Methods:**
- `fit(X, y)`: Train all enabled models
- `predict(X)`: Ensemble prediction using voting
- `predict_proba(X)`: Weighted average probabilities
- `get_model_agreement(X)`: Agreement score between models (0-1)
- `cross_validate(X, y, cv=5)`: Cross-validation per model
- `get_confidence(X)`: Prediction confidence (max probability)

---

## Backtest Module (`src/backtest.py`)

### SlippageConfig

Configuration for dynamic slippage calculation.

```python
from src.backtest import SlippageConfig

config = SlippageConfig(
    base_slippage_bps=2.0,      # Base slippage in basis points
    vix_threshold_low=15.0,     # VIX level for low volatility
    vix_threshold_high=30.0,    # VIX level for high volatility
    vix_multiplier_low=1.0,     # Multiplier at low VIX
    vix_multiplier_high=3.0,    # Multiplier at high VIX
    max_slippage_bps=20.0,      # Maximum slippage cap
    size_impact=True            # Adjust for trade size
)
```

---

### DynamicSlippageCalculator

Calculates VIX-adjusted transaction costs.

```python
from src.backtest import DynamicSlippageCalculator, SlippageConfig

config = SlippageConfig()
calculator = DynamicSlippageCalculator(config)

slippage = calculator.calculate_slippage(
    vix=25.0,        # Current VIX level
    turnover=0.5,    # Portfolio turnover (0-2)
    trade_size=0.1   # Trade size as fraction of portfolio
)
```

---

### BacktestEngine

Full-featured backtesting engine with realistic costs.

```python
from src.backtest import BacktestEngine, TransactionCostConfig

engine = BacktestEngine(cost_config=TransactionCostConfig())

result = engine.run_backtest(
    returns=returns_df,          # DataFrame of asset returns
    regimes=regimes_series,      # Series of regime labels
    weights_by_regime=weights,   # Dict of {regime: Series(weights)}
    vix=vix_series,             # Optional VIX series
    verbose=True
)

# Access results
print(result.total_return)
print(result.sharpe_ratio)
print(result.max_drawdown)
print(result.returns)
print(result.wealth)
```

**BacktestResult attributes:**
- `returns`: Series of portfolio returns
- `wealth`: Series of cumulative wealth
- `positions`: DataFrame of positions over time
- `turnover`: Series of turnover
- `costs`: Series of transaction costs
- `slippage`: Series of slippage costs
- `total_return`, `sharpe_ratio`, `max_drawdown`, `volatility`, `win_rate`

---

## Stress Testing Module (`src/stress_testing.py`)

### StressTester

Comprehensive stress testing and VaR analysis.

```python
from src.stress_testing import StressTester

tester = StressTester(
    strategy_returns=returns_series,
    asset_returns=returns_df,      # Optional
    weights=weights_series         # Optional
)

# VaR Analysis
var_result = tester.calculate_var(confidence=0.95)
print(var_result.var_historical)
print(var_result.var_parametric)
print(var_result.var_montecarlo)
print(var_result.cvar)

# Multiple confidence levels
results = tester.calculate_var_multiple_levels()

# Historical crisis replay
crisis_results = tester.replay_all_crises()

# Hypothetical scenarios
scenario = tester.run_hypothetical_scenario(
    name='market_crash',
    shocks={'SPY': -0.30, 'TLT': 0.10}
)

# Predefined scenarios
scenarios = tester.run_predefined_scenarios()

# Sensitivity analysis
sensitivity = tester.sensitivity_analysis(shock_range=0.10)

# Full report
report = tester.generate_full_report()
```

---

## Walk Forward Module (`src/walk_forward.py`)

### WalkForwardAnalyzer

Rolling window walk-forward analysis.

```python
from src.walk_forward import WalkForwardAnalyzer, WalkForwardConfig

config = WalkForwardConfig(
    train_window=36,    # Training window in months
    test_window=12,     # Test window in months
    step_size=1,        # Step size between windows
    min_train_samples=24
)

analyzer = WalkForwardAnalyzer(
    train_window=36,
    test_window=12,
    config=config
)

results = analyzer.run_walk_forward(
    returns=returns_df,
    features=features_df,
    weights_by_regime=weights,
    verbose=True
)

# Access results
print(results.overall_sharpe)
print(results.overall_return)
print(results.combined_returns)
print(results.combined_wealth)

# Per-window results
for window in results.window_results:
    print(window['test_return'], window['test_sharpe'])
```

---

## Visualization Module (`src/visualization.py`)

### AdvancedVisualizer

Professional-grade visualizations.

```python
from src.visualization import AdvancedVisualizer

viz = AdvancedVisualizer(style='seaborn', figsize=(12, 8))

# Wealth curves
fig = viz.plot_wealth_curves(wealth_dict, title='Strategy Performance')

# Drawdown
fig = viz.plot_drawdown(returns_series)

# Regime timeline
fig = viz.plot_regime_timeline(regimes_series, prices_series)

# Monthly heatmap
fig = viz.plot_monthly_returns_heatmap(returns_series)

# Risk metrics
fig = viz.plot_risk_metrics(returns_series)

# Factor exposures
fig = viz.plot_factor_exposures(exposures_df)

# Rolling metrics
fig = viz.plot_rolling_metrics(returns_series, window=12)

# Save all
viz.save_all_plots(output_dir='output/')
```

---

## Dashboard (`src/dashboard.py`)

Interactive Streamlit dashboard.

```bash
python3 -m streamlit run src/dashboard.py
```

**Features:**
- 5 tabs: Dashboard, Performance, Regimes, Risk, Export
- Interactive regime weight configuration
- Dynamic slippage toggle
- CSV data export
- Real-time backtest updates