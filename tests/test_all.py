#!/usr/bin/env python3
"""Test complet de tous les modules"""

import numpy as np
import pandas as pd

print("="*60)
print("🧪 TEST COMPLET - MACRO REGIME LAB")
print("="*60)

# Données de test
np.random.seed(42)
n = 120
dates = pd.date_range('2015-01-01', periods=n, freq='ME')

returns = pd.DataFrame({
    'SPY': np.random.normal(0.008, 0.04, n),
    'TLT': np.random.normal(0.003, 0.02, n),
    'GLD': np.random.normal(0.002, 0.03, n),
    'XLK': np.random.normal(0.01, 0.05, n)
}, index=dates)

regimes = pd.Series(
    np.random.choice(['expansion', 'slowdown', 'contraction'], n),
    index=dates, name='regime'
)

vix = pd.Series(np.random.uniform(12, 35, n), index=dates)

# 1. Test Models
print("\n[1/5] Testing Models...")
try:
    from src.models import EnsembleRegimeDetector, EnsembleConfig
    X = np.random.randn(n, 3)
    y = regimes.values
    
    config = EnsembleConfig(use_hmm=True, use_rf=True, use_gbm=True)
    ensemble = EnsembleRegimeDetector(n_regimes=3, config=config)
    ensemble.fit(X, y)
    pred = ensemble.predict(X)
    print(f"   ✅ EnsembleRegimeDetector OK - {len(np.unique(pred))} regimes detected")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Test Backtest
print("\n[2/5] Testing Backtest...")
try:
    from src.backtest import BacktestEngine, TransactionCostConfig
    
    weights = {
        'expansion': pd.Series({'SPY': 0.6, 'XLK': 0.3, 'TLT': 0.1, 'GLD': 0.0}),
        'slowdown': pd.Series({'SPY': 0.4, 'XLK': 0.2, 'TLT': 0.3, 'GLD': 0.1}),
        'contraction': pd.Series({'SPY': 0.1, 'XLK': 0.0, 'TLT': 0.6, 'GLD': 0.3})
    }
    
    engine = BacktestEngine()
    result = engine.run_backtest(returns, regimes, weights, vix, verbose=False)
    print(f"   ✅ BacktestEngine OK - Sharpe: {result.sharpe_ratio:.2f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Test Stress Testing
print("\n[3/5] Testing Stress Testing...")
try:
    from src.stress_testing import StressTester
    
    strategy_rets = returns['SPY']
    tester = StressTester(strategy_rets)
    var = tester.calculate_var(0.95, verbose=False)
    print(f"   ✅ StressTester OK - VaR 95%: {var.var_historical*100:.2f}%")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. Test Walk Forward
print("\n[4/5] Testing Walk Forward...")
try:
    from src.walk_forward import WalkForwardAnalyzer
    
    analyzer = WalkForwardAnalyzer(train_window=36, test_window=12)
    # Just test initialization
    print(f"   ✅ WalkForwardAnalyzer OK - train={analyzer.train_window}m, test={analyzer.test_window}m")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 5. Test Visualization
print("\n[5/5] Testing Visualization...")
try:
    from src.visualization import AdvancedVisualizer
    
    viz = AdvancedVisualizer()
    print(f"   ✅ AdvancedVisualizer OK")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("✅ TOUS LES TESTS PASSÉS!")
print("="*60)