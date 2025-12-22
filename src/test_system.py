#!/usr/bin/env python3
"""Test système des modules"""

import sys
import pandas as pd
import numpy as np

print("="*80)
print("TEST DES MODULES")
print("="*80)

print("\n[1/5] Test des imports...")
try:
    from src.data_loader import DataLoader
    from src.features import FeatureEngineer
    from src.models import HMMRegimeDetector, RandomForestRegimeDetector
    from src.backtest import BacktestEngine
    from src.metrics import PerformanceMetrics
    print("✅ Tous les imports OK")
except Exception as e:
    print(f"❌ Erreur import: {e}")
    sys.exit(1)

FRED_API_KEY = "dee74f4224925bf1a3974d668b0e8460"  # <-- Mets ta clé
TICKERS = ["SPY", "TLT"]
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

print("\n[2/5] Test DataLoader...")
try:
    loader = DataLoader(fred_api_key=FRED_API_KEY)
    data = loader.load_all_data(TICKERS, START_DATE, END_DATE, use_cache=False)
    print(f"✅ DataLoader OK - {len(data['returns'])} mois")
except Exception as e:
    print(f"❌ Erreur: {e}")
    sys.exit(1)

print("\n[3/5] Test FeatureEngineer...")
try:
    engineer = FeatureEngineer()
    features = engineer.build_macro_features(data['macro'], data['vix'])
    print(f"✅ Features OK - {len(features.columns)} features")
except Exception as e:
    print(f"❌ Erreur: {e}")
    sys.exit(1)

print("\n[4/5] Test HMM...")
try:
    selected = engineer.select_features_for_modeling(features)
    hmm = HMMRegimeDetector(n_states=3)
    hmm.fit(selected.values)
    print(f"✅ HMM OK")
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()

print("\n[5/5] Test Metrics...")
try:
    calc = PerformanceMetrics()
    spy_returns = data['returns']['SPY']
    metrics = calc.calculate_all_metrics(spy_returns)
    print(f"✅ Metrics OK - Sharpe: {metrics['sharpe_ratio']:.2f}")
except Exception as e:
    print(f"❌ Erreur: {e}")

print("\n" + "="*80)
print("✅ TEST TERMINÉ")
print("="*80)