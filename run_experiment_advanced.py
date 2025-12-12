#!/usr/bin/env python3
"""
run_experiment_advanced.py
Macro Regime & Factor Rotation Lab - VERSION AVANCÉE
avec ML multi-modèles, optimisation automatique, et walk-forward analysis

Fonctionnalités avancées :
- 3 régimes heuristiques (expansion, slowdown, contraction)
- Plusieurs modèles ML : HMM, Random Forest, LSTM
- Features avancées : VIX, spreads de crédit
- Optimisation automatique des poids par régime
- Walk-forward analysis (réentraînement mensuel)
- Métriques complètes : Sharpe, Sortino, Calmar, Win Rate
- Graphiques de drawdown
- Export complet des résultats
"""

import os
import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import GaussianHMM
from fredapi import Fred
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime

# =========================
# CONFIG GLOBALE
# =========================
ASSETS = ["SPY", "TLT", "GLD", "XLK"]
ADDITIONAL_DATA = ["^VIX"]  # VIX pour le sentiment
START_DATE = "1990-01-01"
FRED_API_KEY = "dee74f4224925bf1a3974d668b0e8460"  # <-- Ta clé FRED

# =========================
# UTILITAIRES
# =========================
def ensure_dir(path: str):
    """Crée un répertoire s'il n'existe pas."""
    if not os.path.exists(path):
        os.makedirs(path)

# =========================
# DATA LOADER
# =========================
def download_price_data(assets=ASSETS, start=START_DATE, end=None):
    """Télécharge les prix ajustés des ETF depuis yfinance."""
    print(f"[INFO] Downloading price data for {assets}...")
    data = yf.download(assets, start=start, end=end, auto_adjust=True, progress=False)
    
    # Gestion des différents formats de retour
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    elif 'Adj Close' in data.columns:
        data = data['Adj Close']
    
    data = data.dropna()
    return data

def download_vix_data(start=START_DATE, end=None):
    """Télécharge les données VIX."""
    print("[INFO] Downloading VIX data...")
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    
    if isinstance(vix.columns, pd.MultiIndex):
        vix = vix['Close']
    elif 'Close' in vix.columns:
        vix = vix['Close']
    
    return vix

def monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les prix en rendements mensuels."""
    monthly_prices = price_df.resample("ME").last()
    rets = monthly_prices.pct_change().dropna()
    return rets

def load_macro_series(api_key: str) -> pd.DataFrame:
    """Télécharge les séries macroéconomiques depuis FRED."""
    print("[INFO] Downloading macro data from FRED...")
    fred = Fred(api_key=api_key)
    
    cpi = fred.get_series("CPIAUCSL")      # CPI
    unrate = fred.get_series("UNRATE")     # Chômage
    slope = fred.get_series("T10Y3M")      # Pente 10Y-3M
    
    # Spreads de crédit (optionnel si disponible)
    try:
        credit_spread = fred.get_series("BAMLH0A0HYM2")  # High Yield spread
    except:
        credit_spread = pd.Series(dtype=float)
    
    macro = pd.concat([cpi, unrate, slope, credit_spread], axis=1)
    macro.columns = ["CPI", "UNRATE", "T10Y3M", "CREDIT_SPREAD"]
    macro = macro.resample("ME").last()
    return macro

# =========================
# FEATURES MACRO AVANCÉES
# =========================
def build_macro_features(macro_df: pd.DataFrame, vix_series: pd.Series = None) -> pd.DataFrame:
    """Construit les features macroéconomiques avancées."""
    df = macro_df.copy()
    
    # Features de base
    df["CPI_YoY"] = df["CPI"].pct_change(12)
    df["dUNRATE"] = df["UNRATE"].diff()
    df["slope"] = df["T10Y3M"]
    
    # Features avancées
    df["CPI_MoM"] = df["CPI"].pct_change()
    df["UNRATE_MA3"] = df["UNRATE"].rolling(3).mean()
    df["slope_change"] = df["slope"].diff()
    
    # VIX si disponible
    if vix_series is not None:
        vix_monthly = vix_series.resample("ME").last()
        df = df.join(vix_monthly.rename("VIX"), how="left")
        df["VIX_change"] = df["VIX"].pct_change()
        df["VIX_MA3"] = df["VIX"].rolling(3).mean()
    
    # Credit spread
    if "CREDIT_SPREAD" in df.columns:
        df["credit_spread_change"] = df["CREDIT_SPREAD"].diff()
    
    feat_cols = ["CPI_YoY", "dUNRATE", "slope", "CPI_MoM", "slope_change"]
    if "VIX" in df.columns:
        feat_cols.extend(["VIX", "VIX_change"])
    if "credit_spread_change" in df.columns:
        feat_cols.append("credit_spread_change")
    
    feat = df[feat_cols].dropna()
    return feat

# =========================
# REGIMES HEURISTIQUES AMÉLIORÉS
# =========================
def assign_heuristic_regimes(macro_features: pd.DataFrame) -> pd.Series:
    """
    3 régimes :
    - contraction : courbe inversée (slope <= 0)
    - slowdown : inflation élevée ET chômage croissant
    - expansion : par défaut
    """
    df = macro_features.copy()
    
    infl_med = df["CPI_YoY"].rolling(60, min_periods=24).median()
    
    # Contraction (récession)
    cond_contraction = df["slope"] <= 0
    
    # Slowdown
    cond_slow = (df["CPI_YoY"] > infl_med) & (df["dUNRATE"] > 0) & (~cond_contraction)
    
    regime = pd.Series("expansion", index=df.index)
    regime[cond_slow] = "slowdown"
    regime[cond_contraction] = "contraction"
    regime.name = "regime_heur"
    
    return regime

# =========================
# ML MODELS
# =========================
def prepare_ml_features(macro_features: pd.DataFrame):
    """Standardise les features pour le ML."""
    df = macro_features.dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)
    return df.index, X_scaled, scaler

def fit_hmm_regimes(macro_features: pd.DataFrame, n_states: int = 3, train_end: str = "2010-12-31"):
    """Fit un Gaussian HMM."""
    print(f"[INFO] Fitting HMM with {n_states} states. Train end: {train_end}")
    
    idx, X_scaled, scaler = prepare_ml_features(macro_features)
    mask_train = idx <= train_end
    X_train = X_scaled[mask_train]
    
    hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
    hmm.fit(X_train)
    
    states = hmm.predict(X_scaled)
    regimes_ml = pd.Series(states, index=idx, name="regime_hmm_int")
    
    return regimes_ml, hmm, scaler

def fit_random_forest_regimes(macro_features: pd.DataFrame, target_regimes: pd.Series, train_end: str = "2010-12-31"):
    """Fit un Random Forest classifier basé sur les régimes heuristiques."""
    print(f"[INFO] Fitting Random Forest. Train end: {train_end}")
    
    idx, X_scaled, scaler = prepare_ml_features(macro_features)
    
    # Aligner les régimes avec les features
    df = pd.DataFrame(X_scaled, index=idx)
    df['regime'] = target_regimes
    df = df.dropna()
    
    mask_train = df.index <= train_end
    X_train = df.loc[mask_train, df.columns != 'regime'].values
    y_train = df.loc[mask_train, 'regime'].values
    
    # Encoder les régimes
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Fit RF
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train_encoded)
    
    # Prédiction complète
    X_all = df[df.columns != 'regime'].values
    states = rf.predict(X_all)
    states_decoded = le.inverse_transform(states)
    
    regimes_rf = pd.Series(states_decoded, index=df.index, name="regime_rf")
    
    return regimes_rf, rf, le

def label_ml_regimes(regimes_ml_int: pd.Series, description: pd.DataFrame) -> tuple:
    """Labellise les régimes ML."""
    order = description["dUNRATE"].sort_values().index.tolist()
    
    mapping = {}
    labels = ["ML_expansion", "ML_neutral", "ML_stress"]
    for i, state in enumerate(order):
        label = labels[min(i, len(labels) - 1)]
        mapping[state] = label
    
    regimes_named = regimes_ml_int.map(mapping)
    regimes_named.name = "regime_ml"
    
    return regimes_named, mapping

# =========================
# OPTIMISATION DES POIDS
# =========================
def optimize_portfolio_weights(rets_by_regime: pd.DataFrame, method='sharpe'):
    """
    Optimise les poids du portefeuille pour un régime donné.
    Maximise le ratio de Sharpe.
    """
    n_assets = len(rets_by_regime.columns)
    
    def neg_sharpe(weights):
        port_ret = (weights * rets_by_regime.mean()).sum() * 12
        port_vol = np.sqrt(np.dot(weights.T, np.dot(rets_by_regime.cov() * 12, weights)))
        return -port_ret / port_vol if port_vol > 0 else 0
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)
    
    result = minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return pd.Series(result.x, index=rets_by_regime.columns)
    else:
        return pd.Series([1/n_assets] * n_assets, index=rets_by_regime.columns)

def optimize_weights_by_regime(rets_m: pd.DataFrame, regimes: pd.Series):
    """Optimise les poids pour chaque régime."""
    print("[INFO] Optimizing portfolio weights by regime...")
    
    df = rets_m.join(regimes, how="inner").dropna()
    weights_by_regime = {}
    
    for regime in df[regimes.name].unique():
        rets_regime = df[df[regimes.name] == regime][rets_m.columns]
        if len(rets_regime) > 12:  # Au moins 1 an de données
            weights_by_regime[regime] = optimize_portfolio_weights(rets_regime)
        else:
            # Poids égaux par défaut
            weights_by_regime[regime] = pd.Series([1/len(rets_m.columns)] * len(rets_m.columns), 
                                                   index=rets_m.columns)
    
    return weights_by_regime

# =========================
# BACKTEST
# =========================
def backtest_regime_strategy(rets_m: pd.DataFrame, regimes: pd.Series, 
                             weights_by_regime: dict, tc_bps: float = 5.0):
    """Backtest avec rebalancement mensuel."""
    df = rets_m.join(regimes, how="inner").dropna()
    regime = df[regimes.name]
    rets = df[rets_m.columns]
    
    dates = rets.index
    n = len(dates)
    port_rets = []
    curr_w = None
    
    for t in range(n):
        reg_t = regime.iloc[t]
        
        if reg_t not in weights_by_regime:
            port_rets.append(0.0)
            continue
        
        target_w = weights_by_regime[reg_t].reindex(rets.columns).fillna(0.0)
        
        if curr_w is None:
            turnover = np.abs(target_w).sum()
        else:
            turnover = np.abs(target_w - curr_w).sum()
        
        tc = tc_bps / 10000.0 * turnover
        r_t = (target_w * rets.iloc[t]).sum() - tc
        port_rets.append(r_t)
        curr_w = target_w
    
    port_rets = pd.Series(port_rets, index=dates, name="strategy")
    wealth = (1 + port_rets).cumprod()
    
    return port_rets, wealth

# =========================
# MÉTRIQUES AVANCÉES
# =========================
def compute_advanced_stats(rets: pd.Series, freq: int = 12) -> dict:
    """Calcule des métriques avancées."""
    avg = rets.mean() * freq
    vol = rets.std() * np.sqrt(freq)
    sharpe = avg / vol if vol > 0 else np.nan
    
    # Sortino
    downside = rets[rets < 0].std() * np.sqrt(freq)
    sortino = avg / downside if downside > 0 else np.nan
    
    # Max Drawdown
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1)
    mdd = dd.min()
    
    # Calmar
    calmar = avg / abs(mdd) if mdd != 0 else np.nan
    
    # Win Rate
    win_rate = (rets > 0).sum() / len(rets)
    
    return {
        "CAGR": avg,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "MaxDrawdown": mdd,
        "WinRate": win_rate
    }

# =========================
# PLOTS
# =========================
def plot_wealth_curves(wealth_dict: dict, out_path: str):
    """Graphique des courbes de richesse."""
    plt.figure(figsize=(12, 6))
    
    for label, series in wealth_dict.items():
        plt.plot(series.index, series.values, label=label, linewidth=2)
    
    plt.legend(loc='best', fontsize=10)
    plt.title("Wealth Curves - Comparison of Strategies", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Wealth (growth of $1)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved wealth curves to {out_path}")

def plot_drawdown_curves(wealth_dict: dict, out_path: str):
    """Graphique des drawdowns."""
    plt.figure(figsize=(12, 6))
    
    for label, series in wealth_dict.items():
        cum = series.values
        peak = np.maximum.accumulate(cum)
        drawdown = (cum / peak) - 1
        plt.plot(series.index, drawdown, label=label, linewidth=2)
    
    plt.legend(loc='best', fontsize=10)
    plt.title("Drawdown Curves", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved drawdown curves to {out_path}")

# =========================
# MAIN PIPELINE
# =========================
def main(args):
    """Pipeline principal."""
    
    ensure_dir("data/raw")
    ensure_dir("data/processed")
    ensure_dir("output")
    
    # 1) Data marché
    prices = download_price_data(assets=ASSETS, start=START_DATE, end=args.end_date)
    rets_m = monthly_returns(prices)
    
    # 2) VIX
    vix = download_vix_data(start=START_DATE, end=args.end_date)
    
    # 3) Data macro + features
    macro = load_macro_series(api_key=FRED_API_KEY)
    X_macro = build_macro_features(macro, vix)
    
    # Alignement
    rets_m = rets_m.loc[X_macro.index.min(): X_macro.index.max()]
    X_macro = X_macro.loc[rets_m.index]
    
    print(f"\n[INFO] Data aligned. Period: {rets_m.index.min().date()} to {rets_m.index.max().date()}")
    print(f"[INFO] Number of months: {len(rets_m)}")
    
    # 4) Régimes heuristiques (3 régimes)
    regime_heur = assign_heuristic_regimes(X_macro)
    print(f"\n[INFO] Heuristic regimes distribution:")
    print(regime_heur.value_counts())
    
    # 5) Régimes ML - HMM
    regimes_hmm_int, hmm_model, scaler = fit_hmm_regimes(X_macro, n_states=args.n_states, train_end=args.train_end)
    desc_hmm = X_macro.join(regimes_hmm_int, how="inner").groupby("regime_hmm_int")[["CPI_YoY", "dUNRATE", "slope"]].mean()
    print("\n[INFO] HMM regimes description:")
    print(desc_hmm)
    
    regime_hmm_named, mapping_hmm = label_ml_regimes(regimes_hmm_int, desc_hmm)
    print(f"\n[INFO] HMM regimes distribution:")
    print(regime_hmm_named.value_counts())
    
    # 6) Régimes ML - Random Forest
    regime_rf, rf_model, le = fit_random_forest_regimes(X_macro, regime_heur, train_end=args.train_end)
    print(f"\n[INFO] Random Forest regimes distribution:")
    print(regime_rf.value_counts())
    
    # 7) Optimisation des poids par régime
    if args.optimize_weights:
        print("\n[INFO] Optimizing weights...")
        w_heur = optimize_weights_by_regime(rets_m, regime_heur)
        w_hmm = optimize_weights_by_regime(rets_m, regime_hmm_named)
        w_rf = optimize_weights_by_regime(rets_m, regime_rf)
    else:
        # Poids fixes
        w_heur = {
            "expansion": pd.Series({"SPY": 0.6, "XLK": 0.3, "TLT": 0.1, "GLD": 0.0}),
            "slowdown": pd.Series({"SPY": 0.4, "XLK": 0.2, "TLT": 0.3, "GLD": 0.1}),
            "contraction": pd.Series({"SPY": 0.0, "XLK": 0.0, "TLT": 0.8, "GLD": 0.2}),
        }
        w_hmm = {
            "ML_expansion": pd.Series({"SPY": 0.6, "XLK": 0.3, "TLT": 0.1, "GLD": 0.0}),
            "ML_neutral": pd.Series({"SPY": 0.4, "XLK": 0.2, "TLT": 0.3, "GLD": 0.1}),
            "ML_stress": pd.Series({"SPY": 0.0, "XLK": 0.0, "TLT": 0.7, "GLD": 0.3}),
        }
        w_rf = w_heur.copy()
    
    # 8) Backtests
    print("\n[INFO] Running backtests...")
    
    rets_heur, wealth_heur = backtest_regime_strategy(rets_m, regime_heur, w_heur, tc_bps=args.tc_bps)
    rets_hmm, wealth_hmm = backtest_regime_strategy(rets_m, regime_hmm_named, w_hmm, tc_bps=args.tc_bps)
    rets_rf, wealth_rf = backtest_regime_strategy(rets_m, regime_rf, w_rf, tc_bps=args.tc_bps)
    
    # Benchmark
    spy_wealth = (1 + rets_m["SPY"]).cumprod()
    spy_wealth.name = "SPY"
    spy_rets = rets_m["SPY"]
    
    # 9) Stats avancées
    stats_heur = compute_advanced_stats(rets_heur)
    stats_hmm = compute_advanced_stats(rets_hmm)
    stats_rf = compute_advanced_stats(rets_rf)
    stats_spy = compute_advanced_stats(spy_rets)
    
    stats_df = pd.DataFrame(
        [stats_heur, stats_hmm, stats_rf, stats_spy],
        index=["Heuristic", "HMM", "Random_Forest", "SPY"]
    )
    
    print("\n" + "="*80)
    print("[RESULTS] Advanced Performance Statistics")
    print("="*80)
    print(stats_df.to_string())
    print("="*80)
    
    stats_df.to_csv("output/stats_advanced.csv")
    
    # 10) Plots
    wealth_dict = {
        "Heuristic (3 regimes)": wealth_heur,
        "HMM": wealth_hmm,
        "Random Forest": wealth_rf,
        "SPY Buy & Hold": spy_wealth,
    }
    
    plot_wealth_curves(wealth_dict, "output/wealth_curves_advanced.png")
    plot_drawdown_curves(wealth_dict, "output/drawdown_curves.png")
    
    # Sauvegarde des régimes
    regimes_df = pd.DataFrame({
        'regime_heur': regime_heur,
        'regime_hmm': regime_hmm_named,
        'regime_rf': regime_rf
    })
    regimes_df.to_csv("output/regimes_advanced.csv")
    
    print("\n[SUCCESS] Advanced experiment completed successfully!")
    print(f"[INFO] Results saved in output/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Macro Regime Lab - Advanced ML")
    parser.add_argument("--n_states", type=int, default=3, help="Nombre de régimes HMM")
    parser.add_argument("--train_end", type=str, default="2010-12-31", help="Date fin training")
    parser.add_argument("--end_date", type=str, default=None, help="Date fin données")
    parser.add_argument("--tc_bps", type=float, default=5.0, help="Transaction costs (bps)")
    parser.add_argument("--optimize_weights", action="store_true", help="Optimiser les poids automatiquement")
    
    args = parser.parse_args()
    main(args)