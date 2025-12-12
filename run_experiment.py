#!/usr/bin/env python3
"""
run_experiment.py
Macro Regime & Factor Rotation Lab with ML regime detection (HMM).

Fonctionnalités :
- Télécharge les prix d'ETF (SPY, TLT, GLD, XLK) via yfinance
- Télécharge les séries macro (CPI, UNRATE, T10Y3M) via FRED
- Construit des features macro (CPI_YoY, dUNRATE, slope)
- Définit des régimes heuristiques par règles
- Apprend des régimes via un Gaussian HMM (ML)
- Backteste :
  * Stratégie baseline (régimes heuristiques)
  * Stratégie ML (régimes HMM)
  * Benchmark SPY buy & hold
- Calcule des métriques de performance
- Sauvegarde les courbes de richesse dans output/
"""

import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from fredapi import Fred
import matplotlib.pyplot as plt

# =========================
# CONFIG GLOBALE
# =========================
ASSETS = ["SPY", "TLT", "GLD", "XLK"]
START_DATE = "1990-01-01"

# Option 1: Mettre ta clé FRED directement ici
FRED_API_KEY = "dee74f4224925bf1a3974d668b0e8460 "  # <-- remplace par ta clé FRED

# Option 2: Utiliser une variable d'environnement (recommandé)
# FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")

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
def download_price_data(start=START_DATE, end=None):
    """Télécharge les prix ajustés des ETF depuis yfinance."""
    print("[INFO] Downloading price data from yfinance...")
    data = yf.download(ASSETS, start=start, end=end, auto_adjust=True)
    
    # Gestion des différents formats de retour de yfinance
    if isinstance(data.columns, pd.MultiIndex):
        # Si MultiIndex, prend 'Close' qui contient les prix ajustés avec auto_adjust=True
        data = data['Close']
    elif 'Adj Close' in data.columns:
        data = data['Adj Close']
    # Sinon data contient déjà directement les prix
    
    data = data.dropna()
    return data

def monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les prix en rendements mensuels."""
    monthly_prices = price_df.resample("M").last()
    rets = monthly_prices.pct_change().dropna()
    return rets

def load_macro_series(api_key: str) -> pd.DataFrame:
    """Télécharge les séries macroéconomiques depuis FRED."""
    print("[INFO] Downloading macro data from FRED...")
    fred = Fred(api_key=api_key)
    
    cpi = fred.get_series("CPIAUCSL")      # CPI niveau
    unrate = fred.get_series("UNRATE")     # chômage %
    slope = fred.get_series("T10Y3M")      # 10Y - 3M
    
    macro = pd.concat([cpi, unrate, slope], axis=1)
    macro.columns = ["CPI", "UNRATE", "T10Y3M"]
    macro = macro.resample("M").last()
    return macro

# =========================
# FEATURES MACRO
# =========================
def build_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Construit les features macroéconomiques pour l'analyse."""
    df = macro_df.copy()
    
    # Inflation en taux de croissance annuel
    df["CPI_YoY"] = df["CPI"].pct_change(12)
    
    # Variation mensuelle du chômage
    df["dUNRATE"] = df["UNRATE"].diff()
    
    # Slope : déjà en %
    df["slope"] = df["T10Y3M"]
    
    feat = df[["CPI_YoY", "dUNRATE", "slope"]].dropna()
    return feat

# =========================
# REGIMES HEURISTIQUES
# =========================
def assign_heuristic_regimes(macro_features: pd.DataFrame) -> pd.Series:
    """
    Règles simples :
    - Par défaut: 'expansion'
    - Si CPI_YoY > médiane roulante 5 ans ET dUNRATE > 0 => 'slowdown'
    """
    df = macro_features.copy()
    
    infl_med = df["CPI_YoY"].rolling(60, min_periods=24).median()
    cond_slow = (df["CPI_YoY"] > infl_med) & (df["dUNRATE"] > 0)
    
    regime = pd.Series("expansion", index=df.index)
    regime[cond_slow] = "slowdown"
    regime.name = "regime_heur"
    
    return regime

# =========================
# REGIMES ML (HMM)
# =========================
def prepare_ml_features(macro_features: pd.DataFrame):
    """Standardise les features pour le ML."""
    df = macro_features.dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)
    return df.index, X_scaled, scaler

def fit_hmm_regimes(macro_features: pd.DataFrame,
                    n_states: int = 3,
                    train_end: str = "2010-12-31"):
    """
    Fit un Gaussian HMM sur les features macro, en utilisant uniquement les données
    jusqu'à train_end, puis prédit les états pour toute la période.
    """
    print(f"[INFO] Fitting HMM with {n_states} states. Train end: {train_end}")
    
    idx, X_scaled, scaler = prepare_ml_features(macro_features)
    
    # Split train/test
    mask_train = idx <= train_end
    X_train = X_scaled[mask_train]
    
    # Fit HMM
    hmm = GaussianHMM(n_components=n_states,
                      covariance_type="full",
                      n_iter=200,
                      random_state=42)
    hmm.fit(X_train)
    
    # Prédiction sur toute la période
    states = hmm.predict(X_scaled)
    regimes_ml = pd.Series(states, index=idx, name="regime_ml_int")
    
    return regimes_ml, hmm, scaler

def describe_ml_regimes(macro_features: pd.DataFrame,
                        regimes_ml_int: pd.Series) -> pd.DataFrame:
    """Calcule les statistiques moyennes par régime ML."""
    df = macro_features.join(regimes_ml_int, how="inner")
    desc = df.groupby("regime_ml_int")[["CPI_YoY", "dUNRATE", "slope"]].mean()
    return desc

def label_ml_regimes(regimes_ml_int: pd.Series,
                     description: pd.DataFrame) -> tuple:
    """
    Labellise les régimes ML selon leurs caractéristiques macro.
    Mapping basé sur dUNRATE : faible -> expansion, moyen -> neutral, fort -> stress
    """
    # Ordonne par dUNRATE croissant
    order = description["dUNRATE"].sort_values().index.tolist()
    
    # Crée le mapping
    mapping = {}
    labels = ["ML_expansion", "ML_neutral", "ML_stress"]
    for i, state in enumerate(order):
        label = labels[min(i, len(labels) - 1)]
        mapping[state] = label
    
    regimes_named = regimes_ml_int.map(mapping)
    regimes_named.name = "regime_ml"
    
    return regimes_named, mapping

# =========================
# BACKTEST
# =========================
def backtest_regime_strategy(rets_m: pd.DataFrame,
                             regimes: pd.Series,
                             weights_by_regime: dict,
                             tc_bps: float = 5.0):
    """
    Backtest simple avec rebalancement mensuel et coûts de transaction.
    
    Args:
        rets_m: DataFrame des rendements mensuels
        regimes: Series des régimes
        weights_by_regime: dict {regime: Series(weights)}
        tc_bps: coûts de transaction en basis points
    
    Returns:
        port_rets: Series des rendements du portefeuille
        wealth: Series de la courbe de richesse
    """
    df = rets_m.join(regimes, how="inner").dropna()
    regime = df[regimes.name]
    rets = df[rets_m.columns]
    
    dates = rets.index
    n = len(dates)
    port_rets = []
    curr_w = None
    
    for t in range(n):
        reg_t = regime.iloc[t]
        
        # Si régime pas défini, reste cash
        if reg_t not in weights_by_regime:
            port_rets.append(0.0)
            continue
        
        target_w = weights_by_regime[reg_t].reindex(rets.columns).fillna(0.0)
        
        # Calcul du turnover et des coûts
        if curr_w is None:
            turnover = np.abs(target_w).sum()
        else:
            turnover = np.abs(target_w - curr_w).sum()
        
        tc = tc_bps / 10000.0 * turnover
        
        # Rendement net
        r_t = (target_w * rets.iloc[t]).sum() - tc
        port_rets.append(r_t)
        curr_w = target_w
    
    port_rets = pd.Series(port_rets, index=dates, name="strategy")
    wealth = (1 + port_rets).cumprod()
    
    return port_rets, wealth

# =========================
# METRICS
# =========================
def compute_stats(rets: pd.Series, freq: int = 12) -> dict:
    """Calcule les statistiques de performance."""
    avg = rets.mean() * freq
    vol = rets.std() * np.sqrt(freq)
    sharpe = avg / vol if vol > 0 else np.nan
    
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1)
    mdd = dd.min()
    
    return {
        "CAGR": avg,
        "Vol": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": mdd,
    }

# =========================
# PLOTS
# =========================
def plot_wealth_curves(wealth_dict: dict, out_path: str):
    """Génère et sauvegarde le graphique des courbes de richesse."""
    plt.figure(figsize=(10, 6))
    
    for label, series in wealth_dict.items():
        plt.plot(series.index, series.values, label=label, linewidth=2)
    
    plt.legend(loc='best')
    plt.title("Wealth Curves - Macro Regime Strategies vs SPY", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Wealth (growth of $1)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved wealth curves to {out_path}")

# =========================
# MAIN PIPELINE
# =========================
def main(args):
    """Pipeline principal du projet."""
    
    # Crée les répertoires
    ensure_dir("data/raw")
    ensure_dir("data/processed")
    ensure_dir("output")
    
    # 1) Data marché
    prices = download_price_data(start=START_DATE, end=args.end_date)
    rets_m = monthly_returns(prices)
    
    # 2) Data macro + features
    macro = load_macro_series(api_key=FRED_API_KEY)
    X_macro = build_macro_features(macro)
    
    # Alignement des périodes
    rets_m = rets_m.loc[X_macro.index.min(): X_macro.index.max()]
    X_macro = X_macro.loc[rets_m.index]
    
    print(f"[INFO] Data aligned. Period: {rets_m.index.min()} to {rets_m.index.max()}")
    print(f"[INFO] Number of months: {len(rets_m)}")
    
    # 3) Régimes heuristiques
    regime_heur = assign_heuristic_regimes(X_macro)
    print(f"\n[INFO] Heuristic regimes distribution:")
    print(regime_heur.value_counts())
    
    # 4) Régimes ML (HMM)
    regimes_ml_int, hmm_model, scaler = fit_hmm_regimes(
        X_macro,
        n_states=args.n_states,
        train_end=args.train_end
    )
    
    desc_ml = describe_ml_regimes(X_macro, regimes_ml_int)
    print("\n[INFO] Description of ML regimes (means):")
    print(desc_ml)
    
    regime_ml_named, mapping = label_ml_regimes(regimes_ml_int, desc_ml)
    print("\n[INFO] Mapping int -> label:")
    print(mapping)
    print(f"\n[INFO] ML regimes distribution:")
    print(regime_ml_named.value_counts())
    
    # 5) Définir les poids par régime
    # Baseline heuristique
    w_heur = {
        "expansion": pd.Series({"SPY": 0.6, "XLK": 0.4, "TLT": 0.0, "GLD": 0.0}),
        "slowdown": pd.Series({"SPY": 0.4, "XLK": 0.2, "TLT": 0.3, "GLD": 0.1}),
    }
    
    # ML régimes (adapté selon les régimes découverts)
    w_ml = {
        "ML_expansion": pd.Series({"SPY": 0.6, "XLK": 0.3, "TLT": 0.1, "GLD": 0.0}),
        "ML_neutral": pd.Series({"SPY": 0.4, "XLK": 0.2, "TLT": 0.3, "GLD": 0.1}),
        "ML_stress": pd.Series({"SPY": 0.0, "XLK": 0.0, "TLT": 0.7, "GLD": 0.3}),
    }
    
    # 6) Backtests
    print("\n[INFO] Running backtests...")
    
    rets_heur, wealth_heur = backtest_regime_strategy(
        rets_m, regime_heur, w_heur, tc_bps=args.tc_bps
    )
    
    rets_ml, wealth_ml = backtest_regime_strategy(
        rets_m, regime_ml_named, w_ml, tc_bps=args.tc_bps
    )
    
    # Benchmark SPY
    spy_wealth = (1 + rets_m["SPY"]).cumprod()
    spy_wealth.name = "SPY_buyhold"
    spy_rets = rets_m["SPY"]
    spy_rets.name = "SPY"
    
    # 7) Stats
    stats_heur = compute_stats(rets_heur)
    stats_ml = compute_stats(rets_ml)
    stats_spy = compute_stats(spy_rets)
    
    stats_df = pd.DataFrame(
        [stats_heur, stats_ml, stats_spy],
        index=["Heuristic", "ML_HMM", "SPY"]
    )
    
    print("\n" + "="*60)
    print("[RESULTS] Performance Statistics")
    print("="*60)
    print(stats_df.to_string())
    print("="*60)
    
    stats_df.to_csv("output/stats.csv")
    print("\n[INFO] Saved stats to output/stats.csv")
    
    # 8) Plots
    wealth_dict = {
        "Heuristic Strategy": wealth_heur,
        "ML HMM Strategy": wealth_ml,
        "SPY Buy & Hold": spy_wealth,
    }
    plot_wealth_curves(wealth_dict, "output/wealth_curves.png")
    
    # Sauvegarde des régimes pour analyse
    regimes_df = pd.DataFrame({
        'regime_heur': regime_heur,
        'regime_ml': regime_ml_named,
        'regime_ml_int': regimes_ml_int
    })
    regimes_df.to_csv("output/regimes.csv")
    print("[INFO] Saved regimes to output/regimes.csv")
    
    print("\n[SUCCESS] Experiment completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Macro Regime & Factor Rotation Lab with ML (HMM)"
    )
    parser.add_argument("--n_states", type=int, default=3,
                        help="Nombre de régimes pour le HMM.")
    parser.add_argument("--train_end", type=str, default="2010-12-31",
                        help="Date de fin de période d'entraînement HMM (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default=None,
                        help="Date de fin de téléchargement des prix (YYYY-MM-DD ou None).")
    parser.add_argument("--tc_bps", type=float, default=5.0,
                        help="Frais de transaction en basis points par rebalancement.")
    
    args = parser.parse_args()
    main(args)