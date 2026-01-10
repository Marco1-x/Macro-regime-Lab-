#!/usr/bin/env python3
"""
app_simple.py
Version simplifiée et robuste de l'application Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from fredapi import Fred
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Configuration
st.set_page_config(page_title="Macro Regime Lab", page_icon="📊", layout="wide")

# =========================
# FONCTIONS
# =========================

@st.cache_data(ttl=3600)
def download_price_data(assets, start, end):
    """Télécharge les données de prix."""
    try:
        data = yf.download(assets, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']
        return data.dropna()
    except Exception as e:
        st.error(f"Erreur téléchargement prix : {e}")
        return None

@st.cache_data(ttl=3600)
def load_macro_series(api_key, start, end):
    """Télécharge données macro."""
    try:
        fred = Fred(api_key=api_key)
        cpi = fred.get_series("CPIAUCSL", observation_start=start, observation_end=end)
        unrate = fred.get_series("UNRATE", observation_start=start, observation_end=end)
        slope = fred.get_series("T10Y3M", observation_start=start, observation_end=end)
        
        macro = pd.concat([cpi, unrate, slope], axis=1)
        macro.columns = ["CPI", "UNRATE", "T10Y3M"]
        return macro.resample("ME").last()
    except Exception as e:
        st.error(f"Erreur téléchargement macro : {e}")
        return None

def monthly_returns(price_df):
    """Calcule les rendements mensuels."""
    monthly_prices = price_df.resample("ME").last()
    return monthly_prices.pct_change().dropna()

def build_macro_features(macro_df):
    """Construit les features macro."""
    df = macro_df.copy()
    df["CPI_YoY"] = df["CPI"].pct_change(12)
    df["dUNRATE"] = df["UNRATE"].diff()
    df["slope"] = df["T10Y3M"]
    return df[["CPI_YoY", "dUNRATE", "slope"]].dropna()

def assign_heuristic_regimes(macro_features):
    """Régimes heuristiques."""
    df = macro_features.copy()
    infl_med = df["CPI_YoY"].rolling(60, min_periods=24).median()
    
    cond_contraction = df["slope"] <= 0
    cond_slow = (df["CPI_YoY"] > infl_med) & (df["dUNRATE"] > 0) & (~cond_contraction)
    
    regime = pd.Series("expansion", index=df.index)
    regime[cond_slow] = "slowdown"
    regime[cond_contraction] = "contraction"
    regime.name = "regime_heur"
    return regime

def backtest_strategy(rets_m, regimes, weights_dict, tc_bps=5.0):
    """Backtest simple."""
    df = rets_m.join(regimes, how="inner").dropna()
    regime = df[regimes.name]
    rets = df[rets_m.columns]
    
    port_rets = []
    curr_w = None
    
    for t in range(len(rets)):
        reg_t = regime.iloc[t]
        if reg_t not in weights_dict:
            port_rets.append(0.0)
            continue
        
        target_w = weights_dict[reg_t].reindex(rets.columns).fillna(0.0)
        
        if curr_w is None:
            turnover = np.abs(target_w).sum()
        else:
            turnover = np.abs(target_w - curr_w).sum()
        
        tc = tc_bps / 10000.0 * turnover
        r_t = (target_w * rets.iloc[t]).sum() - tc
        port_rets.append(r_t)
        curr_w = target_w
    
    port_rets = pd.Series(port_rets, index=rets.index)
    wealth = (1 + port_rets).cumprod()
    return port_rets, wealth

def compute_stats(rets, freq=12):
    """Statistiques."""
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
        "MaxDrawdown": mdd
    }

# =========================
# INTERFACE
# =========================

def main():
    st.title("📊 Macro Regime & Factor Rotation Lab")
    st.markdown("### Application Interactive de Trading Quantitatif")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    fred_key = st.sidebar.text_input("🔑 Clé FRED API", type="password")
    
    st.sidebar.subheader("📈 Actifs")
    available_assets = ["SPY", "TLT", "GLD", "XLK", "QQQ", "IWM"]
    selected_assets = st.sidebar.multiselect(
        "ETFs",
        available_assets,
        default=["SPY", "TLT", "GLD", "XLK"]
    )
    
    st.sidebar.subheader("📅 Période")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Début", value=datetime(2000, 1, 1))
    end_date = col2.date_input("Fin", value=datetime.now())
    
    tc_bps = st.sidebar.slider("Coûts transaction (bps)", 0.0, 20.0, 5.0)
    
    run_button = st.sidebar.button("🚀 LANCER", type="primary")
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["📊 Résultats", "📈 Graphiques", "💾 Export"])
    
    if run_button:
        if len(selected_assets) < 2:
            st.error("⚠️ Sélectionnez au moins 2 actifs")
            return
        
        if not fred_key or len(fred_key) != 32:
            st.error("⚠️ Clé FRED invalide (32 caractères)")
            return
        
        with st.spinner("🔄 Chargement des données..."):
            # Téléchargement
            prices = download_price_data(selected_assets, start_date, end_date)
            if prices is None:
                return
            
            rets_m = monthly_returns(prices)
            
            macro = load_macro_series(fred_key, start_date, end_date)
            if macro is None:
                return
            
            X_macro = build_macro_features(macro)
            
            # Alignement
            rets_m = rets_m.loc[X_macro.index.min(): X_macro.index.max()]
            X_macro = X_macro.loc[rets_m.index]
            
            st.success(f"✅ {len(rets_m)} mois chargés")
        
        with st.spinner("🤖 Calcul..."):
            # Régimes
            regime_heur = assign_heuristic_regimes(X_macro)
            
            # Poids fixes
            w_heur = {
                "expansion": pd.Series({a: 0.6/len(selected_assets) if a == selected_assets[0] else 0.4/max(1, len(selected_assets)-1) for a in selected_assets}),
                "slowdown": pd.Series({a: 1/len(selected_assets) for a in selected_assets}),
                "contraction": pd.Series({a: 1/len(selected_assets) for a in selected_assets}),
            }
            
            # Backtest
            rets_heur, wealth_heur = backtest_strategy(rets_m, regime_heur, w_heur, tc_bps)
            
            # Benchmark
            spy_rets = rets_m[selected_assets[0]]
            spy_wealth = (1 + spy_rets).cumprod()
            
            # Stats
            stats_heur = compute_stats(rets_heur)
            stats_spy = compute_stats(spy_rets)
        
        st.success("✅ Analyse terminée !")
        
        # TAB 1: Résultats
        with tab1:
            st.header("📊 Performance")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stratégie Heuristique", f"{stats_heur['CAGR']*100:.2f}%", 
                         f"Sharpe: {stats_heur['Sharpe']:.2f}")
            with col2:
                st.metric("Benchmark SPY", f"{stats_spy['CAGR']*100:.2f}%",
                         f"Sharpe: {stats_spy['Sharpe']:.2f}")
            
            st.subheader("Statistiques Complètes")
            stats_df = pd.DataFrame([stats_heur, stats_spy], index=["Heuristic", "SPY"])
            st.dataframe(stats_df.style.format({
                'CAGR': '{:.2%}',
                'Vol': '{:.2%}',
                'Sharpe': '{:.2f}',
                'MaxDrawdown': '{:.2%}'
            }))
            
            st.subheader("Distribution des Régimes")
            st.write(regime_heur.value_counts())
        
        # TAB 2: Graphiques
        with tab2:
            st.header("📈 Courbes de Richesse")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wealth_heur.index, y=wealth_heur.values, 
                                    name="Heuristic", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=spy_wealth.index, y=spy_wealth.values,
                                    name="SPY", line=dict(width=2)))
            
            fig.update_layout(
                title="Growth of $1",
                xaxis_title="Date",
                yaxis_title="Wealth",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown
            st.header("📉 Drawdowns")
            fig_dd = go.Figure()
            
            cum_h = wealth_heur.values
            peak_h = np.maximum.accumulate(cum_h)
            dd_h = (cum_h / peak_h - 1) * 100
            
            cum_s = spy_wealth.values
            peak_s = np.maximum.accumulate(cum_s)
            dd_s = (cum_s / peak_s - 1) * 100
            
            fig_dd.add_trace(go.Scatter(x=wealth_heur.index, y=dd_h, name="Heuristic", fill='tozeroy'))
            fig_dd.add_trace(go.Scatter(x=spy_wealth.index, y=dd_s, name="SPY", fill='tozeroy'))
            
            fig_dd.update_layout(title="Drawdown (%)", xaxis_title="Date", yaxis_title="Drawdown (%)", height=400)
            st.plotly_chart(fig_dd, use_container_width=True)
        
        # TAB 3: Export
        with tab3:
            st.header("💾 Téléchargement")
            
            stats_df = pd.DataFrame([stats_heur, stats_spy], index=["Heuristic", "SPY"])
            csv = stats_df.to_csv().encode('utf-8')
            st.download_button("📥 Stats (CSV)", csv, "stats.csv", "text/csv")
            
            wealth_df = pd.DataFrame({"Heuristic": wealth_heur, "SPY": spy_wealth})
            csv_w = wealth_df.to_csv().encode('utf-8')
            st.download_button("📥 Wealth (CSV)", csv_w, "wealth.csv", "text/csv")

if __name__ == "__main__":
    main()