#!/usr/bin/env python3
"""
Macro Regime & Factor Rotation Lab - Dashboard
Version avec téléchargement direct Yahoo Finance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from fredapi import Fred
import requests
import warnings
warnings.filterwarnings('ignore')

# Configuration page
st.set_page_config(
    page_title="Macro Regime Lab",
    page_icon="📊",
    layout="wide"
)

# =========================
# FONCTIONS DATA
# =========================

def download_yahoo_data(symbol, start_date, end_date):
    """Télécharge les données depuis Yahoo Finance API"""
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"period1": start_ts, "period2": end_ts, "interval": "1d"}
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        if "chart" not in data or not data["chart"]["result"]:
            return None
            
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quotes = result["indicators"]["quote"][0]
        
        df = pd.DataFrame({
            "Date": pd.to_datetime(timestamps, unit='s'),
            "Close": quotes["close"]
        })
        df.set_index("Date", inplace=True)
        return df.dropna()
    except:
        return None

@st.cache_data(ttl=3600)
def download_etf_prices(symbols, start_date, end_date):
    """Télécharge les prix de plusieurs ETFs"""
    all_data = {}
    for symbol in symbols:
        df = download_yahoo_data(symbol, start_date, end_date)
        if df is not None:
            all_data[symbol] = df["Close"]
    
    if not all_data:
        return pd.DataFrame()
    return pd.DataFrame(all_data).dropna()

@st.cache_data(ttl=3600)
def download_macro_data(api_key, start_date, end_date):
    """Télécharge les données macro depuis FRED"""
    try:
        fred = Fred(api_key=api_key)
        
        cpi = fred.get_series("CPIAUCSL", observation_start=start_date, observation_end=end_date)
        unrate = fred.get_series("UNRATE", observation_start=start_date, observation_end=end_date)
        t10y3m = fred.get_series("T10Y3M", observation_start=start_date, observation_end=end_date)
        fedfunds = fred.get_series("FEDFUNDS", observation_start=start_date, observation_end=end_date)
        
        macro = pd.DataFrame({
            "CPI": cpi,
            "UNRATE": unrate,
            "T10Y3M": t10y3m,
            "FEDFUNDS": fedfunds
        })
        return macro.resample("ME").last()
    except Exception as e:
        st.error(f"Erreur FRED: {e}")
        return pd.DataFrame()

def build_features(macro_df):
    """Construit les features macro"""
    df = macro_df.copy()
    df["CPI_YoY"] = df["CPI"].pct_change(12)
    df["dUNRATE"] = df["UNRATE"].diff()
    df["slope"] = df["T10Y3M"]
    return df[["CPI_YoY", "dUNRATE", "slope"]].dropna()

def assign_regimes(features):
    """Assigne les régimes macro"""
    df = features.copy()
    infl_med = df["CPI_YoY"].rolling(60, min_periods=12).median()
    
    regime = pd.Series("Expansion", index=df.index)
    regime[df["slope"] <= 0] = "Contraction"
    regime[(df["CPI_YoY"] > infl_med) & (df["dUNRATE"] > 0) & (df["slope"] > 0)] = "Slowdown"
    regime.name = "regime"
    return regime

def monthly_returns(prices):
    """Calcule les rendements mensuels"""
    monthly = prices.resample("ME").last()
    return monthly.pct_change().dropna()

def backtest_strategy(returns, regimes, weights_dict, tc_bps=5):
    """Backtest de la stratégie"""
    df = returns.join(regimes, how="inner").dropna()
    
    port_returns = []
    current_weights = None
    
    for date in df.index:
        regime = df.loc[date, "regime"]
        target_weights = weights_dict.get(regime, {})
        
        # Calculer rendement
        ret = sum(target_weights.get(col, 0) * df.loc[date, col] 
                  for col in returns.columns if col in target_weights)
        
        # Coûts de transaction
        if current_weights:
            turnover = sum(abs(target_weights.get(col, 0) - current_weights.get(col, 0)) 
                          for col in returns.columns)
            ret -= tc_bps / 10000 * turnover
        
        port_returns.append(ret)
        current_weights = target_weights
    
    port_returns = pd.Series(port_returns, index=df.index)
    wealth = (1 + port_returns).cumprod()
    return port_returns, wealth

def compute_metrics(returns):
    """Calcule les métriques de performance"""
    annual_ret = returns.mean() * 12
    annual_vol = returns.std() * np.sqrt(12)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    
    wealth = (1 + returns).cumprod()
    drawdown = (wealth / wealth.cummax() - 1)
    max_dd = drawdown.min()
    
    return {
        "CAGR": annual_ret,
        "Volatility": annual_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": (returns > 0).mean()
    }

# =========================
# INTERFACE
# =========================

st.title("📊 Macro Regime & Factor Rotation Lab")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# API Key
fred_key = st.sidebar.text_input("🔑 FRED API Key", type="password", 
                                  help="Obtenez votre clé sur https://fred.stlouisfed.org/docs/api/api_key.html")

# Période
st.sidebar.subheader("📅 Période")
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Début", value=datetime(2005, 1, 1))
end_date = col2.date_input("Fin", value=datetime.now())

# Assets
st.sidebar.subheader("📈 Assets")
available_assets = ["SPY", "TLT", "GLD", "XLK", "QQQ", "IWM", "EFA", "VNQ"]
selected_assets = st.sidebar.multiselect("ETFs", available_assets, default=["SPY", "TLT", "GLD", "XLK"])

# Poids par régime
st.sidebar.subheader("🎯 Poids par Régime")

with st.sidebar.expander("Expansion"):
    w_exp = {}
    for asset in selected_assets:
        default = 0.4 if asset == "SPY" else (0.3 if asset == "XLK" else 0.15)
        w_exp[asset] = st.slider(f"{asset}", 0.0, 1.0, default, 0.05, key=f"exp_{asset}")

with st.sidebar.expander("Slowdown"):
    w_slow = {}
    for asset in selected_assets:
        default = 0.3 if asset in ["SPY", "TLT"] else 0.2
        w_slow[asset] = st.slider(f"{asset}", 0.0, 1.0, default, 0.05, key=f"slow_{asset}")

with st.sidebar.expander("Contraction"):
    w_cont = {}
    for asset in selected_assets:
        default = 0.5 if asset == "TLT" else (0.3 if asset == "GLD" else 0.1)
        w_cont[asset] = st.slider(f"{asset}", 0.0, 1.0, default, 0.05, key=f"cont_{asset}")

weights_by_regime = {
    "Expansion": w_exp,
    "Slowdown": w_slow,
    "Contraction": w_cont
}

# Transaction costs
tc_bps = st.sidebar.slider("💰 Transaction Costs (bps)", 0, 50, 5)

# Run button
run_analysis = st.sidebar.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)

# =========================
# MAIN CONTENT
# =========================

if run_analysis:
    if not fred_key:
        st.error("⚠️ Veuillez entrer votre clé FRED API")
    elif len(selected_assets) < 2:
        st.error("⚠️ Sélectionnez au moins 2 assets")
    else:
        with st.spinner("📥 Téléchargement des données..."):
            # Download data
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            prices = download_etf_prices(selected_assets, start_str, end_str)
            macro = download_macro_data(fred_key, start_str, end_str)
            
            if prices.empty:
                st.error("❌ Impossible de télécharger les prix ETF")
                st.stop()
            
            if macro.empty:
                st.error("❌ Impossible de télécharger les données macro")
                st.stop()
            
            st.success(f"✅ {len(prices)} jours de données téléchargés")
        
        with st.spinner("🔄 Analyse en cours..."):
            # Process data
            features = build_features(macro)
            regimes = assign_regimes(features)
            returns = monthly_returns(prices)
            
            # Align data
            common_idx = returns.index.intersection(regimes.index)
            returns = returns.loc[common_idx]
            regimes = regimes.loc[common_idx]
            
            # Backtest
            strat_returns, strat_wealth = backtest_strategy(returns, regimes, weights_by_regime, tc_bps)
            
            # Benchmark
            spy_returns = returns["SPY"] if "SPY" in returns.columns else returns.iloc[:, 0]
            spy_wealth = (1 + spy_returns).cumprod()
            
            # Metrics
            strat_metrics = compute_metrics(strat_returns)
            spy_metrics = compute_metrics(spy_returns)
        
        # =========================
        # DISPLAY RESULTS
        # =========================
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "📈 Charts", "🔍 Regimes", "📋 Details"])
        
        with tab1:
            st.header("Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategy CAGR", f"{strat_metrics['CAGR']*100:.1f}%")
            col2.metric("Strategy Sharpe", f"{strat_metrics['Sharpe']:.2f}")
            col3.metric("Max Drawdown", f"{strat_metrics['Max Drawdown']*100:.1f}%")
            col4.metric("Win Rate", f"{strat_metrics['Win Rate']*100:.1f}%")
            
            st.subheader("Strategy vs Benchmark")
            metrics_df = pd.DataFrame({
                "Strategy": strat_metrics,
                "SPY (Benchmark)": spy_metrics
            }).T
            metrics_df = metrics_df.style.format({
                "CAGR": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2%}",
                "Win Rate": "{:.2%}"
            })
            st.dataframe(metrics_df, use_container_width=True)
        
        with tab2:
            st.header("Wealth Curves")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strat_wealth.index, y=strat_wealth.values, 
                                     name="Strategy", line=dict(width=2, color="blue")))
            fig.add_trace(go.Scatter(x=spy_wealth.index, y=spy_wealth.values,
                                     name="SPY", line=dict(width=2, color="gray", dash="dash")))
            fig.update_layout(title="Growth of $1", xaxis_title="Date", yaxis_title="Value",
                             hovermode="x unified", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown
            st.subheader("Drawdown")
            strat_dd = (strat_wealth / strat_wealth.cummax() - 1) * 100
            spy_dd = (spy_wealth / spy_wealth.cummax() - 1) * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values,
                                        name="Strategy", fill="tozeroy"))
            fig_dd.add_trace(go.Scatter(x=spy_dd.index, y=spy_dd.values,
                                        name="SPY", fill="tozeroy"))
            fig_dd.update_layout(title="Drawdown (%)", xaxis_title="Date", yaxis_title="Drawdown %",
                                height=400)
            st.plotly_chart(fig_dd, use_container_width=True)
        
        with tab3:
            st.header("Regime Analysis")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Distribution")
                regime_counts = regimes.value_counts()
                fig_pie = px.pie(values=regime_counts.values, names=regime_counts.index,
                                color_discrete_map={"Expansion": "green", "Slowdown": "orange", "Contraction": "red"})
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Regime Timeline")
                regime_df = pd.DataFrame({"Date": regimes.index, "Regime": regimes.values})
                fig_timeline = px.scatter(regime_df, x="Date", y="Regime", color="Regime",
                                         color_discrete_map={"Expansion": "green", "Slowdown": "orange", "Contraction": "red"})
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            st.subheader("Current Weights")
            current_regime = regimes.iloc[-1] if len(regimes) > 0 else "Unknown"
            st.info(f"Current Regime: **{current_regime}**")
            
            current_weights = weights_by_regime.get(current_regime, {})
            if current_weights:
                fig_weights = px.bar(x=list(current_weights.keys()), y=list(current_weights.values()),
                                    labels={"x": "Asset", "y": "Weight"})
                st.plotly_chart(fig_weights, use_container_width=True)
        
        with tab4:
            st.header("Data Details")
            
            st.subheader("Monthly Returns")
            st.dataframe(returns.tail(12).style.format("{:.2%}"), use_container_width=True)
            
            st.subheader("Macro Features")
            st.dataframe(features.tail(12).style.format("{:.4f}"), use_container_width=True)
            
            # Download button
            csv = returns.to_csv()
            st.download_button("📥 Download Returns CSV", csv, "returns.csv", "text/csv")

else:
    st.info("👈 Configurez vos paramètres et cliquez sur **RUN ANALYSIS** pour commencer")
    
    st.markdown("""
    ### 🎯 Comment utiliser ce dashboard
    
    1. **Entrez votre clé FRED API** (gratuite sur [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html))
    2. **Sélectionnez la période** d'analyse
    3. **Choisissez les ETFs** à inclure
    4. **Ajustez les poids** pour chaque régime
    5. **Cliquez sur RUN ANALYSIS**
    
    ### 📊 Les 3 Régimes Macro
    
    - **🟢 Expansion**: Courbe des taux positive, inflation stable
    - **🟠 Slowdown**: Inflation haute + chômage en hausse
    - **🔴 Contraction**: Courbe des taux inversée (récession probable)
    """)
