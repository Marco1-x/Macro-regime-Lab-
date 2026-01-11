#!/usr/bin/env python3
"""
dashboard.py
Application Web Interactive pour Macro Regime & Factor Rotation Lab
Version avec téléchargement Yahoo Finance direct (sans yfinance bugué)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from fredapi import Fred as FredAPI
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import requests

# Configuration de la page
st.set_page_config(
    page_title="Macro Regime Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# FONCTIONS DATA (YAHOO DIRECT)
# =========================

def download_yahoo_data(symbol, start_date, end_date):
    """Télécharge les données depuis Yahoo Finance API directement."""
    try:
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    except:
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"period1": start_ts, "period2": end_ts, "interval": "1d"}
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        if "chart" not in data or not data["chart"]["result"]:
            return None
            
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        if not timestamps:
            return None
            
        quotes = result["indicators"]["quote"][0]
        
        df = pd.DataFrame({
            "Date": pd.to_datetime(timestamps, unit='s'),
            "Open": quotes.get("open"),
            "High": quotes.get("high"),
            "Low": quotes.get("low"),
            "Close": quotes.get("close"),
            "Volume": quotes.get("volume")
        })
        df.set_index("Date", inplace=True)
        return df.dropna()
    except Exception as e:
        st.warning(f"Error downloading {symbol}: {e}")
        return None

@st.cache_data(ttl=3600)
def download_price_data(assets, start, end):
    """Télécharge les données de prix pour plusieurs assets."""
    all_data = {}
    for symbol in assets:
        df = download_yahoo_data(symbol, start, end)
        if df is not None and not df.empty:
            all_data[symbol] = df["Close"]
    
    if not all_data:
        return pd.DataFrame()
    return pd.DataFrame(all_data).dropna()

@st.cache_data(ttl=3600)
def download_vix_data(start, end):
    """Télécharge VIX."""
    df = download_yahoo_data("^VIX", start, end)
    if df is not None:
        return df["Close"]
    return pd.Series()

@st.cache_data(ttl=3600)
def load_macro_series(api_key, start, end):
    """Télécharge données macro depuis FRED."""
    try:
        fred = FredAPI(api_key=api_key)
        
        start_str = start.strftime("%Y-%m-%d") if hasattr(start, 'strftime') else str(start)
        end_str = end.strftime("%Y-%m-%d") if hasattr(end, 'strftime') else str(end)
        
        cpi = fred.get_series("CPIAUCSL", observation_start=start_str, observation_end=end_str)
        unrate = fred.get_series("UNRATE", observation_start=start_str, observation_end=end_str)
        slope = fred.get_series("T10Y3M", observation_start=start_str, observation_end=end_str)
        fedfunds = fred.get_series("FEDFUNDS", observation_start=start_str, observation_end=end_str)
        
        macro = pd.concat([cpi, unrate, slope, fedfunds], axis=1)
        macro.columns = ["CPI", "UNRATE", "T10Y3M", "FEDFUNDS"]
        macro = macro.resample("ME").last()
        
        return macro.dropna()
    except Exception as e:
        st.error(f"FRED API Error: {e}")
        return pd.DataFrame()

# =========================
# FONCTIONS ANALYSE
# =========================

def build_features(macro_df):
    """Construit les features macro."""
    df = macro_df.copy()
    df["CPI_YoY"] = df["CPI"].pct_change(12)
    df["dUNRATE"] = df["UNRATE"].diff()
    df["slope"] = df["T10Y3M"]
    return df[["CPI_YoY", "dUNRATE", "slope"]].dropna()

def assign_regimes_rule_based(features):
    """Assigne les régimes avec règles heuristiques."""
    df = features.copy()
    infl_med = df["CPI_YoY"].rolling(60, min_periods=12).median()
    
    regime = pd.Series("Expansion", index=df.index)
    regime[df["slope"] <= 0] = "Contraction"
    regime[(df["CPI_YoY"] > infl_med) & (df["dUNRATE"] > 0) & (df["slope"] > 0)] = "Slowdown"
    
    return regime

def assign_regimes_hmm(features, n_states=3):
    """Assigne les régimes avec HMM."""
    try:
        scaler = StandardScaler()
        X = scaler.fit_transform(features.values)
        
        model = GaussianHMM(n_components=n_states, covariance_type="full", 
                           n_iter=100, random_state=42)
        model.fit(X)
        states = model.predict(X)
        
        # Map states to regime names based on characteristics
        regime_map = {0: "Expansion", 1: "Slowdown", 2: "Contraction"}
        regime = pd.Series([regime_map.get(s, "Expansion") for s in states], index=features.index)
        
        return regime
    except:
        return assign_regimes_rule_based(features)

def monthly_returns(prices):
    """Calcule les rendements mensuels."""
    monthly = prices.resample("ME").last()
    return monthly.pct_change().dropna()

def calculate_dynamic_slippage(vix, base_bps=2.0):
    """Calcule le slippage dynamique basé sur VIX."""
    if vix is None or len(vix) == 0:
        return base_bps
    vix_level = vix.iloc[-1] if hasattr(vix, 'iloc') else vix
    if vix_level < 15:
        return base_bps
    elif vix_level < 25:
        return base_bps * 1.5
    elif vix_level < 35:
        return base_bps * 2.0
    else:
        return base_bps * 3.0

def run_backtest(returns, regimes, weights_by_regime, cost_bps=5.0, vix=None, dynamic_slippage=False):
    """Backtest de la stratégie avec coûts de transaction."""
    df = returns.join(regimes.rename("regime"), how="inner").dropna()
    
    port_returns = []
    costs_list = []
    current_weights = None
    
    for i, date in enumerate(df.index):
        regime = df.loc[date, "regime"]
        target_weights = weights_by_regime.get(regime, weights_by_regime.get("Expansion", {}))
        
        # Portfolio return
        ret = sum(target_weights.get(col, 0) * df.loc[date, col] 
                  for col in returns.columns if col in target_weights)
        
        # Transaction costs
        cost = 0.0
        if current_weights:
            turnover = sum(abs(target_weights.get(col, 0) - current_weights.get(col, 0)) 
                          for col in returns.columns)
            
            if dynamic_slippage and vix is not None:
                try:
                    vix_at_date = vix.loc[:date].iloc[-1]
                    slippage = calculate_dynamic_slippage(vix_at_date, cost_bps)
                except:
                    slippage = cost_bps
            else:
                slippage = cost_bps
            
            cost = slippage / 10000 * turnover
            ret -= cost
        
        port_returns.append(ret)
        costs_list.append(cost)
        current_weights = target_weights
    
    port_returns = pd.Series(port_returns, index=df.index, name="strategy")
    wealth = (1 + port_returns).cumprod()
    
    return port_returns, wealth, pd.Series(costs_list, index=df.index)

def compute_metrics(returns):
    """Calcule les métriques de performance."""
    if len(returns) == 0:
        return {}
    
    annual_ret = returns.mean() * 12
    annual_vol = returns.std() * np.sqrt(12)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    max_dd = drawdown.min()
    
    return {
        "CAGR": annual_ret,
        "Volatility": annual_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": (returns > 0).mean(),
        "Total Return": wealth.iloc[-1] - 1 if len(wealth) > 0 else 0
    }

# =========================
# INTERFACE PRINCIPALE
# =========================

def main():
    # Header
    st.markdown('<p class="main-header">📊 Macro Regime Lab</p>', unsafe_allow_html=True)
    st.markdown("### 🚀 Quantitative Trading Strategy Dashboard")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input("🔑 FRED API Key", type="password",
                               help="Get your free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        
        # Period
        st.subheader("📅 Period")
        n_months = st.slider("Number of months", 24, 240, 120)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_months * 30)
        
        # Assets
        st.subheader("📈 Assets")
        available_assets = ["SPY", "TLT", "GLD", "XLK", "QQQ", "IWM", "EFA", "VNQ", "AGG", "LQD"]
        selected_assets = st.multiselect("Select ETFs", available_assets, 
                                         default=["SPY", "TLT", "GLD", "XLK"])
        
        # Regime Detection Method
        st.subheader("🎯 Regime Detection")
        regime_method = st.selectbox("Method", ["Rule-Based", "HMM"])
        
        # Weights by Regime
        st.subheader("⚖️ Regime Weights")
        
        with st.expander("🟢 Expansion Weights"):
            w_exp = {}
            for asset in selected_assets:
                default = 0.6 if asset == "SPY" else (0.4 if asset == "XLK" else 0.0)
                w_exp[asset] = st.slider(f"{asset}", 0.0, 1.0, min(default, 1.0), 0.05, key=f"exp_{asset}")
        
        with st.expander("🟡 Slowdown Weights"):
            w_slow = {}
            for asset in selected_assets:
                default = 0.4 if asset in ["SPY", "TLT"] else (0.2 if asset == "GLD" else 0.0)
                w_slow[asset] = st.slider(f"{asset}", 0.0, 1.0, min(default, 1.0), 0.05, key=f"slow_{asset}")
        
        with st.expander("🔴 Contraction Weights"):
            w_cont = {}
            for asset in selected_assets:
                default = 0.7 if asset == "TLT" else (0.3 if asset == "GLD" else 0.0)
                w_cont[asset] = st.slider(f"{asset}", 0.0, 1.0, min(default, 1.0), 0.05, key=f"cont_{asset}")
        
        # Costs
        st.subheader("💰 Costs")
        cost_bps = st.slider("Commission (bps)", 0.0, 20.0, 5.0, 0.5)
        dynamic_slippage = st.checkbox("Dynamic Slippage (VIX)", value=True)
        
        # Run Button
        run_analysis = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)
    
    # Main Content
    if run_analysis:
        if not api_key:
            st.error("⚠️ Please enter your FRED API Key")
            return
        
        if len(selected_assets) < 2:
            st.error("⚠️ Please select at least 2 assets")
            return
        
        weights_by_regime = {
            "Expansion": w_exp,
            "Slowdown": w_slow,
            "Contraction": w_cont
        }
        
        with st.spinner("📥 Downloading data..."):
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            prices = download_price_data(selected_assets, start_str, end_str)
            macro = load_macro_series(api_key, start_date, end_date)
            vix = download_vix_data(start_str, end_str) if dynamic_slippage else None
        
        if prices.empty:
            st.error("❌ Failed to download price data")
            return
        
        if macro.empty:
            st.error("❌ Failed to download macro data")
            return
        
        with st.spinner("🔄 Running analysis..."):
            features = build_features(macro)
            
            if regime_method == "HMM":
                regimes = assign_regimes_hmm(features)
            else:
                regimes = assign_regimes_rule_based(features)
            
            returns = monthly_returns(prices)
            
            # Align data
            common_idx = returns.index.intersection(regimes.index)
            returns = returns.loc[common_idx]
            regimes = regimes.loc[common_idx]
            
            # Run backtest
            strat_returns, strat_wealth, costs = run_backtest(
                returns, regimes, weights_by_regime, cost_bps, vix, dynamic_slippage
            )
            
            # Benchmarks
            spy_returns = returns["SPY"] if "SPY" in returns.columns else returns.iloc[:, 0]
            spy_wealth = (1 + spy_returns).cumprod()
            
            # 60/40 benchmark
            bench_weights = {"SPY": 0.6, "TLT": 0.4}
            bench_ret = sum(bench_weights.get(col, 0) * returns[col] 
                          for col in returns.columns if col in bench_weights)
            bench_wealth = (1 + bench_ret).cumprod()
            
            # Metrics
            strat_metrics = compute_metrics(strat_returns)
            spy_metrics = compute_metrics(spy_returns)
            bench_metrics = compute_metrics(bench_ret)
        
        st.success(f"✅ Analysis complete! {len(prices)} days of data")
        
        # =========================
        # TABS
        # =========================
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 Performance", "🎯 Regimes", "📉 Risk", "📤 Export"])
        
        # TAB 1: Dashboard
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Strategy CAGR", f"{strat_metrics.get('CAGR', 0)*100:.1f}%",
                         f"{(strat_metrics.get('CAGR', 0) - spy_metrics.get('CAGR', 0))*100:+.1f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{strat_metrics.get('Sharpe', 0):.2f}",
                         f"{strat_metrics.get('Sharpe', 0) - spy_metrics.get('Sharpe', 0):+.2f}")
            with col3:
                st.metric("Max Drawdown", f"{strat_metrics.get('Max Drawdown', 0)*100:.1f}%")
            with col4:
                st.metric("Win Rate", f"{strat_metrics.get('Win Rate', 0)*100:.1f}%")
            
            # Wealth Curve
            st.subheader("📈 Wealth Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strat_wealth.index, y=strat_wealth.values,
                                    name="Strategy", line=dict(width=3, color="#2E7D32")))
            fig.add_trace(go.Scatter(x=spy_wealth.index, y=spy_wealth.values,
                                    name="SPY", line=dict(width=2, color="gray", dash="dash")))
            fig.add_trace(go.Scatter(x=bench_wealth.index, y=bench_wealth.values,
                                    name="60/40", line=dict(width=2, color="orange", dash="dot")))
            fig.update_layout(title="Growth of $1", xaxis_title="Date", yaxis_title="Value",
                            hovermode="x unified", height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # TAB 2: Performance
        with tab2:
            st.subheader("📈 Performance Analysis")
            
            # Monthly Returns Heatmap
            st.subheader("📊 Monthly Returns Heatmap")
            monthly_rets = strat_returns.copy()
            monthly_rets.index = pd.to_datetime(monthly_rets.index)
            
            heatmap_data = monthly_rets.groupby([monthly_rets.index.year, monthly_rets.index.month]).sum().unstack()
            heatmap_data.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig_heat = px.imshow(heatmap_data.values * 100,
                                labels=dict(x="Month", y="Year", color="Return %"),
                                x=heatmap_data.columns.tolist(),
                                y=heatmap_data.index.tolist(),
                                color_continuous_scale="RdYlGn",
                                aspect="auto")
            fig_heat.update_traces(text=np.round(heatmap_data.values * 100, 1),
                                  texttemplate="%{text}%")
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # Performance Table
            st.subheader("📋 Performance Comparison")
            perf_df = pd.DataFrame({
                "Strategy": strat_metrics,
                "SPY": spy_metrics,
                "60/40": bench_metrics
            }).T
            
            st.dataframe(perf_df.style.format({
                "CAGR": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2%}",
                "Win Rate": "{:.2%}",
                "Total Return": "{:.2%}"
            }), use_container_width=True)
        
        # TAB 3: Regimes
        with tab3:
            st.subheader("🎯 Regime Analysis")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Distribution")
                regime_counts = regimes.value_counts()
                fig_pie = px.pie(values=regime_counts.values, names=regime_counts.index,
                               color=regime_counts.index,
                               color_discrete_map={"Expansion": "#2E7D32", 
                                                  "Slowdown": "#FF9800", 
                                                  "Contraction": "#D32F2F"})
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Regime Timeline")
                regime_df = pd.DataFrame({"Date": regimes.index, "Regime": regimes.values})
                fig_timeline = px.scatter(regime_df, x="Date", y="Regime", color="Regime",
                                        color_discrete_map={"Expansion": "#2E7D32", 
                                                           "Slowdown": "#FF9800", 
                                                           "Contraction": "#D32F2F"})
                fig_timeline.update_traces(marker=dict(size=10))
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Current Regime
            current_regime = regimes.iloc[-1] if len(regimes) > 0 else "Unknown"
            regime_colors = {"Expansion": "🟢", "Slowdown": "🟡", "Contraction": "🔴"}
            st.info(f"**Current Regime:** {regime_colors.get(current_regime, '⚪')} {current_regime}")
        
        # TAB 4: Risk
        with tab4:
            st.subheader("📉 Risk Analysis")
            
            # Drawdown Chart
            st.subheader("Drawdown")
            strat_dd = (strat_wealth / strat_wealth.cummax() - 1) * 100
            spy_dd = (spy_wealth / spy_wealth.cummax() - 1) * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values,
                                       fill='tozeroy', name="Strategy", 
                                       line=dict(color="#2E7D32")))
            fig_dd.add_trace(go.Scatter(x=spy_dd.index, y=spy_dd.values,
                                       fill='tozeroy', name="SPY",
                                       line=dict(color="gray")))
            fig_dd.update_layout(title="Drawdown (%)", xaxis_title="Date", 
                               yaxis_title="Drawdown %", height=400)
            st.plotly_chart(fig_dd, use_container_width=True)
            
            # Rolling Metrics
            st.subheader("Rolling Sharpe (12 months)")
            rolling_ret = strat_returns.rolling(12).mean() * 12
            rolling_vol = strat_returns.rolling(12).std() * np.sqrt(12)
            rolling_sharpe = rolling_ret / rolling_vol
            
            fig_sharpe = go.Figure()
            fig_sharpe.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                                           name="Rolling Sharpe", line=dict(color="#2E7D32")))
            fig_sharpe.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_sharpe.update_layout(height=300)
            st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # TAB 5: Export
        with tab5:
            st.subheader("📤 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 Download Returns (CSV)",
                    strat_returns.to_csv(),
                    "strategy_returns.csv",
                    "text/csv"
                )
            
            with col2:
                export_df = pd.DataFrame({
                    "Date": regimes.index,
                    "Regime": regimes.values
                })
                st.download_button(
                    "📥 Download Regimes (CSV)",
                    export_df.to_csv(index=False),
                    "regimes.csv",
                    "text/csv"
                )
    
    else:
        # Welcome Screen
        st.info("👈 Configure your parameters and click **RUN ANALYSIS** to start")
        
        st.markdown("""
        ### 🎯 How to use this dashboard
        
        1. **Enter your FRED API Key** (free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html))
        2. **Select the analysis period**
        3. **Choose ETFs** to include
        4. **Adjust weights** for each regime
        5. **Click RUN ANALYSIS**
        
        ### 📊 The 3 Macro Regimes
        
        - **🟢 Expansion**: Positive yield curve, stable inflation
        - **🟡 Slowdown**: High inflation + rising unemployment
        - **🔴 Contraction**: Inverted yield curve (recession likely)
        
        ### 📈 Default Allocation Strategy
        
        | Regime | SPY | TLT | GLD | XLK |
        |--------|-----|-----|-----|-----|
        | Expansion | 60% | 0% | 0% | 40% |
        | Slowdown | 40% | 40% | 20% | 0% |
        | Contraction | 0% | 70% | 30% | 0% |
        """)

if __name__ == "__main__":
    main()
