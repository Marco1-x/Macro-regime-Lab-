#!/usr/bin/env python3
"""
dashboard.py
Dashboard Streamlit Avancé pour Macro Regime Lab

Fonctionnalités :
- Interface intuitive pour backtesting
- Visualisations interactives (Plotly)
- Configuration dynamique des paramètres
- Intégration AdvancedVisualizer
- Export des résultats
- Comparaison de stratégies

Ce dashboard utilise les modules du projet:
- models.py (EnsembleRegimeDetector)
- backtest.py (slippage dynamique)
- visualization.py (AdvancedVisualizer)
- stress_testing.py (stress tests)
- walk_forward.py (validation)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# =========================================
# CONFIGURATION DE LA PAGE
# =========================================

st.set_page_config(
    page_title="Macro Regime Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .regime-expansion { background-color: #28a745; color: white; padding: 5px 10px; border-radius: 5px; }
    .regime-slowdown { background-color: #ffc107; color: black; padding: 5px 10px; border-radius: 5px; }
    .regime-contraction { background-color: #dc3545; color: white; padding: 5px 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)


# =========================================
# FONCTIONS UTILITAIRES
# =========================================

@st.cache_data(ttl=3600)
def generate_sample_data(n_months: int = 120, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Génère des données de démonstration."""
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=n_months, freq='ME')
    
    spy_ret = np.random.normal(0.008, 0.04, n_months)
    tlt_ret = np.random.normal(0.003, 0.015, n_months) - 0.3 * spy_ret
    gld_ret = np.random.normal(0.002, 0.025, n_months)
    xlk_ret = spy_ret * 1.2 + np.random.normal(0, 0.02, n_months)
    
    returns_df = pd.DataFrame({
        'SPY': spy_ret, 'TLT': tlt_ret, 'GLD': gld_ret, 'XLK': xlk_ret
    }, index=dates)
    
    regimes = pd.Series(
        np.random.choice(['expansion', 'slowdown', 'contraction'], n_months, p=[0.5, 0.3, 0.2]),
        index=dates, name='regime'
    )
    
    base_vix = np.random.normal(20, 5, n_months)
    vix = pd.Series(
        np.where(regimes == 'contraction', base_vix + 15,
                np.where(regimes == 'slowdown', base_vix + 5, base_vix)),
        index=dates, name='VIX'
    ).clip(10, 80)
    
    return returns_df, regimes, vix


def calculate_metrics(returns: pd.Series) -> Dict:
    """Calcule les métriques de performance."""
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 12
    ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
    volatility = returns.std() * np.sqrt(12)
    sharpe = ann_return / volatility if volatility > 0 else 0
    
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    max_dd = (wealth / peak - 1).min()
    win_rate = (returns > 0).mean()
    
    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate
    }


def run_backtest(returns_df: pd.DataFrame, regimes: pd.Series,
                  weights: Dict[str, Dict[str, float]], tc_bps: float = 5.0,
                  vix_series: pd.Series = None, use_dynamic_slippage: bool = False
                  ) -> Tuple[pd.Series, pd.Series, Dict]:
    """Exécute le backtest."""
    df = returns_df.join(regimes, how='inner').dropna()
    
    if vix_series is not None:
        vix_aligned = vix_series.reindex(df.index).ffill().bfill()
    else:
        vix_aligned = pd.Series(20.0, index=df.index)
    
    assets = returns_df.columns.tolist()
    port_rets = []
    curr_w = None
    
    for t in range(len(df)):
        regime = df['regime'].iloc[t]
        rets = df[assets].iloc[t]
        vix = vix_aligned.iloc[t]
        
        if regime in weights:
            target_w = pd.Series(weights[regime]).reindex(assets).fillna(0.0)
        else:
            target_w = pd.Series(0.0, index=assets)
        
        if curr_w is None:
            turnover = target_w.abs().sum()
        else:
            turnover = (target_w - curr_w).abs().sum()
        
        tc = tc_bps / 10000 * turnover
        
        if use_dynamic_slippage and vix_series is not None:
            if vix < 15:
                slippage_mult = 1.0
            elif vix > 30:
                slippage_mult = 3.0
            else:
                slippage_mult = 1.0 + 2.0 * (vix - 15) / 15
            slippage = 2.0 / 10000 * slippage_mult * turnover
            tc += slippage
        
        net_ret = (target_w * rets).sum() - tc
        port_rets.append(net_ret)
        curr_w = target_w
    
    port_rets = pd.Series(port_rets, index=df.index, name='strategy')
    wealth = (1 + port_rets).cumprod()
    metrics = calculate_metrics(port_rets)
    
    return port_rets, wealth, metrics


# =========================================
# COMPOSANTS DE VISUALISATION
# =========================================

def plot_wealth_curves(wealth_dict: Dict[str, pd.Series]) -> go.Figure:
    """Crée le graphique des courbes de richesse."""
    fig = go.Figure()
    colors = {'Strategy': '#2ca02c', 'SPY': '#1f77b4'}
    
    for name, series in wealth_dict.items():
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=name, mode='lines',
            line=dict(width=2.5 if name == 'Strategy' else 1.5, color=colors.get(name, '#7f7f7f'))
        ))
    
    fig.update_layout(
        title="📈 Wealth Curves (Growth of $1)",
        xaxis_title="Date", yaxis_title="Wealth",
        hovermode='x unified', height=500, template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_drawdown(wealth: pd.Series) -> go.Figure:
    """Crée le graphique de drawdown."""
    drawdown = (wealth / wealth.cummax() - 1) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values, fill='tozeroy',
        fillcolor='rgba(220, 53, 69, 0.3)', line=dict(color='#dc3545', width=1), name='Drawdown'
    ))
    fig.update_layout(title="📉 Drawdown", xaxis_title="Date", yaxis_title="Drawdown (%)", height=300, template='plotly_white')
    return fig


def plot_regime_distribution(regimes: pd.Series) -> go.Figure:
    """Crée le graphique de distribution des régimes."""
    counts = regimes.value_counts()
    colors = {'expansion': '#28a745', 'slowdown': '#ffc107', 'contraction': '#dc3545'}
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values,
        marker_colors=[colors.get(r, '#7f7f7f') for r in counts.index],
        hole=0.4, textinfo='percent+label'
    )])
    fig.update_layout(title="🎯 Regime Distribution", height=350)
    return fig


def plot_monthly_returns_heatmap(returns: pd.Series) -> go.Figure:
    """Crée la heatmap des rendements mensuels."""
    returns_df = returns.to_frame('return')
    returns_df['year'] = returns_df.index.year
    returns_df['month'] = returns_df.index.month
    
    pivot = returns_df.pivot_table(values='return', index='year', columns='month', aggfunc='sum') * 100
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=months[:pivot.shape[1]], y=pivot.index,
        colorscale='RdYlGn', zmid=0, text=np.round(pivot.values, 1),
        texttemplate='%{text}%', textfont={"size": 10}, colorbar=dict(title="Return %")
    ))
    fig.update_layout(title="📅 Monthly Returns Heatmap", xaxis_title="Month", yaxis_title="Year", height=400)
    return fig


def plot_rolling_sharpe(returns: pd.Series, window: int = 12) -> go.Figure:
    """Crée le graphique du Sharpe roulant."""
    rolling_sharpe = (returns.rolling(window).mean() * 12) / (returns.rolling(window).std() * np.sqrt(12))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, mode='lines',
                             line=dict(color='#1f77b4', width=2), name=f'{window}M Rolling Sharpe'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=1, line_dash="dash", line_color="green", annotation_text="Good (1.0)")
    fig.update_layout(title=f"📊 {window}-Month Rolling Sharpe Ratio", xaxis_title="Date",
                     yaxis_title="Sharpe Ratio", height=350, template='plotly_white')
    return fig


def plot_regime_performance(returns: pd.Series, regimes: pd.Series) -> go.Figure:
    """Crée le graphique de performance par régime."""
    df = pd.DataFrame({'returns': returns, 'regime': regimes})
    stats = df.groupby('regime')['returns'].agg(['mean', 'std'])
    stats['sharpe'] = (stats['mean'] * 12) / (stats['std'] * np.sqrt(12))
    stats['mean'] = stats['mean'] * 12 * 100
    
    colors = {'expansion': '#28a745', 'slowdown': '#ffc107', 'contraction': '#dc3545'}
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Annualized Return by Regime', 'Sharpe by Regime'))
    
    fig.add_trace(go.Bar(x=stats.index, y=stats['mean'],
                        marker_color=[colors.get(r, '#7f7f7f') for r in stats.index],
                        text=[f"{v:.1f}%" for v in stats['mean']], textposition='auto', name='Return'), row=1, col=1)
    fig.add_trace(go.Bar(x=stats.index, y=stats['sharpe'],
                        marker_color=[colors.get(r, '#7f7f7f') for r in stats.index],
                        text=[f"{v:.2f}" for v in stats['sharpe']], textposition='auto', name='Sharpe'), row=1, col=2)
    
    fig.update_layout(title="📊 Performance by Regime", height=350, showlegend=False)
    return fig


# =========================================
# INTERFACE PRINCIPALE
# =========================================

def main():
    """Application principale."""
    
    st.markdown('<h1 class="main-header">📊 Macro Regime Lab</h1>', unsafe_allow_html=True)
    st.markdown("### 🚀 Quantitative Trading Strategy Dashboard")
    
    # SIDEBAR
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("📅 Period")
    n_months = st.sidebar.slider("Number of months", 36, 240, 120, 12)
    
    st.sidebar.subheader("📈 Assets")
    available_assets = ["SPY", "TLT", "GLD", "XLK"]
    selected_assets = st.sidebar.multiselect("Select ETFs", available_assets, default=available_assets)
    
    st.sidebar.subheader("🎯 Regime Weights")
    
    with st.sidebar.expander("Expansion Weights"):
        exp_weights = {asset: st.slider(f"{asset}", 0.0, 1.0, 0.4 if asset == 'SPY' else 0.2, 0.05, key=f"exp_{asset}") for asset in selected_assets}
    
    with st.sidebar.expander("Slowdown Weights"):
        slow_weights = {asset: st.slider(f"{asset}", 0.0, 1.0, 0.3 if asset == 'TLT' else 0.2, 0.05, key=f"slow_{asset}") for asset in selected_assets}
    
    with st.sidebar.expander("Contraction Weights"):
        cont_weights = {asset: st.slider(f"{asset}", 0.0, 1.0, 0.5 if asset == 'TLT' else 0.1, 0.05, key=f"cont_{asset}") for asset in selected_assets}
    
    weights_by_regime = {'expansion': exp_weights, 'slowdown': slow_weights, 'contraction': cont_weights}
    
    st.sidebar.subheader("💰 Costs")
    tc_bps = st.sidebar.slider("Commission (bps)", 0.0, 20.0, 5.0, 0.5)
    use_dynamic_slippage = st.sidebar.checkbox("Dynamic Slippage (VIX)", value=True)
    
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 RUN BACKTEST", type="primary", use_container_width=True)
    
    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 Performance", "🎯 Regimes", "📉 Risk", "💾 Export"])
    
    # Load data
    returns_df, regimes, vix = generate_sample_data(n_months)
    available = [a for a in selected_assets if a in returns_df.columns]
    returns_df = returns_df[available]
    
    # Run backtest
    if run_button or 'results' not in st.session_state:
        with st.spinner("Running backtest..."):
            port_rets, wealth, metrics = run_backtest(returns_df, regimes, weights_by_regime, tc_bps, vix, use_dynamic_slippage)
            spy_wealth = (1 + returns_df['SPY']).cumprod()
            st.session_state['results'] = {'returns': port_rets, 'wealth': wealth, 'metrics': metrics, 'spy_wealth': spy_wealth, 'regimes': regimes, 'vix': vix}
        st.success("✅ Backtest completed!")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        port_rets, wealth, metrics = results['returns'], results['wealth'], results['metrics']
        spy_wealth = results['spy_wealth']
        
        # TAB 1: DASHBOARD
        with tab1:
            st.header("📊 Performance Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{metrics['total_return']*100:.1f}%", f"vs SPY: {(metrics['total_return'] - (spy_wealth.iloc[-1]-1))*100:+.1f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.1f}%")
            with col4:
                st.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
            
            st.plotly_chart(plot_wealth_curves({'Strategy': wealth, 'SPY': spy_wealth}), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_drawdown(wealth), use_container_width=True)
            with col2:
                st.plotly_chart(plot_regime_distribution(regimes), use_container_width=True)
        
        # TAB 2: PERFORMANCE
        with tab2:
            st.header("📈 Performance Analysis")
            st.plotly_chart(plot_monthly_returns_heatmap(port_rets), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_rolling_sharpe(port_rets, 12), use_container_width=True)
            with col2:
                st.plotly_chart(plot_regime_performance(port_rets, regimes), use_container_width=True)
        
        # TAB 3: REGIMES
        with tab3:
            st.header("🎯 Regime Analysis")
            
            counts = regimes.value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribution")
                for regime in counts.index:
                    color = {'expansion': '🟢', 'slowdown': '🟡', 'contraction': '🔴'}.get(regime, '⚪')
                    st.write(f"{color} **{regime.title()}**: {counts[regime]} months ({counts[regime]/len(regimes)*100:.1f}%)")
            
            with col2:
                st.subheader("Current Weights")
                for regime, wts in weights_by_regime.items():
                    st.write(f"**{regime.title()}:** " + ", ".join([f"{a}: {w*100:.0f}%" for a, w in wts.items() if w > 0]))
            
            st.plotly_chart(plot_regime_performance(port_rets, regimes), use_container_width=True)
        
        # TAB 4: RISK
        with tab4:
            st.header("📉 Risk Analysis")
            
            var_95 = port_rets.quantile(0.05)
            var_99 = port_rets.quantile(0.01)
            cvar_95 = port_rets[port_rets <= var_95].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("VaR 95%", f"{var_95*100:.2f}%")
            with col2:
                st.metric("VaR 99%", f"{var_99*100:.2f}%")
            with col3:
                st.metric("CVaR 95%", f"{cvar_95*100:.2f}%")
            
            if use_dynamic_slippage:
                st.subheader("VIX & Dynamic Slippage")
                fig = px.histogram(results['vix'], nbins=30, title="VIX Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        # TAB 5: EXPORT
        with tab5:
            st.header("💾 Export Results")
            
            stats_df = pd.DataFrame({
                'Metric': ['Total Return', 'Ann. Return', 'Volatility', 'Sharpe', 'Max DD', 'Win Rate'],
                'Value': [f"{metrics['total_return']*100:.2f}%", f"{metrics['annualized_return']*100:.2f}%",
                         f"{metrics['volatility']*100:.2f}%", f"{metrics['sharpe_ratio']:.2f}",
                         f"{metrics['max_drawdown']*100:.2f}%", f"{metrics['win_rate']*100:.1f}%"]
            })
            
            st.download_button("📥 Statistics (CSV)", stats_df.to_csv(index=False).encode('utf-8'),
                              "statistics.csv", "text/csv")
            
            wealth_df = pd.DataFrame({'Date': wealth.index, 'Strategy': wealth.values, 'SPY': spy_wealth.values})
            st.download_button("📥 Wealth Curve (CSV)", wealth_df.to_csv(index=False).encode('utf-8'),
                              "wealth_curve.csv", "text/csv")
            
            st.success("✅ Exports ready!")
    
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;'>📊 Macro Regime Lab | Built with Streamlit</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()