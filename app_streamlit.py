#!/usr/bin/env python3
"""
app_streamlit.py
Application Web Interactive pour Macro Regime & Factor Rotation Lab
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from fredapi import Fred as FredAPI
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import io
import traceback
from pathlib import Path

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
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
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
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# FONCTIONS UTILITAIRES
# =========================

@st.cache_data(ttl=3600)
def download_price_data(assets, start, end):
    """Télécharge les données de prix."""
    data = yf.download(assets, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    elif 'Adj Close' in data.columns:
        data = data['Adj Close']
    return data.dropna()

@st.cache_data(ttl=3600)
def download_vix_data(start, end):
    """Télécharge VIX."""
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix = vix['Close']
    elif 'Close' in vix.columns:
        vix = vix['Close']
    return vix

@st.cache_data(ttl=3600)
def load_macro_series(api_key, start, end):
    """Télécharge données macro."""
    try:
        fred = FredAPI(api_key=api_key)
        
        start_str = start.strftime("%Y-%m-%d") if hasattr(start, 'strftime') else str(start)
        end_str = end.strftime("%Y-%m-%d") if hasattr(end, 'strftime') else str(end)
        
        cpi = fred.get_series("CPIAUCSL", observation_start=start_str, observation_end=end_str)
        unrate = fred.get_series("UNRATE", observation_start=start_str, observation_end=end_str)
        slope = fred.get_series("T10Y3M", observation_start=start_str, observation_end=end_str)
        
        macro = pd.concat([cpi, unrate, slope], axis=1)
        macro.columns = ["CPI", "UNRATE", "T10Y3M"]
        macro = macro.resample("ME").last()
        
        return macro
    except Exception as e:
        st.error(f"❌ Erreur FRED: {e}")
        st.code(traceback.format_exc())
        return pd.DataFrame()

def monthly_returns(price_df):
    """Calcule les rendements mensuels."""
    monthly_prices = price_df.resample("ME").last()
    return monthly_prices.pct_change().dropna()

def build_macro_features(macro_df, vix_series=None):
    """Construit les features macro."""
    if macro_df.empty:
        return pd.DataFrame()
    
    df = macro_df.copy()
    df["CPI_YoY"] = df["CPI"].pct_change(12)
    df["dUNRATE"] = df["UNRATE"].diff()
    df["slope"] = df["T10Y3M"]
    df["CPI_MoM"] = df["CPI"].pct_change()
    df["slope_change"] = df["slope"].diff()
    
    if vix_series is not None:
        # Convertir en Series si c'est un DataFrame
        if isinstance(vix_series, pd.DataFrame):
            vix_series = vix_series.squeeze()
        vix_monthly = vix_series.resample("ME").last()
        vix_monthly.name = "VIX"
        df = df.join(vix_monthly, how="left")
        df["VIX_change"] = df["VIX"].pct_change()
    
    feat_cols = ["CPI_YoY", "dUNRATE", "slope", "CPI_MoM", "slope_change"]
    if "VIX" in df.columns:
        feat_cols.extend(["VIX", "VIX_change"])
    
    return df[feat_cols].dropna()

def assign_heuristic_regimes(macro_features):
    """Régimes heuristiques (3 régimes)."""
    df = macro_features.copy()
    infl_med = df["CPI_YoY"].rolling(60, min_periods=24).median()
    
    cond_contraction = df["slope"] <= 0
    cond_slow = (df["CPI_YoY"] > infl_med) & (df["dUNRATE"] > 0) & (~cond_contraction)
    
    regime = pd.Series("expansion", index=df.index)
    regime[cond_slow] = "slowdown"
    regime[cond_contraction] = "contraction"
    regime.name = "regime_heur"
    return regime

def fit_hmm_regimes(macro_features, n_states, train_end):
    """Fit HMM."""
    df = macro_features.dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)
    
    mask_train = df.index <= train_end
    X_train = X_scaled[mask_train]
    
    hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
    hmm.fit(X_train)
    
    states = hmm.predict(X_scaled)
    regimes_ml = pd.Series(states, index=df.index, name="regime_hmm")
    
    # Labellisation
    desc = macro_features.join(regimes_ml, how="inner").groupby("regime_hmm")[["CPI_YoY", "dUNRATE", "slope"]].mean()
    order = desc["dUNRATE"].sort_values().index.tolist()
    
    mapping = {}
    labels = ["ML_expansion", "ML_neutral", "ML_stress"]
    for i, state in enumerate(order):
        mapping[state] = labels[min(i, len(labels) - 1)]
    
    regimes_named = regimes_ml.map(mapping)
    regimes_named.name = "regime_hmm"
    
    return regimes_named, hmm, desc

def fit_random_forest_regimes(macro_features, target_regimes, train_end):
    """Fit Random Forest."""
    # Aligner les données
    common_idx = macro_features.index.intersection(target_regimes.index)
    df = macro_features.loc[common_idx].dropna()
    regimes_aligned = target_regimes.loc[df.index]
    
    # Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)
    
    # Séparer features (X) et target (y)
    feature_cols = list(range(X_scaled.shape[1]))
    X_df = pd.DataFrame(X_scaled, index=df.index, columns=feature_cols)
    y = regimes_aligned.values
    
    # Train/Test split
    mask_train = X_df.index <= train_end
    X_train = X_df.loc[mask_train].values
    y_train = y[mask_train]
    
    # Encoder les labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Fit Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train_encoded)
    
    # Prédire sur tout
    states = rf.predict(X_df.values)
    states_decoded = le.inverse_transform(states)
    
    return pd.Series(states_decoded, index=df.index, name="regime_rf"), rf

def optimize_portfolio_weights(rets_by_regime):
    """Optimise les poids (Sharpe max)."""
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

def optimize_weights_by_regime(rets_m, regimes):
    """Optimise poids par régime."""
    df = rets_m.join(regimes, how="inner").dropna()
    weights_by_regime = {}
    
    for regime in df[regimes.name].unique():
        rets_regime = df[df[regimes.name] == regime][rets_m.columns]
        if len(rets_regime) > 12:
            weights_by_regime[regime] = optimize_portfolio_weights(rets_regime)
        else:
            weights_by_regime[regime] = pd.Series([1/len(rets_m.columns)] * len(rets_m.columns), 
                                                   index=rets_m.columns)
    return weights_by_regime

def backtest_regime_strategy(rets_m, regimes, weights_by_regime, tc_bps=5.0):
    """Backtest."""
    df = rets_m.join(regimes, how="inner").dropna()
    regime_col = df[regimes.name]
    rets = df[rets_m.columns]
    
    dates = rets.index
    port_rets = []
    curr_w = None
    
    for t in range(len(dates)):
        reg_t = regime_col.iloc[t]
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
    
    port_rets = pd.Series(port_rets, index=dates)
    wealth = (1 + port_rets).cumprod()
    return port_rets, wealth

def compute_stats(rets, freq=12):
    """Statistiques avancées."""
    avg = rets.mean() * freq
    vol = rets.std() * np.sqrt(freq)
    sharpe = avg / vol if vol > 0 else np.nan
    
    downside = rets[rets < 0].std() * np.sqrt(freq)
    sortino = avg / downside if downside > 0 else np.nan
    
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1)
    mdd = dd.min()
    
    calmar = avg / abs(mdd) if mdd != 0 else np.nan
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
# INTERFACE STREAMLIT
# =========================

def main():
    # En-tête
    st.markdown('<h1 class="main-header">📊 Macro Regime & Factor Rotation Lab</h1>', unsafe_allow_html=True)
    st.markdown("### 🚀 Application Interactive de Trading Quantitatif Multi-Modèles ML")
    
    # Sidebar - Paramètres
    st.sidebar.header("⚙️ Configuration")
    
    # Clé FRED
    fred_key = st.sidebar.text_input("🔑 Clé FRED API", value="", type="password")
    
    # Sélection des actifs
    st.sidebar.subheader("📈 Sélection des Actifs")
    available_assets = ["SPY", "TLT", "GLD", "XLK", "QQQ", "IWM", "EFA", "VNQ", "DBC"]
    selected_assets = st.sidebar.multiselect(
        "ETFs à inclure",
        available_assets,
        default=["SPY", "TLT", "GLD", "XLK"]
    )
    
    # Période
    st.sidebar.subheader("📅 Période d'Analyse")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Début", value=datetime(1990, 1, 1))
    end_date = col2.date_input("Fin", value=datetime.now())
    
    # Train/Test split
    train_end = st.sidebar.date_input("📊 Fin période training", value=datetime(2010, 12, 31))
    
    # Paramètres ML
    st.sidebar.subheader("🤖 Paramètres ML")
    n_states = st.sidebar.slider("Nombre de régimes HMM", 2, 5, 3)
    
    # Options avancées
    st.sidebar.subheader("⚙️ Options Avancées")
    optimize_weights = st.sidebar.checkbox("Optimiser poids automatiquement", value=False)
    tc_bps = st.sidebar.slider("Coûts de transaction (bps)", 0.0, 20.0, 5.0, 0.5)
    use_vix = st.sidebar.checkbox("Inclure VIX", value=True)
    
    # Sélection des modèles
    st.sidebar.subheader("🎯 Modèles à Comparer")
    use_heuristic = st.sidebar.checkbox("Heuristique (3 régimes)", value=True)
    use_hmm = st.sidebar.checkbox("Hidden Markov Model", value=True)
    use_rf = st.sidebar.checkbox("Random Forest", value=True)
    
    # Bouton principal
    run_button = st.sidebar.button("🚀 LANCER L'ANALYSE", type="primary")
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "📈 Graphiques Interactifs", 
        "🔍 Analyse des Régimes",
        "⚖️ Comparaison Modèles",
        "💾 Export & Rapport"
    ])
    
    # =========================
    # EXÉCUTION
    # =========================
    
    if run_button:
        if len(selected_assets) < 2:
            st.error("⚠️ Sélectionnez au moins 2 actifs!")
            return
        
        if len(fred_key) == 0:
            st.error("⚠️ Entrez votre clé FRED API")
            return
        
        with st.spinner("🔄 Téléchargement et traitement des données..."):
            try:
                # Téléchargement
                prices = download_price_data(selected_assets, start_date, end_date)
                rets_m = monthly_returns(prices)
                
                vix = download_vix_data(start_date, end_date) if use_vix else None
                macro = load_macro_series(fred_key, start_date, end_date)
                
                if macro.empty:
                    st.error("❌ Impossible de charger les données macro. Vérifiez votre clé FRED.")
                    return
                
                X_macro = build_macro_features(macro, vix)
                
                if X_macro.empty:
                    st.error("❌ Pas assez de données pour construire les features.")
                    return
                
                # Alignement
                rets_m = rets_m.loc[X_macro.index.min(): X_macro.index.max()]
                X_macro = X_macro.loc[rets_m.index]
                
                st.success(f"✅ Données chargées : {len(rets_m)} mois de {rets_m.index.min().date()} à {rets_m.index.max().date()}")
                
            except Exception as e:
                st.error(f"❌ Erreur lors du téléchargement : {e}")
                st.code(traceback.format_exc())
                return
        
        # Calcul des régimes et backtests
        results = {}
        weights_dict = {}
        regimes_dict = {}
        
        with st.spinner("🤖 Calcul des régimes et backtests..."):
            try:
                # Heuristique
                if use_heuristic:
                    regime_heur = assign_heuristic_regimes(X_macro)
                    regimes_dict['Heuristic'] = regime_heur
                    
                    if optimize_weights:
                        w_heur = optimize_weights_by_regime(rets_m, regime_heur)
                    else:
                        w_heur = {
                            "expansion": pd.Series({a: 1/len(selected_assets) for a in selected_assets}),
                            "slowdown": pd.Series({a: 1/len(selected_assets) for a in selected_assets}),
                            "contraction": pd.Series({a: 1/len(selected_assets) for a in selected_assets}),
                        }
                    
                    weights_dict['Heuristic'] = w_heur
                    rets_heur, wealth_heur = backtest_regime_strategy(rets_m, regime_heur, w_heur, tc_bps)
                    results['Heuristic'] = {'returns': rets_heur, 'wealth': wealth_heur, 'stats': compute_stats(rets_heur)}
                
                # HMM
                if use_hmm:
                    regime_hmm, hmm_model, hmm_desc = fit_hmm_regimes(X_macro, n_states, pd.Timestamp(train_end))
                    regimes_dict['HMM'] = regime_hmm
                    
                    if optimize_weights:
                        w_hmm = optimize_weights_by_regime(rets_m, regime_hmm)
                    else:
                        unique_regimes = regime_hmm.unique()
                        w_hmm = {r: pd.Series({a: 1/len(selected_assets) for a in selected_assets}) for r in unique_regimes}
                    
                    weights_dict['HMM'] = w_hmm
                    rets_hmm, wealth_hmm = backtest_regime_strategy(rets_m, regime_hmm, w_hmm, tc_bps)
                    results['HMM'] = {'returns': rets_hmm, 'wealth': wealth_hmm, 'stats': compute_stats(rets_hmm)}
                
                # Random Forest
                if use_rf and use_heuristic:
                    regime_rf, rf_model = fit_random_forest_regimes(X_macro, regime_heur, pd.Timestamp(train_end))
                    regimes_dict['Random Forest'] = regime_rf
                    
                    if optimize_weights:
                        w_rf = optimize_weights_by_regime(rets_m, regime_rf)
                    else:
                        w_rf = w_heur.copy()
                    
                    weights_dict['Random Forest'] = w_rf
                    rets_rf, wealth_rf = backtest_regime_strategy(rets_m, regime_rf, w_rf, tc_bps)
                    results['Random Forest'] = {'returns': rets_rf, 'wealth': wealth_rf, 'stats': compute_stats(rets_rf)}
                
                # Benchmark SPY
                spy_rets = rets_m[selected_assets[0]]
                spy_wealth = (1 + spy_rets).cumprod()
                results['Benchmark'] = {'returns': spy_rets, 'wealth': spy_wealth, 'stats': compute_stats(spy_rets)}
            
            except Exception as e:
                st.error(f"❌ Erreur lors du calcul : {e}")
                st.code(traceback.format_exc())
                return
        
        st.success("✅ Analyse terminée!")
        
        # Stocker en session
        st.session_state['results'] = results
        st.session_state['regimes'] = regimes_dict
        st.session_state['weights'] = weights_dict
        st.session_state['rets_m'] = rets_m
        st.session_state['X_macro'] = X_macro
    
    # =========================
    # AFFICHAGE DES RÉSULTATS
    # =========================
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # TAB 1: Dashboard
        with tab1:
            st.header("📊 Performance Dashboard")
            
            # Métriques en cartes
            cols = st.columns(len(results))
            for i, (name, data) in enumerate(results.items()):
                with cols[i]:
                    st.metric(
                        label=f"{name}",
                        value=f"{data['stats']['CAGR']*100:.2f}%",
                        delta=f"Sharpe: {data['stats']['Sharpe']:.2f}"
                    )
            
            # Tableau de stats
            st.subheader("📈 Statistiques Complètes")
            stats_df = pd.DataFrame({name: data['stats'] for name, data in results.items()}).T
            st.dataframe(stats_df.style.format({
                'CAGR': '{:.2%}',
                'Vol': '{:.2%}',
                'Sharpe': '{:.2f}',
                'Sortino': '{:.2f}',
                'Calmar': '{:.2f}',
                'MaxDrawdown': '{:.2%}',
                'WinRate': '{:.2%}'
            }), use_container_width=True)
        
        # TAB 2: Graphiques
        with tab2:
            st.header("📈 Graphiques Interactifs")
            
            # Wealth curves
            fig_wealth = go.Figure()
            for name, data in results.items():
                fig_wealth.add_trace(go.Scatter(
                    x=data['wealth'].index,
                    y=data['wealth'].values,
                    name=name,
                    mode='lines',
                    line=dict(width=2)
                ))
            
            fig_wealth.update_layout(
                title="Courbes de Richesse (Growth of $1)",
                xaxis_title="Date",
                yaxis_title="Wealth",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig_wealth, use_container_width=True)
            
            # Drawdown
            fig_dd = go.Figure()
            for name, data in results.items():
                cum = data['wealth'].values
                peak = np.maximum.accumulate(cum)
                dd = (cum / peak - 1) * 100
                
                fig_dd.add_trace(go.Scatter(
                    x=data['wealth'].index,
                    y=dd,
                    name=name,
                    mode='lines',
                    fill='tozeroy'
                ))
            
            fig_dd.update_layout(
                title="Drawdown (%)",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig_dd, use_container_width=True)
        
        # TAB 3: Régimes
        with tab3:
            st.header("🔍 Analyse des Régimes")
            
            if 'regimes' in st.session_state:
                regimes = st.session_state['regimes']
                
                for name, regime in regimes.items():
                    st.subheader(f"Régimes - {name}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("Distribution:")
                        dist = regime.value_counts()
                        st.dataframe(dist)
                    
                    with col2:
                        fig_regime = px.pie(
                            values=dist.values,
                            names=dist.index,
                            title=f"Répartition {name}"
                        )
                        st.plotly_chart(fig_regime, use_container_width=True)
        
        # TAB 4: Comparaison
        with tab4:
            st.header("⚖️ Comparaison des Modèles")
            
            metrics = ['CAGR', 'Sharpe', 'Sortino', 'Calmar', 'MaxDrawdown']
            
            for metric in metrics:
                values = [results[name]['stats'][metric] for name in results.keys()]
                
                fig = go.Figure(data=[
                    go.Bar(x=list(results.keys()), y=values, text=[f"{v:.2f}" for v in values])
                ])
                
                fig.update_layout(
                    title=f"Comparaison - {metric}",
                    yaxis_title=metric,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # TAB 5: Export
        with tab5:
            st.header("💾 Export & Téléchargement")
            
            # CSV Stats
            stats_df = pd.DataFrame({name: data['stats'] for name, data in results.items()}).T
            csv = stats_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Télécharger Stats (CSV)",
                data=csv,
                file_name="macro_regime_stats.csv",
                mime="text/csv"
            )
            
            # CSV Wealth
            wealth_df = pd.DataFrame({name: data['wealth'] for name, data in results.items()})
            csv_wealth = wealth_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Télécharger Wealth Curves (CSV)",
                data=csv_wealth,
                file_name="macro_regime_wealth.csv",
                mime="text/csv"
            )
            
            st.info("💡 Le PDF sera généré avec matplotlib et disponible prochainement")

    # Page d'accueil si pas de résultats
    else:
        st.info("👆 Configurez vos paramètres dans la barre latérale et cliquez sur **LANCER L'ANALYSE** pour commencer.")
        
        # Afficher les résultats précédents s'ils existent
        try:
            stats_file = Path("output/stats.csv")
            if stats_file.exists():
                st.subheader("📊 Résultats précédents")
                stats = pd.read_csv(stats_file, index_col=0)
                st.dataframe(stats, use_container_width=True)
                
                if Path("output/wealth_curves.png").exists():
                    st.image("output/wealth_curves.png", use_container_width=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()