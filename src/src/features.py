"""
features.py
Module pour la création et l'ingénierie de features macro et techniques.

Fonctionnalités :
- Features macroéconomiques (inflation, chômage, taux, spreads)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Momentum indicators
- Volatility indicators
- Seasonality features
- Interaction features
- Feature scaling et transformation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Classe principale pour créer et gérer les features.
    """
    
    def __init__(self):
        """Initialise le FeatureEngineer."""
        self.feature_names = []
        self.feature_importance = {}
    
    def build_macro_features(self, macro_df: pd.DataFrame, vix_series: pd.Series = None) -> pd.DataFrame:
        """
        Construit les features macroéconomiques de base et avancées.
        
        Args:
            macro_df: DataFrame avec les séries macro brutes
            vix_series: Series VIX (optionnel)
            
        Returns:
            DataFrame avec toutes les features macro
        """
        print("[INFO] Building macro features...")
        df = macro_df.copy()
        
        # =========================
        # FEATURES DE BASE
        # =========================
        
        # Inflation
        df["CPI_YoY"] = df["CPI"].pct_change(12)  # Inflation annuelle
        df["CPI_MoM"] = df["CPI"].pct_change()     # Inflation mensuelle
        df["CPI_QoQ"] = df["CPI"].pct_change(3)    # Inflation trimestrielle
        
        # Chômage
        df["dUNRATE"] = df["UNRATE"].diff()                           # Variation du chômage
        df["UNRATE_MA3"] = df["UNRATE"].rolling(3).mean()            # Moyenne mobile 3 mois
        df["UNRATE_MA12"] = df["UNRATE"].rolling(12).mean()          # Moyenne mobile 1 an
        df["UNRATE_trend"] = df["UNRATE"] - df["UNRATE_MA12"]        # Écart à la tendance
        
        # Yield curve
        df["slope"] = df["T10Y3M"]                                    # Pente de la courbe
        df["slope_change"] = df["slope"].diff()                       # Variation de la pente
        df["slope_MA3"] = df["slope"].rolling(3).mean()              # Moyenne mobile
        df["slope_vol"] = df["slope"].rolling(12).std()              # Volatilité de la pente
        
        # =========================
        # FEATURES AVANCÉES
        # =========================
        
        # Inflation dynamics
        df["CPI_acceleration"] = df["CPI_MoM"].diff()                 # Accélération de l'inflation
        df["CPI_volatility"] = df["CPI_MoM"].rolling(12).std()       # Volatilité de l'inflation
        
        # Chômage dynamics
        df["UNRATE_velocity"] = df["dUNRATE"].rolling(3).mean()      # Vitesse du changement
        df["UNRATE_acceleration"] = df["dUNRATE"].diff()              # Accélération
        
        # Yield curve dynamics
        df["slope_momentum"] = df["slope"].diff(3)                    # Momentum sur 3 mois
        df["slope_reversal"] = (df["slope"] - df["slope"].shift(6))  # Reversal sur 6 mois
        
        # =========================
        # VIX FEATURES
        # =========================
        if vix_series is not None:
            vix_monthly = vix_series.resample("ME").last()
            df = df.join(vix_monthly.rename("VIX"), how="left")
            
            df["VIX_change"] = df["VIX"].pct_change()
            df["VIX_MA3"] = df["VIX"].rolling(3).mean()
            df["VIX_vol"] = df["VIX"].rolling(12).std()
            df["VIX_zscore"] = (df["VIX"] - df["VIX"].rolling(60).mean()) / df["VIX"].rolling(60).std()
            df["VIX_spike"] = (df["VIX"] > df["VIX"].rolling(12).mean() + 2 * df["VIX"].rolling(12).std()).astype(int)
        
        # =========================
        # CREDIT SPREADS
        # =========================
        if "CREDIT_SPREAD" in df.columns:
            df["credit_spread_change"] = df["CREDIT_SPREAD"].diff()
            df["credit_spread_MA6"] = df["CREDIT_SPREAD"].rolling(6).mean()
            df["credit_spread_trend"] = df["CREDIT_SPREAD"] - df["credit_spread_MA6"]
        
        # =========================
        # GDP FEATURES (si disponible)
        # =========================
        if "GDP" in df.columns:
            df["GDP_YoY"] = df["GDP"].pct_change(4)  # Croissance annuelle (trimestrielle)
            df["GDP_trend"] = df["GDP"].rolling(8).mean()
        
        # =========================
        # FED FUNDS FEATURES
        # =========================
        if "FED_FUNDS" in df.columns:
            df["FED_change"] = df["FED_FUNDS"].diff()
            df["FED_direction"] = np.sign(df["FED_change"])
            df["FED_cycle"] = df["FED_direction"].rolling(6).sum()  # Cycle de hausse/baisse
        
        # =========================
        # INTERACTION FEATURES
        # =========================
        
        # Misery Index (Inflation + Unemployment)
        df["misery_index"] = df["CPI_YoY"].fillna(0) + df["UNRATE"].fillna(0)
        
        # Inflation-Unemployment interaction
        df["phillips_curve"] = df["CPI_YoY"] * df["UNRATE"]
        
        # Yield curve + VIX interaction
        if "VIX" in df.columns:
            df["stress_indicator"] = df["VIX"] * (1 - df["slope"] / 10)
        
        # =========================
        # SEASONALITY
        # =========================
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        print(f"[INFO] Created {len(df.columns)} features")
        
        return df
    
    def build_technical_indicators(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule des indicateurs techniques sur les prix.
        
        Args:
            prices: DataFrame des prix
            
        Returns:
            DataFrame des indicateurs techniques
        """
        print("[INFO] Building technical indicators...")
        indicators = pd.DataFrame(index=prices.index)
        
        for asset in prices.columns:
            price_series = prices[asset]
            
            # RSI (Relative Strength Index)
            indicators[f"{asset}_RSI"] = self._calculate_rsi(price_series, period=14)
            
            # MACD
            macd, signal = self._calculate_macd(price_series)
            indicators[f"{asset}_MACD"] = macd
            indicators[f"{asset}_MACD_signal"] = signal
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_mid = self._calculate_bollinger_bands(price_series)
            indicators[f"{asset}_BB_position"] = (price_series - bb_lower) / (bb_upper - bb_lower)
            
            # Momentum
            indicators[f"{asset}_momentum_1m"] = price_series.pct_change(21)   # 1 mois
            indicators[f"{asset}_momentum_3m"] = price_series.pct_change(63)   # 3 mois
            indicators[f"{asset}_momentum_6m"] = price_series.pct_change(126)  # 6 mois
            
            # Volatility
            returns = price_series.pct_change()
            indicators[f"{asset}_vol_21d"] = returns.rolling(21).std() * np.sqrt(252)
            indicators[f"{asset}_vol_63d"] = returns.rolling(63).std() * np.sqrt(252)
            
            # Moving averages
            indicators[f"{asset}_MA50"] = price_series.rolling(50).mean()
            indicators[f"{asset}_MA200"] = price_series.rolling(200).mean()
            indicators[f"{asset}_price_to_MA50"] = price_series / indicators[f"{asset}_MA50"]
            indicators[f"{asset}_price_to_MA200"] = price_series / indicators[f"{asset}_MA200"]
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calcule le MACD."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcule les Bollinger Bands."""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std_dev * std)
        lower_band = ma - (std_dev * std)
        return upper_band, lower_band, ma
    
    def select_features_for_modeling(self, features_df: pd.DataFrame, 
                                     target_features: List[str] = None) -> pd.DataFrame:
        """
        Sélectionne et nettoie les features pour le modeling.
        
        Args:
            features_df: DataFrame avec toutes les features
            target_features: Liste des features à garder (optionnel)
            
        Returns:
            DataFrame nettoyé avec features sélectionnées
        """
        if target_features is None:
            # Features par défaut pour le modeling des régimes
            target_features = [
                "CPI_YoY", "CPI_MoM", "CPI_acceleration",
                "dUNRATE", "UNRATE_trend", "UNRATE_velocity",
                "slope", "slope_change", "slope_momentum",
            ]
            
            # Ajouter VIX si disponible
            if "VIX" in features_df.columns:
                target_features.extend(["VIX", "VIX_change", "VIX_zscore"])
            
            # Ajouter credit spread si disponible
            if "credit_spread_change" in features_df.columns:
                target_features.append("credit_spread_change")
            
            # Ajouter interactions
            if "misery_index" in features_df.columns:
                target_features.append("misery_index")
            
            if "stress_indicator" in features_df.columns:
                target_features.append("stress_indicator")
        
        # Sélectionner les features
        available_features = [f for f in target_features if f in features_df.columns]
        
        if len(available_features) < len(target_features):
            missing = set(target_features) - set(available_features)
            print(f"[WARNING] Missing features: {missing}")
        
        selected = features_df[available_features].copy()
        
        # Nettoyage
        selected = selected.replace([np.inf, -np.inf], np.nan)
        selected = selected.dropna()
        
        print(f"[INFO] Selected {len(available_features)} features for modeling")
        print(f"[INFO] {len(selected)} observations after cleaning")
        
        self.feature_names = available_features
        
        return selected
    
    def create_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 3, 6]) -> pd.DataFrame:
        """
        Crée des features avec lags temporels.
        
        Args:
            df: DataFrame original
            lags: Liste des lags à créer
            
        Returns:
            DataFrame avec features laggées
        """
        print(f"[INFO] Creating lagged features with lags: {lags}")
        
        result = df.copy()
        
        for col in df.columns:
            for lag in lags:
                result[f"{col}_lag{lag}"] = df[col].shift(lag)
        
        print(f"[INFO] Created {len(result.columns)} total features (including lags)")
        
        return result
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """
        Crée des features avec statistiques roulantes.
        
        Args:
            df: DataFrame original
            windows: Liste des fenêtres
            
        Returns:
            DataFrame avec features roulantes
        """
        print(f"[INFO] Creating rolling features with windows: {windows}")
        
        result = df.copy()
        
        for col in df.columns:
            for window in windows:
                result[f"{col}_MA{window}"] = df[col].rolling(window).mean()
                result[f"{col}_std{window}"] = df[col].rolling(window).std()
                result[f"{col}_min{window}"] = df[col].rolling(window).min()
                result[f"{col}_max{window}"] = df[col].rolling(window).max()
        
        print(f"[INFO] Created {len(result.columns)} total features (including rolling)")
        
        return result
    
    def calculate_feature_importance(self, features: pd.DataFrame, 
                                    target: pd.Series, 
                                    model) -> pd.DataFrame:
        """
        Calcule l'importance des features avec un modèle donné.
        
        Args:
            features: DataFrame des features
            target: Series target
            model: Modèle sklearn avec feature_importances_
            
        Returns:
            DataFrame avec importance des features
        """
        if not hasattr(model, 'feature_importances_'):
            print("[WARNING] Model doesn't have feature_importances_ attribute")
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'feature': features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance.set_index('feature')['importance'].to_dict()
        
        print("\n[INFO] Top 10 most important features:")
        print(importance.head(10).to_string(index=False))
        
        return importance


# Fonctions utilitaires standalone
def build_macro_features(macro_df: pd.DataFrame, vix_series: pd.Series = None) -> pd.DataFrame:
    """Wrapper pour compatibilité avec l'ancien code."""
    engineer = FeatureEngineer()
    return engineer.build_macro_features(macro_df, vix_series)


def select_default_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne les features par défaut pour le modeling."""
    engineer = FeatureEngineer()
    return engineer.select_features_for_modeling(features_df)