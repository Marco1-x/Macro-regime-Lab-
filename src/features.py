import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    """Construit et prépare les features pour le ML."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def build_macro_features(self, macro_df: pd.DataFrame, vix_series: pd.Series = None) -> pd.DataFrame:
        """Construit les features macroéconomiques."""
        df = macro_df.copy()
        
        # Inflation annualisée
        if "CPI" in df.columns:
            df["CPI_YoY"] = df["CPI"].pct_change(12)
        
        # Variation du chômage
        if "UNRATE" in df.columns:
            df["dUNRATE"] = df["UNRATE"].diff()
        
        # Yield curve slope
        if "T10Y3M" in df.columns:
            df["slope"] = df["T10Y3M"]
        
        # VIX features
        if vix_series is not None:
            df["VIX"] = vix_series
            df["VIX_chg"] = vix_series.pct_change()
        
        # Sélectionner seulement les colonnes engineered
        feature_cols = ["CPI_YoY", "dUNRATE", "slope"]
        if "VIX" in df.columns:
            feature_cols.extend(["VIX", "VIX_chg"])
        
        features = df[feature_cols].dropna()
        return features
    
    def select_features_for_modeling(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Standardise les features pour le ML."""
        X_scaled = self.scaler.fit_transform(features_df.values)
        return pd.DataFrame(X_scaled, index=features_df.index, columns=features_df.columns)