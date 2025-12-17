"""
data_loader.py
Module pour le téléchargement et la gestion des données de marché et macro.

Fonctionnalités :
- Téléchargement des prix ETF (yfinance)
- Téléchargement des données macro (FRED)
- Téléchargement VIX et autres indicateurs
- Calcul des rendements mensuels
- Gestion du cache pour éviter les téléchargements répétés
- Validation et nettoyage des données
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Classe principale pour charger toutes les données nécessaires au backtest.
    """
    
    def __init__(self, fred_api_key: str, cache_dir: str = "data/raw"):
        """
        Initialise le DataLoader.
        
        Args:
            fred_api_key: Clé API FRED
            cache_dir: Répertoire pour stocker le cache
        """
        self.fred_api_key = fred_api_key
        self.cache_dir = cache_dir
        self.fred = Fred(api_key=fred_api_key)
        
        # Créer le répertoire de cache
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_etf_prices(self, tickers: list, start_date: str, end_date: str = None,
                           use_cache: bool = True) -> pd.DataFrame:
        """
        Télécharge les prix des ETF depuis Yahoo Finance.
        
        Args:
            tickers: Liste des tickers ETF
            start_date: Date de début (format YYYY-MM-DD)
            end_date: Date de fin (optionnel)
            use_cache: Utiliser le cache si disponible
            
        Returns:
            DataFrame avec les prix ajustés
        """
        cache_file = os.path.join(self.cache_dir, f"etf_prices_{'_'.join(tickers)}.csv")
        
        # Vérifier le cache
        if use_cache and os.path.exists(cache_file):
            print(f"[INFO] Loading ETF prices from cache: {cache_file}")
            prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return prices
        
        print(f"[INFO] Downloading ETF prices for {tickers}...")
        try:
            data = yf.download(tickers, start=start_date, end=end_date, 
                             auto_adjust=True, progress=False)
            
            # Gestion des différents formats de retour
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            elif 'Adj Close' in data.columns:
                prices = data['Adj Close']
            else:
                prices = data
            
            # Nettoyage
            prices = prices.dropna()
            
            # Validation
            if prices.empty:
                raise ValueError("No data downloaded")
            
            if len(prices) < 100:
                print(f"[WARNING] Only {len(prices)} data points downloaded")
            
            # Sauvegarder dans le cache
            prices.to_csv(cache_file)
            print(f"[INFO] Saved {len(prices)} days of data to cache")
            
            return prices
            
        except Exception as e:
            print(f"[ERROR] Failed to download ETF prices: {e}")
            raise
    
    def download_vix(self, start_date: str, end_date: str = None,
                    use_cache: bool = True) -> pd.Series:
        """
        Télécharge l'indice VIX.
        
        Args:
            start_date: Date de début
            end_date: Date de fin (optionnel)
            use_cache: Utiliser le cache
            
        Returns:
            Series avec les valeurs VIX
        """
        cache_file = os.path.join(self.cache_dir, "vix.csv")
        
        if use_cache and os.path.exists(cache_file):
            print("[INFO] Loading VIX from cache")
            vix = pd.read_csv(cache_file, index_col=0, parse_dates=True, squeeze=True)
            return vix
        
        print("[INFO] Downloading VIX data...")
        try:
            data = yf.download("^VIX", start=start_date, end=end_date,
                             auto_adjust=True, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                vix = data['Close']
            elif 'Close' in data.columns:
                vix = data['Close']
            else:
                vix = data
            
            vix = vix.dropna()
            vix.to_csv(cache_file)
            
            return vix
            
        except Exception as e:
            print(f"[WARNING] Failed to download VIX: {e}")
            return pd.Series(dtype=float)
    
    def download_fred_series(self, series_id: str, start_date: str, end_date: str = None,
                            use_cache: bool = True) -> pd.Series:
        """
        Télécharge une série FRED.
        
        Args:
            series_id: ID de la série FRED
            start_date: Date de début
            end_date: Date de fin (optionnel)
            use_cache: Utiliser le cache
            
        Returns:
            Series avec les données
        """
        cache_file = os.path.join(self.cache_dir, f"fred_{series_id}.csv")
        
        if use_cache and os.path.exists(cache_file):
            print(f"[INFO] Loading {series_id} from cache")
            series = pd.read_csv(cache_file, index_col=0, parse_dates=True, squeeze=True)
            return series
        
        print(f"[INFO] Downloading FRED series: {series_id}")
        try:
            series = self.fred.get_series(series_id, 
                                         observation_start=start_date,
                                         observation_end=end_date)
            series.to_csv(cache_file)
            return series
            
        except Exception as e:
            print(f"[WARNING] Failed to download {series_id}: {e}")
            return pd.Series(dtype=float)
    
    def download_macro_data(self, start_date: str, end_date: str = None,
                           use_cache: bool = True) -> pd.DataFrame:
        """
        Télécharge toutes les séries macroéconomiques nécessaires.
        
        Args:
            start_date: Date de début
            end_date: Date de fin (optionnel)
            use_cache: Utiliser le cache
            
        Returns:
            DataFrame avec toutes les séries macro
        """
        print("[INFO] Downloading macro data from FRED...")
        
        # Séries principales
        cpi = self.download_fred_series("CPIAUCSL", start_date, end_date, use_cache)
        unrate = self.download_fred_series("UNRATE", start_date, end_date, use_cache)
        slope = self.download_fred_series("T10Y3M", start_date, end_date, use_cache)
        
        # Séries optionnelles
        try:
            credit_spread = self.download_fred_series("BAMLH0A0HYM2", start_date, end_date, use_cache)
        except:
            credit_spread = pd.Series(dtype=float)
        
        try:
            gdp = self.download_fred_series("GDP", start_date, end_date, use_cache)
        except:
            gdp = pd.Series(dtype=float)
        
        try:
            fed_funds = self.download_fred_series("FEDFUNDS", start_date, end_date, use_cache)
        except:
            fed_funds = pd.Series(dtype=float)
        
        # Combiner toutes les séries
        macro = pd.concat([cpi, unrate, slope, credit_spread, gdp, fed_funds], 
                         axis=1, keys=["CPI", "UNRATE", "T10Y3M", "CREDIT_SPREAD", "GDP", "FED_FUNDS"])
        
        # Resample en mensuel
        macro = macro.resample("ME").last()
        
        print(f"[INFO] Downloaded {len(macro)} months of macro data")
        print(f"[INFO] Columns: {list(macro.columns)}")
        
        return macro
    
    def calculate_monthly_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les rendements mensuels à partir des prix quotidiens.
        
        Args:
            prices: DataFrame des prix quotidiens
            
        Returns:
            DataFrame des rendements mensuels
        """
        print("[INFO] Calculating monthly returns...")
        monthly_prices = prices.resample("ME").last()
        returns = monthly_prices.pct_change().dropna()
        
        print(f"[INFO] {len(returns)} months of returns calculated")
        
        # Validation
        if returns.isnull().any().any():
            print("[WARNING] NaN values found in returns")
            returns = returns.dropna()
        
        # Vérifier les valeurs aberrantes
        for col in returns.columns:
            extreme = returns[col].abs() > 0.5  # >50% rendement mensuel
            if extreme.any():
                print(f"[WARNING] Extreme returns detected in {col}: {returns[col][extreme].values}")
        
        return returns
    
    def load_all_data(self, tickers: list, start_date: str, end_date: str = None,
                     use_cache: bool = True) -> dict:
        """
        Charge toutes les données nécessaires pour le backtest.
        
        Args:
            tickers: Liste des tickers ETF
            start_date: Date de début
            end_date: Date de fin (optionnel)
            use_cache: Utiliser le cache
            
        Returns:
            Dictionnaire contenant toutes les données
        """
        print("\n" + "="*60)
        print("LOADING ALL DATA")
        print("="*60)
        
        # Prix ETF
        prices = self.download_etf_prices(tickers, start_date, end_date, use_cache)
        returns = self.calculate_monthly_returns(prices)
        
        # VIX
        vix = self.download_vix(start_date, end_date, use_cache)
        
        # Macro
        macro = self.download_macro_data(start_date, end_date, use_cache)
        
        # Alignement des dates
        common_start = max(returns.index.min(), macro.index.min())
        common_end = min(returns.index.max(), macro.index.max())
        
        returns = returns.loc[common_start:common_end]
        macro = macro.loc[common_start:common_end]
        
        print(f"\n[INFO] Data aligned: {common_start.date()} to {common_end.date()}")
        print(f"[INFO] Total months: {len(returns)}")
        print("="*60 + "\n")
        
        return {
            'prices': prices,
            'returns': returns,
            'vix': vix,
            'macro': macro,
            'start_date': common_start,
            'end_date': common_end
        }


# Fonctions utilitaires pour compatibilité avec le code existant
def download_price_data(assets: list, start: str, end: str = None, 
                       fred_api_key: str = None) -> pd.DataFrame:
    """Wrapper pour compatibilité avec l'ancien code."""
    loader = DataLoader(fred_api_key or "YOUR_KEY_HERE")
    return loader.download_etf_prices(assets, start, end)


def monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper pour compatibilité."""
    monthly_prices = price_df.resample("ME").last()
    return monthly_prices.pct_change().dropna()


def load_macro_series(api_key: str) -> pd.DataFrame:
    """Wrapper pour compatibilité."""
    loader = DataLoader(api_key)
    return loader.download_macro_data("1990-01-01")