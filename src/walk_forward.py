#!/usr/bin/env python3
"""
walk_forward.py
Module de Walk-Forward Analysis pour Macro Regime Lab

Fonctionnalités :
- Rolling window training avec réentraînement périodique
- Anchored (expanding) window analysis
- Out-of-sample testing systématique
- Parameter stability analysis
- Overfitting detection
- Comparaison des méthodes de validation
- Métriques de robustesse

Walk-Forward = Gold standard pour valider les stratégies de trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports internes (si disponibles)
try:
    from models import HMMRegimeDetector, RandomForestRegimeDetector
    from backtest import BacktestEngine
    from metrics import PerformanceMetrics
except ImportError:
    pass


@dataclass
class WalkForwardResult:
    """Résultat d'une fenêtre de walk-forward."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_months: int
    test_months: int
    
    # Métriques in-sample (train)
    train_return: float
    train_sharpe: float
    train_volatility: float
    
    # Métriques out-of-sample (test)
    test_return: float
    test_sharpe: float
    test_volatility: float
    test_max_drawdown: float
    
    # Détails
    model_params: Dict = None
    regimes_predicted: pd.Series = None


@dataclass
class WalkForwardSummary:
    """Résumé complet de l'analyse walk-forward."""
    method: str  # 'rolling' ou 'anchored'
    total_windows: int
    train_window_size: int
    test_window_size: int
    
    # Métriques agrégées OOS
    avg_oos_return: float
    avg_oos_sharpe: float
    avg_oos_volatility: float
    avg_oos_max_drawdown: float
    
    # Stabilité
    sharpe_stability: float  # std des Sharpe OOS
    return_consistency: float  # % de fenêtres avec return > 0
    overfitting_ratio: float  # train_sharpe / test_sharpe moyen
    
    # Détails par fenêtre
    window_results: List[WalkForwardResult] = None


class WalkForwardAnalyzer:
    """
    Classe principale pour l'analyse Walk-Forward.
    
    Deux modes :
    1. Rolling Window : fenêtre fixe qui "roule" dans le temps
    2. Anchored Window : fenêtre qui s'étend depuis le début
    """
    
    def __init__(self, 
                 train_window: int = 36,
                 test_window: int = 12,
                 step_size: int = None,
                 min_train_size: int = 24):
        """
        Initialise le Walk-Forward Analyzer.
        
        Args:
            train_window: Taille de la fenêtre d'entraînement (mois)
            test_window: Taille de la fenêtre de test (mois)
            step_size: Pas entre les fenêtres (défaut = test_window)
            min_train_size: Taille minimale d'entraînement pour anchored
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size or test_window
        self.min_train_size = min_train_size
        
    def _compute_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calcule les métriques de base pour une série de rendements."""
        if len(returns) == 0 or returns.isna().all():
            return {
                'return': np.nan,
                'sharpe': np.nan,
                'volatility': np.nan,
                'max_drawdown': np.nan
            }
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 2:
            return {
                'return': returns_clean.sum() if len(returns_clean) > 0 else np.nan,
                'sharpe': np.nan,
                'volatility': np.nan,
                'max_drawdown': np.nan
            }
        
        # Rendement annualisé
        ann_return = returns_clean.mean() * 12
        
        # Volatilité annualisée
        ann_vol = returns_clean.std() * np.sqrt(12)
        
        # Sharpe Ratio
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        
        # Max Drawdown
        wealth = (1 + returns_clean).cumprod()
        peak = wealth.cummax()
        drawdown = (wealth / peak - 1)
        max_dd = drawdown.min()
        
        return {
            'return': ann_return,
            'sharpe': sharpe,
            'volatility': ann_vol,
            'max_drawdown': max_dd
        }
    
    def generate_windows_rolling(self, 
                                  data_index: pd.DatetimeIndex) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Génère les fenêtres pour le mode Rolling.
        
        Args:
            data_index: Index temporel des données
            
        Returns:
            Liste de tuples (train_index, test_index)
        """
        windows = []
        n = len(data_index)
        
        start = 0
        while start + self.train_window + self.test_window <= n:
            train_end = start + self.train_window
            test_end = train_end + self.test_window
            
            train_idx = data_index[start:train_end]
            test_idx = data_index[train_end:test_end]
            
            windows.append((train_idx, test_idx))
            start += self.step_size
        
        return windows
    
    def generate_windows_anchored(self, 
                                   data_index: pd.DatetimeIndex) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Génère les fenêtres pour le mode Anchored (expanding).
        
        Args:
            data_index: Index temporel des données
            
        Returns:
            Liste de tuples (train_index, test_index)
        """
        windows = []
        n = len(data_index)
        
        train_end = self.min_train_size
        while train_end + self.test_window <= n:
            test_end = train_end + self.test_window
            
            train_idx = data_index[:train_end]
            test_idx = data_index[train_end:test_end]
            
            windows.append((train_idx, test_idx))
            train_end += self.step_size
        
        return windows
    
    def run_walk_forward(self,
                          features: pd.DataFrame,
                          returns: pd.Series,
                          model_factory: Callable,
                          strategy_func: Callable,
                          method: str = 'rolling',
                          verbose: bool = True) -> WalkForwardSummary:
        """
        Exécute l'analyse walk-forward complète.
        
        Args:
            features: DataFrame des features pour le modèle
            returns: Series des rendements pour le backtest
            model_factory: Fonction qui crée un nouveau modèle
            strategy_func: Fonction qui génère les rendements de stratégie
                          signature: (model, features, returns) -> pd.Series
            method: 'rolling' ou 'anchored'
            verbose: Afficher la progression
            
        Returns:
            WalkForwardSummary avec tous les résultats
        """
        # Aligner les données
        common_idx = features.index.intersection(returns.index)
        features = features.loc[common_idx]
        returns = returns.loc[common_idx]
        
        # Générer les fenêtres
        if method == 'rolling':
            windows = self.generate_windows_rolling(features.index)
        else:
            windows = self.generate_windows_anchored(features.index)
        
        if len(windows) == 0:
            raise ValueError(f"Pas assez de données pour le walk-forward. "
                           f"Besoin de {self.train_window + self.test_window} mois minimum.")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"WALK-FORWARD ANALYSIS ({method.upper()})")
            print(f"{'='*60}")
            print(f"Train window: {self.train_window} mois")
            print(f"Test window: {self.test_window} mois")
            print(f"Step size: {self.step_size} mois")
            print(f"Total windows: {len(windows)}")
            print(f"{'='*60}\n")
        
        results = []
        
        for i, (train_idx, test_idx) in enumerate(windows):
            if verbose:
                print(f"[Window {i+1}/{len(windows)}] "
                      f"Train: {train_idx[0].strftime('%Y-%m')} → {train_idx[-1].strftime('%Y-%m')} | "
                      f"Test: {test_idx[0].strftime('%Y-%m')} → {test_idx[-1].strftime('%Y-%m')}")
            
            # Données train/test
            X_train = features.loc[train_idx]
            X_test = features.loc[test_idx]
            y_train = returns.loc[train_idx]
            y_test = returns.loc[test_idx]
            
            # Créer et entraîner le modèle
            model = model_factory()
            
            try:
                # Entraînement
                if hasattr(model, 'fit'):
                    model.fit(X_train.values)
                
                # Générer les rendements de stratégie
                train_strategy_returns = strategy_func(model, X_train, y_train)
                test_strategy_returns = strategy_func(model, X_test, y_test)
                
                # Calculer les métriques
                train_metrics = self._compute_metrics(train_strategy_returns)
                test_metrics = self._compute_metrics(test_strategy_returns)
                
                # Prédiction des régimes (si applicable)
                regimes = None
                if hasattr(model, 'predict'):
                    try:
                        regimes = pd.Series(model.predict(X_test.values), index=test_idx)
                    except:
                        pass
                
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Erreur: {e}")
                train_metrics = {'return': np.nan, 'sharpe': np.nan, 
                               'volatility': np.nan, 'max_drawdown': np.nan}
                test_metrics = train_metrics.copy()
                regimes = None
            
            # Créer le résultat
            result = WalkForwardResult(
                window_id=i + 1,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
                train_months=len(train_idx),
                test_months=len(test_idx),
                train_return=train_metrics['return'],
                train_sharpe=train_metrics['sharpe'],
                train_volatility=train_metrics['volatility'],
                test_return=test_metrics['return'],
                test_sharpe=test_metrics['sharpe'],
                test_volatility=test_metrics['volatility'],
                test_max_drawdown=test_metrics['max_drawdown'],
                regimes_predicted=regimes
            )
            
            results.append(result)
            
            if verbose:
                print(f"   Train Sharpe: {train_metrics['sharpe']:.2f} | "
                      f"Test Sharpe: {test_metrics['sharpe']:.2f}")
        
        # Calculer le résumé
        summary = self._compute_summary(results, method)
        
        if verbose:
            self._print_summary(summary)
        
        return summary
    
    def _compute_summary(self, 
                          results: List[WalkForwardResult], 
                          method: str) -> WalkForwardSummary:
        """Calcule le résumé de l'analyse walk-forward."""
        
        # Extraire les métriques OOS
        oos_returns = [r.test_return for r in results if not np.isnan(r.test_return)]
        oos_sharpes = [r.test_sharpe for r in results if not np.isnan(r.test_sharpe)]
        oos_vols = [r.test_volatility for r in results if not np.isnan(r.test_volatility)]
        oos_dds = [r.test_max_drawdown for r in results if not np.isnan(r.test_max_drawdown)]
        
        # Métriques train pour overfitting ratio
        train_sharpes = [r.train_sharpe for r in results if not np.isnan(r.train_sharpe)]
        
        # Calculer les statistiques
        avg_oos_return = np.mean(oos_returns) if oos_returns else np.nan
        avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else np.nan
        avg_oos_vol = np.mean(oos_vols) if oos_vols else np.nan
        avg_oos_dd = np.mean(oos_dds) if oos_dds else np.nan
        
        # Stabilité du Sharpe
        sharpe_stability = np.std(oos_sharpes) if len(oos_sharpes) > 1 else np.nan
        
        # Consistance des rendements (% positifs)
        return_consistency = sum(1 for r in oos_returns if r > 0) / len(oos_returns) if oos_returns else np.nan
        
        # Overfitting ratio (train/test Sharpe)
        avg_train_sharpe = np.mean(train_sharpes) if train_sharpes else np.nan
        overfitting_ratio = avg_train_sharpe / avg_oos_sharpe if avg_oos_sharpe and avg_oos_sharpe != 0 else np.nan
        
        return WalkForwardSummary(
            method=method,
            total_windows=len(results),
            train_window_size=self.train_window,
            test_window_size=self.test_window,
            avg_oos_return=avg_oos_return,
            avg_oos_sharpe=avg_oos_sharpe,
            avg_oos_volatility=avg_oos_vol,
            avg_oos_max_drawdown=avg_oos_dd,
            sharpe_stability=sharpe_stability,
            return_consistency=return_consistency,
            overfitting_ratio=overfitting_ratio,
            window_results=results
        )
    
    def _print_summary(self, summary: WalkForwardSummary):
        """Affiche le résumé de manière formatée."""
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD SUMMARY ({summary.method.upper()})")
        print(f"{'='*60}")
        print(f"Windows analysées: {summary.total_windows}")
        print(f"Train/Test: {summary.train_window_size}/{summary.test_window_size} mois")
        print(f"\n📊 MÉTRIQUES OUT-OF-SAMPLE (MOYENNES):")
        print(f"   Rendement annualisé: {summary.avg_oos_return*100:.2f}%")
        print(f"   Sharpe Ratio: {summary.avg_oos_sharpe:.2f}")
        print(f"   Volatilité: {summary.avg_oos_volatility*100:.2f}%")
        print(f"   Max Drawdown: {summary.avg_oos_max_drawdown*100:.2f}%")
        print(f"\n🎯 INDICATEURS DE ROBUSTESSE:")
        print(f"   Stabilité Sharpe (std): {summary.sharpe_stability:.2f}")
        print(f"   Consistance rendements: {summary.return_consistency*100:.1f}%")
        print(f"   Ratio overfitting: {summary.overfitting_ratio:.2f}")
        
        # Interprétation
        print(f"\n💡 INTERPRÉTATION:")
        if summary.overfitting_ratio and summary.overfitting_ratio < 1.5:
            print(f"   ✅ Faible overfitting (ratio < 1.5)")
        elif summary.overfitting_ratio and summary.overfitting_ratio < 2.5:
            print(f"   ⚠️ Overfitting modéré (ratio 1.5-2.5)")
        else:
            print(f"   ❌ Fort overfitting (ratio > 2.5)")
        
        if summary.return_consistency and summary.return_consistency > 0.6:
            print(f"   ✅ Bonne consistance ({summary.return_consistency*100:.0f}% fenêtres positives)")
        else:
            print(f"   ⚠️ Consistance faible (<60% fenêtres positives)")
        
        print(f"{'='*60}\n")
    
    def compare_methods(self,
                         features: pd.DataFrame,
                         returns: pd.Series,
                         model_factory: Callable,
                         strategy_func: Callable,
                         verbose: bool = True) -> Dict[str, WalkForwardSummary]:
        """
        Compare les méthodes Rolling et Anchored.
        
        Args:
            features: DataFrame des features
            returns: Series des rendements
            model_factory: Fonction créant le modèle
            strategy_func: Fonction de stratégie
            verbose: Afficher les détails
            
        Returns:
            Dict avec les résumés pour chaque méthode
        """
        results = {}
        
        for method in ['rolling', 'anchored']:
            if verbose:
                print(f"\n{'#'*60}")
                print(f"# MÉTHODE: {method.upper()}")
                print(f"{'#'*60}")
            
            summary = self.run_walk_forward(
                features=features,
                returns=returns,
                model_factory=model_factory,
                strategy_func=strategy_func,
                method=method,
                verbose=verbose
            )
            results[method] = summary
        
        # Comparaison
        if verbose:
            self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[str, WalkForwardSummary]):
        """Affiche la comparaison entre les méthodes."""
        print(f"\n{'='*60}")
        print(f"COMPARAISON ROLLING vs ANCHORED")
        print(f"{'='*60}")
        
        headers = ["Métrique", "Rolling", "Anchored", "Meilleur"]
        rows = []
        
        r = results.get('rolling')
        a = results.get('anchored')
        
        if r and a:
            # Sharpe
            better_sharpe = "Rolling" if (r.avg_oos_sharpe or 0) > (a.avg_oos_sharpe or 0) else "Anchored"
            rows.append(["Sharpe OOS", f"{r.avg_oos_sharpe:.2f}", f"{a.avg_oos_sharpe:.2f}", better_sharpe])
            
            # Return
            better_return = "Rolling" if (r.avg_oos_return or 0) > (a.avg_oos_return or 0) else "Anchored"
            rows.append(["Return OOS", f"{r.avg_oos_return*100:.1f}%", f"{a.avg_oos_return*100:.1f}%", better_return])
            
            # Overfitting
            better_of = "Rolling" if (r.overfitting_ratio or 999) < (a.overfitting_ratio or 999) else "Anchored"
            rows.append(["Overfitting", f"{r.overfitting_ratio:.2f}", f"{a.overfitting_ratio:.2f}", better_of])
            
            # Consistency
            better_cons = "Rolling" if (r.return_consistency or 0) > (a.return_consistency or 0) else "Anchored"
            rows.append(["Consistance", f"{r.return_consistency*100:.0f}%", f"{a.return_consistency*100:.0f}%", better_cons])
        
        # Affichage
        col_widths = [15, 12, 12, 10]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
        
        print(f"{'='*60}\n")


class ParameterStabilityAnalyzer:
    """
    Analyse la stabilité des paramètres du modèle à travers les fenêtres.
    Détecte si les paramètres optimaux changent significativement.
    """
    
    def __init__(self):
        self.parameter_history = []
    
    def analyze_stability(self,
                           features: pd.DataFrame,
                           returns: pd.Series,
                           param_grid: Dict[str, List],
                           model_class: type,
                           strategy_func: Callable,
                           train_window: int = 36,
                           step_size: int = 12) -> pd.DataFrame:
        """
        Analyse la stabilité des paramètres optimaux à travers le temps.
        
        Args:
            features: Features pour le modèle
            returns: Rendements pour évaluation
            param_grid: Grille de paramètres à tester
            model_class: Classe du modèle
            strategy_func: Fonction de stratégie
            train_window: Taille fenêtre d'entraînement
            step_size: Pas entre fenêtres
            
        Returns:
            DataFrame avec les paramètres optimaux par fenêtre
        """
        print(f"\n{'='*60}")
        print(f"PARAMETER STABILITY ANALYSIS")
        print(f"{'='*60}")
        
        results = []
        n = len(features)
        
        start = 0
        window_id = 0
        
        while start + train_window < n:
            window_id += 1
            end = start + train_window
            
            train_features = features.iloc[start:end]
            train_returns = returns.iloc[start:end]
            
            # Grid search sur cette fenêtre
            best_params = None
            best_sharpe = -np.inf
            
            # Générer toutes les combinaisons
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            from itertools import product
            for combo in product(*param_values):
                params = dict(zip(param_names, combo))
                
                try:
                    model = model_class(**params)
                    if hasattr(model, 'fit'):
                        model.fit(train_features.values)
                    
                    strategy_returns = strategy_func(model, train_features, train_returns)
                    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(12) if strategy_returns.std() > 0 else 0
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params.copy()
                        
                except Exception:
                    continue
            
            if best_params:
                result = {
                    'window_id': window_id,
                    'train_start': train_features.index[0],
                    'train_end': train_features.index[-1],
                    'best_sharpe': best_sharpe,
                    **best_params
                }
                results.append(result)
                print(f"[Window {window_id}] Best params: {best_params} (Sharpe: {best_sharpe:.2f})")
            
            start += step_size
        
        results_df = pd.DataFrame(results)
        
        # Analyse de stabilité
        if len(results_df) > 1:
            print(f"\n📊 STABILITÉ DES PARAMÈTRES:")
            for param in param_names:
                if param in results_df.columns:
                    values = results_df[param]
                    if values.dtype in ['int64', 'float64']:
                        print(f"   {param}: mean={values.mean():.2f}, std={values.std():.2f}")
                    else:
                        mode = values.mode().iloc[0] if len(values.mode()) > 0 else None
                        print(f"   {param}: mode={mode}, unique={values.nunique()}")
        
        print(f"{'='*60}\n")
        
        return results_df


class OverfittingDetector:
    """
    Détecte les signes d'overfitting dans la stratégie.
    """
    
    @staticmethod
    def compute_overfitting_score(train_sharpe: float, 
                                   test_sharpe: float,
                                   train_return: float,
                                   test_return: float) -> Dict[str, Any]:
        """
        Calcule un score d'overfitting et des indicateurs.
        
        Args:
            train_sharpe: Sharpe in-sample
            test_sharpe: Sharpe out-of-sample
            train_return: Return in-sample
            test_return: Return out-of-sample
            
        Returns:
            Dict avec score et diagnostics
        """
        # Ratio Sharpe
        sharpe_ratio = train_sharpe / test_sharpe if test_sharpe and test_sharpe != 0 else np.inf
        
        # Différence de return
        return_decay = (train_return - test_return) / abs(train_return) if train_return != 0 else np.nan
        
        # Score composite (0-100, plus haut = plus d'overfitting)
        score = 0
        
        # Contribution du ratio Sharpe
        if sharpe_ratio > 3:
            score += 40
        elif sharpe_ratio > 2:
            score += 25
        elif sharpe_ratio > 1.5:
            score += 10
        
        # Contribution du decay de return
        if return_decay and return_decay > 0.5:
            score += 30
        elif return_decay and return_decay > 0.25:
            score += 15
        elif return_decay and return_decay > 0:
            score += 5
        
        # Sharpe OOS négatif
        if test_sharpe and test_sharpe < 0:
            score += 30
        
        # Classification
        if score >= 60:
            level = "HIGH"
            recommendation = "⚠️ Réduire la complexité du modèle, augmenter les données"
        elif score >= 30:
            level = "MODERATE"
            recommendation = "⚡ Surveiller, considérer la régularisation"
        else:
            level = "LOW"
            recommendation = "✅ Modèle semble robuste"
        
        return {
            'score': score,
            'level': level,
            'sharpe_ratio': sharpe_ratio,
            'return_decay': return_decay,
            'recommendation': recommendation
        }
    
    @staticmethod
    def analyze_walk_forward_results(summary: WalkForwardSummary) -> Dict[str, Any]:
        """
        Analyse les résultats walk-forward pour détecter l'overfitting.
        
        Args:
            summary: Résumé du walk-forward
            
        Returns:
            Analyse d'overfitting
        """
        if not summary.window_results:
            return {'error': 'No window results'}
        
        # Moyennes
        train_sharpes = [r.train_sharpe for r in summary.window_results if not np.isnan(r.train_sharpe)]
        test_sharpes = [r.test_sharpe for r in summary.window_results if not np.isnan(r.test_sharpe)]
        train_returns = [r.train_return for r in summary.window_results if not np.isnan(r.train_return)]
        test_returns = [r.test_return for r in summary.window_results if not np.isnan(r.test_return)]
        
        avg_train_sharpe = np.mean(train_sharpes) if train_sharpes else 0
        avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else 0
        avg_train_return = np.mean(train_returns) if train_returns else 0
        avg_test_return = np.mean(test_returns) if test_returns else 0
        
        return OverfittingDetector.compute_overfitting_score(
            avg_train_sharpe, avg_test_sharpe,
            avg_train_return, avg_test_return
        )


def create_results_dataframe(summary: WalkForwardSummary) -> pd.DataFrame:
    """
    Convertit les résultats walk-forward en DataFrame pour analyse.
    
    Args:
        summary: Résumé du walk-forward
        
    Returns:
        DataFrame avec toutes les fenêtres
    """
    if not summary.window_results:
        return pd.DataFrame()
    
    data = []
    for r in summary.window_results:
        data.append({
            'window_id': r.window_id,
            'train_start': r.train_start,
            'train_end': r.train_end,
            'test_start': r.test_start,
            'test_end': r.test_end,
            'train_months': r.train_months,
            'test_months': r.test_months,
            'train_return': r.train_return,
            'train_sharpe': r.train_sharpe,
            'train_vol': r.train_volatility,
            'test_return': r.test_return,
            'test_sharpe': r.test_sharpe,
            'test_vol': r.test_volatility,
            'test_max_dd': r.test_max_drawdown
        })
    
    return pd.DataFrame(data)


# =========================================
# MAIN (pour test)
# =========================================

if __name__ == "__main__":
    print("="*60)
    print("WALK-FORWARD ANALYSIS - TEST MODULE")
    print("="*60)
    
    # Données de test
    np.random.seed(42)
    dates = pd.date_range('2005-01-01', periods=180, freq='ME')
    
    # Features simulées
    features = pd.DataFrame({
        'feature1': np.random.randn(180),
        'feature2': np.random.randn(180),
        'feature3': np.random.randn(180),
    }, index=dates)
    
    # Rendements simulés avec un léger signal
    signal = (features['feature1'] > 0).astype(float) * 0.005
    noise = np.random.normal(0, 0.04, 180)
    returns = pd.Series(signal + noise, index=dates, name='returns')
    
    # Modèle factory simple
    class SimpleModel:
        def __init__(self, n_states=3):
            self.n_states = n_states
            self.threshold = 0
        
        def fit(self, X):
            self.threshold = np.mean(X[:, 0])
        
        def predict(self, X):
            return (X[:, 0] > self.threshold).astype(int)
    
    def model_factory():
        return SimpleModel(n_states=3)
    
    # Stratégie simple
    def strategy_func(model, features, returns):
        predictions = model.predict(features.values)
        # Long quand prédiction = 1, flat sinon
        strategy_returns = returns * predictions
        return strategy_returns
    
    # Test Walk-Forward
    print("\n[TEST 1] Walk-Forward Rolling...")
    wf = WalkForwardAnalyzer(train_window=36, test_window=12, step_size=12)
    
    summary_rolling = wf.run_walk_forward(
        features=features,
        returns=returns,
        model_factory=model_factory,
        strategy_func=strategy_func,
        method='rolling',
        verbose=True
    )
    
    print("\n[TEST 2] Walk-Forward Anchored...")
    summary_anchored = wf.run_walk_forward(
        features=features,
        returns=returns,
        model_factory=model_factory,
        strategy_func=strategy_func,
        method='anchored',
        verbose=True
    )
    
    print("\n[TEST 3] Comparaison des méthodes...")
    comparison = wf.compare_methods(
        features=features,
        returns=returns,
        model_factory=model_factory,
        strategy_func=strategy_func,
        verbose=True
    )
    
    print("\n[TEST 4] Overfitting Detection...")
    of_analysis = OverfittingDetector.analyze_walk_forward_results(summary_rolling)
    print(f"   Score: {of_analysis['score']}/100")
    print(f"   Level: {of_analysis['level']}")
    print(f"   {of_analysis['recommendation']}")
    
    print("\n[TEST 5] Export DataFrame...")
    results_df = create_results_dataframe(summary_rolling)
    print(results_df.head())
    
    print("\n" + "="*60)
    print("✅ WALK-FORWARD MODULE TEST COMPLETED!")
    print("="*60)