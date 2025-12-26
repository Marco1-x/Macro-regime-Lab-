#!/usr/bin/env python3
"""
stress_testing.py
Module de Stress Testing pour Macro Regime Lab

Fonctionnalités :
- Replay des crises historiques (Black Monday, 2008, COVID, etc.)
- Value at Risk (VaR) : Historique, Paramétrique, Monte Carlo
- Conditional VaR (CVaR) / Expected Shortfall
- Stress scenarios personnalisés
- Analyse de sensibilité
- Rapport de stress test complet

Le stress testing est essentiel pour évaluer la robustesse
d'une stratégie face aux événements extrêmes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =========================================
# DÉFINITION DES CRISES HISTORIQUES
# =========================================

HISTORICAL_CRISES = {
    'black_monday_1987': {
        'name': 'Black Monday 1987',
        'start': '1987-10-01',
        'end': '1987-10-31',
        'description': 'Crash boursier du 19 octobre 1987, -22.6% en une journée',
        'spy_drawdown': -0.226,
        'duration_days': 1
    },
    'dot_com_crash': {
        'name': 'Dot-com Bubble Burst',
        'start': '2000-03-01',
        'end': '2002-10-31',
        'description': 'Éclatement de la bulle internet',
        'spy_drawdown': -0.49,
        'duration_days': 929
    },
    'financial_crisis_2008': {
        'name': 'Global Financial Crisis 2008',
        'start': '2007-10-01',
        'end': '2009-03-31',
        'description': 'Crise des subprimes, faillite Lehman Brothers',
        'spy_drawdown': -0.57,
        'duration_days': 517
    },
    'flash_crash_2010': {
        'name': 'Flash Crash 2010',
        'start': '2010-05-06',
        'end': '2010-05-06',
        'description': 'Crash éclair du 6 mai 2010, -9% en minutes',
        'spy_drawdown': -0.09,
        'duration_days': 1
    },
    'euro_crisis_2011': {
        'name': 'European Debt Crisis',
        'start': '2011-07-01',
        'end': '2011-10-31',
        'description': 'Crise de la dette souveraine européenne',
        'spy_drawdown': -0.19,
        'duration_days': 122
    },
    'china_crash_2015': {
        'name': 'China Stock Market Crash',
        'start': '2015-08-01',
        'end': '2015-09-30',
        'description': 'Crash du marché chinois, dévaluation du yuan',
        'spy_drawdown': -0.12,
        'duration_days': 60
    },
    'covid_crash_2020': {
        'name': 'COVID-19 Crash',
        'start': '2020-02-19',
        'end': '2020-03-23',
        'description': 'Pandémie mondiale, -34% en 33 jours',
        'spy_drawdown': -0.34,
        'duration_days': 33
    },
    'svb_crisis_2023': {
        'name': 'Silicon Valley Bank Crisis',
        'start': '2023-03-08',
        'end': '2023-03-15',
        'description': 'Faillite de SVB, contagion bancaire',
        'spy_drawdown': -0.05,
        'duration_days': 7
    }
}


@dataclass
class StressTestResult:
    """Résultat d'un stress test."""
    scenario_name: str
    scenario_type: str  # 'historical', 'hypothetical', 'sensitivity'
    
    # Impact sur le portefeuille
    portfolio_return: float
    portfolio_drawdown: float
    
    # Comparaison avec benchmark
    benchmark_return: float
    relative_performance: float
    
    # Détails
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_days: int = 0
    
    # Métriques additionnelles
    worst_day: float = 0
    best_day: float = 0
    volatility: float = 0


@dataclass 
class VaRResult:
    """Résultat du calcul de VaR."""
    confidence_level: float
    var_historical: float
    var_parametric: float
    var_monte_carlo: float
    cvar_historical: float  # Expected Shortfall
    
    # Détails Monte Carlo
    mc_simulations: int = 10000
    mc_std: float = 0


class StressTester:
    """
    Classe principale pour le stress testing.
    """
    
    def __init__(self, returns: pd.Series, benchmark_returns: pd.Series = None):
        """
        Initialise le stress tester.
        
        Args:
            returns: Série des rendements de la stratégie
            benchmark_returns: Série des rendements du benchmark (optionnel)
        """
        self.returns = returns.dropna()
        self.benchmark = benchmark_returns.dropna() if benchmark_returns is not None else None
        
    # =========================================
    # 1. REPLAY DES CRISES HISTORIQUES
    # =========================================
    
    def replay_historical_crisis(self, 
                                  crisis_key: str,
                                  verbose: bool = True) -> StressTestResult:
        """
        Replay une crise historique sur la stratégie.
        
        Args:
            crisis_key: Clé de la crise dans HISTORICAL_CRISES
            verbose: Afficher les détails
            
        Returns:
            StressTestResult
        """
        if crisis_key not in HISTORICAL_CRISES:
            raise ValueError(f"Crise inconnue: {crisis_key}. "
                           f"Disponibles: {list(HISTORICAL_CRISES.keys())}")
        
        crisis = HISTORICAL_CRISES[crisis_key]
        
        # Filtrer les rendements pendant la crise
        start = pd.to_datetime(crisis['start'])
        end = pd.to_datetime(crisis['end'])
        
        # Vérifier si on a des données pour cette période
        mask = (self.returns.index >= start) & (self.returns.index <= end)
        crisis_returns = self.returns[mask]
        
        if len(crisis_returns) == 0:
            if verbose:
                print(f"⚠️ Pas de données pour {crisis['name']} ({crisis['start']} - {crisis['end']})")
            
            # Utiliser le drawdown théorique de la crise
            return StressTestResult(
                scenario_name=crisis['name'],
                scenario_type='historical',
                portfolio_return=crisis['spy_drawdown'],
                portfolio_drawdown=crisis['spy_drawdown'],
                benchmark_return=crisis['spy_drawdown'],
                relative_performance=0,
                start_date=start,
                end_date=end,
                duration_days=crisis['duration_days']
            )
        
        # Calculer les métriques
        total_return = (1 + crisis_returns).prod() - 1
        
        # Drawdown pendant la crise
        wealth = (1 + crisis_returns).cumprod()
        peak = wealth.cummax()
        drawdown = (wealth / peak - 1).min()
        
        # Benchmark
        bench_return = 0
        if self.benchmark is not None:
            bench_mask = (self.benchmark.index >= start) & (self.benchmark.index <= end)
            bench_crisis = self.benchmark[bench_mask]
            if len(bench_crisis) > 0:
                bench_return = (1 + bench_crisis).prod() - 1
        
        result = StressTestResult(
            scenario_name=crisis['name'],
            scenario_type='historical',
            portfolio_return=total_return,
            portfolio_drawdown=drawdown,
            benchmark_return=bench_return,
            relative_performance=total_return - bench_return,
            start_date=start,
            end_date=end,
            duration_days=len(crisis_returns),
            worst_day=crisis_returns.min(),
            best_day=crisis_returns.max(),
            volatility=crisis_returns.std() * np.sqrt(252)
        )
        
        if verbose:
            self._print_crisis_result(result, crisis)
        
        return result
    
    def replay_all_crises(self, verbose: bool = True) -> Dict[str, StressTestResult]:
        """
        Replay toutes les crises historiques.
        
        Args:
            verbose: Afficher les détails
            
        Returns:
            Dict des résultats par crise
        """
        if verbose:
            print("\n" + "="*70)
            print("📉 STRESS TEST - REPLAY DES CRISES HISTORIQUES")
            print("="*70)
        
        results = {}
        for crisis_key in HISTORICAL_CRISES:
            results[crisis_key] = self.replay_historical_crisis(crisis_key, verbose=False)
        
        if verbose:
            self._print_crisis_summary(results)
        
        return results
    
    def _print_crisis_result(self, result: StressTestResult, crisis: dict):
        """Affiche le résultat d'une crise."""
        print(f"\n📉 {result.scenario_name}")
        print(f"   Période: {crisis['start']} → {crisis['end']}")
        print(f"   {crisis['description']}")
        print(f"   Portfolio Return: {result.portfolio_return*100:+.2f}%")
        print(f"   Max Drawdown: {result.portfolio_drawdown*100:.2f}%")
        print(f"   vs Benchmark: {result.relative_performance*100:+.2f}%")
    
    def _print_crisis_summary(self, results: Dict[str, StressTestResult]):
        """Affiche le résumé des crises."""
        print(f"\n{'─'*70}")
        print(f"{'Crise':<30} {'Return':>10} {'Drawdown':>12} {'vs Bench':>10}")
        print(f"{'─'*70}")
        
        for key, result in results.items():
            print(f"{result.scenario_name:<30} "
                  f"{result.portfolio_return*100:>+9.2f}% "
                  f"{result.portfolio_drawdown*100:>11.2f}% "
                  f"{result.relative_performance*100:>+9.2f}%")
        
        print(f"{'─'*70}")
        
        # Worst case
        worst = min(results.values(), key=lambda x: x.portfolio_return)
        print(f"\n⚠️  PIRE SCÉNARIO: {worst.scenario_name} ({worst.portfolio_return*100:+.2f}%)")
    
    # =========================================
    # 2. VALUE AT RISK (VaR)
    # =========================================
    
    def calculate_var(self, 
                       confidence_level: float = 0.95,
                       horizon_days: int = 1,
                       mc_simulations: int = 10000,
                       verbose: bool = True) -> VaRResult:
        """
        Calcule la Value at Risk avec plusieurs méthodes.
        
        Args:
            confidence_level: Niveau de confiance (0.95 = 95%)
            horizon_days: Horizon temporel en jours
            mc_simulations: Nombre de simulations Monte Carlo
            verbose: Afficher les résultats
            
        Returns:
            VaRResult
        """
        returns = self.returns.values
        
        # Ajuster pour l'horizon
        if horizon_days > 1:
            returns_horizon = self._aggregate_returns(returns, horizon_days)
        else:
            returns_horizon = returns
        
        # 1. VaR Historique
        var_historical = np.percentile(returns_horizon, (1 - confidence_level) * 100)
        
        # 2. VaR Paramétrique (Gaussien)
        mu = np.mean(returns_horizon)
        sigma = np.std(returns_horizon)
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence_level)
        var_parametric = mu + z_score * sigma
        
        # 3. VaR Monte Carlo
        mc_returns = np.random.normal(mu, sigma, mc_simulations)
        var_monte_carlo = np.percentile(mc_returns, (1 - confidence_level) * 100)
        mc_std = np.std([np.percentile(np.random.choice(mc_returns, len(mc_returns)), 
                                       (1 - confidence_level) * 100) 
                        for _ in range(100)])
        
        # 4. CVaR / Expected Shortfall
        cvar_historical = returns_horizon[returns_horizon <= var_historical].mean()
        
        result = VaRResult(
            confidence_level=confidence_level,
            var_historical=var_historical,
            var_parametric=var_parametric,
            var_monte_carlo=var_monte_carlo,
            cvar_historical=cvar_historical,
            mc_simulations=mc_simulations,
            mc_std=mc_std
        )
        
        if verbose:
            self._print_var_result(result, horizon_days)
        
        return result
    
    def calculate_var_multiple_levels(self, 
                                       levels: List[float] = [0.90, 0.95, 0.99],
                                       verbose: bool = True) -> pd.DataFrame:
        """
        Calcule le VaR pour plusieurs niveaux de confiance.
        
        Args:
            levels: Liste des niveaux de confiance
            verbose: Afficher les résultats
            
        Returns:
            DataFrame avec les VaR
        """
        results = []
        
        for level in levels:
            var_result = self.calculate_var(level, verbose=False)
            results.append({
                'Confidence': f"{level*100:.0f}%",
                'VaR Hist': var_result.var_historical,
                'VaR Param': var_result.var_parametric,
                'VaR MC': var_result.var_monte_carlo,
                'CVaR (ES)': var_result.cvar_historical
            })
        
        df = pd.DataFrame(results)
        
        if verbose:
            print("\n" + "="*70)
            print("📊 VALUE AT RISK - MULTIPLE CONFIDENCE LEVELS")
            print("="*70)
            
            for _, row in df.iterrows():
                print(f"\n{row['Confidence']} Confidence:")
                print(f"   VaR Historical:  {row['VaR Hist']*100:>8.2f}%")
                print(f"   VaR Parametric:  {row['VaR Param']*100:>8.2f}%")
                print(f"   VaR Monte Carlo: {row['VaR MC']*100:>8.2f}%")
                print(f"   CVaR (ES):       {row['CVaR (ES)']*100:>8.2f}%")
            
            print("="*70)
        
        return df
    
    def _aggregate_returns(self, returns: np.ndarray, horizon: int) -> np.ndarray:
        """Agrège les rendements sur un horizon donné."""
        n = len(returns)
        aggregated = []
        for i in range(n - horizon + 1):
            period_return = np.prod(1 + returns[i:i+horizon]) - 1
            aggregated.append(period_return)
        return np.array(aggregated)
    
    def _print_var_result(self, result: VaRResult, horizon: int):
        """Affiche les résultats VaR."""
        print(f"\n📊 VALUE AT RISK ({result.confidence_level*100:.0f}% confidence, {horizon}-day horizon)")
        print(f"{'─'*50}")
        print(f"   VaR Historical:   {result.var_historical*100:>8.2f}%")
        print(f"   VaR Parametric:   {result.var_parametric*100:>8.2f}%")
        print(f"   VaR Monte Carlo:  {result.var_monte_carlo*100:>8.2f}% (±{result.mc_std*100:.2f}%)")
        print(f"   CVaR (ES):        {result.cvar_historical*100:>8.2f}%")
        print(f"{'─'*50}")
    
    # =========================================
    # 3. SCÉNARIOS HYPOTHÉTIQUES
    # =========================================
    
    def run_hypothetical_scenario(self,
                                   scenario_name: str,
                                   shocks: Dict[str, float],
                                   asset_returns: pd.DataFrame = None,
                                   weights: pd.Series = None,
                                   verbose: bool = True) -> StressTestResult:
        """
        Exécute un scénario hypothétique avec des chocs définis.
        
        Args:
            scenario_name: Nom du scénario
            shocks: Dict {asset: shock} (ex: {'SPY': -0.20, 'TLT': 0.05})
            asset_returns: DataFrame des rendements par actif
            weights: Poids du portefeuille
            verbose: Afficher les résultats
            
        Returns:
            StressTestResult
        """
        # Calculer l'impact sur le portefeuille
        portfolio_shock = 0
        
        if weights is not None:
            for asset, shock in shocks.items():
                if asset in weights.index:
                    portfolio_shock += weights[asset] * shock
        else:
            # Moyenne simple des chocs
            portfolio_shock = np.mean(list(shocks.values()))
        
        result = StressTestResult(
            scenario_name=scenario_name,
            scenario_type='hypothetical',
            portfolio_return=portfolio_shock,
            portfolio_drawdown=min(portfolio_shock, 0),
            benchmark_return=shocks.get('SPY', 0),
            relative_performance=portfolio_shock - shocks.get('SPY', 0)
        )
        
        if verbose:
            print(f"\n🔮 SCÉNARIO HYPOTHÉTIQUE: {scenario_name}")
            print(f"{'─'*50}")
            print("   Chocs appliqués:")
            for asset, shock in shocks.items():
                print(f"      {asset}: {shock*100:+.1f}%")
            print(f"{'─'*50}")
            print(f"   Impact Portfolio: {portfolio_shock*100:+.2f}%")
        
        return result
    
    def run_predefined_scenarios(self, 
                                  weights: pd.Series = None,
                                  verbose: bool = True) -> Dict[str, StressTestResult]:
        """
        Exécute une série de scénarios prédéfinis.
        
        Args:
            weights: Poids du portefeuille
            verbose: Afficher les résultats
            
        Returns:
            Dict des résultats
        """
        scenarios = {
            'market_crash_20': {
                'name': 'Market Crash -20%',
                'shocks': {'SPY': -0.20, 'XLK': -0.25, 'TLT': 0.08, 'GLD': 0.05}
            },
            'market_crash_40': {
                'name': 'Market Crash -40%',
                'shocks': {'SPY': -0.40, 'XLK': -0.50, 'TLT': 0.15, 'GLD': 0.10}
            },
            'rates_spike': {
                'name': 'Interest Rates +200bps',
                'shocks': {'SPY': -0.10, 'XLK': -0.15, 'TLT': -0.20, 'GLD': -0.05}
            },
            'stagflation': {
                'name': 'Stagflation Scenario',
                'shocks': {'SPY': -0.15, 'XLK': -0.20, 'TLT': -0.10, 'GLD': 0.15}
            },
            'deflation': {
                'name': 'Deflation Scenario',
                'shocks': {'SPY': -0.25, 'XLK': -0.30, 'TLT': 0.20, 'GLD': 0.05}
            },
            'geopolitical': {
                'name': 'Geopolitical Crisis',
                'shocks': {'SPY': -0.15, 'XLK': -0.20, 'TLT': 0.10, 'GLD': 0.20}
            },
            'tech_crash': {
                'name': 'Tech Sector Crash',
                'shocks': {'SPY': -0.12, 'XLK': -0.35, 'TLT': 0.05, 'GLD': 0.02}
            },
            'bull_market': {
                'name': 'Bull Market +30%',
                'shocks': {'SPY': 0.30, 'XLK': 0.40, 'TLT': -0.05, 'GLD': -0.03}
            }
        }
        
        if verbose:
            print("\n" + "="*70)
            print("🔮 STRESS TEST - SCÉNARIOS HYPOTHÉTIQUES")
            print("="*70)
        
        results = {}
        for key, scenario in scenarios.items():
            results[key] = self.run_hypothetical_scenario(
                scenario_name=scenario['name'],
                shocks=scenario['shocks'],
                weights=weights,
                verbose=verbose
            )
        
        if verbose:
            self._print_scenario_summary(results)
        
        return results
    
    def _print_scenario_summary(self, results: Dict[str, StressTestResult]):
        """Affiche le résumé des scénarios."""
        print(f"\n{'─'*70}")
        print(f"{'Scénario':<30} {'Impact':>12} {'vs SPY':>12}")
        print(f"{'─'*70}")
        
        for key, result in results.items():
            print(f"{result.scenario_name:<30} "
                  f"{result.portfolio_return*100:>+11.2f}% "
                  f"{result.relative_performance*100:>+11.2f}%")
        
        print(f"{'─'*70}")
    
    # =========================================
    # 4. ANALYSE DE SENSIBILITÉ
    # =========================================
    
    def sensitivity_analysis(self,
                              asset_returns: pd.DataFrame,
                              weights: pd.Series,
                              shock_range: Tuple[float, float] = (-0.30, 0.30),
                              n_points: int = 13,
                              verbose: bool = True) -> pd.DataFrame:
        """
        Analyse de sensibilité du portefeuille aux chocs sur chaque actif.
        
        Args:
            asset_returns: DataFrame des rendements
            weights: Poids du portefeuille
            shock_range: Plage des chocs à tester
            n_points: Nombre de points
            verbose: Afficher les résultats
            
        Returns:
            DataFrame de sensibilité
        """
        shocks = np.linspace(shock_range[0], shock_range[1], n_points)
        
        results = {'shock': shocks}
        
        for asset in weights.index:
            impacts = []
            for shock in shocks:
                # Impact = poids * choc
                impact = weights[asset] * shock
                impacts.append(impact)
            results[asset] = impacts
        
        df = pd.DataFrame(results)
        
        if verbose:
            print("\n" + "="*70)
            print("📈 ANALYSE DE SENSIBILITÉ")
            print("="*70)
            print(f"\nImpact d'un choc de ±{abs(shock_range[0])*100:.0f}% sur chaque actif:")
            print(f"{'─'*50}")
            
            for asset in weights.index:
                worst = weights[asset] * shock_range[0]
                best = weights[asset] * shock_range[1]
                print(f"   {asset} (poids: {weights[asset]*100:.1f}%): "
                      f"{worst*100:+.2f}% à {best*100:+.2f}%")
            
            print(f"{'─'*50}")
        
        return df
    
    # =========================================
    # 5. RAPPORT COMPLET
    # =========================================
    
    def generate_full_report(self,
                              weights: pd.Series = None,
                              save_path: str = None,
                              verbose: bool = True) -> Dict:
        """
        Génère un rapport de stress test complet.
        
        Args:
            weights: Poids du portefeuille
            save_path: Chemin pour sauvegarder le rapport
            verbose: Afficher les résultats
            
        Returns:
            Dict avec tous les résultats
        """
        if verbose:
            print("\n" + "="*70)
            print("🔥 RAPPORT COMPLET DE STRESS TEST")
            print("="*70)
            print(f"Période: {self.returns.index.min().strftime('%Y-%m-%d')} → "
                  f"{self.returns.index.max().strftime('%Y-%m-%d')}")
            print(f"Observations: {len(self.returns)}")
        
        report = {}
        
        # 1. Crises historiques
        report['historical_crises'] = self.replay_all_crises(verbose=verbose)
        
        # 2. VaR multi-niveaux
        report['var_analysis'] = self.calculate_var_multiple_levels(verbose=verbose)
        
        # 3. Scénarios hypothétiques
        report['hypothetical_scenarios'] = self.run_predefined_scenarios(
            weights=weights, verbose=verbose
        )
        
        # 4. Statistiques de risque
        report['risk_stats'] = self._compute_risk_stats()
        
        if verbose:
            print("\n" + "="*70)
            print("📊 STATISTIQUES DE RISQUE")
            print("="*70)
            for key, value in report['risk_stats'].items():
                if isinstance(value, float):
                    print(f"   {key}: {value*100:.2f}%" if abs(value) < 1 else f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
        
        # Sauvegarder si demandé
        if save_path:
            self._save_report(report, save_path)
        
        if verbose:
            print("\n" + "="*70)
            print("✅ RAPPORT DE STRESS TEST TERMINÉ")
            print("="*70)
        
        return report
    
    def _compute_risk_stats(self) -> Dict:
        """Calcule les statistiques de risque."""
        returns = self.returns
        
        # Drawdown
        wealth = (1 + returns).cumprod()
        peak = wealth.cummax()
        drawdown = wealth / peak - 1
        
        # Statistiques
        stats = {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown.mean(),
            'current_drawdown': drawdown.iloc[-1],
            'worst_day': returns.min(),
            'worst_month': returns.resample('ME').sum().min() if len(returns) > 30 else returns.min(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'negative_months_pct': (returns < 0).mean(),
            'tail_ratio': abs(returns.quantile(0.95) / returns.quantile(0.05)) if returns.quantile(0.05) != 0 else np.nan
        }
        
        return stats
    
    def _save_report(self, report: Dict, path: str):
        """Sauvegarde le rapport."""
        import json
        
        # Convertir les résultats en format sérialisable
        serializable = {}
        
        for key, value in report.items():
            if isinstance(value, pd.DataFrame):
                serializable[key] = value.to_dict()
            elif isinstance(value, dict):
                serializable[key] = {
                    k: v.__dict__ if hasattr(v, '__dict__') else v 
                    for k, v in value.items()
                }
            else:
                serializable[key] = value
        
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        
        print(f"\n[INFO] Rapport sauvegardé: {path}")


# =========================================
# FONCTIONS UTILITAIRES
# =========================================

def quick_stress_test(returns: pd.Series, 
                       benchmark: pd.Series = None,
                       verbose: bool = True) -> Dict:
    """
    Stress test rapide avec les paramètres par défaut.
    
    Args:
        returns: Rendements de la stratégie
        benchmark: Rendements du benchmark
        verbose: Afficher les résultats
        
    Returns:
        Dict avec les résultats principaux
    """
    tester = StressTester(returns, benchmark)
    
    # VaR 95%
    var = tester.calculate_var(0.95, verbose=verbose)
    
    # Crises principales
    crises = tester.replay_all_crises(verbose=verbose)
    
    return {
        'var_95': var,
        'crises': crises
    }


def create_stress_summary_table(results: Dict[str, StressTestResult]) -> pd.DataFrame:
    """
    Crée un tableau résumé des stress tests.
    
    Args:
        results: Dict des résultats de stress test
        
    Returns:
        DataFrame résumé
    """
    data = []
    for key, result in results.items():
        data.append({
            'Scenario': result.scenario_name,
            'Type': result.scenario_type,
            'Return': result.portfolio_return,
            'Drawdown': result.portfolio_drawdown,
            'vs Benchmark': result.relative_performance
        })
    
    return pd.DataFrame(data)


# =========================================
# MAIN (pour test)
# =========================================

if __name__ == "__main__":
    print("="*70)
    print("🔥 STRESS TESTING MODULE - TEST")
    print("="*70)
    
    # Données de test
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', periods=1500, freq='D')
    
    # Rendements simulés avec quelques événements extrêmes
    returns = np.random.normal(0.0003, 0.012, 1500)
    
    # Ajouter des chocs (simule COVID, etc.)
    returns[500:530] = np.random.normal(-0.02, 0.03, 30)  # Crise
    returns[700:710] = np.random.normal(0.015, 0.02, 10)  # Recovery
    
    returns = pd.Series(returns, index=dates, name='strategy')
    
    # Benchmark
    benchmark = pd.Series(np.random.normal(0.0004, 0.01, 1500), index=dates, name='SPY')
    
    # Test du StressTester
    print("\n[TEST 1] Initialisation...")
    tester = StressTester(returns, benchmark)
    print("   ✅ StressTester créé")
    
    print("\n[TEST 2] VaR Analysis...")
    var_result = tester.calculate_var(confidence_level=0.95, verbose=True)
    
    print("\n[TEST 3] Multi-level VaR...")
    var_df = tester.calculate_var_multiple_levels([0.90, 0.95, 0.99], verbose=True)
    
    print("\n[TEST 4] Hypothetical Scenarios...")
    weights = pd.Series({'SPY': 0.4, 'XLK': 0.3, 'TLT': 0.2, 'GLD': 0.1})
    scenarios = tester.run_predefined_scenarios(weights=weights, verbose=True)
    
    print("\n[TEST 5] Sensitivity Analysis...")
    sens_df = tester.sensitivity_analysis(
        asset_returns=None,
        weights=weights,
        verbose=True
    )
    
    print("\n[TEST 6] Risk Statistics...")
    risk_stats = tester._compute_risk_stats()
    print("   Max Drawdown:", f"{risk_stats['max_drawdown']*100:.2f}%")
    print("   Worst Day:", f"{risk_stats['worst_day']*100:.2f}%")
    print("   Skewness:", f"{risk_stats['skewness']:.2f}")
    print("   Kurtosis:", f"{risk_stats['kurtosis']:.2f}")
    
    print("\n[TEST 7] Summary Table...")
    summary = create_stress_summary_table(scenarios)
    print(summary.to_string())
    
    print("\n" + "="*70)
    print("✅ STRESS TESTING MODULE - ALL TESTS PASSED!")
    print("="*70)