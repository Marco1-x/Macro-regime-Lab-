#!/usr/bin/env python3
"""
backtest.py
Module de backtesting pour Macro Regime Lab

Fonctionnalités :
- Backtest avec rebalancement mensuel
- Coûts de transaction fixes et dynamiques
- Slippage dynamique basé sur VIX (Option C)
- Gestion des poids par régime
- Métriques de performance intégrées

Le slippage dynamique simule l'impact de marché réaliste
qui augmente en période de volatilité élevée (VIX > 30).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# =========================================
# CONFIGURATION DU SLIPPAGE
# =========================================

@dataclass
class SlippageConfig:
    """
    Configuration du slippage dynamique.
    
    Le slippage représente le coût caché dû à l'impact de marché
    lors de l'exécution des ordres. Il augmente avec la volatilité.
    """
    # Slippage de base (bps)
    base_slippage_bps: float = 2.0
    
    # Multiplicateur VIX
    vix_threshold_low: float = 15.0   # VIX "normal"
    vix_threshold_high: float = 30.0  # VIX "stress"
    vix_multiplier_low: float = 1.0   # Multiplicateur quand VIX < 15
    vix_multiplier_high: float = 3.0  # Multiplicateur quand VIX > 30
    
    # Slippage max (bps)
    max_slippage_bps: float = 20.0
    
    # Ajustement par taille de trade
    size_impact: bool = True
    size_threshold: float = 0.1  # 10% du portfolio
    size_multiplier: float = 1.5
    
    def __repr__(self):
        return (f"SlippageConfig(base={self.base_slippage_bps}bps, "
                f"vix_range=[{self.vix_threshold_low}, {self.vix_threshold_high}], "
                f"max={self.max_slippage_bps}bps)")


@dataclass
class TransactionCostConfig:
    """Configuration complète des coûts de transaction."""
    # Coûts fixes
    commission_bps: float = 5.0      # Commission broker
    spread_bps: float = 1.0          # Bid-ask spread moyen
    
    # Slippage dynamique
    slippage_config: SlippageConfig = None
    
    # Options
    apply_slippage: bool = True
    apply_spread: bool = True
    
    def __post_init__(self):
        if self.slippage_config is None:
            self.slippage_config = SlippageConfig()


# =========================================
# CALCUL DU SLIPPAGE DYNAMIQUE
# =========================================

class DynamicSlippageCalculator:
    """
    Calcule le slippage dynamique basé sur le VIX.
    
    Le slippage augmente de manière non-linéaire avec le VIX,
    reflétant la détérioration de la liquidité en période de stress.
    """
    
    def __init__(self, config: SlippageConfig = None):
        """
        Initialise le calculateur.
        
        Args:
            config: Configuration du slippage
        """
        self.config = config or SlippageConfig()
        
    def calculate_slippage(self, 
                            vix: float,
                            turnover: float = 1.0,
                            trade_size: float = 0.0) -> float:
        """
        Calcule le slippage pour un trade.
        
        Args:
            vix: Niveau du VIX
            turnover: Turnover du rebalancement (0-2)
            trade_size: Taille relative du trade (0-1)
            
        Returns:
            Slippage en bps
        """
        cfg = self.config
        
        # 1. Base slippage
        slippage = cfg.base_slippage_bps
        
        # 2. Multiplicateur VIX (interpolation linéaire)
        if vix <= cfg.vix_threshold_low:
            vix_mult = cfg.vix_multiplier_low
        elif vix >= cfg.vix_threshold_high:
            vix_mult = cfg.vix_multiplier_high
        else:
            # Interpolation linéaire
            ratio = (vix - cfg.vix_threshold_low) / (cfg.vix_threshold_high - cfg.vix_threshold_low)
            vix_mult = cfg.vix_multiplier_low + ratio * (cfg.vix_multiplier_high - cfg.vix_multiplier_low)
        
        slippage *= vix_mult
        
        # 3. Ajustement taille de trade
        if cfg.size_impact and trade_size > cfg.size_threshold:
            size_factor = 1 + (trade_size - cfg.size_threshold) * cfg.size_multiplier
            slippage *= size_factor
        
        # 4. Appliquer le turnover
        slippage *= turnover
        
        # 5. Plafonner
        slippage = min(slippage, cfg.max_slippage_bps)
        
        return slippage
    
    def calculate_slippage_series(self, 
                                   vix_series: pd.Series,
                                   turnover_series: pd.Series = None) -> pd.Series:
        """
        Calcule le slippage pour une série temporelle.
        
        Args:
            vix_series: Série du VIX
            turnover_series: Série des turnovers
            
        Returns:
            Série des slippages en bps
        """
        if turnover_series is None:
            turnover_series = pd.Series(1.0, index=vix_series.index)
            
        slippages = []
        for date in vix_series.index:
            vix = vix_series.loc[date]
            turnover = turnover_series.loc[date] if date in turnover_series.index else 1.0
            slippage = self.calculate_slippage(vix, turnover)
            slippages.append(slippage)
            
        return pd.Series(slippages, index=vix_series.index, name='slippage_bps')


# =========================================
# BACKTEST ENGINE
# =========================================

@dataclass
class BacktestResult:
    """Résultat d'un backtest."""
    returns: pd.Series
    wealth: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    slippage: pd.Series
    
    # Métriques
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_costs: float = 0.0
    total_slippage: float = 0.0
    
    def __repr__(self):
        return (f"BacktestResult(Return={self.total_return:.2%}, "
                f"Sharpe={self.sharpe_ratio:.2f}, "
                f"MaxDD={self.max_drawdown:.2%})")


class BacktestEngine:
    """
    Moteur de backtesting avec slippage dynamique.
    
    Supporte:
    - Rebalancement basé sur régimes
    - Coûts de transaction réalistes
    - Slippage dynamique basé sur VIX
    - Contraintes de position
    """
    
    def __init__(self, 
                 cost_config: TransactionCostConfig = None,
                 slippage_calculator: DynamicSlippageCalculator = None):
        """
        Initialise le moteur de backtest.
        
        Args:
            cost_config: Configuration des coûts
            slippage_calculator: Calculateur de slippage
        """
        self.cost_config = cost_config or TransactionCostConfig()
        
        if slippage_calculator is None:
            self.slippage_calculator = DynamicSlippageCalculator(
                self.cost_config.slippage_config
            )
        else:
            self.slippage_calculator = slippage_calculator
            
    def run_backtest(self,
                      returns_df: pd.DataFrame,
                      regimes: pd.Series,
                      weights_by_regime: Dict[str, pd.Series],
                      vix_series: pd.Series = None,
                      initial_capital: float = 1.0,
                      verbose: bool = True) -> BacktestResult:
        """
        Exécute le backtest complet.
        
        Args:
            returns_df: DataFrame des rendements par actif
            regimes: Série des régimes
            weights_by_regime: Dict {regime: Series(poids)}
            vix_series: Série VIX pour slippage dynamique
            initial_capital: Capital initial
            verbose: Afficher les détails
            
        Returns:
            BacktestResult
        """
        # Aligner les données
        df = returns_df.join(regimes, how='inner').dropna()
        regime_col = regimes.name if regimes.name else 'regime'
        
        # Aligner VIX si disponible
        if vix_series is not None:
            vix_aligned = vix_series.reindex(df.index).ffill().bfill()
        else:
            vix_aligned = pd.Series(20.0, index=df.index)  # VIX moyen par défaut
        
        # Initialisation
        dates = df.index
        n_dates = len(dates)
        assets = returns_df.columns.tolist()
        
        port_returns = []
        positions_list = []
        turnover_list = []
        costs_list = []
        slippage_list = []
        
        current_weights = None
        
        if verbose:
            print("\n" + "="*60)
            print("📈 BACKTEST ENGINE - RUNNING")
            print("="*60)
            print(f"Period: {dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}")
            print(f"Assets: {assets}")
            print(f"Regimes: {list(weights_by_regime.keys())}")
            print(f"Dynamic Slippage: {vix_series is not None}")
        
        for t in range(n_dates):
            date = dates[t]
            regime = df[regime_col].iloc[t]
            rets = df[assets].iloc[t]
            vix = vix_aligned.iloc[t]
            
            # Poids cibles pour ce régime
            if regime not in weights_by_regime:
                # Régime inconnu: rester en cash
                target_weights = pd.Series(0.0, index=assets)
            else:
                target_weights = weights_by_regime[regime].reindex(assets).fillna(0.0)
            
            # Calcul du turnover
            if current_weights is None:
                turnover = target_weights.abs().sum()
            else:
                turnover = (target_weights - current_weights).abs().sum()
            
            # Calcul des coûts
            commission = self.cost_config.commission_bps / 10000 * turnover
            spread = self.cost_config.spread_bps / 10000 * turnover if self.cost_config.apply_spread else 0
            
            # Slippage dynamique
            if self.cost_config.apply_slippage:
                slippage_bps = self.slippage_calculator.calculate_slippage(vix, turnover)
                slippage = slippage_bps / 10000 * turnover
            else:
                slippage_bps = 0
                slippage = 0
            
            total_cost = commission + spread + slippage
            
            # Rendement net
            gross_return = (target_weights * rets).sum()
            net_return = gross_return - total_cost
            
            # Stocker
            port_returns.append(net_return)
            positions_list.append(target_weights.values)
            turnover_list.append(turnover)
            costs_list.append(total_cost)
            slippage_list.append(slippage)
            
            current_weights = target_weights.copy()
        
        # Convertir en Series/DataFrame
        port_returns = pd.Series(port_returns, index=dates, name='strategy')
        wealth = (1 + port_returns).cumprod() * initial_capital
        positions = pd.DataFrame(positions_list, index=dates, columns=assets)
        turnover = pd.Series(turnover_list, index=dates, name='turnover')
        costs = pd.Series(costs_list, index=dates, name='costs')
        slippage = pd.Series(slippage_list, index=dates, name='slippage')
        
        # Calculer les métriques
        result = self._create_result(
            port_returns, wealth, positions, turnover, costs, slippage
        )
        
        if verbose:
            self._print_summary(result)
        
        return result
    
    def _create_result(self, returns, wealth, positions, turnover, costs, slippage) -> BacktestResult:
        """Crée le résultat avec métriques."""
        # Calculs
        total_return = wealth.iloc[-1] / wealth.iloc[0] - 1
        n_years = len(returns) / 12  # Assume mensuel
        annualized_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        volatility = returns.std() * np.sqrt(12)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        peak = wealth.cummax()
        drawdown = (wealth / peak) - 1
        max_dd = drawdown.min()
        
        return BacktestResult(
            returns=returns,
            wealth=wealth,
            positions=positions,
            turnover=turnover,
            costs=costs,
            slippage=slippage,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_costs=costs.sum(),
            total_slippage=slippage.sum()
        )
    
    def _print_summary(self, result: BacktestResult):
        """Affiche le résumé."""
        print(f"\n{'─'*60}")
        print("📊 BACKTEST RESULTS")
        print(f"{'─'*60}")
        print(f"   Total Return:      {result.total_return:>10.2%}")
        print(f"   Annualized Return: {result.annualized_return:>10.2%}")
        print(f"   Volatility:        {result.volatility:>10.2%}")
        print(f"   Sharpe Ratio:      {result.sharpe_ratio:>10.2f}")
        print(f"   Max Drawdown:      {result.max_drawdown:>10.2%}")
        print(f"{'─'*60}")
        print(f"   Total Costs:       {result.total_costs*100:>10.2f}%")
        print(f"   Total Slippage:    {result.total_slippage*100:>10.2f}%")
        print(f"   Avg Turnover:      {result.turnover.mean():>10.2f}")
        print(f"{'─'*60}")
    
    def compare_with_without_slippage(self,
                                       returns_df: pd.DataFrame,
                                       regimes: pd.Series,
                                       weights_by_regime: Dict[str, pd.Series],
                                       vix_series: pd.Series,
                                       verbose: bool = True) -> Tuple[BacktestResult, BacktestResult]:
        """
        Compare les résultats avec et sans slippage dynamique.
        
        Args:
            returns_df: DataFrame des rendements
            regimes: Série des régimes
            weights_by_regime: Poids par régime
            vix_series: Série VIX
            verbose: Afficher les résultats
            
        Returns:
            Tuple (result_with_slippage, result_without_slippage)
        """
        if verbose:
            print("\n" + "="*60)
            print("📊 COMPARISON: WITH vs WITHOUT DYNAMIC SLIPPAGE")
            print("="*60)
        
        # Avec slippage dynamique
        if verbose:
            print("\n[1] WITH Dynamic Slippage:")
        result_with = self.run_backtest(
            returns_df, regimes, weights_by_regime, vix_series, verbose=False
        )
        
        # Sans slippage (juste les coûts fixes)
        original_apply = self.cost_config.apply_slippage
        self.cost_config.apply_slippage = False
        
        if verbose:
            print("\n[2] WITHOUT Dynamic Slippage:")
        result_without = self.run_backtest(
            returns_df, regimes, weights_by_regime, None, verbose=False
        )
        
        self.cost_config.apply_slippage = original_apply
        
        if verbose:
            print(f"\n{'─'*60}")
            print(f"{'Metric':<25} {'With Slippage':>15} {'Without':>15} {'Diff':>10}")
            print(f"{'─'*60}")
            print(f"{'Annualized Return':<25} {result_with.annualized_return:>14.2%} "
                  f"{result_without.annualized_return:>14.2%} "
                  f"{(result_with.annualized_return - result_without.annualized_return)*100:>+9.2f}%")
            print(f"{'Sharpe Ratio':<25} {result_with.sharpe_ratio:>15.2f} "
                  f"{result_without.sharpe_ratio:>15.2f} "
                  f"{result_with.sharpe_ratio - result_without.sharpe_ratio:>+10.2f}")
            print(f"{'Total Costs':<25} {result_with.total_costs*100:>14.2f}% "
                  f"{result_without.total_costs*100:>14.2f}% "
                  f"{(result_with.total_costs - result_without.total_costs)*100:>+9.2f}%")
            print(f"{'─'*60}")
            print(f"\n💡 Le slippage dynamique a coûté {(result_without.annualized_return - result_with.annualized_return)*100:.2f}% de return annuel")
        
        return result_with, result_without


# =========================================
# FONCTIONS UTILITAIRES
# =========================================

def quick_backtest(returns_df: pd.DataFrame,
                    regimes: pd.Series,
                    weights_by_regime: Dict[str, pd.Series],
                    tc_bps: float = 5.0,
                    vix_series: pd.Series = None) -> BacktestResult:
    """
    Backtest rapide avec paramètres par défaut.
    
    Args:
        returns_df: DataFrame des rendements
        regimes: Série des régimes
        weights_by_regime: Poids par régime
        tc_bps: Coûts de transaction en bps
        vix_series: Série VIX optionnelle
        
    Returns:
        BacktestResult
    """
    config = TransactionCostConfig(
        commission_bps=tc_bps,
        spread_bps=1.0,
        apply_slippage=vix_series is not None
    )
    
    engine = BacktestEngine(cost_config=config)
    return engine.run_backtest(
        returns_df, regimes, weights_by_regime, vix_series, verbose=False
    )


def simple_backtest(returns_df: pd.DataFrame,
                     regimes: pd.Series,
                     weights_dict: Dict[str, pd.Series],
                     tc_bps: float = 5.0) -> Tuple[pd.Series, pd.Series]:
    """
    Backtest simple compatible avec l'ancien code.
    
    Args:
        returns_df: DataFrame des rendements
        regimes: Série des régimes
        weights_dict: Dict {regime: weights}
        tc_bps: Transaction costs en bps
        
    Returns:
        Tuple (returns, wealth)
    """
    df = returns_df.join(regimes, how='inner').dropna()
    regime_col = regimes.name if regimes.name else 'regime'
    
    rets = df[returns_df.columns]
    regime = df[regime_col]
    
    port_rets = []
    curr_w = None
    
    for t in range(len(rets)):
        reg_t = regime.iloc[t]
        
        if reg_t not in weights_dict:
            port_rets.append(0.0)
            continue
        
        target_w = weights_dict[reg_t].reindex(rets.columns).fillna(0.0)
        
        if curr_w is None:
            turnover = target_w.abs().sum()
        else:
            turnover = (target_w - curr_w).abs().sum()
        
        tc = tc_bps / 10000.0 * turnover
        r_t = (target_w * rets.iloc[t]).sum() - tc
        port_rets.append(r_t)
        curr_w = target_w
    
    port_rets = pd.Series(port_rets, index=rets.index, name='strategy')
    wealth = (1 + port_rets).cumprod()
    
    return port_rets, wealth


def analyze_slippage_impact(vix_series: pd.Series,
                             config: SlippageConfig = None) -> pd.DataFrame:
    """
    Analyse l'impact du slippage sur différents niveaux de VIX.
    
    Args:
        vix_series: Série VIX
        config: Configuration slippage
        
    Returns:
        DataFrame d'analyse
    """
    calculator = DynamicSlippageCalculator(config)
    
    # Calculer slippage pour chaque observation
    slippages = calculator.calculate_slippage_series(vix_series)
    
    # Statistiques par niveau de VIX
    df = pd.DataFrame({
        'VIX': vix_series,
        'Slippage (bps)': slippages
    })
    
    bins = [0, 15, 20, 25, 30, 40, 100]
    labels = ['<15', '15-20', '20-25', '25-30', '30-40', '>40']
    df['VIX_Range'] = pd.cut(vix_series, bins=bins, labels=labels)
    
    summary = df.groupby('VIX_Range').agg({
        'VIX': ['count', 'mean'],
        'Slippage (bps)': ['mean', 'std', 'max']
    }).round(2)
    
    return summary


# =========================================
# MAIN (pour test)
# =========================================

if __name__ == "__main__":
    print("="*70)
    print("📈 BACKTEST ENGINE - TEST")
    print("="*70)
    
    # Données de test
    np.random.seed(42)
    n_months = 120
    dates = pd.date_range('2015-01-01', periods=n_months, freq='ME')
    
    # Rendements simulés
    returns_df = pd.DataFrame({
        'SPY': np.random.normal(0.008, 0.04, n_months),
        'TLT': np.random.normal(0.003, 0.02, n_months),
        'GLD': np.random.normal(0.002, 0.03, n_months),
        'XLK': np.random.normal(0.01, 0.05, n_months)
    }, index=dates)
    
    # Régimes simulés
    regimes = pd.Series(
        np.random.choice(['expansion', 'slowdown', 'contraction'], n_months, p=[0.5, 0.3, 0.2]),
        index=dates,
        name='regime'
    )
    
    # VIX simulé
    vix = pd.Series(
        np.clip(np.random.normal(20, 8, n_months), 10, 80),
        index=dates,
        name='VIX'
    )
    
    # Poids par régime
    weights = {
        'expansion': pd.Series({'SPY': 0.6, 'XLK': 0.3, 'TLT': 0.1, 'GLD': 0.0}),
        'slowdown': pd.Series({'SPY': 0.4, 'XLK': 0.2, 'TLT': 0.3, 'GLD': 0.1}),
        'contraction': pd.Series({'SPY': 0.1, 'XLK': 0.0, 'TLT': 0.6, 'GLD': 0.3})
    }
    
    # Test 1: SlippageConfig
    print("\n[TEST 1] SlippageConfig...")
    config = SlippageConfig(base_slippage_bps=2.0, vix_threshold_high=30.0)
    print(f"   ✅ {config}")
    
    # Test 2: DynamicSlippageCalculator
    print("\n[TEST 2] DynamicSlippageCalculator...")
    calc = DynamicSlippageCalculator(config)
    
    # Test différents VIX
    for vix_level in [12, 20, 30, 50]:
        slip = calc.calculate_slippage(vix_level, turnover=1.0)
        print(f"   VIX={vix_level}: {slip:.2f} bps")
    
    # Test 3: BacktestEngine
    print("\n[TEST 3] BacktestEngine...")
    engine = BacktestEngine()
    result = engine.run_backtest(
        returns_df, regimes, weights, vix, verbose=True
    )
    
    # Test 4: Compare with/without slippage
    print("\n[TEST 4] Compare With/Without Slippage...")
    result_with, result_without = engine.compare_with_without_slippage(
        returns_df, regimes, weights, vix, verbose=True
    )
    
    # Test 5: Quick backtest
    print("\n[TEST 5] Quick Backtest...")
    quick_result = quick_backtest(returns_df, regimes, weights, tc_bps=5.0, vix_series=vix)
    print(f"   ✅ {quick_result}")
    
    # Test 6: Simple backtest (legacy)
    print("\n[TEST 6] Simple Backtest (legacy)...")
    rets, wealth = simple_backtest(returns_df, regimes, weights, tc_bps=5.0)
    print(f"   ✅ Final wealth: {wealth.iloc[-1]:.2f}")
    
    # Test 7: Slippage impact analysis
    print("\n[TEST 7] Slippage Impact Analysis...")
    impact = analyze_slippage_impact(vix, config)
    print(impact)
    
    print("\n" + "="*70)
    print("✅ BACKTEST ENGINE - ALL TESTS PASSED!")
    print("="*70)