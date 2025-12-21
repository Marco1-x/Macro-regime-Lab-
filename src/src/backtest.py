"""
backtest.py
Module pour le backtesting avancé de stratégies de trading.

Fonctionnalités :
- Backtesting avec transaction costs réalistes
- Slippage modeling (fonction de la volatilité)
- Market impact (fonction de la taille des ordres)
- Position sizing intelligent
- Risk management (stop-loss, position limits)
- Portfolio rebalancing avec contraintes
- Performance tracking détaillé
- Multiple strategies comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BacktestEngine:
    """
    Moteur de backtesting avancé pour stratégies de trading.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 transaction_cost_bps: float = 5.0,
                 slippage_bps: float = 2.0,
                 use_market_impact: bool = True,
                 use_dynamic_slippage: bool = True):
        """
        Initialise le moteur de backtesting.
        
        Args:
            initial_capital: Capital initial
            transaction_cost_bps: Frais de transaction en basis points
            slippage_bps: Slippage en basis points
            use_market_impact: Utiliser le market impact
            use_dynamic_slippage: Slippage dynamique basé sur la volatilité
        """
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.use_market_impact = use_market_impact
        self.use_dynamic_slippage = use_dynamic_slippage
        
        # État du portfolio
        self.current_weights = None
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        
        # Tracking
        self.trades = []
        self.positions = []
        self.portfolio_history = []
    
    def calculate_transaction_costs(self, 
                                    turnover: float, 
                                    portfolio_value: float) -> float:
        """
        Calcule les coûts de transaction.
        
        Args:
            turnover: Turnover total (somme des valeurs absolues des changements)
            portfolio_value: Valeur actuelle du portfolio
            
        Returns:
            Coût en dollars
        """
        cost_rate = self.transaction_cost_bps / 10000.0
        cost = turnover * portfolio_value * cost_rate
        
        return cost
    
    def calculate_slippage(self, 
                          returns: pd.Series,
                          volatility: float = None,
                          trade_size: float = None) -> float:
        """
        Calcule le slippage.
        
        Args:
            returns: Série de rendements pour calculer la volatilité
            volatility: Volatilité (si déjà calculée)
            trade_size: Taille du trade (pour market impact)
            
        Returns:
            Slippage en basis points
        """
        base_slippage = self.slippage_bps
        
        if self.use_dynamic_slippage:
            if volatility is None:
                volatility = returns.std() * np.sqrt(252)
            
            # Slippage augmente avec la volatilité
            vol_factor = min(volatility / 0.20, 3.0)  # Cap à 3x
            base_slippage *= vol_factor
        
        if self.use_market_impact and trade_size is not None:
            # Market impact augmente avec la taille du trade
            # Modèle simplifié: impact = sqrt(trade_size) * factor
            impact_factor = np.sqrt(trade_size / self.initial_capital)
            base_slippage *= (1 + impact_factor)
        
        return base_slippage
    
    def apply_position_limits(self, 
                             target_weights: pd.Series,
                             max_position: float = 0.40,
                             min_position: float = 0.0) -> pd.Series:
        """
        Applique des limites de position.
        
        Args:
            target_weights: Poids cibles
            max_position: Position max par actif
            min_position: Position min par actif
            
        Returns:
            Poids ajustés
        """
        adjusted = target_weights.copy()
        
        # Appliquer les limites
        adjusted = adjusted.clip(lower=min_position, upper=max_position)
        
        # Renormaliser pour que la somme = 1
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total
        
        return adjusted
    
    def calculate_turnover(self, 
                          current_weights: pd.Series,
                          target_weights: pd.Series) -> float:
        """
        Calcule le turnover.
        
        Args:
            current_weights: Poids actuels
            target_weights: Poids cibles
            
        Returns:
            Turnover total
        """
        if current_weights is None:
            return target_weights.abs().sum()
        
        # Aligner les index
        current_weights = current_weights.reindex(target_weights.index, fill_value=0.0)
        
        turnover = (target_weights - current_weights).abs().sum()
        
        return turnover
    
    def execute_rebalance(self,
                         date: pd.Timestamp,
                         target_weights: pd.Series,
                         returns: pd.Series,
                         volatility: Optional[pd.Series] = None) -> Dict:
        """
        Exécute un rebalancement.
        
        Args:
            date: Date du rebalancement
            target_weights: Poids cibles
            returns: Rendements de la période
            volatility: Volatilité des actifs (optionnel)
            
        Returns:
            Dictionnaire avec les détails du trade
        """
        # Calculer le turnover
        turnover = self.calculate_turnover(self.current_weights, target_weights)
        
        # Coûts de transaction
        tc = self.calculate_transaction_costs(turnover, self.portfolio_value)
        
        # Slippage
        avg_vol = volatility.mean() if volatility is not None else None
        slippage_bps = self.calculate_slippage(returns, avg_vol, turnover * self.portfolio_value)
        slippage_cost = (slippage_bps / 10000.0) * turnover * self.portfolio_value
        
        # Total des coûts
        total_costs = tc + slippage_cost
        
        # Rendement du portfolio (avant coûts)
        if self.current_weights is not None:
            aligned_weights = self.current_weights.reindex(returns.index, fill_value=0.0)
            portfolio_return = (aligned_weights * returns).sum()
        else:
            portfolio_return = 0.0
        
        # Mise à jour de la valeur du portfolio
        self.portfolio_value = self.portfolio_value * (1 + portfolio_return) - total_costs
        
        # Enregistrer le trade
        trade_info = {
            'date': date,
            'turnover': turnover,
            'transaction_cost': tc,
            'slippage_cost': slippage_cost,
            'total_cost': total_costs,
            'portfolio_return': portfolio_return,
            'portfolio_value': self.portfolio_value,
            'weights': target_weights.to_dict()
        }
        
        self.trades.append(trade_info)
        
        # Mettre à jour les poids actuels
        self.current_weights = target_weights
        
        return trade_info
    
    def run_backtest(self,
                    returns: pd.DataFrame,
                    regimes: pd.Series,
                    weights_by_regime: Dict[str, pd.Series],
                    volatility: Optional[pd.DataFrame] = None,
                    rebalance_frequency: str = 'M') -> pd.DataFrame:
        """
        Exécute le backtest complet.
        
        Args:
            returns: DataFrame des rendements
            regimes: Series des régimes
            weights_by_regime: Dict {regime: weights}
            volatility: DataFrame de volatilité (optionnel)
            rebalance_frequency: Fréquence de rebalancement
            
        Returns:
            DataFrame avec l'historique du portfolio
        """
        print(f"[INFO] Running backtest from {returns.index.min()} to {returns.index.max()}")
        print(f"[INFO] Initial capital: ${self.initial_capital:,.0f}")
        print(f"[INFO] Transaction costs: {self.transaction_cost_bps} bps")
        print(f"[INFO] Base slippage: {self.slippage_bps} bps")
        
        # Aligner les données
        aligned = returns.join(regimes, how='inner').dropna()
        
        portfolio_history = []
        
        for date in aligned.index:
            regime = aligned.loc[date, regimes.name]
            period_returns = aligned.loc[date, returns.columns]
            
            # Obtenir les poids cibles
            if regime not in weights_by_regime:
                # Si régime inconnu, rester en cash
                target_weights = pd.Series(0.0, index=returns.columns)
            else:
                target_weights = weights_by_regime[regime].reindex(returns.columns, fill_value=0.0)
            
            # Appliquer les limites de position
            target_weights = self.apply_position_limits(target_weights)
            
            # Volatilité de la période
            period_vol = volatility.loc[date] if volatility is not None else None
            
            # Exécuter le rebalancement
            trade_info = self.execute_rebalance(date, target_weights, period_returns, period_vol)
            
            # Enregistrer l'historique
            portfolio_history.append({
                'date': date,
                'regime': regime,
                'portfolio_value': self.portfolio_value,
                'return': trade_info['portfolio_return'],
                'costs': trade_info['total_cost'],
                'turnover': trade_info['turnover']
            })
        
        # Créer le DataFrame de résultats
        results_df = pd.DataFrame(portfolio_history).set_index('date')
        
        # Calculer les rendements cumulés
        results_df['wealth'] = results_df['portfolio_value'] / self.initial_capital
        results_df['cumulative_return'] = results_df['wealth'] - 1
        
        # Calculer les coûts cumulés
        results_df['cumulative_costs'] = results_df['costs'].cumsum()
        results_df['cumulative_turnover'] = results_df['turnover'].cumsum()
        
        print(f"\n[BACKTEST SUMMARY]")
        print(f"Final portfolio value: ${results_df['portfolio_value'].iloc[-1]:,.0f}")
        print(f"Total return: {results_df['cumulative_return'].iloc[-1]:.2%}")
        print(f"Total costs: ${results_df['cumulative_costs'].iloc[-1]:,.0f}")
        print(f"Average turnover: {results_df['turnover'].mean():.2%}")
        
        return results_df


class PositionSizer:
    """
    Classe pour le position sizing intelligent.
    """
    
    def __init__(self, method: str = 'equal_weight'):
        """
        Initialise le position sizer.
        
        Args:
            method: Méthode de sizing ('equal_weight', 'risk_parity', 'kelly')
        """
        self.method = method
    
    def calculate_sizes(self,
                       assets: List[str],
                       expected_returns: Optional[pd.Series] = None,
                       volatilities: Optional[pd.Series] = None,
                       covariance: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Calcule les tailles de position.
        
        Args:
            assets: Liste des actifs
            expected_returns: Rendements attendus (optionnel)
            volatilities: Volatilités (optionnel)
            covariance: Matrice de covariance (optionnel)
            
        Returns:
            Series avec les poids
        """
        if self.method == 'equal_weight':
            return self._equal_weight(assets)
        
        elif self.method == 'risk_parity':
            if volatilities is None:
                raise ValueError("Risk parity requires volatilities")
            return self._risk_parity(assets, volatilities)
        
        elif self.method == 'kelly':
            if expected_returns is None or volatilities is None:
                raise ValueError("Kelly criterion requires returns and volatilities")
            return self._kelly_criterion(assets, expected_returns, volatilities)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _equal_weight(self, assets: List[str]) -> pd.Series:
        """Equal weight allocation."""
        n = len(assets)
        return pd.Series(1.0 / n, index=assets)
    
    def _risk_parity(self, assets: List[str], volatilities: pd.Series) -> pd.Series:
        """
        Risk parity: poids inversement proportionnels à la volatilité.
        """
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        return weights
    
    def _kelly_criterion(self, 
                        assets: List[str],
                        expected_returns: pd.Series,
                        volatilities: pd.Series) -> pd.Series:
        """
        Kelly criterion: f = (expected_return) / (variance).
        On applique un fractional Kelly (50%) pour plus de prudence.
        """
        variance = volatilities ** 2
        kelly_fractions = expected_returns / variance
        
        # Fractional Kelly (50%)
        kelly_fractions = kelly_fractions * 0.5
        
        # Clip aux valeurs positives et normaliser
        kelly_fractions = kelly_fractions.clip(lower=0.0)
        
        if kelly_fractions.sum() > 0:
            weights = kelly_fractions / kelly_fractions.sum()
        else:
            weights = self._equal_weight(assets)
        
        return weights


class RiskManager:
    """
    Classe pour la gestion du risque.
    """
    
    def __init__(self, 
                 max_drawdown: float = 0.20,
                 max_leverage: float = 1.0,
                 position_limit: float = 0.30):
        """
        Initialise le risk manager.
        
        Args:
            max_drawdown: Drawdown maximum acceptable
            max_leverage: Leverage maximum
            position_limit: Limite par position
        """
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.position_limit = position_limit
    
    def check_drawdown(self, wealth_series: pd.Series) -> bool:
        """
        Vérifie si le drawdown dépasse la limite.
        
        Args:
            wealth_series: Série de la richesse cumulée
            
        Returns:
            True si drawdown acceptable
        """
        peak = wealth_series.cummax()
        drawdown = (wealth_series / peak - 1).min()
        
        return drawdown >= -self.max_drawdown
    
    def adjust_for_risk(self, 
                       weights: pd.Series,
                       current_drawdown: float) -> pd.Series:
        """
        Ajuste les poids en fonction du risque actuel.
        
        Args:
            weights: Poids initiaux
            current_drawdown: Drawdown actuel
            
        Returns:
            Poids ajustés
        """
        if current_drawdown < -self.max_drawdown * 0.5:
            # Si on approche de la limite, réduire l'exposition
            scale_factor = 0.5
            weights = weights * scale_factor
        
        # Appliquer les limites de position
        weights = weights.clip(upper=self.position_limit)
        
        # Renormaliser
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        return weights


# Fonction utilitaire pour compatibilité
def backtest_regime_strategy(returns: pd.DataFrame,
                             regimes: pd.Series,
                             weights_by_regime: Dict[str, pd.Series],
                             transaction_cost_bps: float = 5.0) -> Tuple[pd.Series, pd.Series]:
    """
    Wrapper pour compatibilité avec l'ancien code.
    
    Returns:
        portfolio_returns, wealth_curve
    """
    engine = BacktestEngine(
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=2.0,
        use_market_impact=True
    )
    
    results = engine.run_backtest(returns, regimes, weights_by_regime)
    
    portfolio_returns = results['return']
    wealth_curve = results['wealth']
    
    return portfolio_returns, wealth_curve