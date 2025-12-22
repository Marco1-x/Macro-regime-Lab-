"""
metrics.py
Module pour le calcul de métriques de performance avancées.

Fonctionnalités :
- Métriques classiques (Sharpe, Sortino, Calmar)
- Métriques de tail risk (VaR, CVaR, Expected Shortfall)
- Métriques de drawdown (Max DD, Average DD, Recovery time)
- Métriques de distribution (Skewness, Kurtosis, Jarque-Bera)
- Métriques avancées (Omega, Information Ratio, Treynor)
- Win/Loss analysis
- Risk-adjusted returns
- Benchmark comparison
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PerformanceMetrics:
    """
    Classe principale pour calculer toutes les métriques de performance.
    """
    
    def __init__(self, freq: int = 12, risk_free_rate: float = 0.02):
        """
        Initialise le calculateur de métriques.
        
        Args:
            freq: Fréquence annuelle (12 pour mensuel, 252 pour daily)
            risk_free_rate: Taux sans risque annuel
        """
        self.freq = freq
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(self, returns: pd.Series, benchmark: pd.Series = None) -> Dict:
        """
        Calcule toutes les métriques disponibles.
        
        Args:
            returns: Série des rendements
            benchmark: Série des rendements du benchmark (optionnel)
            
        Returns:
            Dictionnaire avec toutes les métriques
        """
        metrics = {}
        
        # Métriques de base
        metrics.update(self._basic_metrics(returns))
        
        # Métriques risk-adjusted
        metrics.update(self._risk_adjusted_metrics(returns))
        
        # Métriques de drawdown
        metrics.update(self._drawdown_metrics(returns))
        
        # Métriques de distribution
        metrics.update(self._distribution_metrics(returns))
        
        # Métriques de tail risk
        metrics.update(self._tail_risk_metrics(returns))
        
        # Métriques win/loss
        metrics.update(self._win_loss_metrics(returns))
        
        # Métriques vs benchmark
        if benchmark is not None:
            metrics.update(self._benchmark_metrics(returns, benchmark))
        
        return metrics
    
    def _basic_metrics(self, returns: pd.Series) -> Dict:
        """Métriques de base."""
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        n_years = n_periods / self.freq
        
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        volatility = returns.std() * np.sqrt(self.freq)
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'mean_return': returns.mean() * self.freq,
            'median_return': returns.median() * self.freq,
        }
    
    def _risk_adjusted_metrics(self, returns: pd.Series) -> Dict:
        """Métriques ajustées du risque."""
        mean_return = returns.mean() * self.freq
        volatility = returns.std() * np.sqrt(self.freq)
        
        # Sharpe Ratio
        excess_return = mean_return - self.risk_free_rate
        sharpe = excess_return / volatility if volatility > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.freq) if len(downside_returns) > 0 else 0
        sortino = excess_return / downside_std if downside_std > 0 else 0
        
        # Calmar Ratio
        max_dd = self._calculate_max_drawdown(returns)
        calmar = mean_return / abs(max_dd) if max_dd != 0 else 0
        
        # MAR Ratio (similar to Calmar)
        mar = mean_return / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'mar_ratio': mar,
            'downside_deviation': downside_std,
        }
    
    def _drawdown_metrics(self, returns: pd.Series) -> Dict:
        """Métriques de drawdown."""
        wealth = (1 + returns).cumprod()
        peak = wealth.cummax()
        drawdown = (wealth / peak - 1)
        
        max_dd = drawdown.min()
        avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        
        # Recovery time (jours pour récupérer du max DD)
        max_dd_idx = drawdown.idxmin()
        recovery_idx = wealth[wealth.index > max_dd_idx][wealth >= peak.loc[max_dd_idx]]
        
        if len(recovery_idx) > 0:
            recovery_periods = len(wealth[max_dd_idx:recovery_idx.index[0]])
        else:
            recovery_periods = np.nan  # Pas encore récupéré
        
        # Drawdown duration
        dd_periods = (drawdown < 0).sum()
        
        return {
            'max_drawdown': max_dd,
            'average_drawdown': avg_dd,
            'recovery_periods': recovery_periods,
            'drawdown_duration': dd_periods,
            'current_drawdown': drawdown.iloc[-1],
        }
    
    def _distribution_metrics(self, returns: pd.Series) -> Dict:
        """Métriques de distribution."""
        # Skewness (asymétrie)
        skew = returns.skew()
        
        # Kurtosis (queues de distribution)
        kurt = returns.kurtosis()
        
        # Jarque-Bera test (normalité)
        jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
        
        return {
            'skewness': skew,
            'kurtosis': kurt,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal': jb_pvalue > 0.05,  # p > 0.05 => normale
        }
    
    def _tail_risk_metrics(self, returns: pd.Series, confidence_levels: list = [0.95, 0.99]) -> Dict:
        """Métriques de tail risk."""
        metrics = {}
        
        for conf in confidence_levels:
            # Value at Risk (VaR)
            var = returns.quantile(1 - conf)
            metrics[f'var_{int(conf*100)}'] = var
            
            # Conditional VaR (CVaR / Expected Shortfall)
            cvar = returns[returns <= var].mean()
            metrics[f'cvar_{int(conf*100)}'] = cvar
        
        # Maximum loss
        metrics['max_loss'] = returns.min()
        
        # Maximum gain
        metrics['max_gain'] = returns.max()
        
        # Tail ratio (gain tail / loss tail)
        gain_tail = returns[returns > returns.quantile(0.95)].mean()
        loss_tail = abs(returns[returns < returns.quantile(0.05)].mean())
        tail_ratio = gain_tail / loss_tail if loss_tail != 0 else 0
        metrics['tail_ratio'] = tail_ratio
        
        return metrics
    
    def _win_loss_metrics(self, returns: pd.Series) -> Dict:
        """Métriques win/loss."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        n_wins = len(wins)
        n_losses = len(losses)
        n_total = len(returns)
        
        win_rate = n_wins / n_total if n_total > 0 else 0
        loss_rate = n_losses / n_total if n_total > 0 else 0
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        # Win/Loss ratio
        win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        
        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses != 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        return {
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'best_month': returns.max(),
            'worst_month': returns.min(),
        }
    
    def _benchmark_metrics(self, returns: pd.Series, benchmark: pd.Series) -> Dict:
        """Métriques vs benchmark."""
        # Aligner les séries
        aligned = pd.DataFrame({'portfolio': returns, 'benchmark': benchmark}).dropna()
        
        if len(aligned) == 0:
            return {}
        
        portfolio_returns = aligned['portfolio']
        benchmark_returns = aligned['benchmark']
        
        # Alpha (excess return)
        portfolio_mean = portfolio_returns.mean() * self.freq
        benchmark_mean = benchmark_returns.mean() * self.freq
        alpha = portfolio_mean - benchmark_mean
        
        # Beta
        covariance = portfolio_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # Information Ratio
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(self.freq)
        information_ratio = alpha / tracking_error if tracking_error != 0 else 0
        
        # Treynor Ratio
        excess_return = portfolio_mean - self.risk_free_rate
        treynor = excess_return / beta if beta != 0 else 0
        
        # Correlation
        correlation = portfolio_returns.corr(benchmark_returns)
        
        # Up/Down capture
        up_periods = benchmark_returns > 0
        down_periods = benchmark_returns < 0
        
        up_capture = (portfolio_returns[up_periods].mean() / benchmark_returns[up_periods].mean()) if up_periods.any() else 0
        down_capture = (portfolio_returns[down_periods].mean() / benchmark_returns[down_periods].mean()) if down_periods.any() else 0
        
        return {
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'up_capture': up_capture,
            'down_capture': down_capture,
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calcule le max drawdown."""
        wealth = (1 + returns).cumprod()
        peak = wealth.cummax()
        drawdown = (wealth / peak - 1)
        return drawdown.min()
    
    def calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calcule l'Omega Ratio.
        
        Args:
            returns: Série des rendements
            threshold: Seuil de rendement minimum acceptable
            
        Returns:
            Omega ratio
        """
        excess_returns = returns - threshold / self.freq
        
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        
        omega = gains / losses if losses != 0 else np.inf
        
        return omega
    
    def calculate_rolling_metrics(self, 
                                  returns: pd.Series,
                                  window: int = 12) -> pd.DataFrame:
        """
        Calcule des métriques roulantes.
        
        Args:
            returns: Série des rendements
            window: Fenêtre en périodes
            
        Returns:
            DataFrame avec métriques roulantes
        """
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Sharpe roulant
        rolling_mean = returns.rolling(window).mean() * self.freq
        rolling_std = returns.rolling(window).std() * np.sqrt(self.freq)
        rolling_metrics['sharpe'] = (rolling_mean - self.risk_free_rate) / rolling_std
        
        # Volatilité roulante
        rolling_metrics['volatility'] = rolling_std
        
        # Win rate roulant
        rolling_metrics['win_rate'] = returns.rolling(window).apply(lambda x: (x > 0).mean())
        
        # Max DD roulant
        def rolling_max_dd(x):
            wealth = (1 + x).cumprod()
            peak = wealth.cummax()
            dd = (wealth / peak - 1)
            return dd.min()
        
        rolling_metrics['max_dd'] = returns.rolling(window).apply(rolling_max_dd)
        
        return rolling_metrics
    
    def generate_metrics_report(self, 
                               returns: pd.Series,
                               benchmark: pd.Series = None) -> pd.DataFrame:
        """
        Génère un rapport complet des métriques.
        
        Args:
            returns: Série des rendements
            benchmark: Benchmark (optionnel)
            
        Returns:
            DataFrame formaté avec toutes les métriques
        """
        metrics = self.calculate_all_metrics(returns, benchmark)
        
        # Formater en DataFrame
        report = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        
        # Formater selon le type
        percentage_metrics = [
            'total_return', 'cagr', 'volatility', 'mean_return', 'median_return',
            'downside_deviation', 'max_drawdown', 'average_drawdown', 'current_drawdown',
            'win_rate', 'loss_rate', 'avg_win', 'avg_loss', 'alpha', 'tracking_error',
            'var_95', 'var_99', 'cvar_95', 'cvar_99', 'max_loss', 'max_gain',
            'best_month', 'worst_month', 'expectancy'
        ]
        
        for metric in percentage_metrics:
            if metric in report.index:
                report.loc[metric, 'Formatted'] = f"{report.loc[metric, 'Value']:.2%}"
        
        ratio_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'mar_ratio',
            'win_loss_ratio', 'profit_factor', 'information_ratio', 'treynor_ratio',
            'beta', 'correlation', 'tail_ratio', 'up_capture', 'down_capture'
        ]
        
        for metric in ratio_metrics:
            if metric in report.index:
                report.loc[metric, 'Formatted'] = f"{report.loc[metric, 'Value']:.2f}"
        
        return report


# Fonctions utilitaires pour compatibilité
def compute_stats(returns: pd.Series, freq: int = 12) -> Dict:
    """Wrapper pour compatibilité avec l'ancien code."""
    calculator = PerformanceMetrics(freq=freq)
    return calculator.calculate_all_metrics(returns)


def compute_advanced_stats(returns: pd.Series, freq: int = 12) -> Dict:
    """Version avancée avec toutes les métriques."""
    calculator = PerformanceMetrics(freq=freq)
    return calculator.calculate_all_metrics(returns)