import pandas as pd
import numpy as np

class PerformanceMetrics:
    """Calcule les métriques de performance."""
    
    def calculate_all_metrics(self, returns: pd.Series, freq: int = 12) -> dict:
        """
        Calcule les métriques principales.
        
        Args:
            returns: Series des rendements (mensuels par défaut)
            freq: fréquence annualisée (12 pour mensuel)
        
        Returns:
            dict avec CAGR, Vol, Sharpe, MaxDrawdown
        """
        avg = returns.mean() * freq
        vol = returns.std() * np.sqrt(freq)
        sharpe = avg / vol if vol > 0 else np.nan
        
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum / peak - 1)
        mdd = dd.min()
        
        return {
            "cagr": avg,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd
        }