import pandas as pd
import numpy as np

class BacktestEngine:
    """Motor pour backtest des stratégies."""
    
    def __init__(self, tc_bps: float = 5.0):
        self.tc_bps = tc_bps
    
    def backtest(self, returns_df: pd.DataFrame, regimes: pd.Series, weights_dict: dict) -> dict:
        """
        Backtest une stratégie avec rebalancement.
        
        Args:
            returns_df: DataFrame des rendements
            regimes: Series des régimes
            weights_dict: dict {regime: Series(weights)}
        
        Returns:
            dict avec port_returns, wealth, etc.
        """
        df = returns_df.join(regimes, how="inner").dropna()
        regime_col = regimes.name if regimes.name else "regime"
        
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
            
            # Turnover et coûts de transaction
            if curr_w is None:
                turnover = np.abs(target_w).sum()
            else:
                turnover = np.abs(target_w - curr_w).sum()
            
            tc = (self.tc_bps / 10000.0) * turnover
            r_t = (target_w * rets.iloc[t]).sum() - tc
            port_rets.append(r_t)
            curr_w = target_w
        
        port_rets = pd.Series(port_rets, index=rets.index)
        wealth = (1 + port_rets).cumprod()
        
        return {
            "returns": port_rets,
            "wealth": wealth
        }