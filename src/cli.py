#!/usr/bin/env python3
"""
Macro Regime & Factor Rotation Lab - CLI
Commands: detect-regimes, backtest, report
"""

import typer
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

app = typer.Typer(help="Macro Regime & Factor Rotation Lab CLI")

# =========================
# DATA FUNCTIONS
# =========================

def load_fred_data(data_dir: Path) -> pd.DataFrame:
    """Load FRED data from CSV files"""
    cpi = pd.read_csv(data_dir / "CPIAUCSL.csv", parse_dates=["observation_date"], index_col="observation_date")
    unrate = pd.read_csv(data_dir / "UNRATE.csv", parse_dates=["observation_date"], index_col="observation_date")
    usrec = pd.read_csv(data_dir / "USREC.csv", parse_dates=["observation_date"], index_col="observation_date")
    
    macro = pd.DataFrame({
        "CPI": cpi["CPIAUCSL"],
        "UNRATE": unrate["UNRATE"],
        "USREC": usrec["USREC"]
    })
    return macro.resample("ME").last().dropna()

def load_etf_prices(data_dir: Path) -> pd.DataFrame:
    """Load ETF prices from CSV"""
    prices = pd.read_csv(data_dir / "etf_prices.csv", parse_dates=["Date"], index_col="Date")
    return prices.resample("ME").last().dropna()

def build_features(macro: pd.DataFrame) -> pd.DataFrame:
    """Build macro features"""
    df = macro.copy()
    df["CPI_YoY"] = df["CPI"].pct_change(12)
    df["dUNRATE"] = df["UNRATE"].diff()
    return df.dropna()

def assign_regimes(features: pd.DataFrame) -> pd.Series:
    """Assign regimes using USREC + heuristics"""
    df = features.copy()
    infl_med = df["CPI_YoY"].rolling(60, min_periods=12).median()
    
    regime = pd.Series("Expansion", index=df.index)
    
    # USREC = 1 → Recession
    if "USREC" in df.columns:
        regime[df["USREC"] == 1] = "Recession"
    
    # Slowdown: high inflation + rising unemployment (not recession)
    slowdown_mask = (df["CPI_YoY"] > infl_med) & (df["dUNRATE"] > 0) & (regime != "Recession")
    regime[slowdown_mask] = "Slowdown"
    
    regime.name = "regime"
    return regime

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly returns"""
    return prices.pct_change().dropna()

def run_backtest(returns: pd.DataFrame, regimes: pd.Series, 
                 weights_by_regime: dict, cost_bps: float = 5.0) -> dict:
    """Run backtest with transaction costs"""
    df = returns.join(regimes, how="inner").dropna()
    
    port_returns = []
    costs_list = []
    weights_history = []
    current_weights = None
    
    for date in df.index:
        regime = df.loc[date, "regime"]
        target_weights = weights_by_regime.get(regime, weights_by_regime.get("Expansion", {}))
        
        # Portfolio return
        ret = sum(target_weights.get(col, 0) * df.loc[date, col] 
                  for col in returns.columns if col in target_weights)
        
        # Transaction costs
        cost = 0.0
        if current_weights:
            turnover = sum(abs(target_weights.get(col, 0) - current_weights.get(col, 0)) 
                          for col in returns.columns)
            cost = cost_bps / 10000 * turnover
            ret -= cost
        
        port_returns.append(ret)
        costs_list.append(cost)
        weights_history.append({"date": date, "regime": regime, **target_weights})
        current_weights = target_weights
    
    port_returns = pd.Series(port_returns, index=df.index, name="strategy")
    wealth = (1 + port_returns).cumprod()
    
    # Metrics
    annual_ret = port_returns.mean() * 12
    annual_vol = port_returns.std() * np.sqrt(12)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    max_dd = (wealth / wealth.cummax() - 1).min()
    
    return {
        "returns": port_returns,
        "wealth": wealth,
        "weights": pd.DataFrame(weights_history),
        "costs": pd.Series(costs_list, index=df.index),
        "metrics": {
            "CAGR": annual_ret,
            "Volatility": annual_vol,
            "Sharpe": sharpe,
            "MaxDD": max_dd,
            "Total_Return": wealth.iloc[-1] - 1
        }
    }

def run_benchmark_60_40(returns: pd.DataFrame) -> dict:
    """Run 60/40 benchmark (60% SPY, 40% TLT)"""
    weights = {"SPY": 0.6, "TLT": 0.4}
    bench_ret = sum(weights.get(col, 0) * returns[col] for col in returns.columns if col in weights)
    bench_ret = bench_ret.dropna()
    wealth = (1 + bench_ret).cumprod()
    
    annual_ret = bench_ret.mean() * 12
    annual_vol = bench_ret.std() * np.sqrt(12)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    max_dd = (wealth / wealth.cummax() - 1).min()
    
    return {
        "returns": bench_ret,
        "wealth": wealth,
        "metrics": {
            "CAGR": annual_ret,
            "Volatility": annual_vol,
            "Sharpe": sharpe,
            "MaxDD": max_dd,
            "Total_Return": wealth.iloc[-1] - 1
        }
    }

# =========================
# CLI COMMANDS
# =========================

@app.command()
def detect_regimes(
    data_dir: Path = typer.Option(Path("data/fred"), help="Directory with FRED CSV files"),
    output: Path = typer.Option(Path("output/regimes.csv"), help="Output CSV file")
):
    """Detect macro regimes from FRED data and save to CSV."""
    typer.echo("📊 Loading FRED data...")
    
    try:
        macro = load_fred_data(data_dir)
        typer.echo(f"   Loaded {len(macro)} months of data")
        
        features = build_features(macro)
        regimes = assign_regimes(features)
        
        # Save
        output.parent.mkdir(parents=True, exist_ok=True)
        result = pd.DataFrame({"date": regimes.index, "regime": regimes.values})
        result.to_csv(output, index=False)
        
        # Stats
        counts = regimes.value_counts()
        typer.echo(f"\n✅ Regimes saved to {output}")
        typer.echo("\n📈 Regime distribution:")
        for regime, count in counts.items():
            pct = count / len(regimes) * 100
            typer.echo(f"   {regime}: {count} months ({pct:.1f}%)")
            
    except FileNotFoundError as e:
        typer.echo(f"❌ Error: {e}", err=True)
        typer.echo("   Make sure CSV files exist in data/fred/", err=True)
        raise typer.Exit(1)

@app.command()
def backtest(
    data_dir: Path = typer.Option(Path("data"), help="Directory with data files"),
    output_dir: Path = typer.Option(Path("output"), help="Output directory"),
    cost_bps: float = typer.Option(5.0, help="Transaction costs in basis points"),
    start_date: str = typer.Option("2005-01-01", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, help="End date (YYYY-MM-DD)")
):
    """Run backtest and generate metrics, weights, and figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    typer.echo("📊 Loading data...")
    
    try:
        # Load data
        macro = load_fred_data(data_dir / "fred")
        prices = load_etf_prices(data_dir)
        
        # Filter dates
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        macro = macro[start:end]
        prices = prices[start:end]
        
        typer.echo(f"   Period: {macro.index[0].date()} to {macro.index[-1].date()}")
        
        # Process
        features = build_features(macro)
        regimes = assign_regimes(features)
        returns = compute_returns(prices)
        
        # Default weights
        weights_by_regime = {
            "Expansion": {"SPY": 0.6, "XLK": 0.4},
            "Slowdown": {"SPY": 0.4, "TLT": 0.4, "GLD": 0.2},
            "Recession": {"TLT": 0.7, "GLD": 0.3}
        }
        
        typer.echo("🔄 Running backtest...")
        
        # Strategy
        result = run_backtest(returns, regimes, weights_by_regime, cost_bps)
        
        # Benchmarks
        spy_ret = returns["SPY"].dropna()
        spy_wealth = (1 + spy_ret).cumprod()
        bench_60_40 = run_benchmark_60_40(returns)
        
        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Regimes CSV
        regimes_df = pd.DataFrame({"date": regimes.index, "regime": regimes.values})
        regimes_df.to_csv(output_dir / "regimes.csv", index=False)
        
        # Returns CSV
        all_returns = pd.DataFrame({
            "strategy": result["returns"],
            "spy": spy_ret,
            "benchmark_60_40": bench_60_40["returns"]
        })
        all_returns.to_csv(output_dir / "returns.csv")
        
        # Weights CSV
        result["weights"].to_csv(output_dir / "weights.csv", index=False)
        
        # Metrics JSON
        metrics = {
            "strategy": result["metrics"],
            "spy": {
                "CAGR": float(spy_ret.mean() * 12),
                "Volatility": float(spy_ret.std() * np.sqrt(12)),
                "Sharpe": float((spy_ret.mean() * 12) / (spy_ret.std() * np.sqrt(12))),
                "MaxDD": float((spy_wealth / spy_wealth.cummax() - 1).min()),
                "Total_Return": float(spy_wealth.iloc[-1] - 1)
            },
            "benchmark_60_40": bench_60_40["metrics"]
        }
        # Convert numpy types to Python types
        for key in metrics:
            for k, v in metrics[key].items():
                metrics[key][k] = float(v)
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Figures
        typer.echo("📈 Generating figures...")
        
        # Wealth curve
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(result["wealth"].index, result["wealth"].values, label="Strategy", linewidth=2)
        ax.plot(spy_wealth.index, spy_wealth.values, label="SPY", linestyle="--", alpha=0.7)
        ax.plot(bench_60_40["wealth"].index, bench_60_40["wealth"].values, 
                label="60/40", linestyle=":", alpha=0.7)
        ax.set_title("Growth of $1")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "wealth_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        # Drawdown
        strat_dd = (result["wealth"] / result["wealth"].cummax() - 1) * 100
        spy_dd = (spy_wealth / spy_wealth.cummax() - 1) * 100
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(strat_dd.index, strat_dd.values, 0, label="Strategy", alpha=0.5)
        ax.fill_between(spy_dd.index, spy_dd.values, 0, label="SPY", alpha=0.3)
        ax.set_title("Drawdown (%)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown %")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "drawdown.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        # Print metrics
        typer.echo(f"\n✅ Backtest complete! Results saved to {output_dir}/")
        typer.echo("\n📊 Performance Summary:")
        typer.echo("-" * 50)
        typer.echo(f"{'Metric':<15} {'Strategy':>12} {'SPY':>12} {'60/40':>12}")
        typer.echo("-" * 50)
        for metric in ["CAGR", "Volatility", "Sharpe", "MaxDD"]:
            s = metrics["strategy"][metric]
            b = metrics["spy"][metric]
            b60 = metrics["benchmark_60_40"][metric]
            if metric in ["CAGR", "Volatility", "MaxDD"]:
                typer.echo(f"{metric:<15} {s*100:>11.1f}% {b*100:>11.1f}% {b60*100:>11.1f}%")
            else:
                typer.echo(f"{metric:<15} {s:>12.2f} {b:>12.2f} {b60:>12.2f}")
        typer.echo("-" * 50)
        
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def report(
    output_dir: Path = typer.Option(Path("output"), help="Output directory with backtest results"),
    report_file: Path = typer.Option(Path("output/REPORT.md"), help="Output report file")
):
    """Generate Markdown report from backtest results."""
    typer.echo("📝 Generating report...")
    
    try:
        # Load metrics
        with open(output_dir / "metrics.json") as f:
            metrics = json.load(f)
        
        # Load regimes for stats
        regimes_df = pd.read_csv(output_dir / "regimes.csv") if (output_dir / "regimes.csv").exists() else None
        
        # Generate report
        report_content = f"""# Macro Regime & Factor Rotation Lab — Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. Executive Summary

This report presents the results of a macro regime-based factor rotation strategy.
The strategy identifies three macroeconomic regimes (Expansion, Slowdown, Recession)
and allocates across ETFs (SPY, TLT, GLD, XLK) accordingly.

---

## 2. Methodology

### 2.1 Regime Detection

Regimes are defined using FRED macroeconomic indicators:

| Regime | Definition |
|--------|------------|
| **Recession** | USREC = 1 (NBER recession indicator) |
| **Slowdown** | CPI YoY > rolling median AND ΔUNRATE > 0 |
| **Expansion** | Otherwise |

### 2.2 Portfolio Allocation

| Regime | SPY | TLT | GLD | XLK |
|--------|-----|-----|-----|-----|
| Expansion | 60% | 0% | 0% | 40% |
| Slowdown | 40% | 40% | 20% | 0% |
| Recession | 0% | 70% | 30% | 0% |

### 2.3 Backtest Parameters

- **Rebalancing:** Monthly
- **Transaction costs:** 5 bps per unit of turnover
- **Benchmarks:** SPY buy-and-hold, 60/40 portfolio

---

## 3. Performance Results

### 3.1 Summary Metrics

| Metric | Strategy | SPY | 60/40 |
|--------|----------|-----|-------|
| CAGR | {metrics['strategy']['CAGR']*100:.1f}% | {metrics['spy']['CAGR']*100:.1f}% | {metrics['benchmark_60_40']['CAGR']*100:.1f}% |
| Volatility | {metrics['strategy']['Volatility']*100:.1f}% | {metrics['spy']['Volatility']*100:.1f}% | {metrics['benchmark_60_40']['Volatility']*100:.1f}% |
| Sharpe Ratio | {metrics['strategy']['Sharpe']:.2f} | {metrics['spy']['Sharpe']:.2f} | {metrics['benchmark_60_40']['Sharpe']:.2f} |
| Max Drawdown | {metrics['strategy']['MaxDD']*100:.1f}% | {metrics['spy']['MaxDD']*100:.1f}% | {metrics['benchmark_60_40']['MaxDD']*100:.1f}% |

### 3.2 Wealth Curve

![Wealth Curve](wealth_curve.png)

### 3.3 Drawdown Analysis

![Drawdown](drawdown.png)

---

## 4. Regime Analysis

"""
        if regimes_df is not None:
            counts = regimes_df["regime"].value_counts()
            report_content += "### 4.1 Regime Distribution\n\n"
            report_content += "| Regime | Months | Percentage |\n|--------|--------|------------|\n"
            for regime, count in counts.items():
                pct = count / len(regimes_df) * 100
                report_content += f"| {regime} | {count} | {pct:.1f}% |\n"
        
        report_content += """
---

## 5. Limitations

1. **NBER dating lag:** Official recession dates are announced with delay
2. **Threshold sensitivity:** Rolling median period affects regime detection
3. **Look-ahead bias:** Careful to use only information available at decision time
4. **Transaction costs:** Real costs may vary with market conditions

---

## 6. Possible Improvements

1. **Hidden Markov Models:** Data-driven regime detection
2. **Additional indicators:** Yield curve slope, credit spreads, PMI
3. **Dynamic weights:** Optimize allocation within regimes
4. **Risk parity:** Volatility-adjusted position sizing

---

## 7. Conclusion

The macro regime rotation strategy demonstrates the value of adapting portfolio
allocation to macroeconomic conditions. By reducing equity exposure during
slowdowns and recessions, the strategy aims to improve risk-adjusted returns
compared to static benchmarks.

---

*Report generated by Macro Regime & Factor Rotation Lab*
"""
        
        # Save report
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w") as f:
            f.write(report_content)
        
        typer.echo(f"✅ Report saved to {report_file}")
        
    except FileNotFoundError as e:
        typer.echo(f"❌ Error: {e}", err=True)
        typer.echo("   Run 'backtest' command first to generate metrics.", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
