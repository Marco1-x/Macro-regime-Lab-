#!/usr/bin/env python3
"""
tests/test_walk_forward.py
Unit tests for walk forward analysis module
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range('2010-01-01', periods=120, freq='ME')
    return pd.DataFrame({
        'SPY': np.random.normal(0.008, 0.04, 120),
        'TLT': np.random.normal(0.003, 0.02, 120),
        'GLD': np.random.normal(0.002, 0.03, 120)
    }, index=dates)

@pytest.fixture
def sample_features():
    np.random.seed(42)
    dates = pd.date_range('2010-01-01', periods=120, freq='ME')
    return pd.DataFrame({
        'CPI_YoY': np.random.normal(0.02, 0.01, 120),
        'dUNRATE': np.random.normal(0, 0.1, 120),
        'slope': np.random.normal(1.5, 0.5, 120)
    }, index=dates)

@pytest.fixture
def sample_regimes():
    np.random.seed(42)
    dates = pd.date_range('2010-01-01', periods=120, freq='ME')
    return pd.Series(
        np.random.choice(['expansion', 'slowdown', 'contraction'], 120),
        index=dates, name='regime'
    )


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig."""
    
    def test_default_config(self):
        from src.walk_forward import WalkForwardConfig
        config = WalkForwardConfig()
        assert config.train_window == 36
        assert config.test_window == 12
        assert config.step_size == 1
    
    def test_custom_config(self):
        from src.walk_forward import WalkForwardConfig
        config = WalkForwardConfig(train_window=48, test_window=6, step_size=3)
        assert config.train_window == 48
        assert config.test_window == 6
        assert config.step_size == 3


class TestWalkForwardAnalyzer:
    """Tests for WalkForwardAnalyzer."""
    
    def test_initialization(self):
        from src.walk_forward import WalkForwardAnalyzer
        analyzer = WalkForwardAnalyzer(train_window=36, test_window=12)
        assert analyzer.train_window == 36
        assert analyzer.test_window == 12
    
    def test_generate_windows(self, sample_returns):
        from src.walk_forward import WalkForwardAnalyzer
        analyzer = WalkForwardAnalyzer(train_window=36, test_window=12)
        windows = analyzer.generate_windows(sample_returns)
        assert len(windows) > 0
        for train_idx, test_idx in windows:
            assert len(train_idx) == 36
            assert len(test_idx) == 12
    
    def test_run_walk_forward(self, sample_returns, sample_features, sample_regimes):
        from src.walk_forward import WalkForwardAnalyzer
        
        weights = {
            'expansion': pd.Series({'SPY': 0.6, 'TLT': 0.2, 'GLD': 0.2}),
            'slowdown': pd.Series({'SPY': 0.4, 'TLT': 0.4, 'GLD': 0.2}),
            'contraction': pd.Series({'SPY': 0.1, 'TLT': 0.6, 'GLD': 0.3})
        }
        
        analyzer = WalkForwardAnalyzer(train_window=36, test_window=12)
        results = analyzer.run_walk_forward(
            sample_returns, sample_features, weights, verbose=False
        )
        
        assert results is not None
        assert hasattr(results, 'combined_returns')
        assert hasattr(results, 'window_results')
    
    def test_walk_forward_metrics(self, sample_returns, sample_features, sample_regimes):
        from src.walk_forward import WalkForwardAnalyzer
        
        weights = {
            'expansion': pd.Series({'SPY': 0.6, 'TLT': 0.2, 'GLD': 0.2}),
            'slowdown': pd.Series({'SPY': 0.4, 'TLT': 0.4, 'GLD': 0.2}),
            'contraction': pd.Series({'SPY': 0.1, 'TLT': 0.6, 'GLD': 0.3})
        }
        
        analyzer = WalkForwardAnalyzer(train_window=36, test_window=12)
        results = analyzer.run_walk_forward(
            sample_returns, sample_features, weights, verbose=False
        )
        
        assert results.overall_sharpe is not None
        assert results.overall_return is not None


class TestWalkForwardResult:
    """Tests for WalkForwardResult dataclass."""
    
    def test_result_structure(self, sample_returns, sample_features):
        from src.walk_forward import WalkForwardAnalyzer
        
        weights = {
            'expansion': pd.Series({'SPY': 0.6, 'TLT': 0.2, 'GLD': 0.2}),
            'slowdown': pd.Series({'SPY': 0.4, 'TLT': 0.4, 'GLD': 0.2}),
            'contraction': pd.Series({'SPY': 0.1, 'TLT': 0.6, 'GLD': 0.3})
        }
        
        analyzer = WalkForwardAnalyzer(train_window=36, test_window=12)
        results = analyzer.run_walk_forward(
            sample_returns, sample_features, weights, verbose=False
        )
        
        assert hasattr(results, 'combined_returns')
        assert hasattr(results, 'combined_wealth')
        assert hasattr(results, 'window_results')
        assert hasattr(results, 'overall_sharpe')
        assert hasattr(results, 'overall_return')
        assert hasattr(results, 'overall_volatility')


class TestWindowMetrics:
    """Tests for per-window metrics."""
    
    def test_window_results_structure(self, sample_returns, sample_features):
        from src.walk_forward import WalkForwardAnalyzer
        
        weights = {
            'expansion': pd.Series({'SPY': 0.6, 'TLT': 0.2, 'GLD': 0.2}),
            'slowdown': pd.Series({'SPY': 0.4, 'TLT': 0.4, 'GLD': 0.2}),
            'contraction': pd.Series({'SPY': 0.1, 'TLT': 0.6, 'GLD': 0.3})
        }
        
        analyzer = WalkForwardAnalyzer(train_window=36, test_window=12)
        results = analyzer.run_walk_forward(
            sample_returns, sample_features, weights, verbose=False
        )
        
        for window_result in results.window_results:
            assert 'train_start' in window_result
            assert 'train_end' in window_result
            assert 'test_start' in window_result
            assert 'test_end' in window_result
            assert 'test_return' in window_result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])