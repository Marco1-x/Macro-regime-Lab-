#!/usr/bin/env python3
"""
visualization.py
Module de visualisation avancée pour Macro Regime Lab

Fonctionnalités :
- Heatmaps de corrélation dynamiques
- Matrices de transition entre régimes
- Timeline des régimes avec événements
- Analyse de contribution au rendement (waterfall)
- Drawdown analysis charts
- Rolling metrics visualization
- Regime cluster visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Style professionnel
plt.style.use('seaborn-v0_8-whitegrid')


class AdvancedVisualizer:
    """
    Classe principale pour les visualisations avancées du projet.
    """
    
    # Palette de couleurs professionnelle
    COLORS = {
        'expansion': '#2ecc71',      # Vert
        'slowdown': '#f39c12',       # Orange
        'contraction': '#e74c3c',    # Rouge
        'ML_expansion': '#27ae60',   # Vert foncé
        'ML_neutral': '#3498db',     # Bleu
        'ML_stress': '#c0392b',      # Rouge foncé
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6',
        'primary': '#3498db',
        'secondary': '#9b59b6',
    }
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'dark'):
        """
        Initialise le visualiseur.
        
        Args:
            figsize: Taille par défaut des figures
            style: 'dark' ou 'light'
        """
        self.figsize = figsize
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Configure le style global des graphiques."""
        if self.style == 'dark':
            plt.rcParams.update({
                'figure.facecolor': '#1a1a2e',
                'axes.facecolor': '#16213e',
                'axes.edgecolor': '#e94560',
                'axes.labelcolor': '#eaeaea',
                'text.color': '#eaeaea',
                'xtick.color': '#eaeaea',
                'ytick.color': '#eaeaea',
                'grid.color': '#0f3460',
                'legend.facecolor': '#16213e',
                'legend.edgecolor': '#e94560',
            })
        else:
            plt.rcParams.update({
                'figure.facecolor': '#ffffff',
                'axes.facecolor': '#f8f9fa',
                'axes.edgecolor': '#dee2e6',
                'axes.labelcolor': '#212529',
                'text.color': '#212529',
                'xtick.color': '#212529',
                'ytick.color': '#212529',
                'grid.color': '#e9ecef',
            })
    
    # =========================================
    # 1. HEATMAPS DE CORRÉLATION
    # =========================================
    
    def plot_correlation_heatmap(self, 
                                  data: pd.DataFrame, 
                                  title: str = "Correlation Matrix",
                                  annot: bool = True,
                                  cmap: str = 'RdYlGn',
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Crée une heatmap de corrélation professionnelle.
        
        Args:
            data: DataFrame avec les données
            title: Titre du graphique
            annot: Afficher les valeurs
            cmap: Colormap
            save_path: Chemin pour sauvegarder
            
        Returns:
            Figure matplotlib
        """
        corr = data.corr()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Masque pour le triangle supérieur
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        
        # Heatmap
        im = ax.imshow(np.ma.masked_array(corr, mask), 
                       cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation', fontsize=12)
        
        # Labels
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(corr.columns, fontsize=10)
        
        # Annotations
        if annot:
            for i in range(len(corr)):
                for j in range(len(corr)):
                    if not mask[i, j]:
                        text_color = 'white' if abs(corr.iloc[i, j]) > 0.5 else 'black'
                        ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                               ha='center', va='center', color=text_color, fontsize=9)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved correlation heatmap to {save_path}")
        
        return fig
    
    def plot_rolling_correlation(self,
                                  series1: pd.Series,
                                  series2: pd.Series,
                                  window: int = 60,
                                  title: str = "Rolling Correlation",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Graphique de corrélation roulante entre deux séries.
        
        Args:
            series1: Première série
            series2: Deuxième série
            window: Fenêtre de calcul
            title: Titre
            save_path: Chemin de sauvegarde
            
        Returns:
            Figure matplotlib
        """
        rolling_corr = series1.rolling(window).corr(series2)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot principal
        ax.plot(rolling_corr.index, rolling_corr.values, 
                color=self.COLORS['primary'], linewidth=2, label=f'{window}-period rolling')
        
        # Zones de corrélation
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax.axhline(y=0.5, color=self.COLORS['positive'], linestyle=':', alpha=0.5)
        ax.axhline(y=-0.5, color=self.COLORS['negative'], linestyle=':', alpha=0.5)
        
        # Fill zones
        ax.fill_between(rolling_corr.index, rolling_corr.values, 0,
                        where=rolling_corr.values > 0, alpha=0.3, color=self.COLORS['positive'])
        ax.fill_between(rolling_corr.index, rolling_corr.values, 0,
                        where=rolling_corr.values < 0, alpha=0.3, color=self.COLORS['negative'])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved rolling correlation to {save_path}")
        
        return fig
    
    # =========================================
    # 2. MATRICES DE TRANSITION
    # =========================================
    
    def compute_transition_matrix(self, regimes: pd.Series) -> pd.DataFrame:
        """
        Calcule la matrice de transition entre régimes.
        
        Args:
            regimes: Série des régimes
            
        Returns:
            DataFrame de la matrice de transition
        """
        regimes_clean = regimes.dropna()
        unique_regimes = regimes_clean.unique()
        
        # Initialiser la matrice
        transition_counts = pd.DataFrame(0, index=unique_regimes, columns=unique_regimes)
        
        # Compter les transitions
        for i in range(len(regimes_clean) - 1):
            from_regime = regimes_clean.iloc[i]
            to_regime = regimes_clean.iloc[i + 1]
            transition_counts.loc[from_regime, to_regime] += 1
        
        # Convertir en probabilités
        transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)
        transition_probs = transition_probs.fillna(0)
        
        return transition_probs
    
    def plot_transition_matrix(self,
                                regimes: pd.Series,
                                title: str = "Regime Transition Matrix",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualise la matrice de transition entre régimes.
        
        Args:
            regimes: Série des régimes
            title: Titre
            save_path: Chemin de sauvegarde
            
        Returns:
            Figure matplotlib
        """
        trans_matrix = self.compute_transition_matrix(regimes)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Heatmap
        im = ax.imshow(trans_matrix.values, cmap='YlOrRd', vmin=0, vmax=1)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Transition Probability', fontsize=12)
        
        # Labels
        ax.set_xticks(np.arange(len(trans_matrix.columns)))
        ax.set_yticks(np.arange(len(trans_matrix.index)))
        ax.set_xticklabels(trans_matrix.columns, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(trans_matrix.index, fontsize=11)
        
        # Annotations
        for i in range(len(trans_matrix)):
            for j in range(len(trans_matrix.columns)):
                val = trans_matrix.iloc[i, j]
                text_color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.1%}',
                       ha='center', va='center', color=text_color, fontsize=12, fontweight='bold')
        
        ax.set_xlabel('To Regime', fontsize=12, fontweight='bold')
        ax.set_ylabel('From Regime', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved transition matrix to {save_path}")
        
        return fig
    
    # =========================================
    # 3. TIMELINE DES RÉGIMES
    # =========================================
    
    def plot_regime_timeline(self,
                              regimes: pd.Series,
                              prices: pd.Series = None,
                              title: str = "Regime Timeline",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualise la timeline des régimes avec optionnellement les prix.
        
        Args:
            regimes: Série des régimes
            prices: Série des prix (optionnel)
            title: Titre
            save_path: Chemin de sauvegarde
            
        Returns:
            Figure matplotlib
        """
        fig, axes = plt.subplots(2 if prices is not None else 1, 1, 
                                  figsize=(14, 8 if prices is not None else 4),
                                  sharex=True)
        
        if prices is not None:
            ax1, ax2 = axes
        else:
            ax1 = axes
            ax2 = None
        
        # Palette de couleurs pour les régimes
        unique_regimes = regimes.dropna().unique()
        color_map = {}
        for regime in unique_regimes:
            if regime in self.COLORS:
                color_map[regime] = self.COLORS[regime]
            else:
                # Générer une couleur aléatoire cohérente
                hash_val = hash(regime) % 360
                color_map[regime] = plt.cm.hsv(hash_val / 360)
        
        # Plot des prix si disponible
        if prices is not None and ax2 is not None:
            ax1.plot(prices.index, prices.values, color='white', linewidth=1.5, alpha=0.9)
            ax1.set_ylabel('Price', fontsize=12)
            ax1.set_title(title, fontsize=14, fontweight='bold')
            
            # Colorer le background selon les régimes
            for regime in unique_regimes:
                mask = regimes == regime
                dates = regimes[mask].index
                for i, date in enumerate(dates):
                    if i < len(dates) - 1:
                        ax1.axvspan(date, dates[i+1], alpha=0.2, color=color_map[regime])
        
        # Timeline des régimes (barre horizontale)
        ax_regime = ax2 if ax2 is not None else ax1
        
        # Créer une barre de régimes
        for i, (date, regime) in enumerate(regimes.items()):
            if pd.notna(regime):
                color = color_map.get(regime, 'gray')
                if i < len(regimes) - 1:
                    next_date = regimes.index[i + 1]
                    ax_regime.axvspan(date, next_date, alpha=0.8, color=color)
        
        ax_regime.set_yticks([])
        ax_regime.set_xlabel('Date', fontsize=12)
        
        if ax2 is None:
            ax_regime.set_title(title, fontsize=14, fontweight='bold')
        
        # Légende
        legend_patches = [mpatches.Patch(color=color, label=regime, alpha=0.8) 
                         for regime, color in color_map.items()]
        ax_regime.legend(handles=legend_patches, loc='upper center', 
                        bbox_to_anchor=(0.5, -0.15), ncol=len(unique_regimes),
                        fontsize=10, frameon=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved regime timeline to {save_path}")
        
        return fig
    
    # =========================================
    # 4. ANALYSE DE CONTRIBUTION AU RENDEMENT
    # =========================================
    
    def plot_return_contribution(self,
                                  returns: pd.DataFrame,
                                  weights: pd.DataFrame,
                                  period: str = 'monthly',
                                  title: str = "Return Contribution by Asset",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Graphique de contribution au rendement par actif (stacked bar).
        
        Args:
            returns: DataFrame des rendements par actif
            weights: DataFrame des poids par actif
            period: 'monthly', 'quarterly', 'yearly'
            title: Titre
            save_path: Chemin de sauvegarde
            
        Returns:
            Figure matplotlib
        """
        # Calcul des contributions
        contributions = returns * weights
        
        # Resampling selon la période
        if period == 'quarterly':
            contributions = contributions.resample('Q').sum()
        elif period == 'yearly':
            contributions = contributions.resample('Y').sum()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Stacked bar chart
        bottom_pos = np.zeros(len(contributions))
        bottom_neg = np.zeros(len(contributions))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(contributions.columns)))
        
        for i, col in enumerate(contributions.columns):
            values = contributions[col].values
            pos_values = np.where(values >= 0, values, 0)
            neg_values = np.where(values < 0, values, 0)
            
            ax.bar(range(len(contributions)), pos_values, bottom=bottom_pos,
                   label=col, color=colors[i], alpha=0.8, width=0.8)
            ax.bar(range(len(contributions)), neg_values, bottom=bottom_neg,
                   color=colors[i], alpha=0.8, width=0.8)
            
            bottom_pos += pos_values
            bottom_neg += neg_values
        
        # Ligne du rendement total
        total_return = contributions.sum(axis=1)
        ax.plot(range(len(contributions)), total_return.values, 
                color='white', linewidth=2, marker='o', markersize=4, label='Total')
        
        # Formatting
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(0, len(contributions), max(1, len(contributions)//10)))
        ax.set_xticklabels([str(d.date()) for d in contributions.index[::max(1, len(contributions)//10)]], 
                          rotation=45, ha='right', fontsize=9)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Return Contribution', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved return contribution to {save_path}")
        
        return fig
    
    def plot_waterfall_returns(self,
                                contributions: Dict[str, float],
                                title: str = "Return Attribution (Waterfall)",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Graphique waterfall des contributions au rendement.
        
        Args:
            contributions: Dict {source: contribution}
            title: Titre
            save_path: Chemin de sauvegarde
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        labels = list(contributions.keys()) + ['Total']
        values = list(contributions.values())
        total = sum(values)
        values.append(total)
        
        # Positions cumulatives
        cumulative = np.zeros(len(values))
        for i in range(len(values) - 1):
            if i == 0:
                cumulative[i] = 0
            else:
                cumulative[i] = cumulative[i-1] + values[i-1]
        cumulative[-1] = 0  # Total commence à 0
        
        # Couleurs
        colors = []
        for i, val in enumerate(values):
            if i == len(values) - 1:  # Total
                colors.append(self.COLORS['primary'])
            elif val >= 0:
                colors.append(self.COLORS['positive'])
            else:
                colors.append(self.COLORS['negative'])
        
        # Barres
        bars = ax.bar(range(len(labels)), values, bottom=cumulative, 
                     color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Lignes de connexion
        for i in range(len(values) - 2):
            y = cumulative[i] + values[i]
            ax.hlines(y, i + 0.4, i + 0.6, colors='gray', linestyles='--', alpha=0.5)
        
        # Annotations
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            y_pos = bar.get_y() + height/2
            ax.annotate(f'{val:.2%}',
                       xy=(bar.get_x() + bar.get_width()/2, y_pos),
                       ha='center', va='center',
                       fontsize=10, fontweight='bold',
                       color='white' if abs(val) > 0.02 else 'black')
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Return', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved waterfall chart to {save_path}")
        
        return fig
    
    # =========================================
    # 5. DRAWDOWN ANALYSIS
    # =========================================
    
    def plot_drawdown_analysis(self,
                                returns: pd.Series,
                                title: str = "Drawdown Analysis",
                                top_n: int = 5,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyse complète des drawdowns.
        
        Args:
            returns: Série des rendements
            title: Titre
            top_n: Nombre de pires drawdowns à afficher
            save_path: Chemin de sauvegarde
            
        Returns:
            Figure matplotlib
        """
        # Calcul du drawdown
        wealth = (1 + returns).cumprod()
        peak = wealth.cummax()
        drawdown = (wealth / peak - 1) * 100
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Wealth curve avec peak
        ax1 = axes[0]
        ax1.plot(wealth.index, wealth.values, color=self.COLORS['primary'], 
                 linewidth=2, label='Portfolio Value')
        ax1.plot(wealth.index, peak.values, color=self.COLORS['secondary'], 
                 linewidth=1, linestyle='--', alpha=0.7, label='Peak')
        ax1.fill_between(wealth.index, wealth.values, peak.values, 
                        alpha=0.3, color=self.COLORS['negative'])
        ax1.set_ylabel('Wealth', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        ax2 = axes[1]
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        color=self.COLORS['negative'], alpha=0.7)
        ax2.plot(drawdown.index, drawdown.values, color=self.COLORS['negative'], 
                 linewidth=1)
        
        # Marquer les pires drawdowns
        worst_dd_idx = drawdown.nsmallest(top_n).index
        for idx in worst_dd_idx:
            ax2.annotate(f'{drawdown[idx]:.1f}%',
                        xy=(idx, drawdown[idx]),
                        xytext=(10, -20), textcoords='offset points',
                        fontsize=9, color='white',
                        arrowprops=dict(arrowstyle='->', color='white', alpha=0.7))
        
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Statistiques
        max_dd = drawdown.min()
        avg_dd = drawdown.mean()
        textstr = f'Max DD: {max_dd:.1f}%\nAvg DD: {avg_dd:.1f}%'
        props = dict(boxstyle='round', facecolor='black', alpha=0.7)
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, color='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved drawdown analysis to {save_path}")
        
        return fig
    
    # =========================================
    # 6. ROLLING METRICS
    # =========================================
    
    def plot_rolling_metrics(self,
                              returns: pd.Series,
                              window: int = 36,
                              title: str = "Rolling Performance Metrics",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Graphique des métriques roulantes.
        
        Args:
            returns: Série des rendements
            window: Fenêtre de calcul (mois)
            title: Titre
            save_path: Chemin de sauvegarde
            
        Returns:
            Figure matplotlib
        """
        # Calcul des métriques roulantes
        rolling_return = returns.rolling(window).mean() * 12
        rolling_vol = returns.rolling(window).std() * np.sqrt(12)
        rolling_sharpe = rolling_return / rolling_vol
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Rolling Return
        ax1 = axes[0]
        ax1.plot(rolling_return.index, rolling_return.values * 100, 
                 color=self.COLORS['primary'], linewidth=2)
        ax1.fill_between(rolling_return.index, rolling_return.values * 100, 0,
                        where=rolling_return.values >= 0, alpha=0.3, color=self.COLORS['positive'])
        ax1.fill_between(rolling_return.index, rolling_return.values * 100, 0,
                        where=rolling_return.values < 0, alpha=0.3, color=self.COLORS['negative'])
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Annualized Return (%)', fontsize=11)
        ax1.set_title(f'{title} ({window}-month rolling)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Rolling Volatility
        ax2 = axes[1]
        ax2.plot(rolling_vol.index, rolling_vol.values * 100, 
                 color=self.COLORS['secondary'], linewidth=2)
        ax2.fill_between(rolling_vol.index, rolling_vol.values * 100, 0, 
                        alpha=0.3, color=self.COLORS['secondary'])
        ax2.set_ylabel('Annualized Volatility (%)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Rolling Sharpe
        ax3 = axes[2]
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values, 
                 color='white', linewidth=2)
        ax3.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                        where=rolling_sharpe.values >= 0, alpha=0.3, color=self.COLORS['positive'])
        ax3.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                        where=rolling_sharpe.values < 0, alpha=0.3, color=self.COLORS['negative'])
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.axhline(y=1, color=self.COLORS['positive'], linestyle=':', alpha=0.7, label='Sharpe = 1')
        ax3.set_ylabel('Sharpe Ratio', fontsize=11)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved rolling metrics to {save_path}")
        
        return fig
    
    # =========================================
    # 7. REGIME PERFORMANCE COMPARISON
    # =========================================
    
    def plot_regime_performance(self,
                                 returns: pd.Series,
                                 regimes: pd.Series,
                                 title: str = "Performance by Regime",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare la performance par régime.
        
        Args:
            returns: Série des rendements
            regimes: Série des régimes
            title: Titre
            save_path: Chemin de sauvegarde
            
        Returns:
            Figure matplotlib
        """
        # Aligner les données
        df = pd.DataFrame({'returns': returns, 'regime': regimes}).dropna()
        
        # Stats par régime
        regime_stats = df.groupby('regime')['returns'].agg([
            ('mean', lambda x: x.mean() * 12),
            ('std', lambda x: x.std() * np.sqrt(12)),
            ('sharpe', lambda x: x.mean() * 12 / (x.std() * np.sqrt(12)) if x.std() > 0 else 0),
            ('count', 'count')
        ])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        regimes_list = regime_stats.index.tolist()
        colors = [self.COLORS.get(r, 'gray') for r in regimes_list]
        
        # Plot 1: Annualized Return
        ax1 = axes[0]
        bars1 = ax1.bar(regimes_list, regime_stats['mean'] * 100, color=colors, alpha=0.8, edgecolor='white')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Annualized Return (%)', fontsize=11)
        ax1.set_title('Return by Regime', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars1, regime_stats['mean'] * 100):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Volatility
        ax2 = axes[1]
        bars2 = ax2.bar(regimes_list, regime_stats['std'] * 100, color=colors, alpha=0.8, edgecolor='white')
        ax2.set_ylabel('Annualized Volatility (%)', fontsize=11)
        ax2.set_title('Volatility by Regime', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars2, regime_stats['std'] * 100):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 3: Sharpe Ratio
        ax3 = axes[2]
        bars3 = ax3.bar(regimes_list, regime_stats['sharpe'], color=colors, alpha=0.8, edgecolor='white')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.axhline(y=1, color=self.COLORS['positive'], linestyle=':', alpha=0.7)
        ax3.set_ylabel('Sharpe Ratio', fontsize=11)
        ax3.set_title('Sharpe by Regime', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars3, regime_stats['sharpe']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved regime performance to {save_path}")
        
        return fig
    
    # =========================================
    # 8. COMPREHENSIVE DASHBOARD
    # =========================================
    
    def create_comprehensive_dashboard(self,
                                        returns: pd.Series,
                                        regimes: pd.Series,
                                        prices: pd.Series = None,
                                        title: str = "Macro Regime Lab - Dashboard",
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Crée un dashboard complet avec toutes les visualisations clés.
        
        Args:
            returns: Série des rendements
            regimes: Série des régimes
            prices: Série des prix (optionnel)
            title: Titre
            save_path: Chemin de sauvegarde
            
        Returns:
            Figure matplotlib
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Grid layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Wealth curve (top, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        wealth = (1 + returns).cumprod()
        ax1.plot(wealth.index, wealth.values, color=self.COLORS['primary'], linewidth=2)
        ax1.fill_between(wealth.index, 1, wealth.values, alpha=0.3, color=self.COLORS['primary'])
        ax1.set_title('Portfolio Wealth Curve', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Wealth')
        ax1.grid(True, alpha=0.3)
        
        # 2. Regime distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        regime_counts = regimes.value_counts()
        colors = [self.COLORS.get(r, 'gray') for r in regime_counts.index]
        ax2.pie(regime_counts.values, labels=regime_counts.index, colors=colors,
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('Regime Distribution', fontsize=12, fontweight='bold')
        
        # 3. Drawdown (second row, spans 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        peak = wealth.cummax()
        drawdown = (wealth / peak - 1) * 100
        ax3.fill_between(drawdown.index, drawdown.values, 0, 
                        color=self.COLORS['negative'], alpha=0.7)
        ax3.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Transition matrix (second row right)
        ax4 = fig.add_subplot(gs[1, 2])
        trans_matrix = self.compute_transition_matrix(regimes)
        im = ax4.imshow(trans_matrix.values, cmap='YlOrRd', vmin=0, vmax=1)
        ax4.set_xticks(range(len(trans_matrix.columns)))
        ax4.set_yticks(range(len(trans_matrix.index)))
        ax4.set_xticklabels(trans_matrix.columns, rotation=45, ha='right', fontsize=8)
        ax4.set_yticklabels(trans_matrix.index, fontsize=8)
        ax4.set_title('Transition Matrix', fontsize=12, fontweight='bold')
        for i in range(len(trans_matrix)):
            for j in range(len(trans_matrix.columns)):
                val = trans_matrix.iloc[i, j]
                ax4.text(j, i, f'{val:.0%}', ha='center', va='center', 
                        color='white' if val > 0.5 else 'black', fontsize=8)
        
        # 5. Rolling Sharpe (third row, spans 2 columns)
        ax5 = fig.add_subplot(gs[2, :2])
        rolling_sharpe = (returns.rolling(36).mean() * 12) / (returns.rolling(36).std() * np.sqrt(12))
        ax5.plot(rolling_sharpe.index, rolling_sharpe.values, color='white', linewidth=2)
        ax5.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                        where=rolling_sharpe.values >= 0, alpha=0.3, color=self.COLORS['positive'])
        ax5.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                        where=rolling_sharpe.values < 0, alpha=0.3, color=self.COLORS['negative'])
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.axhline(y=1, color=self.COLORS['positive'], linestyle=':', alpha=0.7)
        ax5.set_title('36-Month Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance by regime (third row right)
        ax6 = fig.add_subplot(gs[2, 2])
        df = pd.DataFrame({'returns': returns, 'regime': regimes}).dropna()
        regime_stats = df.groupby('regime')['returns'].agg(lambda x: x.mean() * 12 * 100)
        colors = [self.COLORS.get(r, 'gray') for r in regime_stats.index]
        bars = ax6.bar(regime_stats.index, regime_stats.values, color=colors, alpha=0.8)
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax6.set_title('Annualized Return by Regime', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Return (%)')
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. Monthly returns distribution (bottom, spans 2 columns)
        ax7 = fig.add_subplot(gs[3, :2])
        ax7.hist(returns.values * 100, bins=50, color=self.COLORS['primary'], 
                alpha=0.7, edgecolor='white')
        ax7.axvline(x=returns.mean() * 100, color=self.COLORS['positive'], 
                   linestyle='--', linewidth=2, label=f'Mean: {returns.mean()*100:.2f}%')
        ax7.axvline(x=0, color='gray', linestyle='-', linewidth=1)
        ax7.set_title('Monthly Returns Distribution', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Return (%)')
        ax7.set_ylabel('Frequency')
        ax7.legend(loc='upper right')
        ax7.grid(True, alpha=0.3)
        
        # 8. Key metrics (bottom right)
        ax8 = fig.add_subplot(gs[3, 2])
        ax8.axis('off')
        
        # Calcul des métriques
        cagr = returns.mean() * 12
        vol = returns.std() * np.sqrt(12)
        sharpe = cagr / vol if vol > 0 else 0
        max_dd = drawdown.min()
        
        metrics_text = f"""
        KEY METRICS
        ══════════════════
        
        CAGR:           {cagr*100:>8.2f}%
        Volatility:     {vol*100:>8.2f}%
        Sharpe Ratio:   {sharpe:>8.2f}
        Max Drawdown:   {max_dd:>8.2f}%
        
        Total Months:   {len(returns):>8d}
        Win Rate:       {(returns > 0).mean()*100:>8.1f}%
        Best Month:     {returns.max()*100:>8.2f}%
        Worst Month:    {returns.min()*100:>8.2f}%
        """
        
        ax8.text(0.1, 0.9, metrics_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved comprehensive dashboard to {save_path}")
        
        return fig


# =========================================
# FONCTIONS UTILITAIRES STANDALONE
# =========================================

def quick_plot_wealth(returns: pd.Series, 
                      benchmark: pd.Series = None,
                      title: str = "Wealth Curve",
                      save_path: str = None) -> plt.Figure:
    """
    Crée rapidement un graphique de wealth curve.
    
    Args:
        returns: Rendements de la stratégie
        benchmark: Rendements du benchmark (optionnel)
        title: Titre
        save_path: Chemin de sauvegarde
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    wealth = (1 + returns).cumprod()
    ax.plot(wealth.index, wealth.values, label='Strategy', linewidth=2, color='#3498db')
    
    if benchmark is not None:
        bench_wealth = (1 + benchmark).cumprod()
        ax.plot(bench_wealth.index, bench_wealth.values, label='Benchmark', 
               linewidth=2, color='#e74c3c', alpha=0.7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Wealth (Growth of $1)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def quick_plot_regimes(regimes: pd.Series,
                       save_path: str = None) -> plt.Figure:
    """
    Crée rapidement un graphique de distribution des régimes.
    
    Args:
        regimes: Série des régimes
        save_path: Chemin de sauvegarde
        
    Returns:
        Figure matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    counts = regimes.value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    axes[0].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
    axes[0].set_title('Regime Distribution', fontsize=12, fontweight='bold')
    
    # Bar chart
    axes[1].bar(counts.index, counts.values, color=colors, edgecolor='white')
    axes[1].set_title('Regime Counts', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =========================================
# MAIN (pour test)
# =========================================

if __name__ == "__main__":
    print("="*60)
    print("VISUALIZATION MODULE - TEST")
    print("="*60)
    
    # Créer des données de test
    np.random.seed(42)
    dates = pd.date_range('2010-01-01', periods=120, freq='M')
    
    # Rendements simulés
    returns = pd.Series(np.random.normal(0.008, 0.04, 120), index=dates, name='strategy')
    
    # Régimes simulés
    regimes_data = np.random.choice(['expansion', 'slowdown', 'contraction'], 120, p=[0.6, 0.25, 0.15])
    regimes = pd.Series(regimes_data, index=dates, name='regime')
    
    # Test du visualizer
    viz = AdvancedVisualizer(style='dark')
    
    print("\n[TEST 1] Correlation heatmap...")
    test_data = pd.DataFrame({
        'SPY': np.random.randn(100),
        'TLT': np.random.randn(100),
        'GLD': np.random.randn(100),
        'XLK': np.random.randn(100),
    })
    viz.plot_correlation_heatmap(test_data, save_path='output/test_correlation.png')
    
    print("\n[TEST 2] Transition matrix...")
    viz.plot_transition_matrix(regimes, save_path='output/test_transition.png')
    
    print("\n[TEST 3] Regime timeline...")
    viz.plot_regime_timeline(regimes, save_path='output/test_timeline.png')
    
    print("\n[TEST 4] Drawdown analysis...")
    viz.plot_drawdown_analysis(returns, save_path='output/test_drawdown.png')
    
    print("\n[TEST 5] Rolling metrics...")
    viz.plot_rolling_metrics(returns, save_path='output/test_rolling.png')
    
    print("\n[TEST 6] Regime performance...")
    viz.plot_regime_performance(returns, regimes, save_path='output/test_regime_perf.png')
    
    print("\n[TEST 7] Comprehensive dashboard...")
    viz.create_comprehensive_dashboard(returns, regimes, save_path='output/test_dashboard.png')
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATION TESTS COMPLETED!")
    print("="*60)