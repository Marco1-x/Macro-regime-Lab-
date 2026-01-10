# 📊 Macro Regime & Factor Rotation Lab

**Analyse de la relation entre les régimes macroéconomiques et la rotation factorielle pour optimiser l'allocation d'un portefeuille ETF (2005-2025)**

Un outil Python qui identifie les régimes macroéconomiques (Expansion, Slowdown, Recession) à partir d'indicateurs publics et ajuste dynamiquement un portefeuille ETF en conséquence.

**Auteur :** Marc Aurel AMOUSSOU  
**Établissement :** HEC Lausanne  
**Cours :** Introduction to Data Science and Advanced Programming (Automne 2025)

---

## 🚀 Démarrage rapide

### Prérequis
- Python 3.9 ou version ultérieure
- Gestionnaire de paquets pip

### Installation et configuration
```bash
# 1. Cloner le repository
git clone https://github.com/Marco1-x/Macro-regime-Lab-.git
cd Macro-regime-Lab-

# 2. Créer l'environnement virtuel
python3 -m venv venv

# 3. Activer l'environnement virtuel
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 4. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 5. Lancer l'analyse
python3 -m src.cli detect-regimes
python3 -m src.cli backtest
python3 -m src.cli report

# 6. (Optionnel) Lancer le dashboard interactif
streamlit run src/dashboard.py
```

---

## 🎯 Aperçu du projet

### Le problème

Les marchés financiers traversent différents régimes macroéconomiques (expansion, ralentissement, récession) qui affectent différemment les classes d'actifs. Ce projet met en place un cadre d'analyse pour :

- **Détecter automatiquement** les régimes macroéconomiques à partir de données FRED
- **Allouer dynamiquement** un portefeuille ETF selon le régime détecté
- **Backtester la stratégie** avec des coûts de transaction réalistes
- **Comparer les performances** avec des benchmarks (SPY, 60/40)

### Questions clés de recherche

1. Peut-on identifier les régimes macroéconomiques à partir d'indicateurs simples ?
2. Une stratégie de rotation basée sur les régimes peut-elle surperformer le marché ?
3. Quel est l'impact des coûts de transaction sur les performances ?
4. La stratégie réduit-elle le drawdown maximum en période de crise ?

---

## 📈 Principaux résultats

### Performance de la stratégie (2005-2025)

| Métrique | Strategy | SPY | 60/40 |
|----------|----------|-----|-------|
| **CAGR** | **13.6%** | 9.5% | 6.0% |
| **Volatilité** | 14.3% | 14.9% | 10.1% |
| **Sharpe Ratio** | **0.95** | 0.64 | 0.60 |
| **Max Drawdown** | **-26.7%** | -52.2% | -31.3% |

### Distribution des régimes (1947-2025)

| Régime | Mois | Pourcentage | Caractéristiques |
|--------|------|-------------|------------------|
| **Expansion** | 705 | 76.5% | Croissance économique, inflation stable |
| **Recession** | 123 | 13.3% | NBER recession officielle |
| **Slowdown** | 94 | 10.2% | Inflation haute + chômage en hausse |

### Principale constatation

> La stratégie de rotation macro **surperforme SPY de 4.1% par an** tout en réduisant le drawdown maximum de moitié (-26.7% vs -52.2%). Le Sharpe Ratio de 0.95 démontre un excellent rendement ajusté au risque.

---

## 🛠️ Méthodologie

### 1. Détection des régimes

Les régimes sont définis par une heuristique transparente utilisant des données FRED :

| Régime | Définition |
|--------|------------|
| **Recession** | USREC = 1 (indicateur officiel NBER) |
| **Slowdown** | CPI YoY > médiane mobile ET ΔUNRATE > 0 |
| **Expansion** | Sinon |

### 2. Allocation du portefeuille

| Régime | SPY | TLT | GLD | XLK | Logique |
|--------|-----|-----|-----|-----|---------|
| **Expansion** | 60% | 0% | 0% | 40% | Risk-on : actions + tech |
| **Slowdown** | 40% | 40% | 20% | 0% | Défensif : bonds + or |
| **Recession** | 0% | 70% | 30% | 0% | Risk-off : bonds + or |

### 3. Paramètres du backtest

- **Rebalancement** : Mensuel
- **Coûts de transaction** : 5 bps par unité de turnover
- **Période** : Janvier 2005 - Novembre 2025
- **Benchmarks** : SPY buy-and-hold, Portefeuille 60/40

---

## 📁 Structure du projet
```
macro-factor-lab/
├── src/
│   ├── cli.py              # CLI Typer (detect-regimes, backtest, report)
│   ├── dashboard.py        # Dashboard interactif Streamlit
│   ├── data_fetcher.py     # Téléchargement Yahoo Finance
│   ├── models.py           # Modèles de détection (HMM, RF, Ensemble)
│   ├── backtest.py         # Moteur de backtest avec coûts
│   ├── stress_testing.py   # VaR et stress testing
│   ├── walk_forward.py     # Analyse walk-forward
│   └── visualization.py    # Utilitaires de visualisation
│
├── data/
│   ├── fred/               # Données macro FRED (offline)
│   │   ├── CPIAUCSL.csv    # Consumer Price Index
│   │   ├── UNRATE.csv      # Taux de chômage
│   │   └── USREC.csv       # Indicateur de récession NBER
│   └── etf_prices.csv      # Prix historiques ETF
│
├── output/
│   ├── regimes.csv         # Régimes détectés
│   ├── returns.csv         # Rendements de la stratégie
│   ├── weights.csv         # Historique des poids
│   ├── metrics.json        # Métriques de performance
│   ├── wealth_curve.png    # Courbe de richesse
│   ├── drawdown.png        # Graphique de drawdown
│   └── REPORT.md           # Rapport généré
│
├── docs/
│   ├── API.md              # Documentation API
│   └── USER_GUIDE.md       # Guide utilisateur
│
├── tests/
│   ├── test_walk_forward.py
│   ├── test_system.py
│   └── test_all.py
│
├── requirements.txt        # Dépendances Python
├── INSTALLATION.md         # Guide d'installation
└── README.md               # Ce fichier
```

---

## 💻 Commandes CLI

Le projet fournit trois commandes principales via Typer :
```bash
# 1. Détecter les régimes macro à partir des données FRED
python3 -m src.cli detect-regimes
# Output: output/regimes.csv

# 2. Exécuter le backtest avec coûts de transaction
python3 -m src.cli backtest
# Output: output/returns.csv, output/metrics.json, output/*.png

# 3. Générer le rapport Markdown
python3 -m src.cli report
# Output: output/REPORT.md
```

### Options avancées
```bash
# Backtest avec période personnalisée
python3 -m src.cli backtest --start-date 2010-01-01 --end-date 2023-12-31

# Backtest avec coûts de transaction différents
python3 -m src.cli backtest --cost-bps 10
```

---

## 🖥️ Dashboard interactif
```bash
streamlit run src/dashboard.py
```

**Fonctionnalités :**
- Configuration interactive des poids par régime
- Visualisation en temps réel des performances
- Analyse des régimes et timeline
- Export des données CSV

---

## 📊 Sources de données

| Source | Indicateurs | Période | Fréquence |
|--------|-------------|---------|-----------|
| **FRED** | CPI, Unemployment, USREC | 1947-2025 | Mensuelle |
| **Yahoo Finance** | SPY, TLT, GLD, XLK | 2000-2025 | Journalière |

---

## 🔧 Technologies utilisées

| Catégorie | Technologies |
|-----------|--------------|
| **Données** | pandas, numpy, fredapi |
| **ML/Stats** | scikit-learn, hmmlearn, scipy |
| **Visualisation** | plotly, matplotlib, seaborn |
| **Dashboard** | Streamlit |
| **CLI** | Typer |
| **Tests** | pytest |

---

## ⚠️ Limitations

1. **Lag NBER** : Les dates de récession officielles sont annoncées avec retard
2. **Sensibilité des seuils** : La période de médiane mobile affecte la détection
3. **Look-ahead bias** : La stratégie n'utilise que l'information disponible à t
4. **Coûts de transaction** : Les coûts réels peuvent varier selon les conditions de marché
5. **Survivorship bias** : Seuls les ETF existants sont analysés

---

## 🔮 Améliorations possibles

- [x] Hidden Markov Models (`src/models.py`) pour détection data-driven des régimes
- [ ] Indicateurs additionnels (yield curve slope, credit spreads, PMI)
- [x] Walk-Forward Analysis (`src/walk_forward.py`) des poids intra-régime
- [x] Stress Testing & VaR (`src/stress_testing.py`) pour le dimensionnement des positions
- [x] Ensemble Models avec voting (`src/models.py`) pour affiner les signaux

---

## 📚 Documentation

- [Guide d'installation](INSTALLATION.md)
- [Référence API](src/API.md)
- [Guide utilisateur](src/user_guide.md)

---

## 👤 Contact

**Marc Aurel AMOUSSOU**  
HEC Lausanne - MSc in Finance  
GitHub : [@Marco1-x](https://github.com/Marco1-x)

---

## 🙏 Remerciements

- **Prof. Simon Scheidegger** - Instructeur du cours
- **Anna Smirnova** - Assistante d'enseignement
- **FRED** - Federal Reserve Economic Data
- **Claude (Anthropic)** - Assistance IA (voir appendice du rapport)

---

*Dernière mise à jour : Janvier 2026*
