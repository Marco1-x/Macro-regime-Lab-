# Installation - Macro Regime Lab

## Prérequis
- Python 3.9+
- Clé API FRED (gratuite sur https://fred.stlouisfed.org/docs/api/api_key.html)

## Installation
```bash
# 1. Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# ou: venv\Scripts\activate  # Windows

# 2. Mettre à jour pip
pip install --upgrade pip

# 3. Installer les dépendances
pip install -r requirements.txt
```

## ⚠️ IMPORTANT - Problème yfinance

Les versions récentes de yfinance ont des problèmes de compatibilité SSL sur macOS.

**Solution**: Ce projet utilise un téléchargement direct depuis l'API Yahoo Finance (voir `src/data_fetcher.py`) au lieu de yfinance.

## Lancer le dashboard
```bash
source venv/bin/activate
streamlit run src/dashboard.py
```

## Fichiers clés

- `src/dashboard.py` - Dashboard principal
- `src/data_fetcher.py` - Module de téléchargement des données
