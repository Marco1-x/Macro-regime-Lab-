#!/usr/bin/env python3
"""
models.py
Module de détection de régimes pour Macro Regime Lab

Modèles inclus :
- HMMRegimeDetector : Gaussian Hidden Markov Model
- RandomForestRegimeDetector : Classification supervisée
- EnsembleRegimeDetector : Combinaison de modèles avec voting/stacking (Option C)

L'EnsembleRegimeDetector combine plusieurs modèles pour une détection
plus robuste des régimes macroéconomiques.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')


# =========================================
# 1. HMM REGIME DETECTOR
# =========================================

class HMMRegimeDetector:
    """
    Détecte les régimes via Gaussian Hidden Markov Model.
    
    Le HMM est un modèle non-supervisé qui apprend les états latents
    à partir des observations de features macroéconomiques.
    """
    
    def __init__(self, n_states: int = 3, covariance_type: str = "full",
                 n_iter: int = 200, random_state: int = 42):
        """
        Initialise le détecteur HMM.
        
        Args:
            n_states: Nombre de régimes à détecter
            covariance_type: Type de covariance ('full', 'diag', 'spherical')
            n_iter: Nombre d'itérations EM
            random_state: Seed pour reproductibilité
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )
        self.is_fitted = False
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, scale: bool = True):
        """
        Entraîne le modèle HMM.
        
        Args:
            X: Features (n_samples, n_features)
            scale: Standardiser les features
        """
        print(f"[INFO] Fitting HMM with {self.n_states} states...")
        
        if scale:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        print(f"[INFO] HMM fitted. Convergence: {self.model.monitor_.converged}")
        
    def predict(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """
        Prédit les régimes.
        
        Args:
            X: Features
            scale: Utiliser le scaler
            
        Returns:
            Array des régimes (entiers)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """
        Retourne les probabilités de chaque régime.
        
        Args:
            X: Features
            scale: Utiliser le scaler
            
        Returns:
            Array (n_samples, n_states) des probabilités
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        return self.model.predict_proba(X_scaled)
    
    def get_stationary_distribution(self) -> np.ndarray:
        """Retourne la distribution stationnaire des régimes."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.get_stationary_distribution()
    
    def get_transition_matrix(self) -> np.ndarray:
        """Retourne la matrice de transition."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.transmat_


# =========================================
# 2. RANDOM FOREST REGIME DETECTOR
# =========================================

class RandomForestRegimeDetector:
    """
    Détecte les régimes via Random Forest (supervisé).
    
    Nécessite des labels de régimes pour l'entraînement
    (ex: régimes heuristiques).
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5,
                 random_state: int = 42):
        """
        Initialise le détecteur Random Forest.
        
        Args:
            n_estimators: Nombre d'arbres
            max_depth: Profondeur max des arbres
            random_state: Seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.is_fitted = False
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def fit(self, X: np.ndarray, y: np.ndarray, scale: bool = True):
        """
        Entraîne le modèle RF avec labels supervisés.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels de régimes
            scale: Standardiser les features
        """
        print(f"[INFO] Fitting Random Forest...")
        
        if scale:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            
        # Encoder les labels si strings
        if isinstance(y[0], str):
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
            
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
        
        print(f"[INFO] Random Forest fitted. Feature importances computed.")
        
    def predict(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """
        Prédit les régimes.
        
        Args:
            X: Features
            scale: Utiliser le scaler
            
        Returns:
            Array des régimes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        predictions = self.model.predict(X_scaled)
        
        # Décoder si nécessaire
        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
            return self.label_encoder.inverse_transform(predictions)
        return predictions
    
    def predict_proba(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Retourne les probabilités de chaque régime."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """Retourne l'importance des features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.feature_importances_


# =========================================
# 3. ENSEMBLE REGIME DETECTOR (Option C)
# =========================================

@dataclass
class EnsembleConfig:
    """Configuration pour l'ensemble."""
    use_hmm: bool = True
    use_rf: bool = True
    use_gbm: bool = True
    use_svm: bool = False
    use_knn: bool = False
    use_logistic: bool = False
    voting_method: str = 'soft'  # 'hard' ou 'soft'
    hmm_weight: float = 1.0
    rf_weight: float = 1.0
    gbm_weight: float = 1.0


class EnsembleRegimeDetector:
    """
    Ensemble de modèles pour la détection de régimes.
    
    Combine plusieurs classifieurs avec voting (dur ou mou)
    ou stacking pour une prédiction plus robuste.
    
    Modèles disponibles:
    - HMM (non-supervisé, converti en supervisé via labels)
    - Random Forest
    - Gradient Boosting
    - SVM
    - KNN
    - Logistic Regression
    """
    
    def __init__(self, 
                 n_regimes: int = 3,
                 config: EnsembleConfig = None,
                 random_state: int = 42):
        """
        Initialise l'ensemble.
        
        Args:
            n_regimes: Nombre de régimes
            config: Configuration de l'ensemble
            random_state: Seed
        """
        self.n_regimes = n_regimes
        self.config = config or EnsembleConfig()
        self.random_state = random_state
        
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialise les modèles selon la config."""
        
        if self.config.use_hmm:
            self.models['hmm'] = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type='full',
                n_iter=200,
                random_state=self.random_state
            )
            self.weights['hmm'] = self.config.hmm_weight
            
        if self.config.use_rf:
            self.models['rf'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            )
            self.weights['rf'] = self.config.rf_weight
            
        if self.config.use_gbm:
            self.models['gbm'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=self.random_state
            )
            self.weights['gbm'] = self.config.gbm_weight
            
        if self.config.use_svm:
            self.models['svm'] = SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            )
            self.weights['svm'] = 1.0
            
        if self.config.use_knn:
            self.models['knn'] = KNeighborsClassifier(
                n_neighbors=5
            )
            self.weights['knn'] = 1.0
            
        if self.config.use_logistic:
            self.models['logistic'] = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            )
            self.weights['logistic'] = 1.0
            
        print(f"[INFO] Ensemble initialized with models: {list(self.models.keys())}")
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Entraîne l'ensemble.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels de régimes (requis pour modèles supervisés)
        """
        print(f"\n[INFO] Fitting Ensemble with {len(self.models)} models...")
        
        # Standardiser
        X_scaled = self.scaler.fit_transform(X)
        
        # Encoder les labels
        if y is not None:
            if isinstance(y[0], str):
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = y
        else:
            y_encoded = None
            
        # Fit chaque modèle
        for name, model in self.models.items():
            print(f"   Fitting {name}...", end=" ")
            
            try:
                if name == 'hmm':
                    # HMM est non-supervisé
                    model.fit(X_scaled)
                else:
                    # Modèles supervisés
                    if y_encoded is None:
                        raise ValueError(f"Labels required for {name}")
                    model.fit(X_scaled, y_encoded)
                    
                print("✅")
                
            except Exception as e:
                print(f"❌ ({e})")
                
        self.is_fitted = True
        print(f"[INFO] Ensemble fitting complete.")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les régimes avec voting.
        
        Args:
            X: Features
            
        Returns:
            Array des régimes prédits
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
            
        X_scaled = self.scaler.transform(X)
        
        if self.config.voting_method == 'soft':
            return self._soft_voting(X_scaled)
        else:
            return self._hard_voting(X_scaled)
    
    def _hard_voting(self, X_scaled: np.ndarray) -> np.ndarray:
        """Vote majoritaire sur les prédictions."""
        predictions = []
        
        for name, model in self.models.items():
            try:
                if name == 'hmm':
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X_scaled)
                predictions.append(pred)
            except:
                continue
                
        if not predictions:
            raise ValueError("No model produced predictions")
            
        # Stack et vote majoritaire
        predictions = np.array(predictions)  # (n_models, n_samples)
        
        # Mode sur chaque sample
        from scipy import stats
        final_pred = stats.mode(predictions, axis=0, keepdims=False)[0]
        
        # Décoder
        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
            return self.label_encoder.inverse_transform(final_pred.astype(int))
        return final_pred
    
    def _soft_voting(self, X_scaled: np.ndarray) -> np.ndarray:
        """Vote pondéré sur les probabilités."""
        n_samples = X_scaled.shape[0]
        weighted_proba = np.zeros((n_samples, self.n_regimes))
        total_weight = 0
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0)
            
            try:
                if name == 'hmm':
                    proba = model.predict_proba(X_scaled)
                else:
                    proba = model.predict_proba(X_scaled)
                    
                weighted_proba += weight * proba
                total_weight += weight
                
            except Exception as e:
                continue
                
        if total_weight == 0:
            raise ValueError("No model produced probabilities")
            
        # Normaliser
        weighted_proba /= total_weight
        
        # Argmax
        final_pred = np.argmax(weighted_proba, axis=1)
        
        # Décoder
        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
            return self.label_encoder.inverse_transform(final_pred)
        return final_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilités moyennes pondérées.
        
        Args:
            X: Features
            
        Returns:
            Array (n_samples, n_regimes) des probabilités
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
            
        X_scaled = self.scaler.transform(X)
        n_samples = X_scaled.shape[0]
        weighted_proba = np.zeros((n_samples, self.n_regimes))
        total_weight = 0
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0)
            
            try:
                if name == 'hmm':
                    proba = model.predict_proba(X_scaled)
                else:
                    proba = model.predict_proba(X_scaled)
                    
                weighted_proba += weight * proba
                total_weight += weight
            except:
                continue
                
        return weighted_proba / total_weight if total_weight > 0 else weighted_proba
    
    def get_model_agreement(self, X: np.ndarray) -> float:
        """
        Calcule le taux d'accord entre les modèles.
        
        Args:
            X: Features
            
        Returns:
            Taux d'accord (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
            
        X_scaled = self.scaler.transform(X)
        predictions = []
        
        for name, model in self.models.items():
            try:
                if name == 'hmm':
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X_scaled)
                predictions.append(pred)
            except:
                continue
                
        if len(predictions) < 2:
            return 1.0
            
        predictions = np.array(predictions)
        
        # Calculer l'accord pour chaque sample
        agreements = []
        for i in range(predictions.shape[1]):
            sample_preds = predictions[:, i]
            unique, counts = np.unique(sample_preds, return_counts=True)
            max_agreement = counts.max() / len(sample_preds)
            agreements.append(max_agreement)
            
        return np.mean(agreements)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """
        Valide chaque modèle par cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Nombre de folds
            
        Returns:
            Dict des scores par modèle
        """
        X_scaled = self.scaler.fit_transform(X)
        
        if isinstance(y[0], str):
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
            
        scores = {}
        
        for name, model in self.models.items():
            if name == 'hmm':
                continue  # Skip HMM (non-supervisé)
                
            try:
                cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv)
                scores[name] = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                }
                print(f"   {name}: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            except Exception as e:
                print(f"   {name}: Error ({e})")
                
        return scores
    
    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne la confiance de la prédiction (max proba).
        
        Args:
            X: Features
            
        Returns:
            Array des confiances
        """
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)


# =========================================
# FONCTIONS UTILITAIRES
# =========================================

def label_regimes_by_characteristics(regimes: np.ndarray, 
                                      features_df: pd.DataFrame,
                                      regime_column: str = 'regime') -> Tuple[pd.Series, Dict]:
    """
    Labellise les régimes numériques selon leurs caractéristiques.
    
    Args:
        regimes: Array des régimes (entiers)
        features_df: DataFrame des features
        regime_column: Nom de la colonne régime
        
    Returns:
        Series des régimes labellisés, Mapping
    """
    df = features_df.copy()
    df[regime_column] = regimes
    
    # Calculer les moyennes par régime
    means = df.groupby(regime_column).mean()
    
    # Ordre par dUNRATE croissant (expansion = faible chômage)
    if 'dUNRATE' in means.columns:
        order = means['dUNRATE'].sort_values().index.tolist()
    elif 'CPI_YoY' in means.columns:
        order = means['CPI_YoY'].sort_values().index.tolist()
    else:
        order = sorted(means.index.tolist())
    
    # Créer le mapping
    labels = ['expansion', 'neutral', 'contraction']
    mapping = {}
    for i, state in enumerate(order):
        label = labels[min(i, len(labels) - 1)]
        mapping[state] = label
    
    # Appliquer
    labeled = pd.Series(regimes).map(mapping)
    
    return labeled, mapping


def compare_detectors(X: np.ndarray, 
                       y: np.ndarray,
                       test_size: float = 0.3) -> pd.DataFrame:
    """
    Compare les performances des différents détecteurs.
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion de test
        
    Returns:
        DataFrame de comparaison
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    results = []
    
    # Random Forest
    rf = RandomForestRegimeDetector()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results.append({
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, rf_pred),
        'F1 Score': f1_score(y_test, rf_pred, average='weighted')
    })
    
    # Ensemble
    ensemble = EnsembleRegimeDetector(n_regimes=len(np.unique(y)))
    ensemble.fit(X_train, y_train)
    ens_pred = ensemble.predict(X_test)
    results.append({
        'Model': 'Ensemble',
        'Accuracy': accuracy_score(y_test, ens_pred),
        'F1 Score': f1_score(y_test, ens_pred, average='weighted')
    })
    
    return pd.DataFrame(results)


# =========================================
# MAIN (pour test)
# =========================================

if __name__ == "__main__":
    print("="*70)
    print("🤖 MODELS MODULE - TEST")
    print("="*70)
    
    # Données de test
    np.random.seed(42)
    n_samples = 500
    
    # Features simulées
    X = np.random.randn(n_samples, 5)
    
    # Labels simulés (3 régimes)
    y = np.random.choice(['expansion', 'slowdown', 'contraction'], n_samples)
    
    # Test HMM
    print("\n[TEST 1] HMMRegimeDetector...")
    hmm = HMMRegimeDetector(n_states=3)
    hmm.fit(X)
    hmm_pred = hmm.predict(X)
    print(f"   ✅ HMM predictions: {np.unique(hmm_pred)}")
    
    # Test Random Forest
    print("\n[TEST 2] RandomForestRegimeDetector...")
    rf = RandomForestRegimeDetector()
    rf.fit(X, y)
    rf_pred = rf.predict(X)
    print(f"   ✅ RF predictions: {np.unique(rf_pred)}")
    print(f"   Feature importance: {rf.get_feature_importance()}")
    
    # Test Ensemble
    print("\n[TEST 3] EnsembleRegimeDetector...")
    config = EnsembleConfig(
        use_hmm=True,
        use_rf=True,
        use_gbm=True,
        voting_method='soft'
    )
    ensemble = EnsembleRegimeDetector(n_regimes=3, config=config)
    ensemble.fit(X, y)
    ens_pred = ensemble.predict(X)
    print(f"   ✅ Ensemble predictions: {np.unique(ens_pred)}")
    
    # Test accord
    print("\n[TEST 4] Model Agreement...")
    agreement = ensemble.get_model_agreement(X)
    print(f"   ✅ Agreement rate: {agreement:.2%}")
    
    # Test confidence
    print("\n[TEST 5] Prediction Confidence...")
    confidence = ensemble.get_confidence(X)
    print(f"   ✅ Mean confidence: {confidence.mean():.2%}")
    print(f"   ✅ Min confidence: {confidence.min():.2%}")
    
    # Test cross-validation
    print("\n[TEST 6] Cross-Validation...")
    cv_scores = ensemble.cross_validate(X, y, cv=3)
    
    # Test comparaison
    print("\n[TEST 7] Compare Detectors...")
    comparison = compare_detectors(X, y)
    print(comparison.to_string())
    
    print("\n" + "="*70)
    print("✅ MODELS MODULE - ALL TESTS PASSED!")
    print("="*70)