"""
models.py
Module pour les modèles de Machine Learning pour la détection de régimes.

Fonctionnalités :
- Hidden Markov Model (HMM) gaussien
- Random Forest Classifier
- LSTM (Long Short-Term Memory)
- Ensemble Model (combinaison des 3)
- Model evaluation et comparison
- Feature scaling et preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')

# Note: Pour LSTM, on simule avec un modèle simple car tensorflow peut être lourd
# Dans un vrai projet, on utiliserait tensorflow/keras


class RegimeDetector:
    """
    Classe de base pour tous les détecteurs de régimes.
    """
    
    def __init__(self, name: str):
        """
        Initialise le détecteur.
        
        Args:
            name: Nom du modèle
        """
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        self.label_encoder = None
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit le modèle."""
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les régimes."""
        raise NotImplementedError("Subclasses must implement predict()")
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """Évalue le modèle."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        y_pred = self.predict(X)
        
        # Aligner les longueurs si nécessaire
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        accuracy = accuracy_score(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }


class HMMRegimeDetector(RegimeDetector):
    """
    Détecteur de régimes basé sur Hidden Markov Model.
    """
    
    def __init__(self, n_states: int = 3, name: str = "HMM"):
        """
        Initialise le HMM.
        
        Args:
            n_states: Nombre de régimes à détecter
            name: Nom du modèle
        """
        super().__init__(name)
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            verbose=False
        )
        self.state_mapping = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit le HMM sur les données.
        
        Args:
            X: Features (n_samples, n_features)
            y: Non utilisé pour HMM (unsupervised)
        """
        print(f"[INFO] Fitting {self.name} with {self.n_states} states...")
        
        # Standardiser
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit HMM
        self.model.fit(X_scaled)
        
        # Prédire les états pour créer le mapping
        states = self.model.predict(X_scaled)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les régimes.
        
        Args:
            X: Features
            
        Returns:
            Array des régimes prédits
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        states = self.model.predict(X_scaled)
        
        return states
    
    def get_state_statistics(self, X: np.ndarray, feature_names: list = None) -> pd.DataFrame:
        """
        Calcule les statistiques moyennes par état.
        
        Args:
            X: Features
            feature_names: Noms des features
            
        Returns:
            DataFrame avec stats par état
        """
        states = self.predict(X)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['state'] = states
        
        stats = df.groupby('state').mean()
        
        return stats
    
    def label_states(self, X: np.ndarray, reference_feature: str = None) -> dict:
        """
        Labellise les états selon leurs caractéristiques.
        
        Args:
            X: Features
            reference_feature: Feature de référence pour le tri (ex: 'dUNRATE')
            
        Returns:
            Mapping état -> label
        """
        stats = self.get_state_statistics(X)
        
        if reference_feature and reference_feature in stats.columns:
            # Trier par la feature de référence
            sorted_states = stats[reference_feature].sort_values().index.tolist()
        else:
            # Trier par la première colonne
            sorted_states = stats.iloc[:, 0].sort_values().index.tolist()
        
        # Créer le mapping
        labels = ["expansion", "neutral", "stress"]
        mapping = {}
        
        for i, state in enumerate(sorted_states):
            label = labels[min(i, len(labels) - 1)]
            mapping[state] = label
        
        self.state_mapping = mapping
        
        return mapping


class RandomForestRegimeDetector(RegimeDetector):
    """
    Détecteur de régimes basé sur Random Forest.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5, name: str = "RandomForest"):
        """
        Initialise le Random Forest.
        
        Args:
            n_estimators: Nombre d'arbres
            max_depth: Profondeur max
            name: Nom du modèle
        """
        super().__init__(name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.label_encoder = LabelEncoder()
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit le Random Forest.
        
        Args:
            X: Features
            y: Target (régimes)
        """
        print(f"[INFO] Fitting {self.name} with {self.n_estimators} trees...")
        
        # Standardiser X
        X_scaled = self.scaler.fit_transform(X)
        
        # Encoder y
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Fit
        self.model.fit(X_scaled, y_encoded)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les régimes.
        
        Args:
            X: Features
            
        Returns:
            Array des régimes prédits (labels originaux)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        y_encoded = self.model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_encoded)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les probabilités par classe.
        
        Args:
            X: Features
            
        Returns:
            Array des probabilités (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        
        return proba
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retourne l'importance des features.
        
        Returns:
            DataFrame avec importance des features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f"f{i}" for i in range(len(self.model.feature_importances_))],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


class LSTMRegimeDetector(RegimeDetector):
    """
    Détecteur de régimes basé sur LSTM (simulation simplifiée).
    
    Note: Dans un vrai projet, on utiliserait TensorFlow/Keras.
    Ici, on simule avec un modèle séquentiel simple pour éviter
    les dépendances lourdes.
    """
    
    def __init__(self, sequence_length: int = 12, name: str = "LSTM"):
        """
        Initialise le LSTM.
        
        Args:
            sequence_length: Longueur des séquences temporelles
            name: Nom du modèle
        """
        super().__init__(name)
        self.sequence_length = sequence_length
        self.label_encoder = LabelEncoder()
        
        # Modèle simplifié (simulation)
        # Dans la vraie vie: self.model = Sequential([LSTM(...), Dense(...)])
        self.model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
        print(f"[INFO] LSTM initialized (simplified version with RF backbone)")
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Crée des séquences temporelles pour LSTM.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Séquences (n_samples - seq_len, seq_len, n_features)
        """
        sequences = []
        
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit le LSTM.
        
        Args:
            X: Features
            y: Target (régimes)
        """
        print(f"[INFO] Fitting {self.name} with sequence length {self.sequence_length}...")
        
        # Standardiser
        X_scaled = self.scaler.fit_transform(X)
        
        # Créer séquences
        X_seq = self._create_sequences(X_scaled)
        
        # Ajuster y pour correspondre aux séquences
        y_seq = y[self.sequence_length:]
        
        # Encoder y
        y_encoded = self.label_encoder.fit_transform(y_seq)
        
        # Flatten séquences pour le modèle simplifié
        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        
        # Fit
        self.model.fit(X_flat, y_encoded)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les régimes.
        
        Args:
            X: Features
            
        Returns:
            Array des régimes prédits
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)
        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        
        y_encoded = self.model.predict(X_flat)
        y_pred = self.label_encoder.inverse_transform(y_encoded)
        
        # Padding pour avoir la même longueur
        y_full = np.concatenate([
            np.array([y_pred[0]] * self.sequence_length),
            y_pred
        ])
        
        return y_full


class EnsembleRegimeDetector:
    """
    Ensemble de plusieurs modèles pour prédictions robustes.
    """
    
    def __init__(self, models: list):
        """
        Initialise l'ensemble.
        
        Args:
            models: Liste des modèles RegimeDetector
        """
        self.models = models
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit tous les modèles.
        
        Args:
            X: Features
            y: Target (optionnel pour HMM)
        """
        print(f"[INFO] Fitting ensemble of {len(self.models)} models...")
        
        for model in self.models:
            if isinstance(model, HMMRegimeDetector):
                model.fit(X)
            else:
                if y is None:
                    raise ValueError(f"{model.name} requires target labels")
                model.fit(X, y)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray, voting: str = 'hard') -> np.ndarray:
        """
        Prédit avec vote majoritaire.
        
        Args:
            X: Features
            voting: 'hard' pour vote majoritaire
            
        Returns:
            Prédictions de l'ensemble
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Vote majoritaire
        predictions = np.array(predictions)
        
        # Mode le plus fréquent pour chaque sample
        ensemble_pred = []
        for i in range(predictions.shape[1]):
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            majority = unique[np.argmax(counts)]
            ensemble_pred.append(majority)
        
        return np.array(ensemble_pred)
    
    def get_prediction_agreement(self, X: np.ndarray) -> float:
        """
        Calcule le taux d'accord entre les modèles.
        
        Args:
            X: Features
            
        Returns:
            Taux d'accord (0-1)
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Pour chaque sample, vérifier si tous les modèles sont d'accord
        agreement = []
        for i in range(predictions.shape[1]):
            votes = predictions[:, i]
            agreement.append(len(np.unique(votes)) == 1)
        
        return np.mean(agreement)


# Fonctions utilitaires pour compatibilité
def fit_hmm_regimes(X: np.ndarray, n_states: int = 3):
    """Wrapper pour compatibilité."""
    model = HMMRegimeDetector(n_states=n_states)
    model.fit(X)
    return model


def fit_random_forest_regimes(X: np.ndarray, y: np.ndarray):
    """Wrapper pour compatibilité."""
    model = RandomForestRegimeDetector()
    model.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    model.fit(X, y)
    return model


def fit_lstm_regimes(X: np.ndarray, y: np.ndarray, sequence_length: int = 12):
    """Wrapper pour compatibilité."""
    model = LSTMRegimeDetector(sequence_length=sequence_length)
    model.fit(X, y)
    return model