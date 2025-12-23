import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier

class HMMRegimeDetector:
    """Détecte les régimes via Gaussian HMM."""
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """Fit le modèle HMM."""
        print(f"[INFO] Fitting HMM with {self.n_states} states...")
        self.model.fit(X)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les états."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.predict(X)

class RandomForestRegimeDetector:
    """Détecte les régimes via Random Forest (supervisé)."""
    
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit le modèle RF avec labels supervisés."""
        print(f"[INFO] Fitting Random Forest...")
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les régimes."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.predict(X)