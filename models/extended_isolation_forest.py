import numpy as np
from sklearn.base import BaseEstimator
from eif import extended_isolation_forest
from typing import Optional, Dict, Any
import yaml
import logging

class ExtendedIsolationForest:
    """
    Extended Isolation Forest (EIF) for genomic anomaly detection.
    Wrapper around the eif package implementation with configurable parameters.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.n_trees = self.config.get('n_trees', 200)
        self.sample_size = self.config.get('sample_size', 512)
        self.extension_level = self.config.get('extension_level', 15)
        self.contamination = self.config.get('contamination', 0.05)
        self.random_state = self.config.get('random_state', 42)
        self._validate_params()     #Validate for error handling in the parameters
        self._init_model()  #Initialize empty model structure
        
    def _init_model(self):      #Initialize model
        self.model = extended_isolation_forest.ExtendedIsolationForest(
            n_trees=self.n_trees,
            sample_size=self.sample_size,
            ExtensionLevel=self.extension_level,
            random_state=self.random_state
        )
    
    def _validate_params(self):     #Validate parameters
        if self.n_trees <= 0:
            raise ValueError("Number of trees must be a positive integer.")
        if self.sample_size <= 0:
            raise ValueError("Sample size must be a positive integer.")
        if self.extension_level < 0:
            raise ValueError("Extension level must be a non-negative integer.")
        if not isinstance(self.contamination, (float, int)) or not (0 <= self.contamination <= 1):
            raise ValueError("Contamination must be a float between 0 and 1.")
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise ValueError("Random state must be an integer or None.")
        if self.random_state is not None and self.random_state < 0:
            raise ValueError("Random state must be a non-negative integer.")
    
    #UNTIL HERE CODE IS PERFECT!!!!!
    
    
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the Extended Isolation Forest model.
        Args:
            X: Input data (n_samples, n_features)
        """
        # Determine extension level if not specified
        if self.extension_level is None:
            self.extension_level = min(X.shape[1] - 1, 1)  # Default to 1 for low-dim data
        
        self.model = extended_isolation_forest.ExtendedIsolationForest(
            n_trees=self.n_trees,
            sample_size=self.sample_size,
            ExtensionLevel=self.extension_level,
            random_state=self.random_state
        )
        
        logging.info(f"Fitting EIF with {self.n_trees} trees, sample_size={self.sample_size}, "
                    f"extension_level={self.extension_level}")
        
        self.model.fit(X)
    
    def detect(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for input data.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Anomaly scores (n_samples,)
        """
        if self.model is None:
            self.fit(X)
        
        # Compute anomaly scores (higher = more anomalous)
        scores = self.model.compute_paths(X)
        
        # Normalize scores to [0, 1] range
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        logging.info(f"EIF detection completed. Score range: {scores.min():.3f}-{scores.max():.3f}")
        
        return scores
    
    @classmethod
    def from_config(cls, config_path: str = 'config.yaml') -> 'ExtendedIsolationForest':
        """
        Create instance from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configured ExtendedIsolationForest instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f).get('models', {}).get('eif', {})
        
        return cls(
            n_trees=config.get('n_trees', 100),
            sample_size=config.get('sample_size', 256),
            extension_level=config.get('extension_level'),
            contamination=config.get('contamination', 0.1),
            random_state=config.get('random_state')
        )