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
    
    Args:
        n_trees: Number of trees in the forest
        sample_size: Subsampling size
        extension_level: Level of extension (0 to n_features-1)
        contamination: Expected proportion of outliers
        random_state: Random seed
    """
    
    def __init__(self, 
                 n_trees: int = 100,
                 sample_size: int = 256,
                 extension_level: Optional[int] = None,
                 contamination: float = 0.1,
                 random_state: Optional[int] = None,
                 **kwargs):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.extension_level = extension_level
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self._configure_from_kwargs(kwargs)
        
    def _configure_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Handle additional configuration parameters."""
        if 'config_path' in kwargs:
            with open(kwargs['config_path'], 'r') as f:
                config = yaml.safe_load(f).get('models', {}).get('eif', {})
            self.n_trees = config.get('n_trees', self.n_trees)
            self.sample_size = config.get('sample_size', self.sample_size)
            self.extension_level = config.get('extension_level', self.extension_level)
            self.contamination = config.get('contamination', self.contamination)
            self.random_state = config.get('random_state', self.random_state)
    
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