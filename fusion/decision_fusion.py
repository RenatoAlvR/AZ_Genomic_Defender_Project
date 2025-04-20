import numpy as np
from typing import Dict, List
import yaml
import logging
from sklearn.preprocessing import MinMaxScaler

class DecisionFusion:
    """
    Non-trainable fusion that combines model scores using:
    - Confidence-weighted voting (dynamic weights based on model performance history)
    - Adaptive thresholding
    """
    
    def __init__(self,
                 method: str = 'confidence_voting',
                 initial_weights: Optional[Dict[str, float]] = None,
                 contamination: float = 0.1,
                 **kwargs):
        self.method = method
        self.weights = initial_weights or {}  # Starting weights (updated dynamically)
        self.contamination = contamination
        self.scaler = MinMaxScaler()
        self.model_performance = {}  # Tracks precision/recall per model
        self._configure_from_kwargs(kwargs)

    def _configure_from_kwargs(self, kwargs):
        """Load from config file if provided."""
        if 'config_path' in kwargs:
            with open(kwargs['config_path'], 'r') as f:
                config = yaml.safe_load(f).get('fusion', {})
            self.method = config.get('method', 'confidence_voting')
            self.weights = config.get('initial_weights', {})
            self.contamination = config.get('contamination', 0.1)

    def update_model_performance(self, model_name: str, precision: float, recall: float):
        """Update weights based on incremental evaluation metrics."""
        self.model_performance[model_name] = {'precision': precision, 'recall': recall}
        # Weight = F1 score (harmonic mean of precision/recall)
        self.weights[model_name] = 2 * (precision * recall) / (precision + recall + 1e-10)

    def confidence_voting(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average where weights = model confidence (F1 score)."""
        weighted_scores = []
        for name, score in scores.items():
            weight = self.weights.get(name, 1.0)  # Default weight = 1 if no history
            weighted_scores.append(weight * score)
        return np.mean(weighted_scores, axis=0)

    def fuse(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine scores without training."""
        if self.method == 'confidence_voting':
            combined = self.confidence_voting(scores)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Scale to [0, 1] and threshold
        combined = self.scaler.fit_transform(combined.reshape(-1, 1)).flatten()
        return combined