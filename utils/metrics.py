import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

class AnomalyMetrics:
    """
    Comprehensive metric calculations for:
    - Binary anomaly detection
    - Incremental performance tracking
    - Model comparison
    """
    
    @staticmethod
    def calculate_binary_metrics(
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        contamination: float = 0.1
    ) -> Dict[str, float]:
        """
        Compute all relevant metrics for anomaly detection.
        
        Args:
            scores: Anomaly scores (higher = more anomalous)
            labels: Ground truth (1=anomaly, 0=normal), optional
            threshold: Custom decision threshold
            contamination: Expected anomaly rate if no threshold provided
            
        Returns:
            Dictionary of metrics including:
            - precision, recall, f1, auc_roc, auc_pr
            - confusion_matrix (if labels available)
        """
        metrics = {}
        
        # Auto-determine threshold if not provided
        if threshold is None:
            threshold = np.percentile(scores, 100 * (1 - contamination))
        
        # Binary predictions
        preds = (scores >= threshold).astype(int)
        metrics['threshold'] = float(threshold)
        
        if labels is not None:
            # Standard classification metrics
            metrics.update({
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds),
                'f1': 2 * (metrics['precision'] * metrics['recall']) / 
                      (metrics['precision'] + metrics['recall'] + 1e-10),
                'auc_roc': roc_auc_score(labels, scores),
                'auc_pr': average_precision_score(labels, scores),
                'confusion_matrix': confusion_matrix(labels, preds).tolist()
            })
        
        # Always available metrics
        metrics.update({
            'anomaly_rate': float(np.mean(preds)),
            'score_stats': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
        })
        
        return metrics

    @staticmethod
    def compare_models(
        model_scores: Dict[str, np.ndarray],
        labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models' performance.
        
        Args:
            model_scores: Dictionary of {model_name: scores}
            labels: Ground truth labels
            
        Returns:
            Nested dictionary of metrics per model
        """
        return {
            name: AnomalyMetrics.calculate_binary_metrics(scores, labels)
            for name, scores in model_scores.items()
        }

    @staticmethod
    def incremental_performance(
        current_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
        alpha: float = 0.1
    ) -> Dict[str, float]:
        """
        Update metrics incrementally using exponential moving average.
        
        Args:
            current_metrics: Previous metric values
            new_metrics: Latest metric values
            alpha: Smoothing factor (0=ignore new, 1=ignore history)
            
        Returns:
            Updated metrics with EMA applied
        """
        updated = {}
        for k in current_metrics.keys():
            if k in new_metrics:
                if isinstance(current_metrics[k], (int, float)):
                    updated[k] = alpha * new_metrics[k] + (1 - alpha) * current_metrics[k]
                else:
                    updated[k] = new_metrics[k]  # Non-numeric values overwritten
        return updated

    @staticmethod
    def detect_drift(
        old_scores: np.ndarray,
        new_scores: np.ndarray,
        threshold: float = 0.05
    ) -> bool:
        """
        Detect significant distribution drift in anomaly scores.
        
        Args:
            old_scores: Reference distribution
            new_scores: Recent scores to compare
            threshold: KS test p-value threshold
            
        Returns:
            True if drift detected (p < threshold)
        """
        from scipy.stats import ks_2samp
        _, p_value = ks_2samp(old_scores, new_scores)
        return p_value < threshold