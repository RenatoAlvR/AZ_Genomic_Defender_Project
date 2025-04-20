import logging
from pathlib import Path
from typing import Dict, Any
import yaml
import sys
import json
import numpy as np
from datetime import datetime

class GenomeGuardianLogger:
    """
    Centralized logging system for:
    - Console output (colored)
    - File logging (persistent)
    - Performance metric tracking
    - Audit trails for model updates
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
        self.log_dir = Path(self.config.get('log_dir', 'logs'))
        self._setup_logger()
        self.metric_history = []

    def _setup_logger(self) -> None:
        """Initialize multi-handler logging."""
        self.log_dir.mkdir(exist_ok=True)
        
        # Main logger
        self.logger = logging.getLogger('genome_guardian')
        self.logger.setLevel(self.config['log_level'])
        
        # Clean old log files (keep N most recent)
        self._rotate_logs()
        
        # File handler (JSON format for parsing)
        log_file = self.log_dir / f"guardian_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JsonFormatter())
        
        # Console handler (colored)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColorFormatter())
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Audit trail setup
        if self.config['audit_log']:
            self.audit_log = self.log_dir / "audit_trail.ndjson"
            self._log_audit_event('system', 'logger_initialized')

    def _rotate_logs(self) -> None:
        """Keep only N most recent log files."""
        log_files = sorted(self.log_dir.glob('guardian_*.log'))
        for old_log in log_files[:-self.config['max_files']]:
            old_log.unlink()

    def _log_audit_event(self, event_type: str, message: str, metadata: dict = {}) -> None:
        """Write structured audit log entry."""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': event_type,
            'message': message,
            **metadata
        }
        with open(self.audit_log, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_metrics(self, 
                   model_name: str, 
                   metrics: Dict[str, float], 
                   dataset: str = 'validation') -> None:
        """Record model performance metrics."""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'model': model_name,
            'dataset': dataset,
            **metrics
        }
        self.metric_history.append(entry)
        self._log_audit_event('metrics', f"{model_name} performance", metrics)
        self.logger.info(f"Metrics recorded for {model_name}")

    def log_training_update(self,
                          model_name: str,
                          incremental: bool,
                          samples: int,
                          duration: float) -> None:
        """Log model training events."""
        event = 'incremental_update' if incremental else 'full_retrain'
        metadata = {
            'model': model_name,
            'samples': samples,
            'duration_sec': round(duration, 2)
        }
        self._log_audit_event('training', event, metadata)
        self.logger.info(
            f"{model_name} {'updated' if incremental else 'retrained'} "
            f"with {samples} samples (took {duration:.2f}s)"
        )

    def log_anomaly_stats(self, 
                         scores: np.ndarray, 
                         threshold: float) -> None:
        """Log detection statistics."""
        n_anomalies = np.sum(scores > threshold)
        stats = {
            'total_samples': len(scores),
            'anomalies': int(n_anomalies),
            'rate': float(n_anomalies / len(scores))
        }
        self._log_audit_event('detection', 'anomaly_stats', stats)
        self.logger.info(
            f"Detection stats - Anomalies: {n_anomalies}/{len(scores)} "
            f"({stats['rate']:.2%})"
        )

class JsonFormatter(logging.Formatter):
    """Structured JSON logging for files."""
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno
        }
        return json.dumps(log_entry)

class ColorFormatter(logging.Formatter):
    """Colored console output."""
    COLORS = {
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[91m',
        'INFO': '\033[92m',     # Green
        'DEBUG': '\033[94m'     # Blue
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        reset = '\033[0m'
        return f"{color}[{record.levelname}] {record.getMessage()}{reset}"

# Singleton logger instance
logger = GenomeGuardianLogger()