"""
GenomeDefender Logging Utilities.

Provides colored console output and JSON file logging with audit trails.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """Structured JSON logging for log files."""
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
    """Colored console output for better readability."""
    COLORS = {
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[91m',  # Red
        'INFO': '\033[92m',      # Green
        'DEBUG': '\033[94m'      # Blue
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        timestamp = datetime.now().strftime('%H:%M:%S')
        return f"{color}[{timestamp}] [{record.levelname}] {record.getMessage()}{self.RESET}"


def setup_logging(
    log_level: str = 'INFO',
    log_dir: str = 'logs',
    log_name: str = 'genome_defender'
) -> logging.Logger:
    """
    Set up logging with colored console output and JSON file logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_name: Base name for log files
        
    Returns:
        Configured logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('genome_defender')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    # File handler (JSON format)
    log_file = log_path / f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler (colored)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger() -> logging.Logger:
    """Get the GenomeDefender logger instance."""
    return logging.getLogger('genome_defender')


def log_detection_stats(
    logger: logging.Logger,
    total_cells: int,
    poisoned_cells: int,
    threshold: float,
    model_name: str
) -> None:
    """Log detection statistics in a formatted way."""
    rate = (poisoned_cells / total_cells) * 100 if total_cells > 0 else 0
    logger.info(f"Detection complete using {model_name.upper()}")
    logger.info(f"  Total cells analyzed: {total_cells:,}")
    logger.info(f"  Poisoned cells found: {poisoned_cells:,} ({rate:.2f}%)")
    logger.info(f"  Detection threshold: {threshold}")


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    loss: float,
    model_name: str
) -> None:
    """Log training progress."""
    logger.info(f"[{model_name.upper()}] Epoch {epoch}/{total_epochs} - Loss: {loss:.6f}")


class AuditTrail:
    """Records audit events for compliance and debugging."""
    
    def __init__(self, log_dir: str = 'logs'):
        self.audit_file = Path(log_dir) / 'audit_trail.ndjson'
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    def log_event(self, event_type: str, message: str, metadata: Dict[str, Any] = None) -> None:
        """Log an audit event to the audit trail file."""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'message': message,
            'metadata': metadata or {}
        }
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def log_detection(self, model: str, dataset: str, poisoned_count: int, total_count: int) -> None:
        """Log a detection run."""
        self.log_event('detection', f'Detection run with {model}', {
            'model': model,
            'dataset': dataset,
            'poisoned_count': poisoned_count,
            'total_count': total_count,
            'poisoning_rate': poisoned_count / total_count if total_count > 0 else 0
        })
    
    def log_training(self, model: str, dataset: str, epochs: int, final_loss: float) -> None:
        """Log a training run."""
        self.log_event('training', f'Training {model}', {
            'model': model,
            'dataset': dataset,
            'epochs': epochs,
            'final_loss': final_loss
        })