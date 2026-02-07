"""
Report Generator for GenomeDefender Detection Results.

Generates comprehensive human-readable and machine-readable reports
for scRNA-seq data poisoning detection results.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging


# Attack type mapping for each model
ATTACK_TYPES = {
    'cae': 'Synthetic Cell Injection',
    'vae': 'Gene Scaling',
    'gnn_ae': 'Label Flips',
    'ddpm': 'Noise Injection'
}


class ReportGenerator:
    """Generates detection reports with percentage poisoned, affected cells/genes."""
    
    def __init__(self, 
                 model_name: str,
                 dataset_path: str,
                 anomaly_scores: np.ndarray,
                 threshold: float,
                 adata,
                 poisoned_cells: Dict[str, List[str]]):
        """
        Initialize the report generator.
        
        Args:
            model_name: Name of the model used ('cae', 'vae', 'gnn_ae', 'ddpm')
            dataset_path: Path to the dataset
            anomaly_scores: Array of anomaly scores per cell
            threshold: Detection threshold used
            adata: AnnData object with cell/gene information
            poisoned_cells: Dict mapping cell names to list of affected genes
        """
        self.model_name = model_name.lower()
        self.dataset_path = dataset_path
        self.anomaly_scores = anomaly_scores
        self.threshold = threshold
        self.adata = adata
        self.poisoned_cells = poisoned_cells
        self.attack_type = ATTACK_TYPES.get(self.model_name, 'Unknown')
        self.timestamp = datetime.now()
        
    @property
    def total_cells(self) -> int:
        return len(self.anomaly_scores)
    
    @property
    def num_poisoned(self) -> int:
        return len(self.poisoned_cells)
    
    @property
    def poisoning_percentage(self) -> float:
        return (self.num_poisoned / self.total_cells) * 100 if self.total_cells > 0 else 0.0
    
    def get_top_affected_genes(self, top_n: int = 10) -> List[tuple]:
        """Get the top N most frequently affected genes across all poisoned cells."""
        gene_counts = {}
        for genes in self.poisoned_cells.values():
            for gene in genes:
                gene_counts[gene] = gene_counts.get(gene, 0) + 1
        
        sorted_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_genes[:top_n]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get detection summary as a dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'model': self.model_name.upper(),
            'attack_type': self.attack_type,
            'dataset': str(self.dataset_path),
            'total_cells': self.total_cells,
            'poisoned_cells': self.num_poisoned,
            'poisoning_percentage': round(self.poisoning_percentage, 2),
            'detection_threshold': self.threshold,
            'anomaly_score_stats': {
                'mean': float(np.mean(self.anomaly_scores)),
                'std': float(np.std(self.anomaly_scores)),
                'min': float(np.min(self.anomaly_scores)),
                'max': float(np.max(self.anomaly_scores))
            },
            'top_affected_genes': [
                {'gene': gene, 'affected_cells': count} 
                for gene, count in self.get_top_affected_genes(10)
            ]
        }
    
    def generate_text_report(self) -> str:
        """Generate human-readable text report."""
        summary = self.get_summary()
        top_genes = self.get_top_affected_genes(10)
        
        report_lines = [
            "=" * 60,
            "       GENOMEDEFENDER DETECTION REPORT",
            "=" * 60,
            "",
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {summary['model']}",
            f"Attack Type Detected: {summary['attack_type']}",
            "",
            "-" * 60,
            "DATASET SUMMARY",
            "-" * 60,
            f"Dataset Path: {summary['dataset']}",
            f"Total Cells Analyzed: {summary['total_cells']:,}",
            "",
            "-" * 60,
            "DETECTION RESULTS",
            "-" * 60,
            f"Poisoned Cells Detected: {summary['poisoned_cells']:,}",
            f"Poisoning Percentage: {summary['poisoning_percentage']:.2f}%",
            f"Detection Threshold: {summary['detection_threshold']}",
            "",
            "Anomaly Score Statistics:",
            f"  Mean: {summary['anomaly_score_stats']['mean']:.4f}",
            f"  Std:  {summary['anomaly_score_stats']['std']:.4f}",
            f"  Min:  {summary['anomaly_score_stats']['min']:.4f}",
            f"  Max:  {summary['anomaly_score_stats']['max']:.4f}",
            "",
        ]
        
        if top_genes:
            report_lines.extend([
                "-" * 60,
                "TOP AFFECTED GENES",
                "-" * 60,
            ])
            for i, (gene, count) in enumerate(top_genes, 1):
                report_lines.append(f"  {i:2d}. {gene}: {count:,} cells")
        
        report_lines.extend([
            "",
            "-" * 60,
            "OUTPUT FILES",
            "-" * 60,
            "  - _labels.csv: Binary cell labels (0=healthy, 1=poisoned)",
            "  - _poisoned.txt: Detailed poisoned cell info with affected genes",
            "  - _umap_poison.png: UMAP visualization colored by poison status",
            "  - _report.txt: This report",
            "  - _report.json: Machine-readable JSON report",
            "",
            "=" * 60,
            "         END OF REPORT",
            "=" * 60,
        ])
        
        return "\n".join(report_lines)
    
    def export_text(self, output_path: str) -> None:
        """Export report as text file."""
        report = self.generate_text_report()
        with open(output_path, 'w') as f:
            f.write(report)
        logging.info(f"Text report saved to {output_path}")
        print(f"Detection report saved to {output_path}")
    
    def export_json(self, output_path: str) -> None:
        """Export report as JSON file."""
        summary = self.get_summary()
        # Add poisoned cells detail
        summary['poisoned_cells_detail'] = {
            cell: genes for cell, genes in list(self.poisoned_cells.items())[:100]  # Limit for large datasets
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logging.info(f"JSON report saved to {output_path}")
    
    def print_summary(self) -> None:
        """Print summary to console."""
        print("\n" + "=" * 50)
        print("  GENOMEDEFENDER DETECTION SUMMARY")
        print("=" * 50)
        print(f"  Model: {self.model_name.upper()} ({self.attack_type})")
        print(f"  Total Cells: {self.total_cells:,}")
        print(f"  Poisoned Cells: {self.num_poisoned:,}")
        print(f"  Poisoning Rate: {self.poisoning_percentage:.2f}%")
        print("=" * 50 + "\n")
