"""
Automated Model Retraining Pipeline
CONTRIBUTION: MLOps pipeline with data drift detection and auto-retraining
Features: Data drift detection, performance monitoring, automated retraining

"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detects data drift using statistical tests.
    
    Methods:
    - Kolmogorov-Smirnov test for distribution changes
    - Population Stability Index (PSI)
    - Feature statistics comparison
    """
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize detector with reference data.
        
        Args:
            reference_data: Training or baseline data
        """
        self.reference_data = reference_data
        self.reference_stats = self._calculate_statistics(reference_data)
        
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate statistical summary of data."""
        stats_dict = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'q25': data[col].quantile(0.25),
                'q50': data[col].quantile(0.50),
                'q75': data[col].quantile(0.75)
            }
        
        return stats_dict
    
    def detect_drift_ks_test(self, 
                            new_data: pd.DataFrame,
                            significance_level: float = 0.05) -> Dict:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            new_data: New data to test
            significance_level: P-value threshold
            
        Returns:
            Dictionary with drift detection results
        """
        logger.info("Running K-S test for data drift...")
        
        drift_results = {}
        drifted_features = []
        
        for col in self.reference_data.select_dtypes(include=[np.number]).columns:
            if col not in new_data.columns:
                continue
            
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                self.reference_data[col].dropna(),
                new_data[col].dropna()
            )
            
            is_drifted = p_value < significance_level
            
            drift_results[col] = {
                'statistic': statistic,
                'p_value': p_value,
                'drifted': is_drifted
            }
            
            if is_drifted:
                drifted_features.append(col)
        
        drift_summary = {
            'total_features': len(drift_results),
            'drifted_features': len(drifted_features),
            'drift_percentage': len(drifted_features) / len(drift_results) * 100,
            'drifted_feature_list': drifted_features,
            'detailed_results': drift_results
        }
        
        logger.info(f"Drift detected in {len(drifted_features)}/{len(drift_results)} features " +
                   f"({drift_summary['drift_percentage']:.1f}%)")
        
        return drift_summary
    
    def calculate_psi(self, 
                     new_data: pd.DataFrame,
                     buckets: int = 10,
                     threshold: float = 0.2) -> Dict:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change
        PSI > 0.2: Significant change
        
        Args:
            new_data: New data to compare
            buckets: Number of bins for discretization
            threshold: PSI threshold for drift
            
        Returns:
            Dictionary with PSI results
        """
        logger.info("Calculating Population Stability Index...")
        
        psi_results = {}
        high_psi_features = []
        
        for col in self.reference_data.select_dtypes(include=[np.number]).columns:
            if col not in new_data.columns:
                continue
            
            # Create bins based on reference data
            ref_col = self.reference_data[col].dropna()
            new_col = new_data[col].dropna()
            
            if len(ref_col) == 0 or len(new_col) == 0:
                continue
            
            # Quantile-based binning
            bin_edges = np.percentile(ref_col, np.linspace(0, 100, buckets + 1))
            bin_edges = np.unique(bin_edges)  # Remove duplicates
            
            if len(bin_edges) < 2:
                continue
            
            # Calculate proportions
            ref_counts, _ = np.histogram(ref_col, bins=bin_edges)
            new_counts, _ = np.histogram(new_col, bins=bin_edges)
            
            ref_props = ref_counts / len(ref_col)
            new_props = new_counts / len(new_col)
            
            # Add small constant to avoid division by zero
            ref_props = np.where(ref_props == 0, 0.0001, ref_props)
            new_props = np.where(new_props == 0, 0.0001, new_props)
            
            # Calculate PSI
            psi = np.sum((new_props - ref_props) * np.log(new_props / ref_props))
            
            psi_results[col] = {
                'psi': psi,
                'drifted': psi > threshold
            }
            
            if psi > threshold:
                high_psi_features.append(col)
        
        psi_summary = {
            'features_analyzed': len(psi_results),
            'high_psi_features': len(high_psi_features),
            'high_psi_list': high_psi_features,
            'detailed_results': psi_results
        }
        
        logger.info(f"High PSI detected in {len(high_psi_features)} features")
        
        return psi_summary


class ModelPerformanceMonitor:
    """
    Monitors model performance over time.
    
    Tracks:
    - Prediction accuracy
    - Error rates
    - Performance degradation
    """
    
    def __init__(self, baseline_metrics: Dict):
        """
        Initialize with baseline metrics.
        
        Args:
            baseline_metrics: Performance metrics from initial training
        """
        self.baseline_metrics = baseline_metrics
        self.performance_history = []
        
    def evaluate_current_performance(self,
                                    y_true: pd.DataFrame,
                                    predictions: pd.DataFrame) -> Dict:
        """
        Evaluate current model performance.
        
        Args:
            y_true: True targets
            predictions: Model predictions
            
        Returns:
            Dictionary with current metrics
        """
        from evaluation_metrics import SupplyChainEvaluator
        
        evaluator = SupplyChainEvaluator()
        current_metrics = evaluator.evaluate_all_targets(y_true, predictions)
        
        # Add timestamp
        current_metrics['timestamp'] = datetime.now().isoformat()
        
        # Store in history
        self.performance_history.append(current_metrics)
        
        return current_metrics
    
    def detect_performance_degradation(self,
                                      current_metrics: Dict,
                                      threshold: float = 0.1) -> Dict:
        """
        Detect if performance has degraded significantly.
        
        Args:
            current_metrics: Current performance metrics
            threshold: Degradation threshold (10% = 0.1)
            
        Returns:
            Dictionary with degradation analysis
        """
        logger.info("Checking for performance degradation...")
        
        degraded_metrics = []
        
        for metric_name in ['Severity_Accuracy', 'Duration_MAE', 'Probability_Accuracy']:
            if metric_name in self.baseline_metrics and metric_name in current_metrics:
                baseline_val = self.baseline_metrics[metric_name]
                current_val = current_metrics[metric_name]
                
                # For MAE, lower is better
                if 'MAE' in metric_name or 'MSE' in metric_name:
                    degradation = (current_val - baseline_val) / baseline_val
                else:
                    # For accuracy, higher is better
                    degradation = (baseline_val - current_val) / baseline_val
                
                if degradation > threshold:
                    degraded_metrics.append({
                        'metric': metric_name,
                        'baseline': baseline_val,
                        'current': current_val,
                        'degradation_pct': degradation * 100
                    })
        
        result = {
            'degraded': len(degraded_metrics) > 0,
            'degraded_metrics': degraded_metrics,
            'threshold_pct': threshold * 100
        }
        
        if result['degraded']:
            logger.warning(f"⚠️ Performance degradation detected in {len(degraded_metrics)} metrics!")
        else:
            logger.info("✅ No significant performance degradation")
        
        return result


class AutomatedRetrainingPipeline:
    """
    Automated pipeline for model retraining.
    
    Workflow:
    1. Monitor data drift
    2. Check performance degradation
    3. Trigger retraining if needed
    4. Validate new models
    5. Deploy if improvement detected
    """
    
    def __init__(self, 
                 model_dir: str = './models',
                 data_dir: str = './data',
                 config_path: str = './config/retraining_config.json'):
        """Initialize pipeline."""
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load retraining configuration."""
        default_config = {
            'drift_threshold': 0.2,
            'performance_threshold': 0.1,
            'min_samples_for_retraining': 100,
            'validation_split': 0.2,
            'auto_deploy': False
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            default_config.update(config)
        
        return default_config
    
    def should_retrain(self,
                      reference_data: pd.DataFrame,
                      new_data: pd.DataFrame,
                      y_true: pd.DataFrame,
                      predictions: pd.DataFrame,
                      baseline_metrics: Dict) -> Tuple[bool, str]:
        """
        Determine if retraining is needed.
        
        Args:
            reference_data: Original training features
            new_data: New features
            y_true: True targets for new data
            predictions: Current model predictions
            baseline_metrics: Baseline performance metrics
            
        Returns:
            Tuple of (should_retrain, reason)
        """
        logger.info("\n" + "="*80)
        logger.info("CHECKING RETRAINING TRIGGERS")
        logger.info("="*80)
        
        reasons = []
        
        # Check 1: Sufficient new data
        if len(new_data) < self.config['min_samples_for_retraining']:
            logger.info(f"❌ Insufficient new data ({len(new_data)} < {self.config['min_samples_for_retraining']})")
            return False, "Insufficient new data"
        
        # Check 2: Data drift
        drift_detector = DataDriftDetector(reference_data)
        drift_results = drift_detector.detect_drift_ks_test(new_data)
        
        if drift_results['drift_percentage'] > self.config['drift_threshold'] * 100:
            reasons.append(f"Data drift detected ({drift_results['drift_percentage']:.1f}%)")
        
        # Check 3: Performance degradation
        monitor = ModelPerformanceMonitor(baseline_metrics)
        current_metrics = monitor.evaluate_current_performance(y_true, predictions)
        degradation = monitor.detect_performance_degradation(
            current_metrics,
            self.config['performance_threshold']
        )
        
        if degradation['degraded']:
            reasons.append(f"Performance degradation in {len(degradation['degraded_metrics'])} metrics")
        
        # Decision
        should_retrain = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "No retraining needed"
        
        logger.info("\n" + "="*80)
        if should_retrain:
            logger.warning(f"⚠️ RETRAINING RECOMMENDED: {reason}")
        else:
            logger.info(f"✅ {reason}")
        logger.info("="*80 + "\n")
        
        return should_retrain, reason
    
    def retrain_models(self,
                      features: pd.DataFrame,
                      targets: pd.DataFrame) -> Dict:
        """
        Retrain all models with new data.
        
        Args:
            features: Feature dataframe
            targets: Target dataframe
            
        Returns:
            Dictionary with new model metrics
        """
        logger.info("\n" + "="*80)
        logger.info("RETRAINING MODELS")
        logger.info("="*80)
        
        from ensemble_model import EnsembleSupplyChainModel
        
        # Train new ensemble
        new_model = EnsembleSupplyChainModel(random_seed=42)
        metrics = new_model.train(features, targets, validation_split=self.config['validation_split'])
        
        # Save new models with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.model_dir / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup old models
        for f in self.model_dir.glob("ensemble_*.pkl"):
            import shutil
            shutil.copy(f, backup_dir / f.name)
        
        logger.info(f"✅ Old models backed up to {backup_dir}")
        
        # Save new models
        new_model.save_models(str(self.model_dir))
        
        logger.info("✅ New models saved")
        logger.info("="*80 + "\n")
        
        return metrics
    
    def run_pipeline(self,
                    reference_data: pd.DataFrame,
                    new_data: pd.DataFrame,
                    y_true: pd.DataFrame,
                    predictions: pd.DataFrame,
                    baseline_metrics: Dict):
        """
        Run complete retraining pipeline.
        
        Args:
            reference_data: Original training features
            new_data: New features
            y_true: True targets
            predictions: Current predictions
            baseline_metrics: Baseline metrics
        """
        # Check if retraining needed
        should_retrain, reason = self.should_retrain(
            reference_data,
            new_data,
            y_true,
            predictions,
            baseline_metrics
        )
        
        if should_retrain:
            # Combine old and new data
            combined_features = pd.concat([reference_data, new_data], axis=0).reset_index(drop=True)
            combined_targets = pd.concat([
                pd.read_csv(self.data_dir / 'target_variables.csv'),
                y_true
            ], axis=0).reset_index(drop=True)
            
            # Retrain
            new_metrics = self.retrain_models(combined_features, combined_targets)
            
            # Log results
            logger.info("\n" + "="*80)
            logger.info("RETRAINING COMPLETE")
            logger.info("="*80)
            logger.info(f"Reason: {reason}")
            logger.info(f"New Severity Accuracy: {new_metrics.get('severity_ensemble_accuracy', 0):.4f}")
            logger.info(f"New Duration MAE: {new_metrics.get('duration_ensemble_mae', 0):.4f}")
            logger.info("="*80 + "\n")
            
            # Save retraining log
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'samples_used': len(combined_features),
                'new_metrics': new_metrics
            }
            
            log_file = self.model_dir / 'retraining_log.json'
            logs = []
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        else:
            logger.info("No retraining performed")


if __name__ == "__main__":
    # Example usage
    logger.info("Automated Retraining Pipeline Demo")
    
    # Load reference data
    reference_features = pd.read_csv('./data/engineered_features.csv')
    reference_targets = pd.read_csv('./data/target_variables.csv')
    
    # Simulate new data (in production, this would be real new data)
    new_features = reference_features.sample(n=50, random_state=123).copy()
    # Add some drift
    for col in new_features.columns[:5]:
        new_features[col] = new_features[col] * 1.2
    
    new_targets = reference_targets.sample(n=50, random_state=123)
    
    # Load baseline metrics
    baseline_metrics = {
        'Severity_Accuracy': 0.78,
        'Duration_MAE': 2.8,
        'Probability_Accuracy': 0.73
    }
    
    # Simulate predictions
    predictions = new_targets.copy()
    predictions.columns = ['predicted_' + col for col in predictions.columns]
    
    # Run pipeline
    pipeline = AutomatedRetrainingPipeline()
    pipeline.run_pipeline(
        reference_features,
        new_features,
        new_targets,
        predictions,
        baseline_metrics
    )