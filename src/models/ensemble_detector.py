"""
Ensemble Anomaly Detector for ICS Networks
Combines multiple ML models for improved accuracy

Models:
- Isolation Forest (unsupervised)
- XGBoost (supervised)
- Random Forest (supervised)
- One-Class SVM (optional)

Author: Sadhana Devarajan
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleICSDetector:
    """
    Ensemble anomaly detector combining multiple algorithms.
    
    Voting Strategy:
    - Weighted voting based on individual model performance
    - Isolation Forest: 40% (best for unknown attacks)
    - XGBoost: 35% (best for known patterns)
    - Random Forest: 25% (robust baseline)
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize ensemble detector."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Models
        self.isolation_forest = None
        self.xgb_model = None
        self.rf_model = None
        self.ocsvm = None
        
        # Scaler
        self.scaler = StandardScaler()
        
        # Weights for voting
        self.weights = {
            'isolation_forest': 0.40,
            'xgboost': 0.35,
            'random_forest': 0.25
        }
        
        # Performance metrics
        self.metrics = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray = None):
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels (optional for unsupervised)
        """
        logger.info("Training Ensemble ICS Detector...")
        logger.info(f"Training samples: {len(X_train)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 1. Train Isolation Forest (unsupervised)
        logger.info("\n[1/3] Training Isolation Forest...")
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=0.1,
            random_state=self.random_seed,
            n_jobs=-1,
            verbose=0
        )
        self.isolation_forest.fit(X_train_scaled)
        logger.info("✅ Isolation Forest trained")
        
        # Check if we have both classes for supervised learning
        if y_train is not None:
            unique_classes = np.unique(y_train)
            logger.info(f"\nLabel analysis: {len(unique_classes)} unique classes found: {unique_classes}")
            
            if len(unique_classes) < 2:
                logger.warning("⚠️  Only one class found - cannot train supervised models")
                logger.warning("    Using Isolation Forest only (unsupervised)")
                logger.info("\n✅ Ensemble training complete (unsupervised mode)")
                return
        
        # If labels available AND multiple classes, train supervised models
        if y_train is not None and len(unique_classes) >= 2:
            # 2. Train XGBoost
            logger.info("\n[2/3] Training XGBoost...")
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_seed,
                n_jobs=-1,
                eval_metric='logloss'
            )
            self.xgb_model.fit(X_train_scaled, y_train)
            logger.info("✅ XGBoost trained")
            
            # 3. Train Random Forest
            logger.info("\n[3/3] Training Random Forest...")
            self.rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=self.random_seed,
                n_jobs=-1
            )
            self.rf_model.fit(X_train_scaled, y_train)
            logger.info("✅ Random Forest trained")
        else:
            logger.info("⚠️  No labels provided or insufficient classes - using Isolation Forest only")
        
        logger.info("\n✅ Ensemble training complete!")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using ensemble voting.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        predictions_dict = {}
        
        # Isolation Forest (-1 = anomaly, 1 = normal)
        iso_pred = self.isolation_forest.predict(X_scaled)
        iso_pred_binary = (iso_pred == -1).astype(int)  # Convert to 0/1
        predictions_dict['isolation_forest'] = iso_pred_binary
        
        # XGBoost (if trained)
        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(X_scaled)
            predictions_dict['xgboost'] = xgb_pred
        
        # Random Forest (if trained)
        if self.rf_model is not None:
            rf_pred = self.rf_model.predict(X_scaled)
            predictions_dict['random_forest'] = rf_pred
        
        # Weighted voting
        weighted_scores = np.zeros(len(X))
        
        for model_name, preds in predictions_dict.items():
            weight = self.weights.get(model_name, 0)
            weighted_scores += preds * weight
        
        # Threshold at 0.5
        ensemble_predictions = (weighted_scores >= 0.5).astype(int)
        
        # Confidence = how unanimous the vote was
        confidence_scores = np.abs(weighted_scores - 0.5) * 2  # Scale to 0-1
        
        return ensemble_predictions, confidence_scores
    
    def predict_with_details(self, X: np.ndarray) -> pd.DataFrame:
        """
        Predict with detailed breakdown by model.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with predictions from each model
        """
        X_scaled = self.scaler.transform(X)
        
        results = pd.DataFrame()
        
        # Isolation Forest
        iso_pred = self.isolation_forest.predict(X_scaled)
        iso_scores = self.isolation_forest.score_samples(X_scaled)
        results['iso_forest_pred'] = (iso_pred == -1).astype(int)
        results['iso_forest_score'] = iso_scores
        
        # XGBoost
        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(X_scaled)
            xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]
            results['xgb_pred'] = xgb_pred
            results['xgb_proba'] = xgb_proba
        
        # Random Forest
        if self.rf_model is not None:
            rf_pred = self.rf_model.predict(X_scaled)
            rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
            results['rf_pred'] = rf_pred
            results['rf_proba'] = rf_proba
        
        # Ensemble
        ensemble_pred, confidence = self.predict(X)
        results['ensemble_pred'] = ensemble_pred
        results['ensemble_confidence'] = confidence
        
        # Agreement level
        if self.xgb_model and self.rf_model:
            results['models_agree'] = (
                (results['iso_forest_pred'] == results['xgb_pred']) &
                (results['xgb_pred'] == results['rf_pred'])
            ).astype(int)
        
        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate ensemble performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        logger.info("\n" + "="*60)
        logger.info("ENSEMBLE MODEL EVALUATION")
        logger.info("="*60)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Check if supervised models exist
        has_supervised = self.xgb_model is not None
        
        if not has_supervised:
            logger.info("\n⚠️  Unsupervised mode - showing anomaly detection stats only")
            
            # Isolation Forest predictions
            iso_pred = self.isolation_forest.predict(X_test_scaled)
            anomalies = (iso_pred == -1).sum()
            normal = (iso_pred == 1).sum()
            anomaly_rate = anomalies / len(iso_pred) * 100
            
            logger.info("\n📊 ANOMALY DETECTION RESULTS:")
            logger.info(f"   Total samples:  {len(iso_pred)}")
            logger.info(f"   Normal:         {normal} ({100-anomaly_rate:.1f}%)")
            logger.info(f"   Anomalies:      {anomalies} ({anomaly_rate:.1f}%)")
            
            metrics = {
                'mode': 'unsupervised',
                'total_samples': len(iso_pred),
                'anomalies': int(anomalies),
                'normal': int(normal),
                'anomaly_rate': anomaly_rate
            }
            
            self.metrics = metrics
            return metrics
        
        # Ensemble predictions (supervised mode)
        ensemble_pred, confidence = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mode': 'supervised',
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred, zero_division=0),
            'recall': recall_score(y_test, ensemble_pred, zero_division=0),
            'f1_score': f1_score(y_test, ensemble_pred, zero_division=0),
        }
        
        # Individual model metrics
        iso_pred = (self.isolation_forest.predict(X_test_scaled) == -1).astype(int)
        metrics['iso_forest_accuracy'] = accuracy_score(y_test, iso_pred)
        
        if self.xgb_model:
            xgb_pred = self.xgb_model.predict(X_test_scaled)
            metrics['xgb_accuracy'] = accuracy_score(y_test, xgb_pred)
        
        if self.rf_model:
            rf_pred = self.rf_model.predict(X_test_scaled)
            metrics['rf_accuracy'] = accuracy_score(y_test, rf_pred)
        
        # Print results
        logger.info("\n📊 ENSEMBLE PERFORMANCE:")
        logger.info(f"   Accuracy:  {metrics['accuracy']:.1%}")
        logger.info(f"   Precision: {metrics['precision']:.1%}")
        logger.info(f"   Recall:    {metrics['recall']:.1%}")
        logger.info(f"   F1-Score:  {metrics['f1_score']:.4f}")
        
        logger.info("\n📊 INDIVIDUAL MODEL ACCURACIES:")
        logger.info(f"   Isolation Forest: {metrics['iso_forest_accuracy']:.1%}")
        if 'xgb_accuracy' in metrics:
            logger.info(f"   XGBoost:          {metrics['xgb_accuracy']:.1%}")
        if 'rf_accuracy' in metrics:
            logger.info(f"   Random Forest:    {metrics['rf_accuracy']:.1%}")
        
        logger.info("\n💡 ENSEMBLE IMPROVEMENT:")
        best_individual = max([
            metrics['iso_forest_accuracy'],
            metrics.get('xgb_accuracy', 0),
            metrics.get('rf_accuracy', 0)
        ])
        improvement = (metrics['accuracy'] - best_individual) * 100
        logger.info(f"   +{improvement:.2f}% over best individual model")
        
        self.metrics = metrics
        return metrics
    
    def save(self, output_dir: str):
        """Save ensemble models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.isolation_forest, output_path / 'ensemble_isolation_forest.pkl')
        joblib.dump(self.scaler, output_path / 'ensemble_scaler.pkl')
        
        if self.xgb_model:
            joblib.dump(self.xgb_model, output_path / 'ensemble_xgboost.pkl')
        
        if self.rf_model:
            joblib.dump(self.rf_model, output_path / 'ensemble_random_forest.pkl')
        
        # Save weights and metrics
        import json
        with open(output_path / 'ensemble_config.json', 'w') as f:
            json.dump({
                'weights': self.weights,
                'metrics': self.metrics,
                'random_seed': self.random_seed
            }, f, indent=2)
        
        logger.info(f"\n✅ Ensemble models saved to {output_path}")
    
    def load(self, model_dir: str):
        """Load ensemble models."""
        model_path = Path(model_dir)
        
        self.isolation_forest = joblib.load(model_path / 'ensemble_isolation_forest.pkl')
        self.scaler = joblib.load(model_path / 'ensemble_scaler.pkl')
        
        if (model_path / 'ensemble_xgboost.pkl').exists():
            self.xgb_model = joblib.load(model_path / 'ensemble_xgboost.pkl')
        
        if (model_path / 'ensemble_random_forest.pkl').exists():
            self.rf_model = joblib.load(model_path / 'ensemble_random_forest.pkl')
        
        logger.info("✅ Ensemble models loaded")


def train_ensemble_detector():
    """Train ensemble detector on ICS data."""
    print("="*80)
    print("ENSEMBLE ICS ANOMALY DETECTOR - TRAINING")
    print("="*80)
    
    # Find data directory - works from any location
    current_dir = Path(__file__).parent
    data_path = current_dir.parent.parent / "data" / "processed" / "ics_features.csv"
    labels_path = current_dir.parent.parent / "data" / "processed" / "ics_labels.csv"
    
    # Try alternative paths if not found
    if not data_path.exists():
        data_path = Path("./data/processed/ics_features.csv")
        labels_path = Path("./data/processed/ics_labels.csv")
    
    if not data_path.exists():
        data_path = Path("../../data/processed/ics_features.csv")
        labels_path = Path("../../data/processed/ics_labels.csv")
    
    if not data_path.exists():
        print(f"❌ Features not found")
        print(f"Tried locations:")
        print(f"  - {current_dir.parent.parent / 'data' / 'processed' / 'ics_features.csv'}")
        print(f"  - ./data/processed/ics_features.csv")
        print(f"  - ../../data/processed/ics_features.csv")
        print(f"\nCurrent directory: {Path.cwd()}")
        print(f"\nPlease ensure data files exist or run from project root")
        return
    
    print(f"\n📥 Loading data...")
    features = pd.read_csv(data_path)
    labels = pd.read_csv(labels_path).values.ravel()
    
    print(f"✅ Loaded {len(features)} samples with {len(features.columns)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train):,}")
    print(f"   Testing:  {len(X_test):,}")
    
    # Train ensemble
    ensemble = EnsembleICSDetector(random_seed=42)
    ensemble.train(X_train.values, y_train)
    
    # Evaluate
    metrics = ensemble.evaluate(X_test.values, y_test)
    
    # Save models
    models_dir = Path("./models")
    if not models_dir.exists():
        models_dir = current_dir.parent.parent / "models"
    
    ensemble.save(str(models_dir))
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    train_ensemble_detector()