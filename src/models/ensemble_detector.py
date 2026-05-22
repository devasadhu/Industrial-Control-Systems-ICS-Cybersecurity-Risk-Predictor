"""
Ensemble Anomaly Detector for ICS Networks
Combines multiple ML models for improved accuracy

Models:
- Isolation Forest (unsupervised, always trained)
- XGBoost (supervised, trained when labeled data with 2+ classes available)
- Random Forest (supervised, trained when labeled data with 2+ classes available)

Author: Sadhana Devarajan
Version: 2.0.0

Fixes applied:
- All models saved/loaded with joblib (not pickle) consistently
- XGBoost and Random Forest pkl files saved alongside isolation forest
- ensemble_config.json updated to reflect which models are actually present
- Weights dynamically adjusted when only isolation forest is available
- Added anomaly_threshold to ensemble_config.json
- Fixed all-zeros label bug (ics_labels.csv regenerated from raw dataset)
- Removed deprecated use_label_encoder XGBoost parameter
- Decision threshold tuned to 0.25 — ICS domain: false negatives cost more
- Session aggregation features (v3, 62 total): src_inter_flow_variance
  (normal/attack ratio=11.76) is primary replay detection signal
- Final metrics: Accuracy=93.8%, Precision=90.6%, Recall=91.1%, F1=0.908
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed. Will use RandomForest only for supervised component.")


class EnsembleICSDetector:
    """
    Ensemble anomaly detector combining multiple algorithms.

    Voting Strategy:
    - When all 3 models trained: IF=0.40, XGB=0.35, RF=0.25
    - When only IF trained (unsupervised): IF=1.0
    - When IF + RF only: IF=0.55, RF=0.45
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.isolation_forest = None
        self.xgb_model = None
        self.rf_model = None

        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.anomaly_threshold: float = -0.0  # Set after training

        # Base weights (adjusted dynamically in predict based on available models)
        self._base_weights = {
            'isolation_forest': 0.40,
            'xgboost': 0.35,
            'random_forest': 0.25
        }
        self.metrics = {}

    @property
    def weights(self) -> Dict[str, float]:
        """Return weights for currently trained models, normalized to sum to 1."""
        available = {'isolation_forest': self._base_weights['isolation_forest']}
        if self.xgb_model is not None:
            available['xgboost'] = self._base_weights['xgboost']
        if self.rf_model is not None:
            available['random_forest'] = self._base_weights['random_forest']

        total = sum(available.values())
        return {k: v / total for k, v in available.items()}

    def train(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        """
        Train all models in the ensemble.

        Args:
            X_train: Training features (numpy array)
            y_train: Optional binary labels (0=normal, 1=attack)
                     If None or single-class, only Isolation Forest is trained.
        """
        logger.info("Training Ensemble ICS Detector...")
        logger.info(f"Training samples: {len(X_train)}")

        X_train_scaled = self.scaler.fit_transform(X_train)

        # 1. Isolation Forest (always trained - unsupervised)
        logger.info("\n[1/3] Training Isolation Forest...")
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=0.1,
            random_state=self.random_seed,
            n_jobs=-1,
            verbose=0
        )
        self.isolation_forest.fit(X_train_scaled)

        # Compute anomaly threshold from training scores
        train_scores = self.isolation_forest.score_samples(X_train_scaled)
        self.anomaly_threshold = float(np.percentile(train_scores, 10))
        logger.info(f"✅ Isolation Forest trained (anomaly_threshold={self.anomaly_threshold:.4f})")

        # Check label quality for supervised models
        can_train_supervised = False
        if y_train is not None:
            unique_classes = np.unique(y_train)
            logger.info(f"\nLabel analysis: {len(unique_classes)} unique classes: {unique_classes}")
            if len(unique_classes) < 2:
                logger.warning(
                    "⚠️  Only one class in labels - cannot train supervised models.\n"
                    "   Check that ics_labels.csv was generated correctly (see ics_feature_engineer.py).\n"
                    "   Running in unsupervised-only mode."
                )
            else:
                can_train_supervised = True
        else:
            logger.info("No labels provided - using Isolation Forest only (unsupervised mode).")

        if can_train_supervised:
            # 2. XGBoost
            if XGBOOST_AVAILABLE:
                logger.info("\n[2/3] Training XGBoost...")
                self.xgb_model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_seed,
                    n_jobs=-1,
                    eval_metric='logloss',
                )
                self.xgb_model.fit(X_train_scaled, y_train)
                logger.info("✅ XGBoost trained")
            else:
                logger.warning("\n[2/3] XGBoost not available - skipping")

            # 3. Random Forest
            logger.info("\n[3/3] Training Random Forest...")
            self.rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=self.random_seed,
                n_jobs=-1
            )
            self.rf_model.fit(X_train_scaled, y_train)
            logger.info("✅ Random Forest trained")

        trained_models = ['IsolationForest']
        if self.xgb_model: trained_models.append('XGBoost')
        if self.rf_model: trained_models.append('RandomForest')
        logger.info(f"\n✅ Ensemble training complete! Active models: {trained_models}")
        logger.info(f"   Active weights: {self.weights}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using ensemble weighted voting.

        Returns:
            (predictions, confidence_scores) where predictions: 0=normal, 1=attack
        """
        if self.isolation_forest is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X_scaled = self.scaler.transform(X)
        active_weights = self.weights

        weighted_scores = np.zeros(len(X))

        # Isolation Forest contribution
        iso_pred = self.isolation_forest.predict(X_scaled)
        iso_binary = (iso_pred == -1).astype(int)
        weighted_scores += iso_binary * active_weights['isolation_forest']

        # XGBoost contribution
        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(X_scaled)
            weighted_scores += xgb_pred * active_weights.get('xgboost', 0)

        # Random Forest contribution
        if self.rf_model is not None:
            rf_pred = self.rf_model.predict(X_scaled)
            weighted_scores += rf_pred * active_weights.get('random_forest', 0)

        ensemble_predictions = (weighted_scores >= 0.25).astype(int)
        confidence_scores = np.abs(weighted_scores - 0.5) * 2

        return ensemble_predictions, confidence_scores

    def predict_with_details(self, X: np.ndarray) -> pd.DataFrame:
        """Predict with per-model breakdown."""
        if self.isolation_forest is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X_scaled = self.scaler.transform(X)
        results = pd.DataFrame()

        iso_pred = self.isolation_forest.predict(X_scaled)
        iso_scores = self.isolation_forest.score_samples(X_scaled)
        results['iso_forest_pred'] = (iso_pred == -1).astype(int)
        results['iso_forest_score'] = iso_scores

        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(X_scaled)
            xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]
            results['xgb_pred'] = xgb_pred
            results['xgb_proba'] = xgb_proba

        if self.rf_model is not None:
            rf_pred = self.rf_model.predict(X_scaled)
            rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
            results['rf_pred'] = rf_pred
            results['rf_proba'] = rf_proba

        ensemble_pred, confidence = self.predict(X)
        results['ensemble_pred'] = ensemble_pred
        results['ensemble_confidence'] = confidence

        if self.xgb_model is not None and self.rf_model is not None:
            results['models_agree'] = (
                (results['iso_forest_pred'] == results['xgb_pred']) &
                (results['xgb_pred'] == results['rf_pred'])
            ).astype(int)

        return results

    def evaluate(self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None) -> Dict:
        """Evaluate ensemble performance."""
        if self.isolation_forest is None:
            raise RuntimeError("Model not trained. Call train() first.")

        logger.info("\n" + "=" * 60)
        logger.info("ENSEMBLE MODEL EVALUATION")
        logger.info("=" * 60)

        X_test_scaled = self.scaler.transform(X_test)

        has_supervised = self.xgb_model is not None or self.rf_model is not None
        has_labels = y_test is not None and len(np.unique(y_test)) >= 2

        if not has_labels or not has_supervised:
            logger.info("\n⚠️  Unsupervised mode - showing anomaly detection stats only")

            iso_pred = self.isolation_forest.predict(X_test_scaled)
            anomalies = (iso_pred == -1).sum()
            normal = (iso_pred == 1).sum()
            anomaly_rate = anomalies / len(iso_pred) * 100

            logger.info(f"\n📊 ANOMALY DETECTION RESULTS:")
            logger.info(f"   Total samples:  {len(iso_pred)}")
            logger.info(f"   Normal:         {normal} ({100 - anomaly_rate:.1f}%)")
            logger.info(f"   Anomalies:      {anomalies} ({anomaly_rate:.1f}%)")

            metrics = {
                'mode': 'unsupervised',
                'total_samples': len(iso_pred),
                'anomalies': int(anomalies),
                'normal': int(normal),
                'anomaly_rate': float(anomaly_rate)
            }
            self.metrics = metrics
            return metrics

        # Supervised evaluation
        ensemble_pred, confidence = self.predict(X_test)

        metrics = {
            'mode': 'supervised',
            'accuracy': float(accuracy_score(y_test, ensemble_pred)),
            'precision': float(precision_score(y_test, ensemble_pred, zero_division=0)),
            'recall': float(recall_score(y_test, ensemble_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, ensemble_pred, zero_division=0)),
        }

        iso_pred_bin = (self.isolation_forest.predict(X_test_scaled) == -1).astype(int)
        metrics['iso_forest_accuracy'] = float(accuracy_score(y_test, iso_pred_bin))

        if self.xgb_model:
            xgb_pred = self.xgb_model.predict(X_test_scaled)
            metrics['xgb_accuracy'] = float(accuracy_score(y_test, xgb_pred))

        if self.rf_model:
            rf_pred = self.rf_model.predict(X_test_scaled)
            metrics['rf_accuracy'] = float(accuracy_score(y_test, rf_pred))

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

        best_individual = max(
            metrics['iso_forest_accuracy'],
            metrics.get('xgb_accuracy', 0),
            metrics.get('rf_accuracy', 0)
        )
        improvement = (metrics['accuracy'] - best_individual) * 100
        logger.info(f"\n💡 Ensemble improvement over best individual: +{improvement:.2f}%")

        self.metrics = metrics
        return metrics

    def save(self, output_dir: str):
        """
        Save ensemble models.

        All files use joblib.dump() - load with joblib.load(), NOT pickle.load().
        """
        if self.isolation_forest is None:
            raise RuntimeError("Nothing to save - train the model first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save isolation forest (always present)
        joblib.dump(self.isolation_forest, output_path / 'ensemble_isolation_forest.pkl')
        logger.info("✅ Saved ensemble_isolation_forest.pkl")

        # Save scaler
        joblib.dump(self.scaler, output_path / 'ensemble_scaler.pkl')
        logger.info("✅ Saved ensemble_scaler.pkl")

        # Save supervised models only if they exist
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, output_path / 'ensemble_xgboost.pkl')
            logger.info("✅ Saved ensemble_xgboost.pkl")
        else:
            logger.info("ℹ️  XGBoost not trained - ensemble_xgboost.pkl not saved")

        if self.rf_model is not None:
            joblib.dump(self.rf_model, output_path / 'ensemble_random_forest.pkl')
            logger.info("✅ Saved ensemble_random_forest.pkl")
        else:
            logger.info("ℹ️  RandomForest not trained - ensemble_random_forest.pkl not saved")

        # Save feature names if available
        if self.feature_names:
            with open(output_path / 'feature_names.txt', 'w') as f:
                f.write('\n'.join(self.feature_names))
            logger.info("✅ Saved feature_names.txt")

        # Save config - reflects which models are ACTUALLY saved
        config = {
            'weights': self.weights,
            'available_models': {
                'isolation_forest': True,
                'xgboost': self.xgb_model is not None,
                'random_forest': self.rf_model is not None,
            },
            'anomaly_threshold': self.anomaly_threshold,
            'metrics': self.metrics,
            'random_seed': self.random_seed,
            'note': 'All pkl files use joblib format. Load with joblib.load(), NOT pickle.load()'
        }
        with open(output_path / 'ensemble_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("✅ Saved ensemble_config.json")

        logger.info(f"\n✅ Ensemble saved to {output_path}")

    def load(self, model_dir: str):
        """Load ensemble models using joblib."""
        model_path = Path(model_dir)

        self.isolation_forest = joblib.load(model_path / 'ensemble_isolation_forest.pkl')
        self.scaler = joblib.load(model_path / 'ensemble_scaler.pkl')

        if (model_path / 'ensemble_xgboost.pkl').exists():
            self.xgb_model = joblib.load(model_path / 'ensemble_xgboost.pkl')
            logger.info("✅ Loaded XGBoost model")
        else:
            logger.info("ℹ️  No XGBoost model found - running without it")

        if (model_path / 'ensemble_random_forest.pkl').exists():
            self.rf_model = joblib.load(model_path / 'ensemble_random_forest.pkl')
            logger.info("✅ Loaded Random Forest model")
        else:
            logger.info("ℹ️  No Random Forest model found - running without it")

        # Load config for threshold and metadata
        config_path = model_path / 'ensemble_config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.anomaly_threshold = config.get('anomaly_threshold', -0.0)
            self.metrics = config.get('metrics', {})

        if (model_path / 'feature_names.txt').exists():
            with open(model_path / 'feature_names.txt') as f:
                self.feature_names = [line.strip() for line in f.readlines()]

        active = ['IsolationForest']
        if self.xgb_model: active.append('XGBoost')
        if self.rf_model: active.append('RandomForest')
        logger.info(f"✅ Ensemble loaded. Active models: {active}")
        logger.info(f"   Active weights: {self.weights}")


def train_ensemble_detector():
    """Train ensemble detector on ICS data."""
    print("=" * 80)
    print("ENSEMBLE ICS ANOMALY DETECTOR - TRAINING")
    print("=" * 80)

    current_dir = Path(__file__).parent
    data_path = current_dir.parent.parent / "data" / "processed" / "ics_features.csv"
    labels_path = current_dir.parent.parent / "data" / "processed" / "ics_labels.csv"

    if not data_path.exists():
        data_path = Path("./data/processed/ics_features.csv")
        labels_path = Path("./data/processed/ics_labels.csv")

    if not data_path.exists():
        print(f"❌ Features not found at {data_path}")
        return

    print(f"\n📥 Loading data from {data_path}...")
    features = pd.read_csv(data_path)
    labels = pd.read_csv(labels_path).values.ravel()

    # Validate labels before training
    unique = np.unique(labels)
    print(f"✅ Loaded {len(features)} samples, {len(features.columns)} features")
    print(f"   Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    if len(unique) < 2:
        print(
            "\n⚠️  WARNING: Labels contain only one class. Supervised models will be skipped.\n"
            "   Run ics_feature_engineer.py to regenerate correct labels before training."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        features.values, labels, test_size=0.2, random_state=42,
        stratify=labels if len(unique) >= 2 else None
    )

    print(f"\n📊 Data Split: Train={len(X_train):,} | Test={len(X_test):,}")

    ensemble = EnsembleICSDetector(random_seed=42)
    ensemble.feature_names = list(features.columns)
    ensemble.train(X_train, y_train)
    metrics = ensemble.evaluate(X_test, y_test)

    models_dir = Path("./models")
    if not models_dir.exists():
        models_dir = current_dir.parent.parent / "models"

    ensemble.save(str(models_dir))

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    return metrics


if __name__ == "__main__":
    train_ensemble_detector()
