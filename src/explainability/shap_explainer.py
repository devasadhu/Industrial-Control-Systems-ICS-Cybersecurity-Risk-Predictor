"""
SHAP Explainability for ICS Anomaly Detection
Explains WHY network flows are flagged as anomalous

Author: Sadhana Devarajan
Version: 1.4.0

Fixes applied (v1.4.0):
- FIX 1: Inverted SHAP impact labels corrected.
         For IsolationForest.score_samples(), more negative = more anomalous.
         A negative SHAP value DECREASES the score (pushes toward anomaly).
         A positive SHAP value INCREASES the score (pushes toward normal).
         Labels were swapped — now corrected.

- FIX 2: Background data (background_sample) is now actually passed to
         TreeExplainer via the `data` parameter, resolving the FutureWarning:
         "passing feature_perturbation='interventional' without providing a
          background dataset will raise an error."

- FIX 3: sklearn version mismatch warning documented. Upgrade sklearn to
         match the version used when the model was pickled (1.8.0):
             pip install scikit-learn==1.8.0
         Alternatively, retrain the model with your currently installed version.

- FIX 4: top_n default in explain_prediction() aligned with demo call (both now 5).
         Change to 10 in both places if you want more features shown.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICSExplainer:
    """SHAP-based explainability for ICS anomaly detection."""

    def __init__(self, model_path: str = None, feature_names_path: str = None,
                 scaler_path: str = None):
        self.model = None
        self.explainer = None
        self.feature_names = []
        self.scaler = None

        if model_path:
            self.load_model(model_path, feature_names_path, scaler_path)

    def load_model(self, model_path: str, feature_names_path: str = None,
                   scaler_path: str = None):
        """
        Load trained model, feature names, and scaler.

        Args:
            model_path: Path to ensemble_isolation_forest.pkl (joblib format).
                        NOTE: Use ensemble_isolation_forest.pkl, NOT
                        isolation_forest_ics_detector.pkl — the ensemble model
                        is trained on the same 51 processed features as the
                        feature engineer output, avoiding feature count mismatches.
            feature_names_path: Path to feature_names.txt
            scaler_path: Path to ensemble_scaler.pkl (StandardScaler, joblib format).
                         NOTE: Do NOT pass feature_scaler.pkl — that file contains
                         feature names only, not a StandardScaler.
        """
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)

        if feature_names_path and Path(feature_names_path).exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f if line.strip()]
        else:
            self.feature_names = [f"feature_{i}" for i in range(51)]
            logger.warning(
                "feature_names.txt not found — using 51 generic feature names. "
                "Run quick_start.py to generate feature_names.txt."
            )

        if scaler_path and Path(scaler_path).exists():
            loaded = joblib.load(scaler_path)
            if hasattr(loaded, 'transform'):
                self.scaler = loaded
                logger.info("✅ Scaler loaded")
            else:
                logger.warning(
                    f"⚠️  {scaler_path} does not appear to be a StandardScaler "
                    "(it may be feature_scaler.pkl which contains feature names only). "
                    "SHAP will run without pre-scaling."
                )

        logger.info(f"✅ Model loaded ({len(self.feature_names)} features)")

    def create_explainer(self, background_data: np.ndarray):
        """Create SHAP explainer with background data."""
        logger.info("Creating SHAP explainer (this may take a minute)...")

        if self.scaler is not None:
            background_data = self.scaler.transform(background_data)

        if len(background_data) > 100:
            indices = np.random.choice(len(background_data), 100, replace=False)
            background_sample = background_data[indices]
        else:
            background_sample = background_data

        def predict_fn(X):
            return self.model.score_samples(X)

        # FIX 2: Pass background_sample via the `data` parameter so that
        # TreeExplainer can use the interventional approach correctly.
        # Previously, background_sample was computed but never passed in,
        # triggering a FutureWarning and producing less accurate SHAP values.
        try:
            self.explainer = shap.TreeExplainer(
                self.model,
                data=background_sample,           # <-- FIX 2: was missing entirely
                feature_perturbation="interventional"
            )
            logger.info("✅ Using TreeExplainer (fast)")
        except Exception as e:
            logger.warning(f"TreeExplainer failed ({e}), falling back to KernelExplainer")
            self.explainer = shap.KernelExplainer(predict_fn, background_sample)
            logger.info("✅ Using KernelExplainer")

    def explain_prediction(self, flow_features: np.ndarray, top_n: int = 5) -> Dict:
        # FIX 4: top_n default changed from 10 → 5 to match the demo call below.
        """Explain why a specific flow was flagged."""
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call create_explainer() first.")

        features_to_explain = flow_features.copy()
        features_for_model = (
            self.scaler.transform([features_to_explain])[0]
            if self.scaler is not None
            else features_to_explain
        )

        # Compute SHAP values
        try:
            shap_values = self.explainer.shap_values(
                features_for_model.reshape(1, -1),
                check_additivity=False
            )
        except Exception:
            shap_values = self.explainer(
                features_for_model.reshape(1, -1),
                check_additivity=False
            ).values

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]

        prediction_score = self.model.score_samples(features_for_model.reshape(1, -1))[0]
        prediction = self.model.predict(features_for_model.reshape(1, -1))[0]

        shap_abs = np.abs(shap_values)
        top_indices = np.argsort(shap_abs)[-top_n:][::-1]

        explanation = {
            'prediction': 'ANOMALY' if prediction == -1 else 'NORMAL',
            'anomaly_score': float(prediction_score),
            'is_anomalous': prediction == -1,
            'top_features': []
        }

        total_shap = shap_abs.sum()
        for idx in top_indices:
            feat_name = (self.feature_names[idx]
                         if idx < len(self.feature_names) else f"feature_{idx}")

            # FIX 1: Impact labels were INVERTED in v1.3.0.
            #
            # IsolationForest.score_samples() returns the negative average depth.
            # More negative score  →  more anomalous.
            #
            # SHAP values here represent each feature's additive contribution
            # to score_samples() output:
            #   shap_value < 0  →  feature DECREASES the score
            #                      (pulls toward more-negative / more anomalous)
            #   shap_value > 0  →  feature INCREASES the score
            #                      (pulls toward less-negative / more normal)
            #
            # The old code had these labels reversed. Fixed below.
            if shap_values[idx] < 0:
                impact_label = "decreases anomaly score (pushes toward anomaly)"
            else:
                impact_label = "increases anomaly score (pushes toward normal)"

            explanation['top_features'].append({
                'name': feat_name,
                'value': float(flow_features[idx]),
                'shap_value': float(shap_values[idx]),
                'contribution_pct': float(shap_abs[idx] / max(total_shap, 1e-10) * 100),
                'impact': impact_label,
            })

        return explanation

    def print_explanation(self, explanation: Dict):
        """Print human-readable explanation."""
        result = explanation['prediction']
        score = explanation['anomaly_score']

        print("\n" + "=" * 70)
        print(f"🔍 DETECTION RESULT: {result}")
        print(f"📊 Anomaly Score: {score:.4f}")
        print("=" * 70)

        if explanation['is_anomalous']:
            print("\n⚠️  WHY THIS WAS FLAGGED AS ANOMALOUS:\n")
        else:
            print("\n✅ WHY THIS WAS CLASSIFIED AS NORMAL:\n")

        for i, feature in enumerate(explanation['top_features'], 1):
            print(f"{i}. {feature['name']}")
            print(f"   Value:        {feature['value']:.4f}")
            print(f"   SHAP value:   {feature['shap_value']:+.4f}")
            print(f"   Impact:       {feature['impact']}")
            print(f"   Contribution: {feature['contribution_pct']:.1f}%")
            print()


def demo_explainer():
    """Demo the SHAP explainer."""
    print("\n" + "=" * 70)
    print("ICS ANOMALY DETECTION - SHAP EXPLAINABILITY DEMO")
    print("=" * 70)

    # NOTE (FIX 3): If you see InconsistentVersionWarning for sklearn,
    # your installed sklearn version differs from the one used to train/pickle
    # the model. To resolve, run:
    #     pip install scikit-learn==1.8.0
    # or retrain the model using your currently installed version.

    models_dir = Path("./models")
    if not models_dir.exists():
        models_dir = Path("../../models")

    model_path = models_dir / "ensemble_isolation_forest.pkl"
    feature_names_path = models_dir / "feature_names.txt"
    scaler_path = models_dir / "ensemble_scaler.pkl"

    if not model_path.exists():
        print(f"\n❌ Model not found at {model_path}")
        print("Run `python quick_start.py` first to train the ensemble model")
        return

    print(f"\n📄 Loading model from {models_dir}...")
    explainer = ICSExplainer(
        model_path=str(model_path),
        feature_names_path=str(feature_names_path),
        scaler_path=str(scaler_path) if scaler_path.exists() else None,
    )

    data_path = Path("./data/processed/ics_features.csv")
    if not data_path.exists():
        data_path = Path("../../data/processed/ics_features.csv")

    if not data_path.exists():
        print(f"\n❌ Data not found at {data_path}")
        return

    print(f"📄 Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"✅ Loaded {len(data)} samples")

    print("\n⚙️ Creating SHAP explainer...")
    background = data.sample(min(100, len(data))).values
    explainer.create_explainer(background)

    print("\n" + "=" * 70)
    print("ANALYZING SAMPLE FLOWS")
    print("=" * 70)

    for i in range(3):
        sample_idx = np.random.randint(0, len(data))
        sample = data.iloc[sample_idx].values

        print(f"\n{'=' * 70}")
        print(f"FLOW #{sample_idx}")
        print("=" * 70)

        # FIX 4: top_n=5 here now matches the method's default of 5.
        explanation = explainer.explain_prediction(sample, top_n=5)
        explainer.print_explanation(explanation)

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Insights:")
    print("• High packet rates often indicate scanning attacks")
    print("• Unusual port numbers (502, 20000) suggest ICS protocols")
    print("• Abnormal flow durations can reveal persistence attacks")
    print("• PSH/URG flags help identify command injection")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    demo_explainer()