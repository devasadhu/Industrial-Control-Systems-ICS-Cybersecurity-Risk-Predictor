"""
SHAP Explainability for ICS Anomaly Detection
Explains WHY network flows are flagged as anomalous

Author: Sadhana Devarajan
Version: 1.0.0
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICSExplainer:
    """SHAP-based explainability for ICS anomaly detection."""
    
    def __init__(self, model_path: str = None, feature_names_path: str = None):
        """Initialize explainer with trained model."""
        self.model = None
        self.explainer = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path, feature_names_path)
    
    def load_model(self, model_path: str, feature_names_path: str):
        """Load trained model."""
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        # Load feature names
        if feature_names_path and Path(feature_names_path).exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
        else:
            # Default feature names
            self.feature_names = [f"feature_{i}" for i in range(52)]
        
        logger.info("✅ Model loaded successfully")
    
    def create_explainer(self, background_data: np.ndarray):
        """Create SHAP explainer with background data."""
        logger.info("Creating SHAP explainer (this may take a minute)...")
        
        # Sample background data (100 samples is enough)
        if len(background_data) > 100:
            indices = np.random.choice(len(background_data), 100, replace=False)
            background_sample = background_data[indices]
        else:
            background_sample = background_data
        
        # Create predictor function
        def predict_fn(X):
            """Wrapper for Isolation Forest predictions."""
            return self.model.score_samples(X)
        
        # Use TreeExplainer (faster for tree-based models)
        try:
            self.explainer = shap.TreeExplainer(self.model, background_sample)
            logger.info("✅ Using TreeExplainer (fast)")
        except:
            # Fallback to KernelExplainer
            self.explainer = shap.KernelExplainer(predict_fn, background_sample)
            logger.info("✅ Using KernelExplainer")
    
    def explain_prediction(self, flow_features: np.ndarray, top_n: int = 10) -> Dict:
        """Explain why a specific flow was flagged."""
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call create_explainer() first.")
        
        # Get SHAP values
        if hasattr(self.explainer, 'shap_values'):
            shap_values = self.explainer.shap_values(flow_features.reshape(1, -1))
        else:
            shap_values = self.explainer(flow_features.reshape(1, -1)).values
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Get prediction
        prediction_score = self.model.score_samples(flow_features.reshape(1, -1))[0]
        prediction = self.model.predict(flow_features.reshape(1, -1))[0]
        
        # Sort features by importance
        shap_abs = np.abs(shap_values)
        top_indices = np.argsort(shap_abs)[-top_n:][::-1]
        
        # Build explanation
        explanation = {
            'prediction': 'ANOMALY' if prediction == -1 else 'NORMAL',
            'anomaly_score': float(prediction_score),
            'is_anomalous': prediction == -1,
            'top_features': []
        }
        
        for idx in top_indices:
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            explanation['top_features'].append({
                'name': feature_name,
                'value': float(flow_features[idx]),
                'shap_value': float(shap_values[idx]),
                'contribution_pct': float(shap_abs[idx] / shap_abs.sum() * 100),
                'impact': 'increases anomaly' if shap_values[idx] < 0 else 'decreases anomaly'
            })
        
        return explanation
    
    def print_explanation(self, explanation: Dict):
        """Print human-readable explanation."""
        result = explanation['prediction']
        score = explanation['anomaly_score']
        
        print("\n" + "="*70)
        print(f"🔍 DETECTION RESULT: {result}")
        print(f"📊 Anomaly Score: {score:.4f}")
        print("="*70)
        
        if explanation['is_anomalous']:
            print("\n⚠️  WHY THIS WAS FLAGGED AS ANOMALOUS:\n")
        else:
            print("\n✅ WHY THIS WAS CLASSIFIED AS NORMAL:\n")
        
        for i, feature in enumerate(explanation['top_features'], 1):
            print(f"{i}. {feature['name']}")
            print(f"   Value: {feature['value']:.4f}")
            print(f"   Impact: {feature['impact']}")
            print(f"   Contribution: {feature['contribution_pct']:.1f}%")
            print()


def demo_explainer():
    """Demo the SHAP explainer."""
    print("\n" + "="*70)
    print("ICS ANOMALY DETECTION - SHAP EXPLAINABILITY DEMO")
    print("="*70)
    
    # Find model directory
    models_dir = Path("./models")
    if not models_dir.exists():
        models_dir = Path("../../models")
    
    model_path = models_dir / "isolation_forest_ics_detector.pkl"
    feature_names_path = models_dir / "feature_names.txt"
    
    if not model_path.exists():
        print(f"\n❌ Model not found at {model_path}")
        print("Run `python quick_start.py` first to train the model")
        return
    
    # Load explainer
    print(f"\n📦 Loading model from {models_dir}...")
    explainer = ICSExplainer(
        model_path=str(model_path),
        feature_names_path=str(feature_names_path)
    )
    
    # Find data directory
    data_path = Path("./data/processed/ics_features.csv")
    if not data_path.exists():
        data_path = Path("../../data/processed/ics_features.csv")
    
    if not data_path.exists():
        print(f"\n❌ Data not found at {data_path}")
        return
    
    print(f"📦 Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"✅ Loaded {len(data)} samples")
    
    # Create explainer with background data
    print("\n🔧 Creating SHAP explainer...")
    background = data.sample(min(100, len(data))).values
    explainer.create_explainer(background)
    
    # Explain a few random samples
    print("\n" + "="*70)
    print("ANALYZING SAMPLE FLOWS")
    print("="*70)
    
    num_samples = 3
    for i in range(num_samples):
        sample_idx = np.random.randint(0, len(data))
        sample = data.iloc[sample_idx].values
        
        print(f"\n{'='*70}")
        print(f"FLOW #{sample_idx}")
        print("="*70)
        
        # Get explanation
        explanation = explainer.explain_prediction(sample, top_n=5)
        explainer.print_explanation(explanation)
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\nKey Insights:")
    print("• High packet rates often indicate scanning attacks")
    print("• Unusual port numbers (502, 20000) suggest ICS protocols")
    print("• Abnormal flow durations can reveal persistence attacks")
    print("• PSH/URG flags help identify command injection")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_explainer()