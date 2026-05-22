"""
fix_and_retrain.py
------------------
Step 1: Regenerate ics_labels.csv from raw dataset (fixes all-zeros bug).
Step 2: Train the full supervised ensemble (IF + XGBoost + RF).
Step 3: Save all models and update model_metadata.json with real metrics.

Run from project root:
    python fix_and_retrain.py

Author: Sadhana Devarajan
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DATA   = Path('./data/raw/kaggle/icssim/Dataset.csv')
FEATURES   = Path('./data/processed/ics_features.csv')
LABELS_OUT = Path('./data/processed/ics_labels.csv')
MODELS_DIR = Path('./models')

sys.path.insert(0, './src')


# ── Step 1: Fix labels ─────────────────────────────────────────────────────────
def fix_labels() -> pd.Series:
    """
    Re-extract IT_B_Label from raw dataset and overwrite ics_labels.csv.

    Root cause of the all-zeros bug: quick_start.py calls train_models_directly()
    in unsupervised mode (single-class path), which re-saves models/feature_scaler.pkl
    but never touches labels. However a prior run of engineer_features() loaded the
    raw df correctly yet the label Series it returned had dtype object '0'/'1' strings
    on some pandas versions — astype(int) on '0'/'1' strings works, but if IT_B_Label
    was read as float then fillna(0) before astype(int) collapses NaN rows to 0 and
    masks attacks. The safe fix: read the raw CSV fresh and re-derive labels directly.
    """
    print("\n" + "="*70)
    print("STEP 1 — Fix ics_labels.csv")
    print("="*70)

    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA}")

    logger.info(f"Reading raw dataset from {RAW_DATA} ...")
    df = pd.read_csv(RAW_DATA, low_memory=False)

    if 'IT_B_Label' not in df.columns:
        raise ValueError(
            "IT_B_Label column missing from raw dataset. "
            "Check column names with: python -c \"import pandas as pd; "
            "print(pd.read_csv('data/raw/kaggle/icssim/Dataset.csv', nrows=2).columns.tolist())\""
        )

    # Cast carefully: handle both numeric and string representations
    raw_col = df['IT_B_Label']
    if raw_col.dtype == object:
        # String values: 'Normal' -> 0, anything else -> 1
        labels = (raw_col.str.strip().str.lower() != 'normal').astype(int)
    else:
        # Numeric 0/1 — convert via float first to handle NaN safely, then int
        labels = raw_col.fillna(0).astype(float).astype(int)

    dist = labels.value_counts().sort_index()
    print(f"✅ Label distribution in raw data:")
    print(f"   Normal (0): {dist.get(0, 0):,}")
    print(f"   Attack (1): {dist.get(1, 0):,}")

    if dist.get(1, 0) == 0:
        raise ValueError(
            "Still no attack samples after re-extraction. "
            "Print raw IT_B_Label unique values: "
            "df['IT_B_Label'].unique()"
        )

    # Save — single column 'label', no index
    labels.to_csv(LABELS_OUT, index=False, header=['label'])
    print(f"✅ Saved corrected labels → {LABELS_OUT}")
    return labels


# ── Step 2 + 3: Train full supervised ensemble ─────────────────────────────────
def train_supervised_ensemble(labels: pd.Series):
    print("\n" + "="*70)
    print("STEP 2 — Train Supervised Ensemble (IF + XGBoost + RF)")
    print("="*70)

    if not FEATURES.exists():
        raise FileNotFoundError(
            f"Features file not found: {FEATURES}\n"
            "Run quick_start.py first to generate ics_features.csv, then re-run this script."
        )

    logger.info(f"Loading features from {FEATURES} ...")
    features = pd.read_csv(FEATURES)

    # Sanity check alignment
    if len(features) != len(labels):
        raise ValueError(
            f"Row mismatch: features has {len(features)} rows, "
            f"labels has {len(labels)} rows. "
            "Re-run feature engineering (python src/ics_feature_engineer.py) "
            "then re-run this script."
        )

    X = features.values
    y = labels.values

    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Dataset: {len(X):,} samples | {X.shape[1]} features")
    print(f"   Class distribution: { {int(k): int(v) for k, v in zip(unique, counts)} }")

    # Stratified split — preserves class ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n   Train: {len(X_train):,} | Test: {len(X_test):,}")

    from src.models.ensemble_detector import EnsembleICSDetector

    ensemble = EnsembleICSDetector(random_seed=42)
    ensemble.feature_names = list(features.columns)

    print("\n🚀 Training ensemble ...")
    ensemble.train(X_train, y_train)   # passes y_train → triggers XGB + RF

    print("\n📊 Evaluating on held-out test set ...")
    metrics = ensemble.evaluate(X_test, y_test)

    # Detailed per-class report
    ensemble_pred, _ = ensemble.predict(X_test)
    print("\n" + classification_report(
        y_test, ensemble_pred, target_names=['Normal', 'Attack'], digits=4
    ))

    # Save all models
    MODELS_DIR.mkdir(exist_ok=True)
    ensemble.save(str(MODELS_DIR))

    # Update model_metadata.json with real supervised metrics
    _update_metadata(metrics, features.columns.tolist(), len(X_train), len(X_test))

    return metrics


def _update_metadata(metrics: dict, feature_names: list, n_train: int, n_test: int):
    meta_path = MODELS_DIR / 'model_metadata.json'

    # Load existing metadata if present, else start fresh
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    meta.update({
        'model_type': 'EnsembleICSDetector',
        'mode': metrics.get('mode', 'supervised'),
        'n_features': len(feature_names),
        'n_train_samples': n_train,
        'n_test_samples': n_test,
        'performance': {
            'accuracy':  round(metrics.get('accuracy', 0), 6),
            'precision': round(metrics.get('precision', 0), 6),
            'recall':    round(metrics.get('recall', 0), 6),
            'f1_score':  round(metrics.get('f1_score', 0), 6),
        },
        'individual_model_accuracy': {
            'isolation_forest': round(metrics.get('iso_forest_accuracy', 0), 6),
            'xgboost':          round(metrics.get('xgb_accuracy', 0), 6),
            'random_forest':    round(metrics.get('rf_accuracy', 0), 6),
        },
        'contamination': 0.1,
        'note': (
            'Supervised ensemble trained on ICSSIM dataset. '
            'Models saved via joblib — load with joblib.load(), not pickle.load(). '
            'Use ensemble_isolation_forest.pkl + ensemble_scaler.pkl for inference. '
            'XGBoost and RF available at ensemble_xgboost.pkl / ensemble_random_forest.pkl.'
        )
    })

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n✅ Updated model_metadata.json → {meta_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*70)
    print("ICS ENSEMBLE FIX + RETRAIN")
    print("="*70)

    try:
        labels = fix_labels()
        metrics = train_supervised_ensemble(labels)

        print("\n" + "="*70)
        print("✅ DONE")
        print("="*70)

        if metrics.get('mode') == 'supervised':
            print(f"\n📊 Final Ensemble Metrics (test set):")
            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1-Score:  {metrics['f1_score']:.4f}")

        print(f"\n📁 Artifacts saved to {MODELS_DIR}/:")
        print(f"   ensemble_isolation_forest.pkl")
        print(f"   ensemble_xgboost.pkl")
        print(f"   ensemble_random_forest.pkl")
        print(f"   ensemble_scaler.pkl")
        print(f"   ensemble_config.json")
        print(f"   model_metadata.json  ← updated with real F1/precision/recall")
        print(f"\n   data/processed/ics_labels.csv  ← corrected (was all-zeros)")

    except Exception as e:
        logger.error(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)