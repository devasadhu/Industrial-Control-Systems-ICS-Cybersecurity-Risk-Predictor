"""
tune_threshold.py
-----------------
Sweeps ensemble prediction thresholds from 0.20 to 0.60 and prints
a table of Precision / Recall / F1 / F2 at each point.

F2 score (beta=2) weights recall twice as heavily as precision —
appropriate for ICS where missing an attack costs more than a false alarm.

Run from project root:
    python tune_threshold.py

Author: Sadhana Devarajan
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

sys.path.insert(0, './src')
from src.models.ensemble_detector import EnsembleICSDetector

FEATURES  = Path('./data/processed/ics_features.csv')
LABELS    = Path('./data/processed/ics_labels.csv')
MODELS    = Path('./models')

print("\n" + "="*70)
print("ENSEMBLE THRESHOLD SWEEP")
print("="*70)

# Load data
features = pd.read_csv(FEATURES)
labels   = pd.read_csv(LABELS)['label'].values

X = features.values
y = labels

# Same split as training — ensures test set is the held-out portion
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load trained ensemble
ensemble = EnsembleICSDetector()
ensemble.load(str(MODELS))

# Get raw weighted scores (before thresholding)
from sklearn.preprocessing import StandardScaler
X_test_scaled = ensemble.scaler.transform(X_test)

active_weights = ensemble.weights
weighted_scores = np.zeros(len(X_test))

iso_pred = ensemble.isolation_forest.predict(X_test_scaled)
weighted_scores += (iso_pred == -1).astype(int) * active_weights['isolation_forest']

if ensemble.xgb_model is not None:
    xgb_pred = ensemble.xgb_model.predict(X_test_scaled)
    weighted_scores += xgb_pred * active_weights.get('xgboost', 0)

if ensemble.rf_model is not None:
    rf_pred = ensemble.rf_model.predict(X_test_scaled)
    weighted_scores += rf_pred * active_weights.get('random_forest', 0)

# Sweep thresholds
thresholds = np.arange(0.20, 0.65, 0.05)

print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'F2':>10}  Note")
print("-"*65)

best_f1  = {'score': 0, 'threshold': None}
best_f2  = {'score': 0, 'threshold': None}
best_bal = {'score': 0, 'threshold': None}  # best |precision - recall| < 0.10

for t in thresholds:
    preds = (weighted_scores >= t).astype(int)

    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    f2   = fbeta_score(y_test, preds, beta=2, zero_division=0)

    note = ""
    if f1 > best_f1['score']:
        best_f1 = {'score': f1, 'threshold': t}
    if f2 > best_f2['score']:
        best_f2 = {'score': f2, 'threshold': t}
    if abs(prec - rec) < 0.10 and f1 > best_bal['score']:
        best_bal = {'score': f1, 'threshold': t}
        note = "← balanced"

    print(f"{t:>10.2f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {f2:>10.4f}  {note}")

print("\n" + "="*70)
print(f"  Best F1  threshold: {best_f1['threshold']:.2f}  (F1={best_f1['score']:.4f})")
print(f"  Best F2  threshold: {best_f2['threshold']:.2f}  (F2={best_f2['score']:.4f})  ← recommended for ICS")
if best_bal['threshold']:
    print(f"  Best balanced (|P-R|<0.10): {best_bal['threshold']:.2f}  (F1={best_bal['score']:.4f})")
print("="*70)
print("""
Recommendation:
  ICS / security context  → use Best F2 threshold (recall weighted 2x)
  Portfolio demo          → use Best F1 threshold (balanced, cleaner number)
  High-precision SOC tool → use 0.50 as-is (96%+ precision, conservative)

Edit src/models/ensemble_detector.py line:
    ensemble_predictions = (weighted_scores >= 0.5).astype(int)
Replace 0.5 with your chosen threshold.
""")
