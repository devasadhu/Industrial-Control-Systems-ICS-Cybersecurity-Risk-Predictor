"""
session_feature_experiment.py
------------------------------
Ablation experiment: does adding session-level features improve the ensemble?

Reads:
  data/processed/ics_features_v2.csv   ← pre-session features (now written by ICSFeatureEngineer)
  data/processed/ics_labels.csv
  data/raw/kaggle/icssim/Dataset.csv   ← raw data for per-attack breakdown + session computation

Fixes applied in this rewrite:
  - [BUG FIX #2] Reads ics_features_v2.csv instead of a file that no longer exists.
    ICSFeatureEngineer.save_features() now writes both v2 (no session) and v3 (full).
  - [BUG FIX #4] model_metadata.json is written unconditionally (not gated on recall improvement).
    If recall does NOT improve we still persist metadata with a flag so the decision is auditable.
  - Import path cleaned up: compute_session_features is sourced from ics_feature_engineer
    directly so this script works even if src/features/session_features.py doesn't exist.

NOTE
----
Core session feature logic lives in ICSFeatureEngineer.create_session_features().
This script only calls it to build the v2→v3 delta for ablation comparison.
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA    = Path("./data/raw/kaggle/icssim/Dataset.csv")
FEATURES_V2 = Path("./data/processed/ics_features_v2.csv")   # ← fixed (was ics_features_v2.csv but file wasn't written)
LABELS      = Path("./data/processed/ics_labels.csv")
MODELS_DIR  = Path("./models")
FEAT_OUT    = Path("./data/processed/ics_features_v3.csv")

# Window size used in session feature computation
WINDOW_SECONDS = 60


def _compute_session_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Thin wrapper: instantiate ICSFeatureEngineer and call create_session_features().
    This avoids duplicating logic and removes the dependency on
    src/features/session_features.py which may not exist.
    """
    import sys
    from pathlib import Path as _Path
    # Insert the project root so `from src.X import ...` resolves correctly.
    _root = str(_Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from src.ics_feature_engineer import ICSFeatureEngineer

    eng = ICSFeatureEngineer(random_seed=42)
    return eng.create_session_features(raw)


# =============================================================================
# TRAIN + EVALUATE
# =============================================================================

def train_and_evaluate(
    features: pd.DataFrame,
    labels: np.ndarray,
    raw: pd.DataFrame,
    label: str,
    threshold: float = 0.25,
):
    import sys
    from pathlib import Path as _Path
    _root = str(_Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from src.models.ensemble_detector import EnsembleICSDetector

    X = features.values
    y = labels

    X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
        X, y, raw.index, test_size=0.2, random_state=42, stratify=y
    )

    ensemble = EnsembleICSDetector(random_seed=42)
    ensemble.feature_names = list(features.columns)
    ensemble.train(X_train, y_train)

    Xs      = ensemble.scaler.transform(X_test)
    weights = ensemble.weights
    scores  = np.zeros(len(X_test))

    scores += (ensemble.isolation_forest.predict(Xs) == -1).astype(int) * weights["isolation_forest"]
    if ensemble.xgb_model:
        scores += ensemble.xgb_model.predict(Xs) * weights.get("xgboost", 0)
    if ensemble.rf_model:
        scores += ensemble.rf_model.predict(Xs) * weights.get("random_forest", 0)

    preds = (scores >= threshold).astype(int)

    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"  {label}  (threshold={threshold})")
    print(f"{'=' * 60}")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

    if "IT_M_Label" in raw.columns:
        attack_labels = raw.loc[idx_test, "IT_M_Label"].values
        attack_mask   = y_test == 1

        print(f"\n  Per-attack recall:")
        for attack_type in ["replay", "ddos", "port-scan", "ip-scan", "mitm"]:
            mask = attack_mask & (attack_labels == attack_type)
            if mask.sum() == 0:
                continue
            caught = preds[mask].sum()
            total  = mask.sum()
            bar    = "█" * int(caught / total * 20)
            print(f"    {attack_type:<12} {caught:>4}/{total:<4} ({caught / total:>5.1%})  {bar}")

    return ensemble, {"precision": prec, "recall": rec, "f1": f1}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SESSION ABLATION — BASELINE vs V3 FEATURES")
    print("=" * 60)

    # ── Validate inputs ───────────────────────────────────────────────────────
    for path in (RAW_DATA, FEATURES_V2, LABELS):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path}\n"
                "Run quick_start.py (or ics_feature_engineer.py) first to generate "
                "ics_features_v2.csv and ics_labels.csv."
            )

    print("\n📥 Loading data...")
    raw         = pd.read_csv(RAW_DATA, low_memory=False)
    features_v2 = pd.read_csv(FEATURES_V2)
    labels      = pd.read_csv(LABELS)["label"].values

    print(f"   Features v2 : {features_v2.shape[1]} columns (no session)")
    print(f"   Flows       : {len(raw):,}")

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n📊 BASELINE (v2 features — no session):")
    _, baseline = train_and_evaluate(features_v2, labels, raw, "BASELINE", threshold=0.25)

    # ── Session Features ──────────────────────────────────────────────────────
    print(f"\n⚙️  Computing session-level features (window={WINDOW_SECONDS}s)...")
    session_feats = _compute_session_features(raw)

    print(f"\n   Session features : {session_feats.shape[1]} columns")
    print(f"   {list(session_feats.columns)}")

    # ── Distribution sanity check ─────────────────────────────────────────────
    print(f"\n📊 Feature distributions (normal vs attack):")
    attack_mask = labels == 1

    for col in session_feats.columns:
        normal_mean = session_feats.loc[~attack_mask, col].mean()
        attack_mean = session_feats.loc[attack_mask, col].mean()
        ratio = attack_mean / (normal_mean + 1e-6)
        indicator = "← discriminative" if abs(ratio - 1) > 0.3 else ""
        print(f"   {col:<30} normal={normal_mean:.3f}  attack={attack_mean:.3f}  ratio={ratio:.2f} {indicator}")

    # ── Merge v2 + session ────────────────────────────────────────────────────
    features_v3 = pd.concat(
        [features_v2.reset_index(drop=True), session_feats.reset_index(drop=True)],
        axis=1,
    )
    features_v3 = features_v3.fillna(0).replace([np.inf, -np.inf], 0)
    print(f"\n   Total features v3 : {features_v3.shape[1]}")

    # ── Retrain ───────────────────────────────────────────────────────────────
    print("\n🚀 RETRAIN (with session features):")
    ensemble_v3, v3_metrics = train_and_evaluate(
        features_v3, labels, raw,
        f"V3 — {features_v3.shape[1]} features",
        threshold=0.25,
    )

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("IMPROVEMENT OVER BASELINE")
    print("=" * 60)

    for m in ["precision", "recall", "f1"]:
        delta = v3_metrics[m] - baseline[m]
        sign  = "+" if delta >= 0 else ""
        print(f"{m:<10}: {baseline[m]:.4f} → {v3_metrics[m]:.4f} ({sign}{delta:.4f})")

    improved = v3_metrics["recall"] > baseline["recall"]

    # ── Persist results (unconditionally) ────────────────────────────────────
    # BUG FIX #4: metadata is now always written so every run is auditable,
    # not just runs where recall improves.
    MODELS_DIR.mkdir(exist_ok=True)

    meta_path = MODELS_DIR / "model_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    meta.update(
        {
            "ablation_experiment": {
                "session_window_seconds": WINDOW_SECONDS,
                "session_features": list(session_feats.columns),
                "baseline_metrics": baseline,
                "v3_metrics": v3_metrics,
                "recall_improved": improved,
                "saved": improved,
            }
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n📄 Ablation results appended to {meta_path}")

    if improved:
        print("\n✅ Recall improved — saving v3 features + model...")

        features_v3.to_csv(FEAT_OUT, index=False)
        print(f"   ✅ Saved features : {FEAT_OUT}")

        with open(MODELS_DIR / "feature_names.txt", "w") as f:
            f.write("\n".join(features_v3.columns))
        print(f"   ✅ Saved feature names")

        ensemble_v3.save(str(MODELS_DIR))
        print(f"   ✅ Saved model")

    else:
        print(
            "\n⚠️  Recall did NOT improve — v3 features and model not saved.\n"
            "    Ablation results written to model_metadata.json for reference."
        )
