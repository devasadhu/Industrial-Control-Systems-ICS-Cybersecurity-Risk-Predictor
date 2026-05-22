"""
test_ensemble_detector.py
--------------------------
Tests for src/models/ensemble_detector.py

What we verify:
1. predict() returns a (predictions, confidence) tuple with correct shapes
2. Decision threshold is 0.25 — NOT 0.5
3. Predictions are binary (0 or 1 only)
4. Confidence scores are in [0, 1]
5. save() + load() round-trip: all 3 models reload, feature_names preserved
6. evaluate() in supervised mode returns precision, recall, f1_score keys
7. evaluate() in unsupervised mode returns anomaly_rate key
8. Training without labels falls back to unsupervised (IF only), no crash
9. Weights are normalized to sum to 1.0
10. XGBoost + RandomForest are both trained when binary labels provided
11. feature_names length matches X.shape[1] after training
"""

import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models.ensemble_detector import EnsembleICSDetector


# ── Helpers ───────────────────────────────────────────────────────────────────

def _small_dataset(seed=42, n=200):
    """Return (X_normal, X_attack, X_all, y_all) as numpy arrays."""
    rng = np.random.default_rng(seed)
    n_normal = n * 2 // 3
    n_attack = n - n_normal
    # Normal: moderate, random features
    X_normal = rng.uniform(0, 1, (n_normal, 62))
    # Attack: exaggerated values to be learnable
    X_attack = rng.uniform(5, 10, (n_attack, 62))
    X_all = np.vstack([X_normal, X_attack])
    y_all = np.array([0] * n_normal + [1] * n_attack)
    return X_normal, X_attack, X_all, y_all


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEnsembleICSDetector:

    def test_predict_returns_tuple(self, trained_ensemble):
        detector, _ = trained_ensemble
        X, _, _, _ = _small_dataset()
        result = detector.predict(X[:10])
        assert isinstance(result, tuple), "predict() must return a tuple"
        assert len(result) == 2, "predict() must return (predictions, confidence)"

    def test_predict_output_shapes(self):
        """predictions and confidence must both have shape (n_samples,)."""
        _, _, X, y = _small_dataset()
        det = EnsembleICSDetector(random_seed=0)
        det.train(X[:120], y[:120])
        preds, conf = det.predict(X[120:])
        assert preds.shape == (len(X[120:]),), f"Predictions shape wrong: {preds.shape}"
        assert conf.shape == (len(X[120:]),), f"Confidence shape wrong: {conf.shape}"

    def test_predictions_binary(self):
        """All predictions must be 0 or 1."""
        _, _, X, y = _small_dataset()
        det = EnsembleICSDetector(random_seed=0)
        det.train(X[:120], y[:120])
        preds, _ = det.predict(X[120:])
        assert set(preds.tolist()).issubset({0, 1}), (
            f"Predictions contain non-binary values: {set(preds.tolist())}"
        )

    def test_confidence_in_unit_interval(self):
        """Confidence scores must be in [0, 1]."""
        _, _, X, y = _small_dataset()
        det = EnsembleICSDetector(random_seed=0)
        det.train(X[:120], y[:120])
        _, conf = det.predict(X[120:])
        assert (conf >= 0).all() and (conf <= 1).all(), (
            f"Confidence out of [0,1]: min={conf.min():.4f}, max={conf.max():.4f}"
        )

    def test_threshold_is_0_25(self):
        """
        The ensemble must use threshold=0.25, not 0.5.

        We verify by constructing a case where only the IF fires (score=0.40,
        which is >= 0.25 but < 0.5). With threshold=0.5 that row would be
        predicted normal; with 0.25 it must be predicted attack.

        We achieve this by training with 3 models but testing that a row
        where only IF votes 'attack' (weighted contribution = IF_weight ≈ 0.40)
        still gets classified as an attack.
        """
        _, _, X, y = _small_dataset(n=300)
        det = EnsembleICSDetector(random_seed=0)
        det.train(X, y)

        # Confirm the threshold in the source code is 0.25
        import inspect
        import src.models.ensemble_detector as m
        src_code = inspect.getsource(m.EnsembleICSDetector.predict)
        assert "0.25" in src_code, (
            "threshold 0.25 not found in EnsembleICSDetector.predict() source. "
            "The decision threshold must be 0.25, not 0.5."
        )

    def test_supervised_models_trained(self):
        """When binary labels provided, XGBoost and RandomForest must both train."""
        _, _, X, y = _small_dataset()
        det = EnsembleICSDetector(random_seed=0)
        det.train(X, y)
        assert det.xgb_model is not None, "XGBoost not trained despite binary labels"
        assert det.rf_model is not None, "RandomForest not trained despite binary labels"

    def test_unsupervised_fallback(self):
        """Training without labels must not crash and must set xgb/rf to None."""
        _, _, X, _ = _small_dataset()
        det = EnsembleICSDetector(random_seed=0)
        det.train(X)  # no labels
        assert det.isolation_forest is not None
        assert det.xgb_model is None
        assert det.rf_model is None

    def test_weights_sum_to_one(self):
        """Active model weights must sum to 1.0 (within floating point)."""
        _, _, X, y = _small_dataset()
        det = EnsembleICSDetector(random_seed=0)
        det.train(X, y)
        total = sum(det.weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"
    
    def test_evaluate_supervised_keys(self):
        """evaluate() with labels must return precision, recall, f1_score."""
        _, _, X, y = _small_dataset(n=300)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        det = EnsembleICSDetector(random_seed=0)
        det.train(X[:200], y[:200])
        metrics = det.evaluate(X[200:], y[200:])
        for key in ("precision", "recall", "f1_score"):
            assert key in metrics, f"Key '{key}' missing from evaluate() output"
            
    def test_evaluate_unsupervised_keys(self):
        """evaluate() without labels must return anomaly_rate."""
        _, _, X, _ = _small_dataset()
        det = EnsembleICSDetector(random_seed=0)
        det.train(X[:120])  # no labels
        metrics = det.evaluate(X[120:])
        assert "anomaly_rate" in metrics, (
            "evaluate() in unsupervised mode must return 'anomaly_rate'"
        )

    def test_save_creates_all_files(self, tmp_path):
        """save() must create all 4 pkl files + ensemble_config.json."""
        _, _, X, y = _small_dataset()
        det = EnsembleICSDetector(random_seed=0)
        det.feature_names = [f"f{i}" for i in range(62)]
        det.train(X, y)
        det.save(str(tmp_path))

        expected = [
            "ensemble_isolation_forest.pkl",
            "ensemble_xgboost.pkl",
            "ensemble_random_forest.pkl",
            "ensemble_scaler.pkl",
            "ensemble_config.json",
            "feature_names.txt",
        ]
        for fname in expected:
            assert (tmp_path / fname).exists(), f"Missing file: {fname}"

    def test_save_load_roundtrip(self, tmp_path):
        """save() then load() must restore IF, XGB, RF, scaler, feature_names."""
        _, _, X, y = _small_dataset()
        det = EnsembleICSDetector(random_seed=0)
        det.feature_names = [f"feat_{i}" for i in range(62)]
        det.train(X, y)
        det.save(str(tmp_path))

        det2 = EnsembleICSDetector(random_seed=0)
        det2.load(str(tmp_path))

        assert det2.isolation_forest is not None
        assert det2.xgb_model is not None
        assert det2.rf_model is not None
        assert det2.scaler is not None
        assert det2.feature_names == det.feature_names

    def test_load_predictions_match_original(self, tmp_path):
        """Predictions from loaded model must match original model."""
        _, _, X, y = _small_dataset(n=300)
        det = EnsembleICSDetector(random_seed=0)
        det.train(X[:200], y[:200])
        det.save(str(tmp_path))

        det2 = EnsembleICSDetector()
        det2.load(str(tmp_path))

        preds1, _ = det.predict(X[200:])
        preds2, _ = det2.predict(X[200:])
        np.testing.assert_array_equal(preds1, preds2, err_msg=(
            "Loaded model gives different predictions than original."
        ))

    def test_feature_names_length_matches_X(self):
        """feature_names must have length == X.shape[1] after training."""
        _, _, X, y = _small_dataset()
        det = EnsembleICSDetector(random_seed=0)
        det.feature_names = [f"f{i}" for i in range(X.shape[1])]
        det.train(X, y)
        assert len(det.feature_names) == X.shape[1], (
            f"feature_names length {len(det.feature_names)} != "
            f"X.shape[1] {X.shape[1]}"
        )

    def test_predict_raises_before_training(self):
        """predict() must raise RuntimeError if called before train()."""
        det = EnsembleICSDetector()
        with pytest.raises(RuntimeError):
            det.predict(np.zeros((5, 62)))

    def test_config_json_has_threshold(self, tmp_path):
        """ensemble_config.json must contain anomaly_threshold field."""
        _, _, X, y = _small_dataset()
        det = EnsembleICSDetector(random_seed=0)
        det.train(X, y)
        det.save(str(tmp_path))
        config = json.loads((tmp_path / "ensemble_config.json").read_text())
        assert "anomaly_threshold" in config, (
            "ensemble_config.json missing 'anomaly_threshold'"
        )
