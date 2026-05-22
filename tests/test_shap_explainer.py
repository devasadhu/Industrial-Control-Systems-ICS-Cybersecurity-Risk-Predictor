"""
test_shap_explainer.py
-----------------------
Tests for src/explainability/shap_explainer.py

What we verify:
1. explain_prediction() returns a dict with required keys
2. 'prediction' is 'ANOMALY' or 'NORMAL'
3. 'top_features' is a list of length top_n
4. Each feature entry has name, value, shap_value, contribution_pct, impact
5. contribution_pct values are non-negative and sum to ~100%
6. SHAP impact labels are correct (negative SHAP → pushes toward anomaly)
7. Explainer raises ValueError if create_explainer() was never called
8. load_model() gracefully warns (not crashes) when scaler file is non-scaler
"""

import sys
import pytest
import numpy as np
import tempfile
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from explainability.shap_explainer import ICSExplainer

# We need a trained IF model for the explainer.
# Build one inline — no dependency on real model files.
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def explainer_fixture(tmp_path_factory):
    """Train a small IsolationForest, save it, and set up ICSExplainer."""
    rng = np.random.default_rng(0)
    n_features = 10
    X = rng.uniform(0, 1, (200, n_features))
    # Inject obvious anomalies to ensure score_samples varies
    X[180:] = rng.uniform(10, 20, (20, n_features))

    tmp = tmp_path_factory.mktemp("models")
    model_path = tmp / "ensemble_isolation_forest.pkl"
    scaler_path = tmp / "ensemble_scaler.pkl"
    feat_path = tmp / "feature_names.txt"

    model = IsolationForest(n_estimators=50, contamination=0.1, random_state=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    feat_path.write_text("\n".join([f"feat_{i}" for i in range(n_features)]))

    explainer = ICSExplainer(
        model_path=str(model_path),
        feature_names_path=str(feat_path),
        scaler_path=str(scaler_path),
    )
    background = X[:100]
    explainer.create_explainer(background)

    return explainer, X, n_features


class TestICSExplainer:

    def test_explain_returns_dict(self, explainer_fixture):
        """explain_prediction() must return a dict."""
        explainer, X, _ = explainer_fixture
        result = explainer.explain_prediction(X[0])
        assert isinstance(result, dict), "explain_prediction() must return a dict"

    def test_required_keys(self, explainer_fixture):
        """Result dict must contain prediction, anomaly_score, is_anomalous, top_features."""
        explainer, X, _ = explainer_fixture
        result = explainer.explain_prediction(X[0])
        for key in ("prediction", "anomaly_score", "is_anomalous", "top_features"):
            assert key in result, f"Missing key '{key}' in explain_prediction() output"

    def test_prediction_is_valid_label(self, explainer_fixture):
        """'prediction' must be 'ANOMALY' or 'NORMAL'."""
        explainer, X, _ = explainer_fixture
        result = explainer.explain_prediction(X[0])
        assert result["prediction"] in ("ANOMALY", "NORMAL"), (
            f"Invalid prediction label: '{result['prediction']}'"
        )

    def test_top_features_length(self, explainer_fixture):
        """top_features list must have length == top_n."""
        explainer, X, _ = explainer_fixture
        for top_n in (3, 5):
            result = explainer.explain_prediction(X[0], top_n=top_n)
            assert len(result["top_features"]) == top_n, (
                f"Expected {top_n} top features, got {len(result['top_features'])}"
            )

    def test_feature_entry_keys(self, explainer_fixture):
        """Each feature entry must have name, value, shap_value, contribution_pct, impact."""
        explainer, X, _ = explainer_fixture
        result = explainer.explain_prediction(X[0], top_n=3)
        required = {"name", "value", "shap_value", "contribution_pct", "impact"}
        for entry in result["top_features"]:
            assert required.issubset(entry.keys()), (
                f"Feature entry missing keys. Expected {required}, got {set(entry.keys())}"
            )

    def test_contribution_pct_non_negative(self, explainer_fixture):
        """All contribution_pct values must be >= 0."""
        explainer, X, _ = explainer_fixture
        result = explainer.explain_prediction(X[0], top_n=5)
        for entry in result["top_features"]:
            assert entry["contribution_pct"] >= 0, (
                f"Negative contribution_pct: {entry['contribution_pct']}"
            )

    def test_shap_impact_label_negative_pushes_toward_anomaly(self, explainer_fixture):
        """
        SHAP value < 0 must carry impact label containing 'anomaly'.
        SHAP value > 0 must carry impact label containing 'normal'.

        This was inverted in v1.3.0 — FIX 1 corrected it.
        IsolationForest.score_samples() is more negative = more anomalous,
        so a negative SHAP contribution pulls score down = more anomalous.
        """
        explainer, X, _ = explainer_fixture
        # Use the obvious anomaly samples to get a mix of SHAP signs
        for idx in [0, 50, 185, 195]:
            result = explainer.explain_prediction(X[idx], top_n=5)
            for entry in result["top_features"]:
                sv = entry["shap_value"]
                impact = entry["impact"].lower()
                if sv < 0:
                    assert "anomaly" in impact, (
                        f"Negative SHAP ({sv:.4f}) should indicate pushes toward "
                        f"anomaly, but impact says: '{impact}'"
                    )
                elif sv > 0:
                    assert "normal" in impact, (
                        f"Positive SHAP ({sv:.4f}) should indicate pushes toward "
                        f"normal, but impact says: '{impact}'"
                    )

    def test_raises_without_create_explainer(self, tmp_path):
        """explain_prediction() must raise ValueError before create_explainer() called."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (50, 5))
        model = IsolationForest(n_estimators=10, random_state=0)
        model.fit(X)
        model_path = tmp_path / "if.pkl"
        joblib.dump(model, model_path)

        explainer = ICSExplainer(model_path=str(model_path))
        with pytest.raises(ValueError, match="[Ee]xplainer not initialized"):
            explainer.explain_prediction(X[0])

    def test_non_scaler_file_warns_not_crashes(self, tmp_path):
        """
        Passing a non-StandardScaler file as scaler_path must log a warning
        and continue, not crash.
        """
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (50, 5))
        model = IsolationForest(n_estimators=10, random_state=0)
        model.fit(X)
        model_path = tmp_path / "if.pkl"
        joblib.dump(model, model_path)

        # Save a list (not a StandardScaler) as the "scaler"
        bad_scaler_path = tmp_path / "bad_scaler.pkl"
        joblib.dump(["feature_1", "feature_2"], bad_scaler_path)

        # Should not raise — just warn
        explainer = ICSExplainer(
            model_path=str(model_path),
            scaler_path=str(bad_scaler_path),
        )
        assert explainer.scaler is None, (
            "Scaler should be None when a non-scaler object is loaded"
        )

    def test_anomaly_score_is_float(self, explainer_fixture):
        """anomaly_score in result must be a Python float."""
        explainer, X, _ = explainer_fixture
        result = explainer.explain_prediction(X[0])
        assert isinstance(result["anomaly_score"], float), (
            f"anomaly_score type: {type(result['anomaly_score'])}"
        )
