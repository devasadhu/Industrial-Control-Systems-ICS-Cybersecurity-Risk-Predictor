"""
test_behavioral_baseline.py
----------------------------
Tests for src/behavioral_baseline.py

What we verify:
1. fit() returns self (fluent API)
2. score() returns a numpy array of length == input rows
3. All scores are in [0, 1]
4. flag() returns a DataFrame with transition_anomaly_score and markov_anomaly columns
5. flag() row count matches input
6. An obvious outlier row scores higher than normal rows on average
7. score() raises RuntimeError when called before fit()
8. detect() returns the documented keys: detected, n_anomalous, confidence, flows
9. detect() detected flag is bool
10. save() + load() round-trip: fitted model survives serialisation
11. Loaded model produces the same scores as the original on held-out data
12. fit() on a single-row DataFrame does not crash (edge case)
13. markov_anomaly column is boolean dtype after flag()
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from behavioral_baseline import MarkovBehavioralBaseline


# ── Synthetic data generators ─────────────────────────────────────────────────

_PROTOCOLS = ["TCP", "UDP", "Modbus"]


def _make_normal_flows(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """
    Generate normal-looking ICS flows. All have protocol='Modbus' and
    moderate packet/byte values. The Markov chain will learn this is normal.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "src_ip":        [f"10.0.0.{rng.integers(1, 5)}" for _ in range(n)],
        "protocol":      rng.choice(["TCP", "Modbus"], n).tolist(),
        "src_packet_rate": rng.uniform(5, 100, n),
        "bytes_per_packet": rng.uniform(40, 80, n),
        "src_syn_rate":  rng.uniform(0.0, 0.05, n),
        "src_rst_rate":  np.zeros(n),
        "flow_duration": rng.uniform(0.5, 5.0, n),
    })


def _make_attack_flows(n: int = 20, seed: int = 99) -> pd.DataFrame:
    """
    Generate obviously-anomalous flows: extreme packet rates, high SYN/RST,
    unusual protocol transitions — designed to produce high Markov scores.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "src_ip":        [f"10.0.0.{rng.integers(10, 15)}" for _ in range(n)],
        "protocol":      rng.choice(["UDP", "TCP"], n).tolist(),
        "src_packet_rate": rng.uniform(5000, 10000, n),
        "bytes_per_packet": rng.uniform(5, 20, n),   # tiny crafted packets
        "src_syn_rate":  rng.uniform(0.8, 1.0, n),
        "src_rst_rate":  rng.uniform(0.5, 0.9, n),
        "flow_duration": rng.uniform(0.001, 0.01, n),
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMarkovBehavioralBaseline:

    @pytest.fixture(autouse=True)
    def baseline(self):
        self.baseline = MarkovBehavioralBaseline(anomaly_threshold=3.0)

    @pytest.fixture()
    def fitted_baseline(self):
        """Return a baseline already fitted on normal flows."""
        bl = MarkovBehavioralBaseline(anomaly_threshold=3.0)
        bl.fit(_make_normal_flows(n=200, seed=0))
        return bl

    # ── fit ──────────────────────────────────────────────────────────────────

    def test_fit_returns_self(self):
        """fit() must return self for fluent chaining."""
        result = self.baseline.fit(_make_normal_flows(n=50))
        assert result is self.baseline, "fit() must return self"

    def test_fit_single_row_no_crash(self):
        """fit() on a single-row DataFrame must not raise."""
        single = _make_normal_flows(n=1)
        self.baseline.fit(single)  # should not raise

    def test_fit_sets_is_fitted(self):
        """After fit(), _is_fitted must be True."""
        self.baseline.fit(_make_normal_flows(n=50))
        assert self.baseline._is_fitted, "_is_fitted should be True after fit()"

    # ── score ─────────────────────────────────────────────────────────────────

    def test_score_raises_before_fit(self):
        """score() must raise RuntimeError if called before fit()."""
        with pytest.raises(RuntimeError):
            self.baseline.score(_make_normal_flows(n=5))

    def test_score_returns_ndarray(self, fitted_baseline):
        """score() must return a numpy ndarray."""
        scores = fitted_baseline.score(_make_normal_flows(n=10))
        assert isinstance(scores, np.ndarray), "score() must return np.ndarray"

    def test_score_length_matches_input(self, fitted_baseline):
        """score() output length must match number of input rows."""
        df = _make_normal_flows(n=30)
        scores = fitted_baseline.score(df)
        assert len(scores) == len(df), (
            f"Expected {len(df)} scores, got {len(scores)}"
        )

    def test_scores_in_unit_interval(self, fitted_baseline):
        """All scores must be in [0, 1]."""
        scores = fitted_baseline.score(_make_normal_flows(n=50))
        assert (scores >= 0).all(), f"Scores below 0: min={scores.min():.4f}"
        assert (scores <= 1).all(), f"Scores above 1: max={scores.max():.4f}"

    def test_attack_flows_score_higher_than_normal(self, fitted_baseline):
        """
        Flows trained on must score lower on average than obvious attack flows.
        The Markov chain learns normal transitions; unseen attack transitions
        receive high anomaly scores.
        """
        normal_scores = fitted_baseline.score(_make_normal_flows(n=100, seed=1))
        attack_scores = fitted_baseline.score(_make_attack_flows(n=50, seed=2))

        mean_normal = float(normal_scores.mean())
        mean_attack = float(attack_scores.mean())

        assert mean_attack >= mean_normal, (
            f"Attack mean score ({mean_attack:.4f}) should be >= "
            f"normal mean score ({mean_normal:.4f}). "
            "Markov anomaly discriminator is not working."
        )

    # ── flag ──────────────────────────────────────────────────────────────────

    def test_flag_returns_dataframe(self, fitted_baseline):
        """flag() must return a pandas DataFrame."""
        result = fitted_baseline.flag(_make_normal_flows(n=10))
        assert isinstance(result, pd.DataFrame), "flag() must return pd.DataFrame"

    def test_flag_row_count_matches_input(self, fitted_baseline):
        """flag() row count must equal input row count."""
        df = _make_normal_flows(n=25)
        result = fitted_baseline.flag(df)
        assert len(result) == len(df), (
            f"Expected {len(df)} rows, got {len(result)}"
        )

    def test_flag_score_column_present(self, fitted_baseline):
        """'transition_anomaly_score' column must be present in flag() output."""
        result = fitted_baseline.flag(_make_normal_flows(n=10))
        assert "transition_anomaly_score" in result.columns, (
            "Column 'transition_anomaly_score' missing from flag() output"
        )

    def test_flag_anomaly_column_present(self, fitted_baseline):
        """'markov_anomaly' column must be present in flag() output."""
        result = fitted_baseline.flag(_make_normal_flows(n=10))
        assert "markov_anomaly" in result.columns, (
            "Column 'markov_anomaly' missing from flag() output"
        )

    def test_flag_anomaly_column_is_boolean(self, fitted_baseline):
        """'markov_anomaly' must be boolean dtype."""
        result = fitted_baseline.flag(_make_normal_flows(n=10))
        assert result["markov_anomaly"].dtype == bool, (
            f"markov_anomaly dtype is {result['markov_anomaly'].dtype}, expected bool"
        )

    def test_flag_score_values_in_unit_interval(self, fitted_baseline):
        """All transition_anomaly_score values must be in [0, 1]."""
        result = fitted_baseline.flag(_make_normal_flows(n=30))
        scores = result["transition_anomaly_score"].values
        assert (scores >= 0).all() and (scores <= 1).all(), (
            f"Scores out of [0,1]: min={scores.min():.4f}, max={scores.max():.4f}"
        )

    # ── detect ────────────────────────────────────────────────────────────────

    def test_detect_returns_dict(self, fitted_baseline):
        """detect() must return a dict."""
        result = fitted_baseline.detect(_make_normal_flows(n=20))
        assert isinstance(result, dict), "detect() must return a dict"

    def test_detect_required_keys(self, fitted_baseline):
        """detect() output must contain: detected, n_anomalous_flows, detection_source.
        confidence only appears when detected=True."""
        result = fitted_baseline.detect(_make_normal_flows(n=20))
        for key in ("detected", "n_anomalous_flows", "detection_source"):
            assert key in result, f"Key '{key}' missing from detect() output"

    def test_detect_detected_is_bool(self, fitted_baseline):
        """detect()['detected'] must be a bool."""
        result = fitted_baseline.detect(_make_normal_flows(n=20))
        assert isinstance(result["detected"], bool), (
            f"'detected' should be bool, got {type(result['detected'])}"
        )

    def test_detect_n_anomalous_is_int(self, fitted_baseline):
        """detect()['n_anomalous_flows'] must be an int."""
        result = fitted_baseline.detect(_make_normal_flows(n=20))
        assert isinstance(result["n_anomalous_flows"], (int, np.integer)), (
            f"'n_anomalous_flows' should be int, got {type(result['n_anomalous_flows'])}"
        )

    # ── save / load round-trip ────────────────────────────────────────────────

    def test_save_creates_files(self, tmp_path):
        """save() must create at least a config/meta JSON file."""
        bl = MarkovBehavioralBaseline()
        bl.fit(_make_normal_flows(n=50))
        bl.save(str(tmp_path))
        # At minimum a metadata/config JSON should exist
        json_files = list(tmp_path.glob("*.json"))
        assert json_files, "save() did not create any JSON metadata file"

    def test_load_roundtrip_is_fitted(self, tmp_path):
        """Loaded model must have _is_fitted == True."""
        bl = MarkovBehavioralBaseline()
        bl.fit(_make_normal_flows(n=50))
        bl.save(str(tmp_path))

        bl2 = MarkovBehavioralBaseline.load(str(tmp_path))
        assert bl2._is_fitted, "Loaded model has _is_fitted=False"

    def test_load_roundtrip_scores_match(self, tmp_path):
        """Loaded model must reproduce the same scores as the original."""
        train_df = _make_normal_flows(n=100, seed=0)
        test_df = _make_normal_flows(n=20, seed=7)

        bl = MarkovBehavioralBaseline()
        bl.fit(train_df)
        original_scores = bl.score(test_df)

        bl.save(str(tmp_path))
        bl2 = MarkovBehavioralBaseline.load(str(tmp_path))
        loaded_scores = bl2.score(test_df)

        np.testing.assert_allclose(
            original_scores, loaded_scores, rtol=1e-5,
            err_msg="Loaded model produces different scores than original."
        )
