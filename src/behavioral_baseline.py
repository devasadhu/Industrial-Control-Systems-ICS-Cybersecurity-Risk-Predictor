"""
Markov Chain Behavioral Baseline v1.0.0
=========================================
Models expected ICS flow-state transitions using a first-order Markov chain
trained on normal traffic. Flags flows with low transition probability as
behavioral anomalies — catching stealthy attacks that stay below ML thresholds.

Background
----------
ICS/SCADA systems are deterministic. A PLC polling sequence is a fixed Markov
chain: HMI reads register A, then register B, then A again. Attackers who
replay or inject single packets break this sequence even if each individual
flow looks "normal" to feature-based ML.

This directly implements the approach in:
  Ghazi et al., "Markov Chain-Based Anomaly Detection for Industrial Control
  Systems", Sensors 2025.

State definition
----------------
A flow state is a discretized tuple: (src_port_bucket, dst_port, protocol_id, fc_code)
  src_port_bucket: "ephemeral" (>=49152) or "well-known" (<49152)
  dst_port: raw integer (502 = Modbus, else 'other')
  protocol_id: 'tcp' | 'udp' | 'other'
  fc_code: Modbus function code (1, 3, 6, 16, …) or 0 if non-Modbus

Transition matrix
-----------------
  T[s_i, s_j] = count(s_i → s_j) / count(s_i)
Laplace smoothed: T[s_i, s_j] = (count + alpha) / (sum_counts + alpha * n_states)

Anomaly score
-------------
  score = -log(P(s_{t} | s_{t-1}))
  Normalised to [0, 1] via sigmoid: 1 / (1 + exp(-k*(score - threshold)))

Usage
-----
  from src.behavioral_baseline import MarkovBehavioralBaseline
  import pandas as pd

  baseline = MarkovBehavioralBaseline()
  baseline.fit(normal_flows_df)
  scores = baseline.score(new_flows_df)   # np.ndarray of float [0,1]
  flags  = baseline.flag(new_flows_df)    # pd.DataFrame with 'transition_anomaly_score'

No new dependencies — numpy and scipy only.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

def _extract_state(row: pd.Series) -> Tuple:
    """
    Convert a flow row to a discrete Markov state.

    Expected columns (all optional with fallback):
      src_port, dst_port, protocol (string), modbus_function_code
    """
    # Source port bucket
    src_port = int(row.get("src_port", row.get("sport", 0)) or 0)
    src_bucket = "ephemeral" if src_port >= 49152 else "well-known"

    # Destination port
    dst_port = int(row.get("dst_port", row.get("dport", 0)) or 0)
    dst_key = dst_port if dst_port in (502, 20000, 44818, 2404) else "other"

    # Protocol
    proto_raw = str(row.get("protocol", row.get("proto", "tcp"))).lower()
    if "tcp" in proto_raw:
        proto = "tcp"
    elif "udp" in proto_raw:
        proto = "udp"
    else:
        proto = "other"

    # Modbus FC code
    fc = int(row.get("modbus_function_code", row.get("fc", 0)) or 0)

    return (src_bucket, dst_key, proto, fc)


def _extract_states_from_df(df: pd.DataFrame) -> List[Tuple]:
    """Extract state sequence from a flows DataFrame."""
    return [_extract_state(row) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# Markov Chain
# ---------------------------------------------------------------------------

class _MarkovChain:
    """First-order Markov chain with Laplace smoothing."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.states: List[Tuple] = []
        self._state_index: Dict[Tuple, int] = {}
        self._counts: np.ndarray | None = None   # (n_states, n_states)
        self._transition_matrix: np.ndarray | None = None

    def fit(self, state_sequences: List[List[Tuple]]) -> None:
        """
        Build transition matrix from a list of state sequences.
        Each sequence = one source IP's ordered flow stream.
        """
        # Build state vocabulary
        all_states = set()
        for seq in state_sequences:
            all_states.update(seq)
        self.states = sorted(all_states, key=str)
        self._state_index = {s: i for i, s in enumerate(self.states)}
        n = len(self.states)

        counts = np.zeros((n, n), dtype=np.float64)
        for seq in state_sequences:
            for i in range(len(seq) - 1):
                s_from = seq[i]
                s_to   = seq[i + 1]
                if s_from in self._state_index and s_to in self._state_index:
                    counts[self._state_index[s_from], self._state_index[s_to]] += 1

        # Laplace smoothing
        counts += self.alpha
        row_sums = counts.sum(axis=1, keepdims=True)
        self._transition_matrix = counts / row_sums
        self._counts = counts

    def log_prob(self, s_from: Tuple, s_to: Tuple) -> float:
        """
        log P(s_to | s_from). Returns log(alpha / (N*alpha)) for unseen states.
        """
        if self._transition_matrix is None:
            raise RuntimeError("Call fit() before log_prob().")

        n = len(self.states)
        fallback = np.log(self.alpha / (n * self.alpha + self.alpha))

        i = self._state_index.get(s_from)
        j = self._state_index.get(s_to)
        if i is None or j is None:
            return fallback
        p = self._transition_matrix[i, j]
        return np.log(p) if p > 0 else fallback

    def transition_probability(self, s_from: Tuple, s_to: Tuple) -> float:
        """P(s_to | s_from), in [0, 1]."""
        n = len(self.states)
        fallback = self.alpha / (n * self.alpha + self.alpha)
        i = self._state_index.get(s_from)
        j = self._state_index.get(s_to)
        if i is None or j is None or self._transition_matrix is None:
            return fallback
        return float(self._transition_matrix[i, j])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "states": [list(s) for s in self.states],
            "transition_matrix": self._transition_matrix.tolist(),
            "alpha": self.alpha,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "_MarkovChain":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        mc = cls(alpha=payload["alpha"])
        mc.states = [tuple(s) for s in payload["states"]]
        mc._state_index = {s: i for i, s in enumerate(mc.states)}
        mc._transition_matrix = np.array(payload["transition_matrix"])
        return mc


# ---------------------------------------------------------------------------
# Behavioral Baseline
# ---------------------------------------------------------------------------

class MarkovBehavioralBaseline:
    """
    Markov Chain Behavioral Baseline for ICS flow anomaly detection.

    Trains per-source-IP Markov chains on normal traffic. At inference,
    computes transition_anomaly_score ∈ [0, 1] for each flow:
      - 0 = expected transition (normal)
      - 1 = highly anomalous transition

    Parameters
    ----------
    alpha : float
        Laplace smoothing factor (default 0.1).
    anomaly_threshold : float
        Negative log-prob threshold above which a transition is flagged (default 3.0).
        Tuned on ICSSIM dataset. Higher = fewer false positives.
    sigmoid_k : float
        Sharpness of sigmoid transformation (default 0.5).
    min_sequence_length : int
        Minimum flows per source IP to train a per-IP chain (default 5).
        IPs with fewer flows use the global chain.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        anomaly_threshold: float = 3.0,
        sigmoid_k: float = 0.5,
        min_sequence_length: int = 5,
    ):
        self.alpha = alpha
        self.anomaly_threshold = anomaly_threshold
        self.sigmoid_k = sigmoid_k
        self.min_sequence_length = min_sequence_length

        self._global_chain: Optional[_MarkovChain] = None
        self._per_ip_chains: Dict[str, _MarkovChain] = {}
        self._is_fitted = False

    # ---- fit ---------------------------------------------------------------

    def fit(self, df: pd.DataFrame, src_col: str = "src_ip") -> "MarkovBehavioralBaseline":
        """
        Train on a DataFrame of normal flows.

        Parameters
        ----------
        df : pd.DataFrame
            Normal traffic flows. Must be sorted by time if you want
            chronological transition modelling.
        src_col : str
            Column name for source IP (used for per-IP chains).
        """
        if df.empty:
            raise ValueError("Training DataFrame is empty.")

        # Global chain (all flows)
        global_states = _extract_states_from_df(df)
        self._global_chain = _MarkovChain(alpha=self.alpha)
        self._global_chain.fit([global_states])

        # Per-IP chains
        if src_col in df.columns:
            for src_ip, group in df.groupby(src_col):
                if len(group) < self.min_sequence_length:
                    continue
                seq = _extract_states_from_df(group)
                chain = _MarkovChain(alpha=self.alpha)
                chain.fit([seq])
                self._per_ip_chains[str(src_ip)] = chain

        self._is_fitted = True
        return self

    # ---- score -------------------------------------------------------------

    def score(self, df: pd.DataFrame, src_col: str = "src_ip") -> np.ndarray:
        """
        Compute transition anomaly score for each row in df.

        Returns np.ndarray of shape (len(df),) with values in [0, 1].
        First flow per source (no prior state) gets score = 0.0.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before score().")

        scores = np.zeros(len(df), dtype=np.float64)
        states = _extract_states_from_df(df)

        src_ips = df[src_col].tolist() if src_col in df.columns else ["_global"] * len(df)
        prev_state_by_ip: Dict[str, Tuple] = {}

        for idx, (state, src_ip) in enumerate(zip(states, src_ips)):
            src_ip = str(src_ip)
            prev = prev_state_by_ip.get(src_ip)
            if prev is None:
                scores[idx] = 0.0
            else:
                chain = self._per_ip_chains.get(src_ip, self._global_chain)
                log_p = chain.log_prob(prev, state)
                neg_log_p = -log_p   # higher = more anomalous
                scores[idx] = self._sigmoid(neg_log_p)
            prev_state_by_ip[src_ip] = state

        return scores

    # ---- flag --------------------------------------------------------------

    def flag(
        self,
        df: pd.DataFrame,
        src_col: str = "src_ip",
        score_col: str = "transition_anomaly_score",
        flag_col: str = "markov_anomaly",
    ) -> pd.DataFrame:
        """
        Return df with two new columns:
          transition_anomaly_score : float [0,1]
          markov_anomaly           : bool  (score above threshold)
        """
        result = df.copy()
        scores = self.score(df, src_col=src_col)
        result[score_col] = scores

        # Threshold in sigmoid space
        sigmoid_threshold = self._sigmoid(self.anomaly_threshold)
        result[flag_col] = scores >= sigmoid_threshold

        return result

    # ---- detection source interface (matches attack_patterns.py style) -----

    def detect(self, df: pd.DataFrame, src_col: str = "src_ip") -> dict:
        """
        Returns a detection dict compatible with detect_all_patterns() output:
          {
            "detected": bool,
            "detection_source": "markov_behavioral",
            "n_anomalous_flows": int,
            "severity": "low"|"medium"|"high"|"critical",
            "flows": [...],
            "confidence": float,
          }
        """
        flagged_df = self.flag(df, src_col=src_col)
        anomalous = flagged_df[flagged_df["markov_anomaly"]]
        n = len(anomalous)

        if n == 0:
            return {
                "detected": False,
                "detection_source": "markov_behavioral",
                "n_anomalous_flows": 0,
            }

        frac = n / len(df)
        if frac > 0.3:
            severity = "critical"
        elif frac > 0.1:
            severity = "high"
        elif frac > 0.03:
            severity = "medium"
        else:
            severity = "low"

        return {
            "detected": True,
            "detection_source": "markov_behavioral",
            "n_anomalous_flows": n,
            "fraction_anomalous": round(frac, 4),
            "severity": severity,
            "confidence": float(anomalous["transition_anomaly_score"].mean()),
            "flows": anomalous[["src_ip"] if "src_ip" in anomalous.columns else []
                               + ["transition_anomaly_score"]].head(50).to_dict("records"),
        }

    # ---- persistence -------------------------------------------------------

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        meta = {
            "alpha": self.alpha,
            "anomaly_threshold": self.anomaly_threshold,
            "sigmoid_k": self.sigmoid_k,
            "min_sequence_length": self.min_sequence_length,
            "per_ip_keys": list(self._per_ip_chains.keys()),
        }
        (dir_path / "markov_meta.json").write_text(json.dumps(meta, indent=2))

        if self._global_chain:
            self._global_chain.save(dir_path / "markov_global.json")

        for ip, chain in self._per_ip_chains.items():
            safe = ip.replace(".", "_").replace(":", "_")
            chain.save(dir_path / f"markov_ip_{safe}.json")

    @classmethod
    def load(cls, dir_path: str | Path) -> "MarkovBehavioralBaseline":
        dir_path = Path(dir_path)
        meta = json.loads((dir_path / "markov_meta.json").read_text())
        obj = cls(
            alpha=meta["alpha"],
            anomaly_threshold=meta["anomaly_threshold"],
            sigmoid_k=meta["sigmoid_k"],
            min_sequence_length=meta["min_sequence_length"],
        )
        global_path = dir_path / "markov_global.json"
        if global_path.exists():
            obj._global_chain = _MarkovChain.load(global_path)
        for ip in meta.get("per_ip_keys", []):
            safe = ip.replace(".", "_").replace(":", "_")
            chain_path = dir_path / f"markov_ip_{safe}.json"
            if chain_path.exists():
                obj._per_ip_chains[ip] = _MarkovChain.load(chain_path)
        obj._is_fitted = True
        return obj

    # ---- helper ------------------------------------------------------------

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-self.sigmoid_k * (x - self.anomaly_threshold)))

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        n_ip_chains = len(self._per_ip_chains)
        return (
            f"MarkovBehavioralBaseline("
            f"status={status}, "
            f"global_states={len(self._global_chain.states) if self._global_chain else 0}, "
            f"per_ip_chains={n_ip_chains}, "
            f"threshold={self.anomaly_threshold})"
        )


# ---------------------------------------------------------------------------
# Convenience: train from ICSSIM CSV
# ---------------------------------------------------------------------------

def train_from_icssim(
    csv_path: str,
    label_col: str = "NST_M_Label",
    normal_label: str = "Normal",
    save_dir: Optional[str] = None,
) -> MarkovBehavioralBaseline:
    """
    Load ICSSIM dataset CSV, filter to normal flows, train baseline.

    Parameters
    ----------
    csv_path : str
        Path to the ICSSIM Dataset.csv.
    label_col : str
        Column name for the class label.
    normal_label : str
        Value of the normal class label.
    save_dir : str | None
        If given, save the trained baseline here.

    Returns
    -------
    MarkovBehavioralBaseline (fitted)
    """
    df = pd.read_csv(csv_path)
    normal_df = df[df[label_col] == normal_label].copy()
    print(f"[markov] Training on {len(normal_df)} normal flows...")

    baseline = MarkovBehavioralBaseline()
    baseline.fit(normal_df)
    print(f"[markov] {baseline}")

    if save_dir:
        baseline.save(save_dir)
        print(f"[markov] Saved → {save_dir}")

    return baseline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train/evaluate Markov Chain Behavioral Baseline on ICSSIM data."
    )
    parser.add_argument("--csv", required=True, help="Path to ICSSIM Dataset.csv")
    parser.add_argument("--save-dir", default=None, help="Save trained model here")
    parser.add_argument("--eval", action="store_true",
                        help="Run self-evaluation on attack flows after training")
    args = parser.parse_args()

    baseline = train_from_icssim(args.csv, save_dir=args.save_dir)

    if args.eval:
        df = pd.read_csv(args.csv)
        attack_df = df[df["NST_M_Label"] != "Normal"].head(500)
        if attack_df.empty:
            print("[markov] No attack flows found for evaluation.")
        else:
            result = baseline.detect(attack_df)
            print(f"[markov] Detection result: {result}")
