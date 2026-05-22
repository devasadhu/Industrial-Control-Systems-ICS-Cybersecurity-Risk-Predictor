import numpy as np
import pandas as pd

WINDOW_SECONDS = 60


def compute_session_features(raw: pd.DataFrame, window_seconds: int = WINDOW_SECONDS) -> pd.DataFrame:
    """
    Compute session-level features per source IP within a time window.

    Returns DataFrame aligned with raw index.
    """

    if 'start' not in raw.columns:
        raise ValueError("'start' column not found — cannot compute time windows.")

    times   = raw['start'].fillna(0).values.astype(float)
    src_ips = raw['sIPs'].fillna('unknown').values
    dst_ips = raw['rIPs'].fillna('unknown').values
    payloads = raw['sPayloadAvg'].fillna(0).values

    n = len(raw)

    src_unique_dst_count    = np.zeros(n, dtype=np.float32)
    src_flow_count          = np.ones(n,  dtype=np.float32)
    src_inter_flow_interval = np.zeros(n, dtype=np.float32)
    src_inter_flow_variance = np.zeros(n, dtype=np.float32)
    src_dst_flow_ratio      = np.ones(n,  dtype=np.float32)
    src_payload_entropy     = np.zeros(n, dtype=np.float32)

    from collections import defaultdict
    ip_to_indices = defaultdict(list)
    for i, ip in enumerate(src_ips):
        ip_to_indices[ip].append(i)

    for src_ip, indices in ip_to_indices.items():
        indices_arr = np.array(indices)
        ip_times    = times[indices_arr]

        sort_order   = np.argsort(ip_times)
        sorted_idx   = indices_arr[sort_order]
        sorted_times = ip_times[sort_order]

        for pos, flow_idx in enumerate(sorted_idx):
            t = sorted_times[pos]

            window_mask = (sorted_times >= t - window_seconds) & \
                          (sorted_times <= t + window_seconds)

            window_indices = sorted_idx[window_mask]
            window_times   = sorted_times[window_mask]

            flow_count = len(window_indices)
            src_flow_count[flow_idx] = float(flow_count)

            window_dsts = dst_ips[window_indices]
            unique_dsts = len(set(window_dsts))
            src_unique_dst_count[flow_idx] = float(unique_dsts)

            src_dst_flow_ratio[flow_idx] = float(flow_count) / max(unique_dsts, 1)

            if flow_count >= 2:
                intervals = np.diff(np.sort(window_times))
                src_inter_flow_interval[flow_idx] = float(intervals.mean())
                src_inter_flow_variance[flow_idx] = float(intervals.var())

            window_payloads = payloads[window_indices]
            window_payloads = window_payloads[window_payloads > 0]

            if len(window_payloads) >= 2:
                counts, _ = np.histogram(window_payloads, bins=10)
                probs = counts / counts.sum()
                probs = probs[probs > 0]
                entropy = float(-np.sum(probs * np.log2(probs)))
                src_payload_entropy[flow_idx] = entropy

    return pd.DataFrame({
        'src_unique_dst_count':    src_unique_dst_count,
        'src_flow_count':          src_flow_count,
        'src_inter_flow_interval': src_inter_flow_interval,
        'src_inter_flow_variance': src_inter_flow_variance,
        'src_dst_flow_ratio':      src_dst_flow_ratio,
        'src_payload_entropy':     src_payload_entropy,
    }, index=raw.index)