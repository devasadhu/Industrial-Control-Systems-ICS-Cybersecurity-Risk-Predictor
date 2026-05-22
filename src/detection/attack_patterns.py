"""
ICS Attack Pattern Detection Library
Detects known attack patterns in industrial control systems

Based on MITRE ATT&CK for ICS and real-world incidents:
- Stuxnet, Triton, BlackEnergy, Industroyer

Author: Sadhana Devarajan
Version: 3.0.0

Changes in v2.0.0:
- FIX 1: All 10 detectors now use the actual 51 engineered features present in
         ics_features.csv. Previously, detect_modbus_flooding() and
         detect_plc_scanning() depended on 'dst_port' which is dropped during
         feature engineering — causing immediate early returns and 0 detections.
- FIX 2: All 10 patterns now have a corresponding detect_* method and are called
         from detect_all_patterns(). Previously only 2 of 10 were wired up.
- FIX 3: detect_unauthorized_writes() no longer depends on external PCAP protocol
         analysis (which was always None in the demo).
- FIX 4: Added _check_required_columns() helper so each detector gracefully skips
         with a warning instead of crashing when a column is unexpectedly absent.

Changes in v2.1.0 — threshold tuning based on ICSSIM describe() output:

Changes in v3.0.0 — Modbus DPI integration:
- detect_unauthorized_writes() now accepts and uses protocol_analysis results from
  ICSProtocolAnalyzer. When Modbus write function codes (FC 05, 06, 0F, 10, 15, 16)
  are confirmed via PCAP DPI, those flows are added as CONFIRMED detections at higher
  confidence (0.95) with function code detail, register address, and severity from
  the analyzer (CRITICAL for FC 10/protected registers, HIGH for others).
- Added detect_modbus_writes_from_pcap(): pure DPI-only detector that catches
  Modbus write FC events regardless of flow-feature thresholds. This handles cases
  where the flow statistics look normal (legitimate-seeming command) but the FC
  is a write — the protocol layer is ground truth.
- detect_all_patterns() now accepts optional pcap_path argument. When provided,
  it runs ICSProtocolAnalyzer and passes results to both detect_unauthorized_writes()
  and detect_modbus_writes_from_pcap(). The 'protocol_aware' key in results
  indicates whether DPI was active.
- ICSProtocolAnalyzer import is lazy (inside method) so the file still loads
  cleanly when Scapy is not installed (flow-based detection is unaffected).

Changes in v2.1.0 — threshold tuning based on ICSSIM describe() output:
  Dataset stats used for calibration (45,718 flows):
    src_packet_rate  : mean=11063  p25=20   p50=49   p75=148   max=1,000,000
    src_inter_pkt_avg: mean=0.030  p25=0.001 p50=0.021 p75=0.047 max=0.500
    byte_ratio       : (inferred symmetric — replay window tightened)
    packet_ratio     : (inferred symmetric — replay window tightened)
    bytes_per_packet : mean=56  p25=57  p50=59  p75=59  max=74
    total_packets    : mean=550  p25=18  p50=30  p75=36  max=17284
    syn_ack_imbalance: (used for credential stuffing)
    src_fin_rate     : mean=0.0002  max=1.0

  TUNING 1 — modbus_flooding: raised flood_packet_rate 100→10000,
             flood_min_packets 200→5000. Previously p75 (148 pkt/s) was
             above threshold, catching ~25% of all normal flows.

  TUNING 2 — time_based_attack: raised time_iat_max 0.01→0.002 and added
             IAT ratio condition (dst/src IAT must be close, confirming
             machine-driven regularity). Mean IAT is 0.030s so 0.01 was
             catching normal fast polling. Also raised min_packets 50→100.

  TUNING 3 — replay_attack: tightened byte_ratio and packet_ratio windows
             from 0.90–1.10 to 0.995–1.005. The wide window was matching
             any balanced bidirectional flow (1,700 false positives).

  TUNING 4 — firmware_modification: lowered bytes_per_packet threshold
             800→70 (dataset max is only 74, so 800 was unreachable).
             Raised firmware_min_bytes 50000→100000 to avoid small flows.
             Added packet_ratio < 0.15 (heavily one-way upload).

  TUNING 5 — unauthorized_write: lowered write_packet_rate_min 20→50
             to reduce overlap with normal low-rate command flows.

  TUNING 6 — protocol_fuzzing: lowered fuzz_bytes_per_pkt_max 40→25
             (dataset min bytes_per_packet is 21, so 40 caught normal frames).
             Kept size_anomaly_min at 300 (deviations >300 from 500-byte
             baseline are genuinely unusual given dataset std of ~6.5).

  TUNING 7 — credential_stuffing: raised cred_max_packets 20→15 and
             added src_syn_rate > 0.3 as a required condition so random
             short flows don't trigger it.

Feature → Attack pattern mapping used:
  modbus_flooding     → src_packet_rate > 10000, total_packets > 5000
  plc_scanning        → packet_ratio < 0.05, dst_packets ≤ 3, byte_ratio < 0.05
  unauthorized_write  → src_psh_rate > 0.4, src_packet_rate > 50, src_rst_rate > 0
  protocol_fuzzing    → packet_size_anomaly > 300, src_urg_rate > 0.01,
                        bytes_per_packet < 25
  mitm                → traffic_symmetry > 0.95, byte_ratio 0.85–1.15, src_ttl=128
  command_injection   → src_psh_rate > 0.5, dst_psh_rate < 0.1, pkt_rate > 15
  time_based_attack   → src_inter_packet_avg < 0.002, total_packets > 100,
                        dst/src IAT ratio 0.8–1.2 (machine regularity)
  replay_attack       → byte_ratio 0.995–1.005, packet_ratio 0.995–1.005,
                        traffic_symmetry > 0.98
  credential_stuffing → syn_ack_imbalance > 0.3, total_packets < 15,
                        src_fin_rate > 0.1, src_syn_rate > 0.3
  firmware_mod        → total_bytes > 100000, bytes_per_packet > 70,
                        packet_ratio < 0.15
"""

import json
import time
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AttackPattern:
    """Attack pattern definition."""
    name: str
    mitre_technique: str
    description: str
    indicators: Dict
    severity: str
    mitigation: str
    real_world_example: str


# ── Thresholds — calibrated against ICSSIM describe() output (45,718 flows) ──
#
#  Key stats used:
#    src_packet_rate  p50=49    p75=148   max=1,000,000
#    src_inter_pkt    p50=0.021 p75=0.047 mean=0.030
#    bytes_per_packet mean=56   p75=59    max=74
#    total_packets    p50=30    p75=36    max=17,284
#    src_fin_rate     mean=0.0002         max=1.0
#
THRESHOLDS = {
    # ── modbus_flooding ───────────────────────────────────────────────────────
    # TUNING 1b: raised from 10,000→50,000. Detections at 10,202 pkt/s were
    # suspiciously round (exactly dataset simulation rate, not real flooding).
    # Genuine flooding attacks in ICSSIM reach 1,000,000 pkt/s (dataset max).
    'flood_packet_rate':         50_000,   # pkt/s  (was 10,000)
    'flood_min_packets':         10_000,   # total  (was 5,000)

    # ── plc_scanning ─────────────────────────────────────────────────────────
    # Unchanged — dst_packets ≤ 3 and ratios near 0 are solid scan signals.
    'scan_packet_ratio_max':      0.05,
    'scan_byte_ratio_max':        0.05,
    'scan_max_dst_packets':       3,

    # ── unauthorized_write ────────────────────────────────────────────────────
    # TUNING 5: raised packet_rate_min 20→50 to avoid overlap with normal
    # low-rate command flows (p25 of src_packet_rate is 20).
    'write_psh_rate_min':         0.4,
    'write_packet_rate_min':      50.0,    # pkt/s  (was 20.0)
    'write_rst_nonzero':          True,

    # ── protocol_fuzzing ─────────────────────────────────────────────────────
    # TUNING 6: lowered bytes_per_pkt threshold 40→25. Dataset min is 21
    # and mean is 56, so only genuinely tiny/malformed frames fall below 25.
    'fuzz_size_anomaly_min':      300,
    'fuzz_urg_rate_min':          0.01,
    'fuzz_bytes_per_pkt_max':     25,      # bytes  (was 40)

    # ── man_in_the_middle ─────────────────────────────────────────────────────
    # TUNING: tightened symmetry 0.85→0.95 and narrowed byte_ratio band.
    # Also requires src_ttl == 128 (Windows TTL is anomalous in ICS networks
    # which typically use Linux/embedded devices with TTL 64).
    'mitm_symmetry_min':          0.95,    # (was 0.85)
    'mitm_byte_ratio_min':        0.85,    # (was 0.7)
    'mitm_byte_ratio_max':        1.15,    # (was 1.3)
    'mitm_ttl_suspicious':        128,

    # ── command_injection ────────────────────────────────────────────────────
    # Unchanged — thresholds remain well-separated from normal traffic.
    'inject_src_psh_min':         0.5,
    'inject_dst_psh_max':         0.1,
    'inject_packet_rate_min':     15.0,

    # ── time_based_attack ─────────────────────────────────────────────────────
    # TUNING 2: lowered IAT threshold 0.01→0.002. Dataset mean IAT is 0.030s
    # and p25 is 0.001s, so 0.01 caught normal polling. Only sub-2ms IAT with
    # sustained packet counts (>100) and symmetric timing suggests automation.
    # Added iat_ratio check: dst_inter_pkt / src_inter_pkt must be close to 1,
    # confirming machine-driven request/response regularity.
    # TUNING 2c: added time_max_packet_rate = 500 pkt/s. Dataset shows normal
    # ICS polling at ~6,500 pkt/s with sub-2ms IAT. Stuxnet-style attacks are
    # precisely timed but moderate-rate — capping at 500 separates them.
    'time_iat_max':               0.002,   # seconds (was 0.01)
    'time_min_packets':           100,     # (was 50)
    'time_iat_ratio_min':         0.8,     # dst_iat / src_iat lower bound
    'time_iat_ratio_max':         1.2,     # dst_iat / src_iat upper bound
    'time_max_packet_rate':       500,     # pkt/s — NEW: excludes high-rate polling

    # ── replay_attack ────────────────────────────────────────────────────────
    # TUNING 3: tightened ratio windows 0.90–1.10 → 0.995–1.005 and raised
    # symmetry 0.85→0.98. The wide window was matching any balanced flow
    # (1,700 false positives). Genuine replays mirror byte/packet counts
    # almost exactly.
    'replay_byte_ratio_min':      0.995,   # (was 0.90)
    'replay_byte_ratio_max':      1.005,   # (was 1.10)
    'replay_packet_ratio_min':    0.995,   # (was 0.90)
    'replay_packet_ratio_max':    1.005,   # (was 1.10)
    'replay_symmetry_min':        0.98,    # (was 0.85)

    # ── credential_stuffing ───────────────────────────────────────────────────
    # TUNING 7: lowered max_packets 20→15 and added src_syn_rate condition
    # (> 0.3) so random short flows don't trigger. src_fin_rate max is 1.0
    # so 0.1 is a reasonable lower bound for rapid teardowns.
    'cred_syn_imbalance_min':     0.3,
    'cred_max_packets':           15,      # (was 20)
    'cred_fin_rate_min':          0.1,
    'cred_syn_rate_min':          0.3,     # NEW: requires actual SYN activity

    # ── firmware_modification ────────────────────────────────────────────────
    # TUNING 4: lowered bytes_per_pkt 800→70. Dataset max bytes_per_packet is
    # only 74, so 800 was completely unreachable. 70 catches the top of the
    # distribution (above p75=59) indicating unusually large ICS frames.
    # Raised min_bytes 50000→100000 to avoid small uploads.
    # Tightened packet_ratio < 0.15 (was 0.3) — firmware uploads are heavily
    # one-directional.
    'firmware_min_bytes':        100_000,  # (was 50,000)
    'firmware_bytes_per_pkt':     70,      # (was 800)
    'firmware_packet_ratio_max':  0.15,    # (was 0.3)
}
# ──────────────────────────────────────────────────────────────────────────────

# ── CVE enrichment ────────────────────────────────────────────────────────────

_CVE_CACHE: dict = {}       # pattern_name → list of CVE dicts
_CVE_CACHE_TS: dict = {}    # pattern_name → fetch timestamp (epoch)
_CVE_CACHE_TTL = 3600       # re-fetch after 1 hour

PATTERN_TO_NVD_KEYWORDS = {
    'modbus_flooding':      'Modbus denial of service',
    'plc_scanning':         'PLC remote discovery SCADA',
    'unauthorized_write':   'Modbus write unauthorized ICS',
    'man_in_the_middle':    'SCADA man in the middle',
    'replay_attack':        'ICS replay attack Modbus',
    'protocol_fuzzing':     'Modbus protocol fuzzing malformed',
    'command_injection':    'ICS command injection SCADA',
    'time_based_attack':    'ICS timing attack PLC',
    'credential_stuffing':  'SCADA authentication brute force',
    'firmware_modification':'PLC firmware modification unauthorized',
    'modbus_write_dpi':     'Modbus write unauthorized ICS',
}


class ICSAttackPatternLibrary:
    """
    Library of known ICS attack patterns.

    Flow-based detectors (10 patterns) work on the 62 engineered features written
    to data/processed/ics_features.csv by ics_feature_engineer.py.
    No 'dst_port' or raw PCAP data required for flow-based detection.

    Protocol-aware detection (v3.0.0):
    - Pass pcap_path to detect_all_patterns() to enable Modbus DPI via
      ICSProtocolAnalyzer. DPI confirms write function codes at the packet level,
      adding CONFIRMED detections independent of flow-feature thresholds.
    - detect_modbus_writes_from_pcap() is a pure DPI detector — it fires on any
      Modbus write FC regardless of flow statistics.
    - detect_unauthorized_writes() merges flow-feature detections with DPI
      confirmations so each detection carries its source ('flow_features' vs
      'modbus_dpi').
    """

    def __init__(self):
        self.patterns = self._load_patterns()
        self.detected_patterns = []

    # ── Pattern definitions ───────────────────────────────────────────────────

    def _load_patterns(self) -> Dict[str, AttackPattern]:
        return {
            'modbus_flooding': AttackPattern(
                name='Modbus Traffic Flooding',
                mitre_technique='T0823 - Denial of Service',
                description='Overwhelming Modbus/TCP service with excessive requests',
                indicators={
                    'src_packet_rate': '> 100 pkt/s',
                    'total_packets': f'> {THRESHOLDS["flood_min_packets"]}',
                },
                severity='HIGH',
                mitigation='Implement rate limiting on Modbus gateway, deploy IPS rules',
                real_world_example='Used in BlackEnergy attacks against Ukrainian power grid'
            ),
            'plc_scanning': AttackPattern(
                name='PLC Network Scanning',
                mitre_technique='T0846 - Remote System Discovery',
                description='Sequential scanning of ICS ports to discover PLCs/RTUs',
                indicators={
                    'packet_ratio': f'< {THRESHOLDS["scan_packet_ratio_max"]} (no responses)',
                    'dst_packets': f'< {THRESHOLDS["scan_max_dst_packets"]}',
                },
                severity='CRITICAL',
                mitigation='Deploy network segmentation, enable port-based access control',
                real_world_example='Pre-attack reconnaissance in Stuxnet campaign'
            ),
            'unauthorized_write': AttackPattern(
                name='Unauthorized PLC Write Command',
                mitre_technique='T0836 - Modify Control Logic',
                description='Unauthorized write to PLC registers or memory',
                indicators={
                    'src_psh_rate': f'> {THRESHOLDS["write_psh_rate_min"]}',
                    'src_packet_rate': f'> {THRESHOLDS["write_packet_rate_min"]} pkt/s',
                    'src_rst_rate': '> 0 (connection resets present)',
                },
                severity='CRITICAL',
                mitigation='Enforce write access controls, implement change management',
                real_world_example='Core technique in Stuxnet and Triton attacks'
            ),
            'protocol_fuzzing': AttackPattern(
                name='Protocol Fuzzing / Malformed Packets',
                mitre_technique='T0851 - Protocol Exploitation',
                description='Sending malformed protocol packets to crash ICS systems',
                indicators={
                    'packet_size_anomaly': f'> {THRESHOLDS["fuzz_size_anomaly_min"]}',
                    'src_urg_rate': f'> {THRESHOLDS["fuzz_urg_rate_min"]}',
                    'bytes_per_packet': f'< {THRESHOLDS["fuzz_bytes_per_pkt_max"]} (tiny packets)',
                },
                severity='HIGH',
                mitigation='Update firmware, deploy protocol-aware firewall',
                real_world_example='ICS vulnerability discovery technique'
            ),
            'man_in_the_middle': AttackPattern(
                name='Man-in-the-Middle Attack',
                mitre_technique='T0830 - Adversary-in-the-Middle',
                description='Intercepting and modifying ICS communications',
                indicators={
                    'traffic_symmetry': f'> {THRESHOLDS["mitm_symmetry_min"]} (relay-like)',
                    'byte_ratio': f'{THRESHOLDS["mitm_byte_ratio_min"]}–{THRESHOLDS["mitm_byte_ratio_max"]}',
                    'src_ttl': f'= {THRESHOLDS["mitm_ttl_suspicious"]} (Windows TTL in ICS)',
                },
                severity='CRITICAL',
                mitigation='Enable TLS/SSL, deploy network monitoring, use certificates',
                real_world_example='Used in advanced APT campaigns against ICS'
            ),
            'command_injection': AttackPattern(
                name='Command Injection',
                mitre_technique='T0871 - Execution through API',
                description='Injecting malicious commands through ICS protocols',
                indicators={
                    'src_psh_rate': f'> {THRESHOLDS["inject_src_psh_min"]}',
                    'dst_psh_rate': f'< {THRESHOLDS["inject_dst_psh_max"]} (controller silent)',
                    'src_packet_rate': f'> {THRESHOLDS["inject_packet_rate_min"]} pkt/s',
                },
                severity='CRITICAL',
                mitigation='Input validation, command whitelisting, audit logging',
                real_world_example='Triton safety system attack'
            ),
            'time_based_attack': AttackPattern(
                name='Time-Based Logic Attack',
                mitre_technique='T0889 - Modify Program',
                description='Exploiting timing in control logic (à la Stuxnet)',
                indicators={
                    'src_inter_packet_avg': f'< {THRESHOLDS["time_iat_max"]}s (machine-precise)',
                    'total_packets': f'> {THRESHOLDS["time_min_packets"]}',
                },
                severity='CRITICAL',
                mitigation='Monitor process timing, detect logic modifications',
                real_world_example='Stuxnet centrifuge attack'
            ),
            'replay_attack': AttackPattern(
                name='Replay Attack',
                mitre_technique='T0843 - Replay Attack',
                description='Replaying captured legitimate commands to manipulate state',
                indicators={
                    'byte_ratio': f'≈ 1.0 ({THRESHOLDS["replay_byte_ratio_min"]}–{THRESHOLDS["replay_byte_ratio_max"]})',
                    'packet_ratio': f'≈ 1.0 ({THRESHOLDS["replay_packet_ratio_min"]}–{THRESHOLDS["replay_packet_ratio_max"]})',
                    'traffic_symmetry': f'> {THRESHOLDS["replay_symmetry_min"]}',
                },
                severity='HIGH',
                mitigation='Implement nonces, timestamps, sequence numbers',
                real_world_example='Common in unsecured SCADA systems'
            ),
            'credential_stuffing': AttackPattern(
                name='Credential Stuffing',
                mitre_technique='T0859 - Valid Accounts',
                description='Using stolen credentials to access ICS HMI/engineering workstations',
                indicators={
                    'syn_ack_imbalance': f'> {THRESHOLDS["cred_syn_imbalance_min"]}',
                    'total_packets': f'< {THRESHOLDS["cred_max_packets"]} (short burst)',
                    'src_fin_rate': f'> {THRESHOLDS["cred_fin_rate_min"]} (rapid teardown)',
                },
                severity='HIGH',
                mitigation='Enforce MFA, monitor authentication logs, lockout policies',
                real_world_example='Common initial access vector in ICS breaches'
            ),
            'firmware_modification': AttackPattern(
                name='Firmware Modification',
                mitre_technique='T0857 - System Firmware',
                description='Unauthorized upload of modified firmware to PLC/RTU',
                indicators={
                    'total_bytes': f'> {THRESHOLDS["firmware_min_bytes"]} (large upload)',
                    'bytes_per_packet': f'> {THRESHOLDS["firmware_bytes_per_pkt"]}',
                    'packet_ratio': f'< {THRESHOLDS["firmware_packet_ratio_max"]} (one-way upload)',
                },
                severity='CRITICAL',
                mitigation='Code signing, firmware validation, secure boot',
                real_world_example='BlackEnergy firmware attacks on Siemens devices'
            ),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_required_columns(self,
                                 df: pd.DataFrame,
                                 required: List[str],
                                 detector_name: str) -> bool:
        """Return True if all required columns are present, else warn and return False."""
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(
                f"[{detector_name}] Skipping — missing columns: {missing}. "
                f"Ensure ics_feature_engineer.py has run."
            )
            return False
        return True

    def _confidence(self, value: float, low: float, high: float) -> float:
        """Linearly scale value in [low, high] → confidence in [0.5, 1.0]."""
        if high <= low:
            return 0.5
        return float(np.clip(0.5 + 0.5 * (value - low) / (high - low), 0.5, 1.0))

    # ── Detectors ─────────────────────────────────────────────────────────────

    def detect_modbus_flooding(self, df: pd.DataFrame) -> List[Dict]:
        """
        High packet rate + high total packets.
        Proxy for Modbus/TCP flooding when dst_port is unavailable.
        """
        required = ['src_packet_rate', 'total_packets', 'flow_duration']
        if not self._check_required_columns(df, required, 'modbus_flooding'):
            return []

        mask = (
            (df['src_packet_rate'] > THRESHOLDS['flood_packet_rate']) &
            (df['total_packets'] > THRESHOLDS['flood_min_packets'])
        )
        detections = []
        for _, row in df[mask].iterrows():
            detections.append({
                'pattern': 'modbus_flooding',
                'severity': 'HIGH',
                'confidence': self._confidence(
                    row['src_packet_rate'],
                    THRESHOLDS['flood_packet_rate'],
                    THRESHOLDS['flood_packet_rate'] * 3
                ),
                'details': {
                    'src_packet_rate': round(row['src_packet_rate'], 2),
                    'total_packets': int(row['total_packets']),
                    'flow_duration_s': round(row['flow_duration'], 4),
                }
            })
        return detections

    def detect_plc_scanning(self, df: pd.DataFrame, **_) -> List[Dict]:
        """
        Near-zero responses from target (dst_packets ≈ 0, packet_ratio tiny).
        Classic port-scan signature: attacker sends, nothing comes back.
        """
        required = ['packet_ratio', 'dst_packets', 'byte_ratio']
        if not self._check_required_columns(df, required, 'plc_scanning'):
            return []

        mask = (
            (df['packet_ratio'] < THRESHOLDS['scan_packet_ratio_max']) &
            (df['dst_packets'] <= THRESHOLDS['scan_max_dst_packets']) &
            (df['byte_ratio'] < THRESHOLDS['scan_byte_ratio_max'])
        )
        detections = []
        for _, row in df[mask].iterrows():
            detections.append({
                'pattern': 'plc_scanning',
                'severity': 'CRITICAL',
                'confidence': self._confidence(
                    1.0 / max(row['packet_ratio'], 1e-6),
                    1.0 / THRESHOLDS['scan_packet_ratio_max'],
                    200.0
                ),
                'details': {
                    'packet_ratio': round(row['packet_ratio'], 4),
                    'dst_packets': int(row['dst_packets']),
                    'byte_ratio': round(row['byte_ratio'], 4),
                }
            })
        return detections

    def detect_unauthorized_writes(self,
                                    df: pd.DataFrame,
                                    protocol_analysis: List[Dict] = None) -> List[Dict]:
        """
        Detects unauthorized PLC write commands via two complementary paths:

        PATH 1 — Flow features (statistical):
          High PSH rate (command bursts) + high packet rate + RST presence.
          Catches write activity observable at the flow-aggregation level.

        PATH 2 — Modbus DPI (protocol-aware, v3.0.0):
          Uses confirmed Modbus write function codes from ICSProtocolAnalyzer.
          Write FCs: 05 (Write Single Coil), 06 (Write Single Register),
                     0F (Write Multiple Coils), 10 (Write Multiple Registers),
                     15 (Write File Record), 16 (Mask Write Register).
          These are ground-truth evidence — no threshold tuning required.
          Severity escalates to CRITICAL for FC 0x10 (bulk register write) or
          when the analyzer flags a critical/protected register address.

        Each detection carries a 'detection_source' key:
          'flow_features' — triggered by PATH 1
          'modbus_dpi'    — confirmed by PATH 2 (higher confidence: 0.95)
          'both'          — triggered by PATH 1 AND confirmed by PATH 2
        """
        detections = []

        # ── PATH 1: flow-feature detection ───────────────────────────────────
        required = ['src_psh_rate', 'src_packet_rate', 'src_rst_rate']
        if self._check_required_columns(df, required, 'unauthorized_write'):
            mask = (
                (df['src_psh_rate'] > THRESHOLDS['write_psh_rate_min']) &
                (df['src_packet_rate'] > THRESHOLDS['write_packet_rate_min']) &
                (df['src_rst_rate'] > 0)
            )
            for _, row in df[mask].iterrows():
                detections.append({
                    'pattern': 'unauthorized_write',
                    'severity': 'CRITICAL',
                    'detection_source': 'flow_features',
                    'confidence': self._confidence(
                        row['src_psh_rate'],
                        THRESHOLDS['write_psh_rate_min'], 1.0
                    ),
                    'details': {
                        'src_psh_rate': round(row['src_psh_rate'], 3),
                        'src_packet_rate': round(row['src_packet_rate'], 2),
                        'src_rst_rate': round(row['src_rst_rate'], 4),
                    }
                })

        # ── PATH 2: Modbus DPI confirmation ───────────────────────────────────
        # MODBUS_WRITE_FCS: function codes that modify PLC state
        MODBUS_WRITE_FCS = {0x05, 0x06, 0x0F, 0x10, 0x15, 0x16}

        if protocol_analysis:
            write_events = [
                a for a in protocol_analysis
                if (
                    a.get('protocol') == 'Modbus/TCP'
                    and a.get('function_code') in MODBUS_WRITE_FCS
                )
            ]

            for event in write_events:
                fc = event.get('function_code', 0)
                fc_name = event.get('function_name', f'FC 0x{fc:02X}')
                details = event.get('details', {})

                # FC 0x10 (Write Multiple Registers) or critical register = CRITICAL
                # FC 0x06 (Write Single Register) = HIGH unless critical register
                if fc == 0x10 or event.get('severity') == 'CRITICAL':
                    severity = 'CRITICAL'
                else:
                    severity = 'HIGH'

                dpi_detection = {
                    'pattern': 'unauthorized_write',
                    'severity': severity,
                    'detection_source': 'modbus_dpi',
                    'confidence': 0.95,   # DPI is ground truth — high base confidence
                    'details': {
                        'function_code': f'0x{fc:02X}',
                        'function_name': fc_name,
                        'unit_id': event.get('unit_id'),
                        'transaction_id': event.get('transaction_id'),
                        'alert': event.get('alert', ''),
                    }
                }

                # Add register address/value if available
                if 'register_address' in details:
                    dpi_detection['details']['register_address'] = details['register_address']
                if 'register_value' in details:
                    dpi_detection['details']['register_value'] = details['register_value']
                if 'start_address' in details:
                    dpi_detection['details']['start_address'] = details['start_address']
                if 'register_count' in details:
                    dpi_detection['details']['register_count'] = details['register_count']

                # Mark flow-feature detections that are also DPI-confirmed
                # (unit_id match is best we can do without per-flow port info)
                unit_id = event.get('unit_id')
                for det in detections:
                    if (det['detection_source'] == 'flow_features'
                            and unit_id is not None):
                        det['detection_source'] = 'both'
                        det['dpi_confirmation'] = {
                            'function_code': f'0x{fc:02X}',
                            'function_name': fc_name,
                            'unit_id': unit_id,
                        }

                detections.append(dpi_detection)

            if write_events:
                logger.info(
                    f"[unauthorized_write] Modbus DPI confirmed {len(write_events)} "
                    f"write FC events "
                    f"({sum(1 for e in write_events if e.get('function_code') == 0x10)} "
                    f"bulk-register FC 0x10)"
                )

        return detections

    def detect_modbus_writes_from_pcap(self,
                                        protocol_analysis: List[Dict]) -> List[Dict]:
        """
        Pure DPI detector — fires on any Modbus write function code found in
        PCAP analysis, independent of flow-feature thresholds.

        This is the protocol-layer ground truth path. A Modbus write FC is a write
        FC regardless of how the flow statistics look. This matters for low-volume
        or single-packet write commands that fall below flow-feature thresholds
        (e.g. a single FC 0x06 write that doesn't produce enough traffic to trigger
        detect_unauthorized_writes() via PSH rate / packet rate).

        FC severity mapping:
          CRITICAL: FC 0x10 (Write Multiple Registers) — bulk PLC state change
                    FC 0x0F (Write Multiple Coils)     — bulk output state change
                    Any FC to a protected/critical register range
          HIGH:     FC 0x05 (Write Single Coil)
                    FC 0x06 (Write Single Register)
                    FC 0x15 (Write File Record)
                    FC 0x16 (Mask Write Register)

        Returns list of detections with detection_source='modbus_dpi'.
        Empty list if protocol_analysis is None or has no write events.
        """
        MODBUS_WRITE_FCS = {0x05, 0x06, 0x0F, 0x10, 0x15, 0x16}
        BULK_WRITE_FCS   = {0x0F, 0x10}   # always CRITICAL

        if not protocol_analysis:
            return []

        detections = []
        for event in protocol_analysis:
            if event.get('protocol') != 'Modbus/TCP':
                continue
            fc = event.get('function_code')
            if fc not in MODBUS_WRITE_FCS:
                continue

            fc_name = event.get('function_name', f'FC 0x{fc:02X}')
            details = event.get('details', {})

            if fc in BULK_WRITE_FCS or event.get('severity') == 'CRITICAL':
                severity = 'CRITICAL'
            else:
                severity = 'HIGH'

            det = {
                'pattern': 'modbus_write_dpi',
                'severity': severity,
                'detection_source': 'modbus_dpi',
                'confidence': 0.95,
                'details': {
                    'function_code': f'0x{fc:02X}',
                    'function_name': fc_name,
                    'unit_id': event.get('unit_id'),
                    'transaction_id': event.get('transaction_id'),
                    'alert': event.get('alert', ''),
                    'is_bulk_write': fc in BULK_WRITE_FCS,
                }
            }

            # Forward register address fields if present
            for key in ('register_address', 'register_value',
                        'start_address', 'register_count', 'byte_count'):
                if key in details:
                    det['details'][key] = details[key]

            detections.append(det)

        if detections:
            logger.info(
                f"[modbus_write_dpi] {len(detections)} Modbus write FC detections "
                f"({sum(1 for d in detections if d['severity'] == 'CRITICAL')} CRITICAL)"
            )

        return detections

    def detect_protocol_fuzzing(self, df: pd.DataFrame) -> List[Dict]:
        """
        Large packet_size_anomaly (malformed sizes) + URG flags + tiny packets.
        """
        required = ['packet_size_anomaly', 'src_urg_rate', 'bytes_per_packet']
        if not self._check_required_columns(df, required, 'protocol_fuzzing'):
            return []

        mask = (
            (df['packet_size_anomaly'] > THRESHOLDS['fuzz_size_anomaly_min']) &
            (
                (df['src_urg_rate'] > THRESHOLDS['fuzz_urg_rate_min']) |
                (df['bytes_per_packet'] < THRESHOLDS['fuzz_bytes_per_pkt_max'])
            )
        )
        detections = []
        for _, row in df[mask].iterrows():
            detections.append({
                'pattern': 'protocol_fuzzing',
                'severity': 'HIGH',
                'confidence': self._confidence(
                    row['packet_size_anomaly'],
                    THRESHOLDS['fuzz_size_anomaly_min'], 800.0
                ),
                'details': {
                    'packet_size_anomaly': round(row['packet_size_anomaly'], 2),
                    'src_urg_rate': round(row['src_urg_rate'], 4),
                    'bytes_per_packet': round(row['bytes_per_packet'], 2),
                }
            })
        return detections

    def detect_man_in_the_middle(self, df: pd.DataFrame) -> List[Dict]:
        """
        Relay-like traffic: high symmetry + balanced byte ratio + Windows TTL.
        A relay node forwards traffic almost perfectly symmetrically.
        """
        required = ['traffic_symmetry', 'byte_ratio', 'src_ttl']
        if not self._check_required_columns(df, required, 'man_in_the_middle'):
            return []

        mask = (
            (df['traffic_symmetry'] > THRESHOLDS['mitm_symmetry_min']) &
            (df['byte_ratio'] >= THRESHOLDS['mitm_byte_ratio_min']) &
            (df['byte_ratio'] <= THRESHOLDS['mitm_byte_ratio_max']) &
            (df['src_ttl'] == THRESHOLDS['mitm_ttl_suspicious'])
        )
        detections = []
        for _, row in df[mask].iterrows():
            detections.append({
                'pattern': 'man_in_the_middle',
                'severity': 'CRITICAL',
                'confidence': self._confidence(
                    row['traffic_symmetry'],
                    THRESHOLDS['mitm_symmetry_min'], 1.0
                ),
                'details': {
                    'traffic_symmetry': round(row['traffic_symmetry'], 4),
                    'byte_ratio': round(row['byte_ratio'], 4),
                    'src_ttl': int(row['src_ttl']),
                }
            })
        return detections

    def detect_command_injection(self, df: pd.DataFrame) -> List[Dict]:
        """
        Attacker pushes many commands (high src_psh_rate) but controller
        barely responds (low dst_psh_rate) — asymmetric PSH pattern.
        """
        required = ['src_psh_rate', 'dst_psh_rate', 'src_packet_rate']
        if not self._check_required_columns(df, required, 'command_injection'):
            return []

        mask = (
            (df['src_psh_rate'] > THRESHOLDS['inject_src_psh_min']) &
            (df['dst_psh_rate'] < THRESHOLDS['inject_dst_psh_max']) &
            (df['src_packet_rate'] > THRESHOLDS['inject_packet_rate_min'])
        )
        detections = []
        for _, row in df[mask].iterrows():
            detections.append({
                'pattern': 'command_injection',
                'severity': 'CRITICAL',
                'confidence': self._confidence(
                    row['src_psh_rate'] - row['dst_psh_rate'],
                    THRESHOLDS['inject_src_psh_min'], 1.0
                ),
                'details': {
                    'src_psh_rate': round(row['src_psh_rate'], 3),
                    'dst_psh_rate': round(row['dst_psh_rate'], 3),
                    'src_packet_rate': round(row['src_packet_rate'], 2),
                }
            })
        return detections

    def detect_time_based_attack(self, df: pd.DataFrame) -> List[Dict]:
        """
        Machine-precision timing: sub-2ms IAT on sustained flows WITH symmetric
        request/response timing (dst IAT mirrors src IAT).

        TUNING 2: threshold lowered 0.01→0.002s and min_packets raised 50→100.
        Dataset mean IAT is 0.030s so 0.01 caught normal fast ICS polling.
        The IAT ratio condition (dst_iat / src_iat ≈ 1.0) confirms the timing
        is machine-driven rather than coincidentally fast.

        TUNING 2b: added src_packet_rate < flood_packet_rate exclusion to
        prevent double-counting flows already flagged as modbus_flooding.
        High-rate simulation flows have IAT≈0 and ratio=1.0 exactly, which
        would otherwise trigger both detectors on the same flows.
        """
        required = ['src_inter_packet_avg', 'dst_inter_packet_avg',
                    'total_packets', 'src_packet_rate']
        if not self._check_required_columns(df, required, 'time_based_attack'):
            return []

        # Compute dst/src IAT ratio safely (avoid div-by-zero)
        src_iat = df['src_inter_packet_avg'].clip(lower=1e-9)
        iat_ratio = df['dst_inter_packet_avg'] / src_iat

        mask = (
            (df['src_inter_packet_avg'] > 0) &
            (df['src_inter_packet_avg'] < THRESHOLDS['time_iat_max']) &
            (df['total_packets'] > THRESHOLDS['time_min_packets']) &
            (iat_ratio >= THRESHOLDS['time_iat_ratio_min']) &
            (iat_ratio <= THRESHOLDS['time_iat_ratio_max']) &
            # exclude flows already captured by modbus_flooding detector
            (df['src_packet_rate'] < THRESHOLDS['flood_packet_rate']) &
            # TUNING 2c: exclude high-rate ICS polling (normal simulation traffic).
            # Stuxnet-style timing attacks are precisely timed but NOT high-throughput.
            # Dataset shows legitimate polling at ~6,500 pkt/s with sub-2ms IAT --
            # capping at 500 pkt/s separates suspicious automation from normal polling.
            (df['src_packet_rate'] < THRESHOLDS['time_max_packet_rate'])
        )
        detections = []
        for idx, row in df[mask].iterrows():
            detections.append({
                'pattern': 'time_based_attack',
                'severity': 'CRITICAL',
                'confidence': self._confidence(
                    THRESHOLDS['time_iat_max'] - row['src_inter_packet_avg'],
                    0, THRESHOLDS['time_iat_max']
                ),
                'details': {
                    'src_inter_packet_avg_s': round(row['src_inter_packet_avg'], 6),
                    'dst_inter_packet_avg_s': round(row['dst_inter_packet_avg'], 6),
                    'iat_ratio': round(iat_ratio[idx], 3),
                    'total_packets': int(row['total_packets']),
                    'src_packet_rate': round(row['src_packet_rate'], 2),
                }
            })
        return detections

    def detect_replay_attack(self, df: pd.DataFrame) -> List[Dict]:
        """
        Replayed flows mirror legitimate ones: byte_ratio ≈ 1.0,
        packet_ratio ≈ 1.0, and high traffic symmetry all together.
        """
        required = ['byte_ratio', 'packet_ratio', 'traffic_symmetry']
        if not self._check_required_columns(df, required, 'replay_attack'):
            return []

        mask = (
            (df['byte_ratio'] >= THRESHOLDS['replay_byte_ratio_min']) &
            (df['byte_ratio'] <= THRESHOLDS['replay_byte_ratio_max']) &
            (df['packet_ratio'] >= THRESHOLDS['replay_packet_ratio_min']) &
            (df['packet_ratio'] <= THRESHOLDS['replay_packet_ratio_max']) &
            (df['traffic_symmetry'] > THRESHOLDS['replay_symmetry_min'])
        )
        detections = []
        for _, row in df[mask].iterrows():
            detections.append({
                'pattern': 'replay_attack',
                'severity': 'HIGH',
                'confidence': self._confidence(
                    row['traffic_symmetry'],
                    THRESHOLDS['replay_symmetry_min'], 1.0
                ),
                'details': {
                    'byte_ratio': round(row['byte_ratio'], 4),
                    'packet_ratio': round(row['packet_ratio'], 4),
                    'traffic_symmetry': round(row['traffic_symmetry'], 4),
                }
            })
        return detections

    def detect_credential_stuffing(self, df: pd.DataFrame) -> List[Dict]:
        """
        Many SYNs, few ACKs, short flows, rapid FIN teardowns.

        TUNING 7: lowered max_packets 20→15 and added src_syn_rate > 0.3
        so random short flows without SYN activity don't trigger.
        """
        required = ['syn_ack_imbalance', 'total_packets', 'src_fin_rate', 'src_syn_rate']
        if not self._check_required_columns(df, required, 'credential_stuffing'):
            return []

        mask = (
            (df['syn_ack_imbalance'] > THRESHOLDS['cred_syn_imbalance_min']) &
            (df['total_packets'] < THRESHOLDS['cred_max_packets']) &
            (df['src_fin_rate'] > THRESHOLDS['cred_fin_rate_min']) &
            (df['src_syn_rate'] > THRESHOLDS['cred_syn_rate_min'])   # NEW
        )
        detections = []
        for _, row in df[mask].iterrows():
            detections.append({
                'pattern': 'credential_stuffing',
                'severity': 'HIGH',
                'confidence': self._confidence(
                    row['syn_ack_imbalance'],
                    THRESHOLDS['cred_syn_imbalance_min'], 1.0
                ),
                'details': {
                    'syn_ack_imbalance': round(row['syn_ack_imbalance'], 3),
                    'src_syn_rate': round(row['src_syn_rate'], 3),
                    'total_packets': int(row['total_packets']),
                    'src_fin_rate': round(row['src_fin_rate'], 3),
                }
            })
        return detections

    def detect_firmware_modification(self, df: pd.DataFrame) -> List[Dict]:
        """
        Large one-way data transfer: big total_bytes, large bytes_per_packet,
        heavily asymmetric packet_ratio (attacker uploading firmware).
        """
        required = ['total_bytes', 'bytes_per_packet', 'packet_ratio']
        if not self._check_required_columns(df, required, 'firmware_modification'):
            return []

        mask = (
            (df['total_bytes'] > THRESHOLDS['firmware_min_bytes']) &
            (df['bytes_per_packet'] > THRESHOLDS['firmware_bytes_per_pkt']) &
            (df['packet_ratio'] < THRESHOLDS['firmware_packet_ratio_max'])
        )
        detections = []
        for _, row in df[mask].iterrows():
            detections.append({
                'pattern': 'firmware_modification',
                'severity': 'CRITICAL',
                'confidence': self._confidence(
                    row['total_bytes'],
                    THRESHOLDS['firmware_min_bytes'],
                    THRESHOLDS['firmware_min_bytes'] * 10
                ),
                'details': {
                    'total_bytes': int(row['total_bytes']),
                    'bytes_per_packet': round(row['bytes_per_packet'], 2),
                    'packet_ratio': round(row['packet_ratio'], 4),
                }
            })
        return detections

    # ── Orchestrator ──────────────────────────────────────────────────────────

    def detect_all_patterns(self,
                             flows_df: pd.DataFrame,
                             protocol_analysis: List[Dict] = None,
                             pcap_path: Optional[str] = None) -> Dict:
        """
        Run all 10 flow-based detectors + optional Modbus DPI and aggregate results.

        Args:
            flows_df:         DataFrame of engineered features (62 columns).
            protocol_analysis: Pre-computed ICSProtocolAnalyzer results. If provided,
                               used directly (pcap_path is ignored).
            pcap_path:        Path to a PCAP file. When provided and protocol_analysis
                              is None, ICSProtocolAnalyzer is run on it automatically.
                              Requires Scapy. If Scapy is not installed, a warning is
                              logged and detection continues in flow-only mode.

        Returns dict with keys:
            total_detections       — count across all patterns
            patterns_found         — list of pattern names with ≥1 detection
            detections_by_pattern  — {pattern_name: [detection, ...]}
            severity_breakdown     — {CRITICAL/HIGH/MEDIUM/LOW: count}
            protocol_aware         — bool: True if DPI results were used
            dpi_summary            — present only when protocol_aware=True
        """
        # ── Step 1: Run DPI if pcap_path provided and no pre-computed results ──
        dpi_summary = None
        if protocol_analysis is None and pcap_path is not None:
            try:
                import sys
                import os
                # Support running from project root or src/detection/
                for candidate in ['.', './src', os.path.dirname(__file__) + '/../..']:
                    if candidate not in sys.path:
                        sys.path.insert(0, candidate)
                from src.ics_protocol_analyzer import ICSProtocolAnalyzer

                logger.info(f"Running Modbus DPI on: {pcap_path}")
                analyzer = ICSProtocolAnalyzer()
                protocol_analysis = analyzer.analyze_pcap_file(pcap_path)
                stats = analyzer.get_statistics()
                dpi_summary = {
                    'pcap_path': pcap_path,
                    'total_ics_packets': stats['total_packets'],
                    'protocols_detected': stats['protocols_detected'],
                    'high_severity_alerts': stats['alerts'],
                }
                logger.info(
                    f"DPI complete — {len(protocol_analysis)} ICS packets, "
                    f"{stats['alerts']} high-severity alerts"
                )
            except ImportError:
                logger.warning(
                    "[detect_all_patterns] Scapy not installed — "
                    "DPI skipped, running in flow-only mode. "
                    "Install with: pip install scapy"
                )
                protocol_analysis = None
            except Exception as e:
                logger.warning(
                    f"[detect_all_patterns] DPI failed ({e}) — "
                    "continuing in flow-only mode."
                )
                protocol_analysis = None

        protocol_aware = protocol_analysis is not None

        logger.info(
            f"Scanning for known attack patterns "
            f"({'protocol-aware' if protocol_aware else 'flow-only'} mode)..."
        )

        # ── Step 2: Run all 10 flow-based detectors ───────────────────────────
        all_detections = {
            'modbus_flooding':       self.detect_modbus_flooding(flows_df),
            'plc_scanning':          self.detect_plc_scanning(flows_df),
            'unauthorized_write':    self.detect_unauthorized_writes(
                                         flows_df, protocol_analysis),
            'protocol_fuzzing':      self.detect_protocol_fuzzing(flows_df),
            'man_in_the_middle':     self.detect_man_in_the_middle(flows_df),
            'command_injection':     self.detect_command_injection(flows_df),
            'time_based_attack':     self.detect_time_based_attack(flows_df),
            'replay_attack':         self.detect_replay_attack(flows_df),
            'credential_stuffing':   self.detect_credential_stuffing(flows_df),
            'firmware_modification': self.detect_firmware_modification(flows_df),
        }

        # ── Step 3: Pure DPI detector (Modbus write FCs below flow thresholds) ─
        if protocol_aware:
            modbus_dpi_detections = self.detect_modbus_writes_from_pcap(protocol_analysis)
            # Only include events not already covered by unauthorized_write
            # (avoid double-counting DPI events injected into unauthorized_write)
            existing_dpi_txids = {
                d['details'].get('transaction_id')
                for d in all_detections['unauthorized_write']
                if d.get('detection_source') in ('modbus_dpi', 'both')
                and d['details'].get('transaction_id') is not None
            }
            novel_dpi = [
                d for d in modbus_dpi_detections
                if d['details'].get('transaction_id') not in existing_dpi_txids
            ]
            all_detections['modbus_write_dpi'] = novel_dpi
        else:
            all_detections['modbus_write_dpi'] = []

        # ── Step 4: Aggregate ─────────────────────────────────────────────────
        total_detections = sum(len(v) for v in all_detections.values())

        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for pattern_detections in all_detections.values():
            for detection in pattern_detections:
                sev = detection.get('severity', 'LOW')
                if sev in severity_counts:
                    severity_counts[sev] += 1

        results = {
            'total_detections':      total_detections,
            'patterns_found':        [k for k, v in all_detections.items() if v],
            'detections_by_pattern': all_detections,
            'severity_breakdown':    severity_counts,
            'protocol_aware':        protocol_aware,
        }

        if dpi_summary:
            results['dpi_summary'] = dpi_summary

        pattern_count = len([k for k, v in all_detections.items()
                             if v and k != 'modbus_write_dpi'])
        logger.info(f"✅ Pattern detection complete")
        logger.info(f"   Mode             : {'protocol-aware (DPI + flow)' if protocol_aware else 'flow-only'}")
        logger.info(f"   Total detections : {total_detections}")
        logger.info(f"   Critical         : {severity_counts['CRITICAL']}")
        logger.info(f"   High             : {severity_counts['HIGH']}")
        logger.info(f"   Patterns active  : {pattern_count}/10"
                    + (f" + {len(all_detections['modbus_write_dpi'])} DPI-only"
                       if all_detections['modbus_write_dpi'] else ""))

        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_pattern_info(self, pattern_name: str) -> Optional[AttackPattern]:
        return self.patterns.get(pattern_name)

    def list_all_patterns(self) -> List[str]:
        return list(self.patterns.keys())

    # ── CVE enrichment ────────────────────────────────────────────────────────

    def fetch_cves(self, pattern_name: str, max_results: int = 5) -> list:
        """
        Fetch CVEs from NVD for a given attack pattern.
        Results are cached for _CVE_CACHE_TTL seconds to respect NVD rate limits
        (5 requests / 30 seconds without API key).

        Returns list of dicts: [{cve_id, cvss_score, description}]
        """
        now = time.time()
        if pattern_name in _CVE_CACHE and (now - _CVE_CACHE_TS.get(pattern_name, 0)) < _CVE_CACHE_TTL:
            return _CVE_CACHE[pattern_name]

        keyword = PATTERN_TO_NVD_KEYWORDS.get(pattern_name)
        if not keyword:
            return []

        try:
            params = urllib.parse.urlencode({
                'keywordSearch': keyword,
                'resultsPerPage': max_results,
            })
            url = f'https://services.nvd.nist.gov/rest/json/cves/2.0?{params}'
            req = urllib.request.Request(url, headers={'User-Agent': 'ICS-Detector/3.0'})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())

            results = []
            for item in data.get('vulnerabilities', []):
                cve = item.get('cve', {})
                cve_id = cve.get('id', '')
                metrics = cve.get('metrics', {})
                score = None
                for key in ('cvssMetricV31', 'cvssMetricV30', 'cvssMetricV2'):
                    entries = metrics.get(key, [])
                    if entries:
                        score = entries[0].get('cvssData', {}).get('baseScore')
                        break
                desc_list = cve.get('descriptions', [])
                desc = next((d['value'] for d in desc_list if d.get('lang') == 'en'), '')
                results.append({
                    'cve_id': cve_id,
                    'cvss_score': score,
                    'description': desc[:200],
                })

            _CVE_CACHE[pattern_name] = results
            _CVE_CACHE_TS[pattern_name] = now
            logger.info(f"[CVE] {pattern_name}: fetched {len(results)} CVEs from NVD")
            return results

        except Exception as e:
            logger.warning(f"[CVE] NVD fetch failed for '{pattern_name}': {e}")
            _CVE_CACHE[pattern_name] = []
            _CVE_CACHE_TS[pattern_name] = now
            return []

    def enrich_with_cves(self, detection_results: dict) -> dict:
        """
        Add 'related_cves' to each detected pattern in detection_results.
        Mutates detection_results in-place and returns it.

        Usage:
            results = library.detect_all_patterns(flows_df)
            results = library.enrich_with_cves(results)
        """
        for pattern_name in detection_results.get('patterns_found', []):
            cves = self.fetch_cves(pattern_name)
            if cves:
                detection_results['detections_by_pattern'][pattern_name + '_cves'] = cves
                # Also attach directly to the top-level result dict for easy access
                detection_results.setdefault('cve_enrichment', {})[pattern_name] = cves
        return detection_results

    def get_cve_summary(self, detection_results: dict) -> str:
        """Return a formatted CVE summary string for terminal output / reports."""
        enrichment = detection_results.get('cve_enrichment', {})
        if not enrichment:
            return "No CVE enrichment available."
        lines = ["\n── CVE Enrichment ──────────────────────────────────────────────────────────"]
        for pattern, cves in enrichment.items():
            lines.append(f"\n  {pattern.upper()}")
            for c in cves:
                score_str = f"CVSS {c['cvss_score']}" if c['cvss_score'] else "CVSS N/A"
                lines.append(f"    [{score_str}] {c['cve_id']}: {c['description'][:120]}")
        return "\n".join(lines)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def generate_threat_report(self, detection_results: Dict) -> str:
        report = []
        report.append("=" * 80)
        report.append("ICS ATTACK PATTERN DETECTION REPORT")
        report.append("=" * 80)

        protocol_aware = detection_results.get('protocol_aware', False)
        mode_str = "Protocol-Aware (DPI + Flow)" if protocol_aware else "Flow-Only"
        report.append(f"\nDetection Mode  : {mode_str}")
        report.append(f"Total Detections: {detection_results['total_detections']}")

        # Pattern count: exclude modbus_write_dpi (it's DPI-only, not one of the 10 MITRE patterns)
        flow_patterns_found = [p for p in detection_results['patterns_found']
                               if p != 'modbus_write_dpi']
        report.append(f"Active Patterns : {len(flow_patterns_found)}/10"
                      + (" + Modbus DPI" if 'modbus_write_dpi' in detection_results['patterns_found'] else ""))

        if protocol_aware and 'dpi_summary' in detection_results:
            s = detection_results['dpi_summary']
            report.append(f"\nDPI Summary:")
            report.append(f"   PCAP            : {s.get('pcap_path', 'N/A')}")
            report.append(f"   ICS packets     : {s.get('total_ics_packets', 0)}")
            report.append(f"   High alerts     : {s.get('high_severity_alerts', 0)}")
            if s.get('protocols_detected'):
                for proto, count in s['protocols_detected'].items():
                    report.append(f"   {proto:<20}: {count} packets")

        if detection_results['total_detections'] == 0:
            report.append("\n✅ No known attack patterns detected")
            report.append("\nNote: If this is unexpected, check THRESHOLDS in attack_patterns.py")
            report.append("      and compare against your dataset's feature distributions.")
            return "\n".join(report)

        report.append(f"\n⚠️  THREATS DETECTED:")
        report.append("\nSeverity Breakdown:")
        icons = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
        for severity, count in detection_results['severity_breakdown'].items():
            if count > 0:
                report.append(f"  {icons[severity]} {severity}: {count}")

        report.append(f"\n{'=' * 80}")
        report.append("DETECTED ATTACK PATTERNS")
        report.append("=" * 80)

        for pattern_name, detections in detection_results['detections_by_pattern'].items():
            if not detections:
                continue

            pattern_info = self.get_pattern_info(pattern_name)
            if pattern_name == 'modbus_write_dpi':
                display_name = "MODBUS WRITE — DPI CONFIRMED (Protocol Ground Truth)"
            else:
                display_name = pattern_info.name if pattern_info else pattern_name.upper()

            report.append(f"\n🚨 {display_name}")
            report.append(f"   Instances : {len(detections)}")

            if pattern_info and pattern_name != 'modbus_write_dpi':
                report.append(f"   MITRE     : {pattern_info.mitre_technique}")
                report.append(f"   Severity  : {pattern_info.severity}")
                report.append(f"   Detail    : {pattern_info.description}")
                report.append(f"   Mitigation: {pattern_info.mitigation}")
                report.append(f"   Example   : {pattern_info.real_world_example}")
            elif pattern_name == 'modbus_write_dpi':
                report.append(f"   MITRE     : T0836 - Modify Control Logic")
                report.append(f"   Source    : Modbus DPI — packet-level FC confirmation")
                report.append(f"   Note      : These events are protocol ground truth.")
                report.append(f"               Write FCs confirmed regardless of flow statistics.")

            for i, det in enumerate(detections[:3], 1):
                src = det.get('detection_source', '')
                src_label = f" [{src}]" if src else ""
                report.append(f"\n   Detection {i}{src_label} "
                               f"(confidence: {det['confidence']:.0%}):")
                for key, value in det['details'].items():
                    report.append(f"      {key}: {value}")
                # Show DPI confirmation for flow_feature detections that got confirmed
                if det.get('dpi_confirmation'):
                    dc = det['dpi_confirmation']
                    report.append(f"      ✅ DPI confirmed: {dc['function_name']} "
                                  f"(unit_id={dc['unit_id']})")

            if len(detections) > 3:
                report.append(f"   ... and {len(detections) - 3} more")

        # CVE enrichment block (only printed if enrich_with_cves() was called first)
        if detection_results.get('cve_enrichment'):
            report.append(self.get_cve_summary(detection_results))

        report.append("\n" + "=" * 80)
        report.append("✅ REPORT COMPLETE")
        report.append("=" * 80)
        return "\n".join(report)


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo_attack_patterns():
    print("=" * 80)
    print("ICS ATTACK PATTERN DETECTION DEMO  (v3.0.0)")
    print("=" * 80)

    library = ICSAttackPatternLibrary()

    print(f"\n📚 Available Attack Patterns: {len(library.patterns)}")
    for name, pattern in library.patterns.items():
        print(f"\n• {pattern.name}")
        print(f"  MITRE    : {pattern.mitre_technique}")
        print(f"  Severity : {pattern.severity}")

    from pathlib import Path
    data_path = Path("./data/processed/ics_features.csv")
    if not data_path.exists():
        data_path = Path("../../data/processed/ics_features.csv")

    if not data_path.exists():
        print(f"\n⚠️  No data found at {data_path} — run quick_start.py first")
        return

    # Auto-detect PCAP for DPI (use first .pcap found in ./data/)
    pcap_path = None
    pcap_candidates = list(Path("./data").glob("*.pcap")) + \
                      list(Path("./data").glob("*.pcapng"))
    if pcap_candidates:
        # Prefer Modbus PCAPs over DNS/generic ones
        modbus_pcaps = [p for p in pcap_candidates
                        if 'modbus' in p.name.lower() or 'ics' in p.name.lower()]
        pcap_path = str(modbus_pcaps[0] if modbus_pcaps else pcap_candidates[0])
        print(f"\n🔬 PCAP detected — DPI will run on: {Path(pcap_path).name}")
        if 'dns' in Path(pcap_path).name.lower():
            print(f"   ⚠️  Note: {Path(pcap_path).name} is DNS traffic, not Modbus.")
            print(f"   DPI will find 0 Modbus packets — this is expected.")
            print(f"   For real Modbus DPI, use a Modbus PCAP from:")
            print(f"   https://www.netresec.com/?page=PCAPNG")
    else:
        print(f"\n⚠️  No PCAP files found in ./data/ — running in flow-only mode")
        print(f"   To enable DPI: place a Modbus PCAP in ./data/ and re-run")

    print(f"\n{'=' * 80}")
    print("ANALYZING NETWORK DATA")
    print("=" * 80)

    flows = pd.read_csv(data_path)
    print(f"\nLoaded {len(flows)} network flows ({len(flows.columns)} features)")

    results = library.detect_all_patterns(flows, pcap_path=pcap_path)
    report = library.generate_threat_report(results)
    print(f"\n{report}")

    # Per-pattern summary table
    print(f"\n{'=' * 80}")
    print("DETECTION SUMMARY BY PATTERN")
    print("=" * 80)
    print(f"{'Pattern':<30} {'Detections':>12} {'Status'}")
    print("-" * 55)
    for pattern_name in library.list_all_patterns():
        count = len(results['detections_by_pattern'].get(pattern_name, []))
        status = "⚠️  DETECTED" if count > 0 else "✅ Clean"
        print(f"{pattern_name:<30} {count:>12}     {status}")

    # DPI-only row
    dpi_count = len(results['detections_by_pattern'].get('modbus_write_dpi', []))
    if results.get('protocol_aware'):
        dpi_status = "⚠️  DETECTED" if dpi_count > 0 else "✅ Clean"
        print(f"{'modbus_write_dpi [DPI]':<30} {dpi_count:>12}     {dpi_status}")
    else:
        print(f"{'modbus_write_dpi [DPI]':<30} {'N/A':>12}     ⬜ No PCAP provided")

    mode = "Protocol-Aware (DPI + Flow)" if results.get('protocol_aware') else "Flow-Only"
    print(f"\nDetection mode: {mode}")


if __name__ == "__main__":
    demo_attack_patterns()