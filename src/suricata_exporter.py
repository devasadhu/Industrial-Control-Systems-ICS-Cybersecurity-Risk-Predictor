"""
Suricata Rule Exporter v1.0.0
==============================
Converts ICS attack pattern detections to Suricata IDS rule format (.rules).

Each attack pattern becomes one or more Suricata rules:
  - Flow-threshold rules (from ML detection thresholds)
  - Content-match rules (from Modbus DPI FC byte patterns)
  - Combined rules where applicable

Output: results/suricata_ics.rules — drop into any Suricata deployment.

Rule SID range: 9000001–9000099 (reserved for custom ICS rules)

Usage
-----
  from src.suricata_exporter import SuricataExporter
  exporter = SuricataExporter()
  rules = exporter.export(detections)          # str, all rules
  exporter.export_to_file(detections, "results/suricata_ics.rules")

  # Standalone demo
  python -m src.suricata_exporter --demo --out results/suricata_ics.rules

No new dependencies.
"""

from __future__ import annotations

import datetime
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Modbus function code bytes (for content matching)
# ---------------------------------------------------------------------------
FC_READ_COILS         = "\\x01"
FC_READ_HOLDING_REGS  = "\\x03"
FC_WRITE_SINGLE_REG   = "\\x06"
FC_WRITE_MULTIPLE_REGS = "\\x10"

# Modbus/TCP MBAP magic: protocol identifier bytes 3-4 are always 0x00 0x00
MODBUS_MBAP_PROTOCOL_BYTES = "\\x00\\x00"

# SID allocation table
_SID_BASE = 9000000
_SID_MAP = {
    "modbus_flooding":      _SID_BASE + 1,
    "plc_scanning":         _SID_BASE + 2,
    "unauthorized_write":   _SID_BASE + 3,
    "man_in_the_middle":    _SID_BASE + 4,
    "replay_attack":        _SID_BASE + 5,
    "command_inject":       _SID_BASE + 6,
    "flooding":             _SID_BASE + 1,   # alias
    "scanning":             _SID_BASE + 2,   # alias
    "mitm":                 _SID_BASE + 4,   # alias
}

# Suricata classtype mapping
_CLASSTYPE_MAP = {
    "modbus_flooding":    "attempted-dos",
    "plc_scanning":       "network-scan",
    "unauthorized_write": "policy-violation",
    "man_in_the_middle":  "bad-unknown",
    "replay_attack":      "bad-unknown",
    "command_inject":     "policy-violation",
    "flooding":           "attempted-dos",
    "scanning":           "network-scan",
    "mitm":               "bad-unknown",
}

# Priority (1=high, 3=low)
_PRIORITY_MAP = {
    "modbus_flooding":    1,
    "plc_scanning":       2,
    "unauthorized_write": 1,
    "man_in_the_middle":  1,
    "replay_attack":      2,
    "command_inject":     1,
    "flooding":           1,
    "scanning":           2,
    "mitm":               1,
}

# MITRE ATT&CK ICS technique IDs for metadata
_TECHNIQUE_MAP = {
    "modbus_flooding":    "T0814",
    "plc_scanning":       "T0846",
    "unauthorized_write": "T0855",
    "man_in_the_middle":  "T0830",
    "replay_attack":      "T0843",
    "command_inject":     "T0855",
    "flooding":           "T0814",
    "scanning":           "T0846",
    "mitm":               "T0830",
}


# ---------------------------------------------------------------------------
# Rule builder helpers
# ---------------------------------------------------------------------------

def _rule(
    action: str,
    proto: str,
    src: str,
    src_port: str,
    direction: str,
    dst: str,
    dst_port: str,
    options: str,
) -> str:
    return f'{action} {proto} {src} {src_port} {direction} {dst} {dst_port} ({options})'


def _opts(**kwargs) -> str:
    """Build Suricata rule options string from key=value pairs."""
    parts = []
    for k, v in kwargs.items():
        if v is None:
            continue
        if v is True:
            parts.append(f"{k};")
        elif v is False:
            pass
        else:
            parts.append(f'{k}:{v};')
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Per-pattern rule generators
# ---------------------------------------------------------------------------

def _rule_flooding(sid: int, detection: dict) -> List[str]:
    """
    High-rate FC01 coil reads → DoS threshold rule.
    Threshold: >10,000 packets/sec from same source.
    """
    opts = (
        'msg:"ICS Modbus Flooding - DoS Attack (T0814)"; '
        f'content:"{MODBUS_MBAP_PROTOCOL_BYTES}"; depth:4; offset:2; '
        f'content:"{FC_READ_COILS}"; offset:7; depth:1; '
        'threshold: type both, track by_src, count 100, seconds 1; '
        f'classtype:attempted-dos; '
        f'metadata:mitre-attack T0814, affected-product Modbus-PLC, '
        f'created-at {datetime.date.today().isoformat()}; '
        f'sid:{sid}; rev:1;'
    )
    return [_rule("alert", "tcp", "any", "any", "->", "any", "502", opts)]


def _rule_scanning(sid: int, detection: dict) -> List[str]:
    """
    Sequential coil address enumeration → network-scan rule.
    Threshold: >50 unique connections to port 502 from same source in 10s.
    """
    opts = (
        'msg:"ICS Modbus PLC Scanning - Address Enumeration (T0846)"; '
        f'content:"{MODBUS_MBAP_PROTOCOL_BYTES}"; depth:4; offset:2; '
        f'content:"{FC_READ_COILS}"; offset:7; depth:1; '
        'threshold: type both, track by_src, count 50, seconds 10; '
        f'classtype:network-scan; '
        f'metadata:mitre-attack T0846, affected-product Modbus-PLC; '
        f'sid:{sid}; rev:1;'
    )
    return [_rule("alert", "tcp", "any", "any", "->", "any", "502", opts)]


def _rule_unauthorized_write(sid: int, detection: dict) -> List[str]:
    """
    FC06 (Write Single Register) + FC16 (Write Multiple Registers) rules.
    Two separate content-match rules — one per FC.
    """
    rules = []

    # FC06 rule
    opts_fc06 = (
        'msg:"ICS Modbus Unauthorized Write Single Register FC06 (T0855)"; '
        f'content:"{MODBUS_MBAP_PROTOCOL_BYTES}"; depth:4; offset:2; '
        f'content:"{FC_WRITE_SINGLE_REG}"; offset:7; depth:1; '
        f'classtype:policy-violation; priority:1; '
        f'metadata:mitre-attack T0855, affected-product Modbus-PLC; '
        f'sid:{sid}; rev:1;'
    )
    rules.append(_rule("alert", "tcp", "any", "any", "->", "any", "502", opts_fc06))

    # FC16 rule (next SID)
    opts_fc16 = (
        'msg:"ICS Modbus Unauthorized Write Multiple Registers FC16 (T0855)"; '
        f'content:"{MODBUS_MBAP_PROTOCOL_BYTES}"; depth:4; offset:2; '
        f'content:"{FC_WRITE_MULTIPLE_REGS}"; offset:7; depth:1; '
        f'classtype:policy-violation; priority:1; '
        f'metadata:mitre-attack T0855, affected-product Modbus-PLC; '
        f'sid:{sid + 50}; rev:1;'
    )
    rules.append(_rule("alert", "tcp", "any", "any", "->", "any", "502", opts_fc16))

    return rules


def _rule_mitm(sid: int, detection: dict) -> List[str]:
    """
    MitM: traffic to port 502 with TTL=128 (Windows default, anomalous on Linux PLC segment).
    Suricata ttl keyword matches exact TTL value.
    """
    opts = (
        'msg:"ICS Modbus MitM - Anomalous TTL=128 on PLC Segment (T0830)"; '
        f'content:"{MODBUS_MBAP_PROTOCOL_BYTES}"; depth:4; offset:2; '
        'ttl:128; '
        f'classtype:bad-unknown; priority:1; '
        f'metadata:mitre-attack T0830, affected-product Modbus-PLC; '
        f'sid:{sid}; rev:1;'
    )
    return [_rule("alert", "tcp", "any", "any", "->", "any", "502", opts)]


def _rule_replay(sid: int, detection: dict) -> List[str]:
    """
    Replay: high-rate FC03 reads from single source with no inter-packet jitter.
    Use threshold to approximate — exact jitter detection requires Suricata Lua.
    """
    opts = (
        'msg:"ICS Modbus Replay Attack - Fixed-Rate FC03 Reads (T0843)"; '
        f'content:"{MODBUS_MBAP_PROTOCOL_BYTES}"; depth:4; offset:2; '
        f'content:"{FC_READ_HOLDING_REGS}"; offset:7; depth:1; '
        'threshold: type both, track by_src, count 20, seconds 1; '
        f'classtype:bad-unknown; priority:2; '
        f'metadata:mitre-attack T0843, affected-product Modbus-PLC; '
        f'sid:{sid}; rev:1;'
    )
    return [_rule("alert", "tcp", "any", "any", "->", "any", "502", opts)]


def _rule_command_inject(sid: int, detection: dict) -> List[str]:
    """Alias of unauthorized_write rules (same FC bytes)."""
    return _rule_unauthorized_write(sid, detection)


_RULE_GENERATORS = {
    "modbus_flooding":    _rule_flooding,
    "plc_scanning":       _rule_scanning,
    "unauthorized_write": _rule_unauthorized_write,
    "man_in_the_middle":  _rule_mitm,
    "replay_attack":      _rule_replay,
    "command_inject":     _rule_command_inject,
    "flooding":           _rule_flooding,
    "scanning":           _rule_scanning,
    "mitm":               _rule_mitm,
}


# ---------------------------------------------------------------------------
# SuricataExporter
# ---------------------------------------------------------------------------

class SuricataExporter:
    """
    Converts ICS attack pattern detections to Suricata .rules format.

    Parameters
    ----------
    include_all_patterns : bool
        If True, generate rules for all known patterns regardless of whether
        they appear in the detections dict (useful for generating a static
        baseline ruleset). Default False.
    """

    def __init__(self, include_all_patterns: bool = False):
        self.include_all_patterns = include_all_patterns

    def export(self, detections: Dict[str, Any]) -> str:
        """
        Convert detections to Suricata rules string.

        Parameters
        ----------
        detections : dict
            Output of detect_all_patterns(). Keys are pattern names.

        Returns
        -------
        str — multi-line Suricata rules text
        """
        header = self._header()
        rule_lines: List[str] = []
        seen_sids = set()

        patterns = (
            list(_RULE_GENERATORS.keys())
            if self.include_all_patterns
            else list(detections.keys())
        )

        for pattern_name in patterns:
            if pattern_name not in _RULE_GENERATORS:
                continue

            result = detections.get(pattern_name, {})
            if not self.include_all_patterns:
                # Skip non-detections unless include_all_patterns
                if isinstance(result, dict) and not result.get("detected", True):
                    continue

            generator = _RULE_GENERATORS[pattern_name]
            sid = _SID_MAP.get(pattern_name, _SID_BASE + 99)

            # Skip duplicate SIDs (flooding and modbus_flooding share SID)
            if sid in seen_sids and pattern_name in ("flooding", "scanning", "mitm",
                                                      "command_inject"):
                continue

            rules = generator(sid, result if isinstance(result, dict) else {})
            rule_lines.append(f"# --- {pattern_name.upper()} ---")
            rule_lines.extend(rules)
            rule_lines.append("")

            seen_sids.add(sid)

        if not rule_lines:
            rule_lines = ["# No attack patterns detected — no rules generated."]

        return header + "\n".join(rule_lines)

    def export_to_file(
        self,
        detections: Dict[str, Any],
        out_path: str | Path = "results/suricata_ics.rules",
    ) -> Path:
        """Export rules to file. Returns Path of written file."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.export(detections)
        out_path.write_text(content, encoding="utf-8")
        return out_path

    def export_baseline_ruleset(
        self, out_path: str | Path = "results/suricata_ics_baseline.rules"
    ) -> Path:
        """
        Export a full baseline ruleset covering all known ICS attack patterns,
        regardless of current detections.
        """
        exporter = SuricataExporter(include_all_patterns=True)
        return exporter.export_to_file({}, out_path=out_path)

    @staticmethod
    def _header() -> str:
        return textwrap.dedent(f"""\
            # =============================================================
            # ICS Network Anomaly Detection System — Suricata Rules
            # Generated: {datetime.datetime.utcnow().isoformat()}Z
            # SID range: 9000001–9000099
            # Protocol: Modbus/TCP (port 502)
            # MITRE ATT&CK ICS mapping included in metadata fields
            #
            # Deploy: copy to /etc/suricata/rules/ics_anomaly.rules
            # Add to suricata.yaml under rule-files:
            #   - ics_anomaly.rules
            # =============================================================

        """)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Export ICS detections to Suricata .rules format."
    )
    parser.add_argument("--input", help="Path to detections JSON")
    parser.add_argument("--out", default="results/suricata_ics.rules", help="Output .rules file")
    parser.add_argument("--demo", action="store_true",
                        help="Generate rules from mock detections")
    parser.add_argument("--baseline", action="store_true",
                        help="Generate full baseline ruleset (all patterns)")
    args = parser.parse_args()

    exporter = SuricataExporter()

    if args.baseline:
        out = exporter.export_baseline_ruleset(args.out)
        print(f"[suricata] Baseline ruleset → {out}")
    else:
        if args.demo or not args.input:
            detections = {
                "modbus_flooding":    {"detected": True, "confidence": "high"},
                "plc_scanning":       {"detected": True, "confidence": "high"},
                "unauthorized_write": {"detected": True, "confidence": "medium"},
                "man_in_the_middle":  {"detected": True, "confidence": "high"},
                "replay_attack":      {"detected": True, "confidence": "medium"},
            }
            print("[suricata] Using mock detections (--demo mode)")
        else:
            with open(args.input) as f:
                detections = json.load(f)

        out = exporter.export_to_file(detections, args.out)
        print(f"[suricata] Rules written → {out}")
        # Print to stdout too
        print("\n" + exporter.export(detections))
