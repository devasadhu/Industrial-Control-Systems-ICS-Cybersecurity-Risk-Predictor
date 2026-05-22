"""
STIX 2.1 Threat Intelligence Exporter v1.0.0
==============================================
Converts ICS attack pattern detections to a STIX 2.1 Bundle consumable by
any SIEM or threat intel platform (Splunk, IBM QRadar, OpenCTI, MISP).

Produces:
  - STIX 2.1 Indicator objects (one per detection)
  - STIX 2.1 AttackPattern objects (MITRE ATT&CK ICS technique stubs)
  - Relationship objects linking each Indicator → AttackPattern
  - Optional ThreatActor stub
  - Full Bundle wrapping all of the above

Output formats:
  - JSON (STIX bundle) — default
  - TAXII 2.1-compatible envelope (--taxii flag)

Usage (standalone):
  python -m src.stix_exporter --input results/detections.json --out results/bundle.json

Usage (from code):
  from src.stix_exporter import STIXExporter
  exporter = STIXExporter()
  bundle_json = exporter.export(detections_dict)

Dependencies:
  stix2>=3.0.0   (pip install stix2)
"""

from __future__ import annotations

import json
import uuid
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# stix2 import — soft dependency
# ---------------------------------------------------------------------------
try:
    import stix2
    from stix2 import (
        Bundle, Indicator, AttackPattern, ThreatActor,
        Relationship, Identity,
    )
    _STIX2_AVAILABLE = True
except ImportError:
    _STIX2_AVAILABLE = False


# ---------------------------------------------------------------------------
# MITRE ATT&CK ICS technique mapping
# ---------------------------------------------------------------------------
# Source: https://attack.mitre.org/matrices/ics/
PATTERN_TO_ATTACK = {
    "modbus_flooding": {
        "technique_id":   "T0814",
        "technique_name": "Denial of Control",
        "tactic":         "Inhibit Response Function",
        "description":    (
            "Adversary floods Modbus/TCP with high-rate FC01 read requests, "
            "exhausting PLC connection resources and preventing legitimate HMI control."
        ),
    },
    "plc_scanning": {
        "technique_id":   "T0846",
        "technique_name": "Remote System Discovery",
        "tactic":         "Discovery",
        "description":    (
            "Sequential enumeration of Modbus coil/register addresses to map PLC "
            "data model and identify writable outputs."
        ),
    },
    "unauthorized_write": {
        "technique_id":   "T0855",
        "technique_name": "Unauthorized Command Message",
        "tactic":         "Impair Process Control",
        "description":    (
            "Adversary sends FC06/FC16 write commands to unexpected register addresses, "
            "manipulating PLC output values without operator knowledge."
        ),
    },
    "man_in_the_middle": {
        "technique_id":   "T0830",
        "technique_name": "Man in the Middle",
        "tactic":         "Collection",
        "description":    (
            "Adversary intercepts Modbus/TCP traffic between HMI and PLC, "
            "forwarding modified commands and injecting forged responses. "
            "Identified by TTL=128 on traffic originating from Linux PLC segment."
        ),
    },
    "replay_attack": {
        "technique_id":   "T0843",
        "technique_name": "Program Download",
        "tactic":         "Lateral Movement",
        "description":    (
            "Captured Modbus FC03 read requests replayed at fixed mechanical intervals "
            "with no timing jitter — characteristic of automated replay tooling."
        ),
    },
    "command_inject": {
        "technique_id":   "T0855",
        "technique_name": "Unauthorized Command Message",
        "tactic":         "Impair Process Control",
        "description":    (
            "FC06/FC16 writes injected to register addresses outside the normal "
            "HMI polling range (>= 0x0100), indicating unauthorized PLC manipulation."
        ),
    },
    "flooding": {
        "technique_id":   "T0814",
        "technique_name": "Denial of Control",
        "tactic":         "Inhibit Response Function",
        "description":    "High-rate Modbus FC01 coil reads exhausting PLC resources.",
    },
    "scanning": {
        "technique_id":   "T0846",
        "technique_name": "Remote System Discovery",
        "tactic":         "Discovery",
        "description":    "Sequential Modbus coil address enumeration (FC01 sweep).",
    },
    "mitm": {
        "technique_id":   "T0830",
        "technique_name": "Man in the Middle",
        "tactic":         "Collection",
        "description":    "Modbus/TCP MitM identified by TTL=128 on PLC-segment traffic.",
    },
}

# Default STIX patterns per attack type (STIX Pattern Language 2.1)
PATTERN_TO_STIX_PATTERN = {
    "modbus_flooding": (
        "[network-traffic:dst_port = 502 AND "
        "network-traffic:dst_ref.type = 'ipv4-addr' AND "
        "network-traffic:protocols[0] = 'tcp']"
    ),
    "plc_scanning": (
        "[network-traffic:dst_port = 502 AND "
        "network-traffic:extensions.'tcp-ext'.src_port_range = '49152-65535']"
    ),
    "unauthorized_write": (
        "[network-traffic:dst_port = 502 AND "
        "network-traffic:extensions.'tcp-ext'.flags = 'PA']"
    ),
    "man_in_the_middle": (
        "[network-traffic:dst_port = 502 AND "
        "network-traffic:src_ref.type = 'ipv4-addr' AND "
        "network-traffic:extensions.'tcp-ext'.flags = 'PA']"
    ),
    "replay_attack": (
        "[network-traffic:dst_port = 502 AND "
        "network-traffic:dst_ref.type = 'ipv4-addr']"
    ),
    "command_inject": (
        "[network-traffic:dst_port = 502 AND "
        "network-traffic:dst_ref.type = 'ipv4-addr']"
    ),
    "flooding":     "[network-traffic:dst_port = 502]",
    "scanning":     "[network-traffic:dst_port = 502]",
    "mitm":         "[network-traffic:dst_port = 502]",
}


# ---------------------------------------------------------------------------
# STIXExporter
# ---------------------------------------------------------------------------

class STIXExporter:
    """
    Converts ICSAttackPatternLibrary detection results to STIX 2.1 Bundle.

    Parameters
    ----------
    author_name : str
        Name of the producing organisation/tool (used in Identity object).
    include_threat_actor : bool
        Whether to add a generic ThreatActor stub to the bundle.
    """

    STIX_SPEC_VERSION = "2.1"

    def __init__(
        self,
        author_name: str = "ICS Network Anomaly Detection System",
        include_threat_actor: bool = True,
    ):
        if not _STIX2_AVAILABLE:
            raise ImportError(
                "stix2 is required: pip install stix2>=3.0.0"
            )
        self.author_name = author_name
        self.include_threat_actor = include_threat_actor
        self._now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- public API --------------------------------------------------------

    def export(
        self,
        detections: Dict[str, Any],
        taxii_envelope: bool = False,
    ) -> str:
        """
        Convert detections dict (from detect_all_patterns()) to STIX 2.1 JSON.

        Parameters
        ----------
        detections : dict
            Output of ICSAttackPatternLibrary.detect_all_patterns().
            Keys are pattern names; values are detection result dicts.
        taxii_envelope : bool
            If True, wrap bundle in a TAXII 2.1 envelope object.

        Returns
        -------
        str — JSON string (STIX Bundle or TAXII envelope)
        """
        objects = []

        # Identity (tool/author)
        identity = Identity(
            id=f"identity--{uuid.uuid4()}",
            name=self.author_name,
            identity_class="system",
        )
        objects.append(identity)

        # Optional ThreatActor stub
        threat_actor = None
        if self.include_threat_actor:
            threat_actor = ThreatActor(
                id=f"threat-actor--{uuid.uuid4()}",
                name="Unknown ICS Adversary",
                description=(
                    "Unattributed threat actor conducting reconnaissance and "
                    "manipulation of industrial control systems via Modbus/TCP."
                ),
                threat_actor_types=["criminal"],
            )
            objects.append(threat_actor)

        # One Indicator + AttackPattern + Relationship per detection
        attack_pattern_cache: Dict[str, AttackPattern] = {}

        for pattern_name, result in detections.items():
            if not result:
                continue
            # Skip top-level summary keys (total_detections, severity_breakdown, etc.)
            # which are ints/lists, not per-pattern detection dicts.
            if not isinstance(result, dict):
                continue
            # Some detect_all_patterns() implementations return empty dicts for
            # non-detections; skip those gracefully.
            if not result.get("detected", True):
                continue

            indicator = self._build_indicator(pattern_name, result, identity)
            objects.append(indicator)

            ap = self._get_or_create_attack_pattern(pattern_name, attack_pattern_cache)
            objects.append(ap)

            rel = Relationship(
                id=f"relationship--{uuid.uuid4()}",
                relationship_type="indicates",
                source_ref=indicator.id,
                target_ref=ap.id,
                created_by_ref=identity.id,
            )
            objects.append(rel)

            if threat_actor:
                uses_rel = Relationship(
                    id=f"relationship--{uuid.uuid4()}",
                    relationship_type="uses",
                    source_ref=threat_actor.id,
                    target_ref=ap.id,
                    created_by_ref=identity.id,
                )
                objects.append(uses_rel)

        bundle = Bundle(objects=objects)
        bundle_json = bundle.serialize(pretty=True)

        if taxii_envelope:
            return self._taxii_envelope(bundle_json)
        return bundle_json

    def export_to_file(
        self,
        detections: Dict[str, Any],
        out_path: str | Path,
        taxii_envelope: bool = False,
    ) -> Path:
        """Export to file. Returns the written path."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_json = self.export(detections, taxii_envelope=taxii_envelope)
        out_path.write_text(bundle_json, encoding="utf-8")
        return out_path

    # ---- internal builders -------------------------------------------------

    def _build_indicator(
        self,
        pattern_name: str,
        result: Dict[str, Any],
        identity: Identity,
    ) -> Indicator:
        attack_info = PATTERN_TO_ATTACK.get(pattern_name, {})
        stix_pattern = PATTERN_TO_STIX_PATTERN.get(
            pattern_name,
            "[network-traffic:dst_port = 502]",
        )
        technique_id = attack_info.get("technique_id", "T0000")
        technique_name = attack_info.get("technique_name", "Unknown Technique")
        tactic = attack_info.get("tactic", "Unknown")

        # Build description from detection metadata
        n_flows = len(result.get("flows", []))
        confidence = result.get("confidence", result.get("severity", "medium"))
        description = (
            f"ICS anomaly detection: {pattern_name.replace('_', ' ').title()}. "
            f"MITRE ATT&CK ICS {technique_id} ({technique_name}) — Tactic: {tactic}. "
            f"Detected in {n_flows} flow(s). Confidence: {confidence}."
        )
        if attack_info.get("description"):
            description += f" {attack_info['description']}"

        return Indicator(
            id=f"indicator--{uuid.uuid4()}",
            name=f"ICS {pattern_name.replace('_', ' ').title()} Detected",
            description=description,
            indicator_types=["malicious-activity", "anomalous-activity"],
            pattern=stix_pattern,
            pattern_type="stix",
            valid_from=self._now,
            labels=[pattern_name, "ics", "modbus", "scada"],
            created_by_ref=identity.id,
            external_references=[
                {
                    "source_name": "MITRE ATT&CK ICS",
                    "external_id": technique_id,
                    "url": f"https://attack.mitre.org/techniques/{technique_id}/",
                }
            ],
        )

    def _get_or_create_attack_pattern(
        self,
        pattern_name: str,
        cache: Dict[str, AttackPattern],
    ) -> AttackPattern:
        """Deduplicate AttackPattern objects by technique_id."""
        attack_info = PATTERN_TO_ATTACK.get(pattern_name, {})
        technique_id = attack_info.get("technique_id", "T0000")

        if technique_id in cache:
            return cache[technique_id]

        ap = AttackPattern(
            id=f"attack-pattern--{uuid.uuid4()}",
            name=attack_info.get("technique_name", pattern_name),
            description=attack_info.get("description", ""),
            external_references=[
                {
                    "source_name": "MITRE ATT&CK ICS",
                    "external_id": technique_id,
                    "url": f"https://attack.mitre.org/techniques/{technique_id}/",
                }
            ],
        )
        cache[technique_id] = ap
        return ap

    @staticmethod
    def _taxii_envelope(bundle_json: str) -> str:
        """Wrap STIX bundle in a TAXII 2.1 envelope."""
        bundle = json.loads(bundle_json)
        envelope = {
            "type": "bundle",
            "id": bundle.get("id", f"bundle--{uuid.uuid4()}"),
            "objects": bundle.get("objects", []),
            "spec_version": "2.1",
        }
        return json.dumps(envelope, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_mock_detections() -> Dict[str, Any]:
    """For --demo mode: build a plausible fake detections dict."""
    return {
        "modbus_flooding": {
            "detected": True,
            "confidence": "high",
            "severity": "critical",
            "flows": [{"src_ip": "10.0.0.52", "packet_rate": 9800}] * 5,
        },
        "plc_scanning": {
            "detected": True,
            "confidence": "high",
            "severity": "high",
            "flows": [{"src_ip": "10.0.0.53", "unique_dst_count": 255}],
        },
        "unauthorized_write": {
            "detected": True,
            "confidence": "medium",
            "severity": "high",
            "flows": [{"src_ip": "10.0.0.51", "fc": 6}],
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export ICS detections to STIX 2.1 Bundle")
    parser.add_argument("--input", help="Path to detections JSON (from detect_all_patterns)")
    parser.add_argument("--out", default="results/stix_bundle.json", help="Output path")
    parser.add_argument("--taxii", action="store_true", help="Wrap in TAXII 2.1 envelope")
    parser.add_argument("--demo", action="store_true",
                        help="Generate a demo bundle from mock detections")
    args = parser.parse_args()

    if args.demo or not args.input:
        detections = _build_mock_detections()
        print("[stix_exporter] Using mock detections (--demo mode)")
    else:
        with open(args.input) as f:
            detections = json.load(f)

    exporter = STIXExporter()
    out_path = exporter.export_to_file(detections, args.out, taxii_envelope=args.taxii)
    print(f"[stix_exporter] Bundle written → {out_path}")
    print(f"[stix_exporter] TAXII envelope: {args.taxii}")
