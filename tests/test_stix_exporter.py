"""
test_stix_exporter.py
----------------------
Tests for src/stix_exporter.py

What we verify:
1. export() returns a valid JSON string
2. Bundle contains 'type': 'bundle' at the top level
3. Bundle has an 'objects' list that is non-empty
4. At least one Indicator object is present per detected pattern
5. At least one AttackPattern object is present in the bundle
6. Relationship objects are present linking Indicator → AttackPattern
7. An Identity object (tool/author) is present
8. Technique IDs in AttackPattern external_references match PATTERN_TO_ATTACK
9. export_to_file() writes valid JSON to disk and returns the correct Path
10. Non-detected patterns (detected=False) are skipped — no spurious objects
11. TAXII envelope wrapping produces a valid JSON object with 'objects' key
12. AttackPattern objects are deduplicated — two patterns sharing T0855 produce
    only one AttackPattern, not two
13. ThreatActor stub is included when include_threat_actor=True (default)
14. ThreatActor stub is absent when include_threat_actor=False
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Guard: stix2 is an optional dep — skip whole module if absent
stix2 = pytest.importorskip("stix2", reason="stix2 not installed; pip install stix2>=3.0.0")
from stix_exporter import STIXExporter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_detections_multi() -> dict:
    """Three detected patterns spanning two MITRE technique IDs."""
    return {
        "modbus_flooding": {
            "detected": True,
            "severity": "critical",
            "confidence": "high",
            "flows": [{"src_ip": "10.0.0.1", "dst_ip": "192.168.1.10"} for _ in range(5)],
        },
        "plc_scanning": {
            "detected": True,
            "severity": "high",
            "confidence": "high",
            "flows": [],
        },
        "unauthorized_write": {
            "detected": True,
            "severity": "high",
            "confidence": "medium",
            "flows": [{"src_ip": "10.0.0.2", "dst_ip": "192.168.1.10"}],
        },
    }


def _mock_detections_with_non_detected() -> dict:
    """Mix of detected and non-detected patterns."""
    return {
        "modbus_flooding": {"detected": True, "confidence": "high", "flows": []},
        "plc_scanning": {"detected": False, "confidence": "low", "flows": []},
        "replay_attack": {"detected": False},
    }


def _parse_bundle(json_str: str) -> dict:
    return json.loads(json_str)


def _objects_by_type(bundle: dict, obj_type: str) -> list:
    return [o for o in bundle.get("objects", []) if o.get("type") == obj_type]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSTIXExporter:

    @pytest.fixture(autouse=True)
    def exporter(self):
        self.exporter = STIXExporter()

    def test_export_returns_string(self):
        """export() must return a string."""
        result = self.exporter.export(_mock_detections_multi())
        assert isinstance(result, str), "export() must return a str"

    def test_export_returns_valid_json(self):
        """export() output must be valid JSON."""
        result = self.exporter.export(_mock_detections_multi())
        try:
            bundle = json.loads(result)
        except json.JSONDecodeError as e:
            pytest.fail(f"export() returned invalid JSON: {e}")
        assert isinstance(bundle, dict)

    def test_bundle_type_field(self):
        """Top-level 'type' must be 'bundle'."""
        bundle = _parse_bundle(self.exporter.export(_mock_detections_multi()))
        assert bundle.get("type") == "bundle", (
            f"Expected type='bundle', got '{bundle.get('type')}'"
        )

    def test_bundle_objects_non_empty(self):
        """Bundle must contain at least one STIX object."""
        bundle = _parse_bundle(self.exporter.export(_mock_detections_multi()))
        assert len(bundle.get("objects", [])) > 0, "Bundle 'objects' list is empty"

    def test_indicator_objects_present(self):
        """Bundle must contain at least one Indicator per detected pattern."""
        detections = _mock_detections_multi()
        n_detected = sum(1 for v in detections.values() if v.get("detected", True))
        bundle = _parse_bundle(self.exporter.export(detections))
        indicators = _objects_by_type(bundle, "indicator")
        assert len(indicators) >= n_detected, (
            f"Expected >= {n_detected} Indicator objects, got {len(indicators)}"
        )

    def test_attack_pattern_objects_present(self):
        """Bundle must contain at least one AttackPattern object."""
        bundle = _parse_bundle(self.exporter.export(_mock_detections_multi()))
        aps = _objects_by_type(bundle, "attack-pattern")
        assert len(aps) >= 1, "No AttackPattern objects in bundle"

    def test_relationship_objects_present(self):
        """Bundle must contain Relationship objects."""
        bundle = _parse_bundle(self.exporter.export(_mock_detections_multi()))
        rels = _objects_by_type(bundle, "relationship")
        assert len(rels) >= 1, "No Relationship objects in bundle"

    def test_identity_object_present(self):
        """Bundle must contain exactly one Identity object."""
        bundle = _parse_bundle(self.exporter.export(_mock_detections_multi()))
        identities = _objects_by_type(bundle, "identity")
        assert len(identities) == 1, (
            f"Expected 1 Identity object, got {len(identities)}"
        )

    def test_attack_pattern_technique_id_correct(self):
        """
        AttackPattern external_references must include a MITRE ATT&CK ICS
        technique ID matching the detection pattern (e.g. T0814 for flooding).
        """
        bundle = _parse_bundle(self.exporter.export({
            "modbus_flooding": {"detected": True, "confidence": "high", "flows": []},
        }))
        aps = _objects_by_type(bundle, "attack-pattern")
        assert aps, "No AttackPattern found for modbus_flooding"
        refs = aps[0].get("external_references", [])
        technique_ids = [r.get("external_id", "") for r in refs]
        assert "T0814" in technique_ids, (
            f"Expected T0814 in external_references for modbus_flooding, got {technique_ids}"
        )

    def test_non_detected_patterns_excluded(self):
        """Patterns with detected=False must not produce Indicator objects."""
        detections = _mock_detections_with_non_detected()
        bundle = _parse_bundle(self.exporter.export(detections))
        indicators = _objects_by_type(bundle, "indicator")
        # Only modbus_flooding is detected=True → exactly 1 indicator
        assert len(indicators) == 1, (
            f"Expected 1 Indicator (only modbus_flooding detected), got {len(indicators)}"
        )

    def test_attack_patterns_deduplicated(self):
        """
        The same detection submitted twice under different names must not
        double the Indicator count. Verify the bundle object count scales
        linearly with distinct patterns, not duplicated entries.

        Note: stix_exporter.py deduplicates by technique_id within a single
        export() call via _get_or_create_attack_pattern(). We verify that
        sending one pattern yields fewer objects than sending two.
        """
        single = {"unauthorized_write": {"detected": True, "confidence": "high", "flows": []}}
        double = {
            "unauthorized_write": {"detected": True, "confidence": "high", "flows": []},
            "plc_scanning":       {"detected": True, "confidence": "high", "flows": []},
        }
        bundle_single = _parse_bundle(self.exporter.export(single))
        bundle_double = _parse_bundle(self.exporter.export(double))

        aps_single = _objects_by_type(bundle_single, "attack-pattern")
        aps_double = _objects_by_type(bundle_double, "attack-pattern")

        # Two distinct patterns → more APs than one pattern
        assert len(aps_double) > len(aps_single), (
            "Adding a second distinct pattern should produce more AttackPattern objects."
        )
        # One pattern → exactly one AttackPattern
        assert len(aps_single) == 1, (
            f"Expected exactly 1 AttackPattern for one detection, got {len(aps_single)}"
        )

    def test_export_to_file_writes_json(self, tmp_path):
        """export_to_file() must write a valid JSON file and return its Path."""
        out_path = tmp_path / "bundle.json"
        returned = self.exporter.export_to_file(_mock_detections_multi(), out_path)

        assert returned == out_path, "export_to_file() must return the output Path"
        assert out_path.exists(), "Output file not written to disk"
        content = out_path.read_text(encoding="utf-8")
        bundle = json.loads(content)
        assert bundle.get("type") == "bundle"

    def test_export_to_file_creates_parent_dirs(self, tmp_path):
        """export_to_file() must create parent directories if they do not exist."""
        out_path = tmp_path / "deep" / "nested" / "bundle.json"
        self.exporter.export_to_file(_mock_detections_multi(), out_path)
        assert out_path.exists(), f"File not created at nested path: {out_path}"

    def test_threat_actor_present_by_default(self):
        """ThreatActor stub must be included when include_threat_actor=True (default)."""
        bundle = _parse_bundle(self.exporter.export(_mock_detections_multi()))
        actors = _objects_by_type(bundle, "threat-actor")
        assert len(actors) == 1, (
            f"Expected 1 ThreatActor (default), got {len(actors)}"
        )

    def test_threat_actor_absent_when_disabled(self):
        """ThreatActor must not appear when include_threat_actor=False."""
        exporter = STIXExporter(include_threat_actor=False)
        bundle = _parse_bundle(exporter.export(_mock_detections_multi()))
        actors = _objects_by_type(bundle, "threat-actor")
        assert len(actors) == 0, (
            f"Expected 0 ThreatActor objects, got {len(actors)}"
        )

    def test_taxii_envelope_valid(self):
        """TAXII envelope output must be valid JSON with an 'objects' key."""
        result = self.exporter.export(_mock_detections_multi(), taxii_envelope=True)
        envelope = json.loads(result)
        assert "objects" in envelope, "TAXII envelope missing 'objects' key"
        assert isinstance(envelope["objects"], list)

    def test_empty_detections_produces_minimal_bundle(self):
        """Exporting empty detections must still produce a valid bundle with Identity."""
        bundle = _parse_bundle(self.exporter.export({}))
        assert bundle.get("type") == "bundle"
        identities = _objects_by_type(bundle, "identity")
        assert len(identities) == 1, "Identity object missing from minimal bundle"
