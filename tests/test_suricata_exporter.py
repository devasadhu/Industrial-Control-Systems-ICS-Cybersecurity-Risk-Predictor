"""
test_suricata_exporter.py
--------------------------
Tests for src/suricata_exporter.py

What we verify:
1. export() returns a non-empty string
2. Output contains the Suricata header comment block
3. Each detected pattern produces at least one 'alert tcp' rule line
4. SID values are unique across a single export call
5. Modbus port 502 appears in every generated rule
6. MITRE ATT&CK metadata field is present in rules for known patterns
7. Non-detected patterns (detected=False) are excluded unless include_all_patterns=True
8. export_to_file() writes the rules file and returns the correct Path
9. export_to_file() creates the parent directory if absent
10. export_baseline_ruleset() generates rules for all known patterns
11. Content-match bytes (\x01, \x03, \x06, \x10) appear in appropriate rules
12. export() on empty detections returns the 'no rules generated' comment
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from suricata_exporter import SuricataExporter, _SID_MAP


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_detections_all() -> dict:
    """All five main patterns detected."""
    return {
        "modbus_flooding":    {"detected": True, "confidence": "high"},
        "plc_scanning":       {"detected": True, "confidence": "high"},
        "unauthorized_write": {"detected": True, "confidence": "medium"},
        "man_in_the_middle":  {"detected": True, "confidence": "high"},
        "replay_attack":      {"detected": True, "confidence": "medium"},
    }


def _mock_detections_partial() -> dict:
    """Only modbus_flooding detected; others explicitly not detected."""
    return {
        "modbus_flooding": {"detected": True,  "confidence": "high"},
        "plc_scanning":    {"detected": False, "confidence": "low"},
        "replay_attack":   {"detected": False},
    }


def _mock_detections_empty() -> dict:
    """All patterns not detected."""
    return {
        "modbus_flooding": {"detected": False},
        "plc_scanning":    {"detected": False},
    }


def _count_rule_lines(rules_str: str) -> int:
    """Count lines that start with 'alert' (actual Suricata rules)."""
    return sum(1 for line in rules_str.splitlines() if line.startswith("alert"))


def _extract_sids(rules_str: str) -> list:
    """Extract all sid values from a rules string."""
    sids = []
    for line in rules_str.splitlines():
        if "sid:" in line:
            for part in line.split(";"):
                part = part.strip()
                if part.startswith("sid:"):
                    try:
                        sids.append(int(part.split(":")[1].strip().rstrip(";")))
                    except ValueError:
                        pass
    return sids


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSuricataExporter:

    @pytest.fixture(autouse=True)
    def exporter(self):
        self.exporter = SuricataExporter()

    def test_export_returns_string(self):
        """export() must return a str."""
        result = self.exporter.export(_mock_detections_all())
        assert isinstance(result, str), "export() must return a str"

    def test_export_non_empty(self):
        """export() output must be non-empty."""
        result = self.exporter.export(_mock_detections_all())
        assert len(result) > 0, "export() returned an empty string"

    def test_header_block_present(self):
        """Output must contain the ICS rules header comment."""
        result = self.exporter.export(_mock_detections_all())
        assert "ICS Network Anomaly Detection System" in result, (
            "Header comment block missing from Suricata rules output."
        )

    def test_alert_rules_generated(self):
        """At least one 'alert tcp' rule must be generated per detection."""
        result = self.exporter.export(_mock_detections_all())
        n_rules = _count_rule_lines(result)
        assert n_rules >= len(_mock_detections_all()), (
            f"Expected >= {len(_mock_detections_all())} alert rules, got {n_rules}"
        )

    def test_modbus_port_502_in_all_rules(self):
        """Every alert rule must reference Modbus port 502."""
        result = self.exporter.export(_mock_detections_all())
        rule_lines = [l for l in result.splitlines() if l.startswith("alert")]
        assert rule_lines, "No alert rule lines found"
        for line in rule_lines:
            assert "502" in line, (
                f"Modbus port 502 missing from rule: {line[:80]}..."
            )

    def test_sids_are_unique(self):
        """SID values in the exported ruleset must be unique."""
        result = self.exporter.export(_mock_detections_all())
        sids = _extract_sids(result)
        assert len(sids) == len(set(sids)), (
            f"Duplicate SIDs found: {[s for s in sids if sids.count(s) > 1]}"
        )

    def test_mitre_metadata_present(self):
        """Rules for patterns with MITRE mappings must include 'mitre-attack' metadata."""
        result = self.exporter.export({
            "modbus_flooding": {"detected": True, "confidence": "high"},
        })
        assert "T0814" in result, (
            "MITRE ATT&CK technique ID T0814 missing from modbus_flooding rule metadata."
        )

    def test_non_detected_patterns_excluded(self):
        """Patterns with detected=False must not produce alert rules."""
        result = self.exporter.export(_mock_detections_partial())
        n_rules = _count_rule_lines(result)
        # Only modbus_flooding is detected — unauthorized_write generates 2 rules (FC06 + FC16)
        # but modbus_flooding should generate exactly 1
        assert n_rules >= 1, "At least one rule expected for modbus_flooding"

        # plc_scanning is not detected — its section comment should not appear
        # (not guaranteed by all implementations; check SID 9000002 is absent)
        # Rely on the rule count being low rather than scanning for absent text
        sids = _extract_sids(result)
        assert _SID_MAP["plc_scanning"] not in sids, (
            "SID for plc_scanning present even though detected=False"
        )

    def test_empty_detections_returns_no_rules_comment(self):
        """No detected patterns → output must include the 'no rules generated' message."""
        result = self.exporter.export(_mock_detections_empty())
        n_rules = _count_rule_lines(result)
        assert n_rules == 0, f"Expected 0 alert rules for empty detections, got {n_rules}"
        assert "no rules generated" in result.lower() or "no attack" in result.lower(), (
            "Expected 'no rules generated' comment in empty-detections output."
        )

    # ── Content-match bytes ───────────────────────────────────────────────────

    def test_flooding_rule_contains_fc01_byte(self):
        """Flooding rule must content-match FC01 (\\x01)."""
        result = self.exporter.export({
            "modbus_flooding": {"detected": True, "confidence": "high"},
        })
        assert "\\x01" in result, (
            "FC01 byte (\\x01) missing from modbus_flooding content match rule."
        )

    def test_unauthorized_write_rule_contains_fc06_byte(self):
        """Unauthorized write rule must content-match FC06 (\\x06)."""
        result = self.exporter.export({
            "unauthorized_write": {"detected": True, "confidence": "high"},
        })
        assert "\\x06" in result, (
            "FC06 byte (\\x06) missing from unauthorized_write content match rule."
        )

    def test_unauthorized_write_rule_contains_fc16_byte(self):
        """Unauthorized write rule must content-match FC16 (\\x10)."""
        result = self.exporter.export({
            "unauthorized_write": {"detected": True, "confidence": "high"},
        })
        assert "\\x10" in result, (
            "FC16 byte (\\x10) missing from unauthorized_write content match rule."
        )

    def test_replay_rule_contains_fc03_byte(self):
        """Replay attack rule must content-match FC03 (\\x03)."""
        result = self.exporter.export({
            "replay_attack": {"detected": True, "confidence": "medium"},
        })
        assert "\\x03" in result, (
            "FC03 byte (\\x03) missing from replay_attack content match rule."
        )

    # ── export_to_file ────────────────────────────────────────────────────────

    def test_export_to_file_creates_file(self, tmp_path):
        """export_to_file() must create the specified file."""
        out_path = tmp_path / "test_rules.rules"
        returned = self.exporter.export_to_file(_mock_detections_all(), out_path)
        assert out_path.exists(), f"Rules file not created at {out_path}"
        assert returned == out_path, "export_to_file() must return the output Path"

    def test_export_to_file_content_matches_export(self, tmp_path):
        """File content from export_to_file() must contain the same rules as export()."""
        detections = _mock_detections_all()
        out_path = tmp_path / "rules.rules"
        self.exporter.export_to_file(detections, out_path)
        file_content = out_path.read_text(encoding="utf-8")

        # The header embeds a timestamp that may differ by milliseconds between
        # the two calls.  Compare only the alert rule lines, which are stable.
        file_rules  = [l for l in file_content.splitlines() if l.startswith("alert")]
        direct_rules = [l for l in self.exporter.export(detections).splitlines() if l.startswith("alert")]
        assert file_rules == direct_rules, (
            "Alert rules from export_to_file() differ from export() output."
        )

    def test_export_to_file_creates_parent_dirs(self, tmp_path):
        """export_to_file() must create parent directories if absent."""
        out_path = tmp_path / "deep" / "nested" / "ics.rules"
        self.exporter.export_to_file(_mock_detections_all(), out_path)
        assert out_path.exists(), f"File not created at nested path: {out_path}"

    # ── baseline ruleset ──────────────────────────────────────────────────────

    def test_baseline_ruleset_generates_all_patterns(self, tmp_path):
        """export_baseline_ruleset() must produce rules for all known patterns."""
        out_path = tmp_path / "baseline.rules"
        returned = self.exporter.export_baseline_ruleset(out_path)
        assert returned.exists(), "Baseline ruleset file not created"
        content = returned.read_text(encoding="utf-8")
        n_rules = _count_rule_lines(content)
        # At minimum one rule per major pattern (flooding, scanning, write, mitm, replay)
        assert n_rules >= 5, (
            f"Expected >= 5 rules in baseline ruleset, got {n_rules}"
        )

    def test_include_all_patterns_flag(self, tmp_path):
        """SuricataExporter(include_all_patterns=True) with empty detections must still produce rules."""
        exporter_all = SuricataExporter(include_all_patterns=True)
        result = exporter_all.export({})
        n_rules = _count_rule_lines(result)
        assert n_rules >= 5, (
            f"Expected >= 5 rules when include_all_patterns=True, got {n_rules}"
        )
