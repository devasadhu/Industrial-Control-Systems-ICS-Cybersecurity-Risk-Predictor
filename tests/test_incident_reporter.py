"""
test_incident_reporter.py
--------------------------
Tests for src/incident_reporter.py

What we verify:
1. generate() returns a dict with at least the 'markdown' key
2. Markdown file is written to disk and is non-empty
3. Markdown contains the required section headings
4. Executive Summary reflects the correct detection count
5. MITRE ATT&CK section appears when a known pattern is detected
6. IEC 62443 section appears when a known pattern is detected
7. Mitigations section appears and references the detected pattern
8. No-detection case produces a clean "no attack patterns" summary
9. flows_df=None is handled gracefully (no crash, file still written)
10. generate() with formats=["markdown"] skips PDF without crashing
11. Incident ID passed to constructor appears in the report header
12. generate() creates the output directory if it does not exist
"""

import sys
import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from incident_reporter import ICSIncidentReporter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_detections_multi() -> dict:
    """Four detected patterns — mirrors the demo fixture in incident_reporter.py."""
    return {
        "modbus_flooding": {
            "detected": True,
            "severity": "critical",
            "confidence": "high",
            "detection_source": "ensemble_ml",
            "flows": [
                {"src_ip": "10.0.0.52", "dst_ip": "192.168.1.10"} for _ in range(12)
            ],
        },
        "plc_scanning": {
            "detected": True,
            "severity": "high",
            "confidence": "high",
            "detection_source": "dpi_modbus",
            "flows": [
                {"src_ip": "10.0.0.53", "dst_ip": "192.168.1.10"} for _ in range(20)
            ],
        },
        "unauthorized_write": {
            "detected": True,
            "severity": "high",
            "confidence": "medium",
            "detection_source": "dpi_modbus",
            "related_cves": [
                {"cve_id": "CVE-2022-29952", "cvss_score": 9.8,
                 "description": "Advantech iView unauthenticated write via Modbus"},
            ],
            "flows": [
                {"src_ip": "10.0.0.51", "dst_ip": "192.168.1.10"} for _ in range(4)
            ],
        },
        "man_in_the_middle": {
            "detected": True,
            "severity": "critical",
            "confidence": "high",
            "detection_source": "dpi_modbus",
            "flows": [
                {"src_ip": "10.0.0.54", "dst_ip": "192.168.1.10"} for _ in range(8)
            ],
        },
    }


def _mock_detections_empty() -> dict:
    """No detections — all patterns present but marked not detected."""
    return {
        "modbus_flooding": {"detected": False},
        "plc_scanning": {"detected": False},
    }


def _mock_flows_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "src_packets": rng.integers(5, 200, n).astype(float),
        "dst_packets": rng.integers(1, 50, n).astype(float),
        "flow_duration": rng.uniform(0.1, 5.0, n),
        "packet_rate": rng.uniform(10, 500, n),
        "byte_rate": rng.uniform(1000, 50000, n),
    })


REQUIRED_SECTIONS = [
    "# ICS Security Incident Report",
    "## Executive Summary",
    "## References",
]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestICSIncidentReporter:

    def test_generate_returns_dict(self, tmp_path):
        """generate() must return a dict."""
        reporter = ICSIncidentReporter(incident_id="ICS-TEST-001")
        result = reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        assert isinstance(result, dict), "generate() must return a dict"

    def test_markdown_key_present(self, tmp_path):
        """Return dict must contain 'markdown' key when formats=['markdown']."""
        reporter = ICSIncidentReporter(incident_id="ICS-TEST-001")
        result = reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        assert "markdown" in result, "Return dict missing 'markdown' key"

    def test_markdown_file_written(self, tmp_path):
        """Markdown file must exist on disk after generate()."""
        reporter = ICSIncidentReporter(incident_id="ICS-TEST-001")
        result = reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        md_path = result["markdown"]
        assert md_path.exists(), f"Markdown file not found: {md_path}"
        assert md_path.stat().st_size > 0, "Markdown file is empty"

    def test_markdown_content_non_empty(self, tmp_path):
        """Markdown file must contain substantial content."""
        reporter = ICSIncidentReporter(incident_id="ICS-TEST-002")
        result = reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        content = result["markdown"].read_text(encoding="utf-8")
        assert len(content) > 200, (
            f"Markdown too short: {len(content)} chars. Report is incomplete."
        )

    @pytest.mark.parametrize("section", REQUIRED_SECTIONS)
    def test_markdown_required_sections(self, tmp_path, section):
        """Each required section heading must appear in the Markdown output."""
        reporter = ICSIncidentReporter(incident_id="ICS-TEST-003")
        result = reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        content = result["markdown"].read_text(encoding="utf-8")
        assert section in content, (
            f"Required section '{section}' missing from Markdown report."
        )

    def test_executive_summary_mentions_detection_count(self, tmp_path):
        """Executive Summary must mention the number of detected patterns."""
        detections = _mock_detections_multi()
        n_detected = sum(
            1 for v in detections.values()
            if isinstance(v, dict) and v.get("detected", True)
        )
        reporter = ICSIncidentReporter(incident_id="ICS-TEST-004")
        result = reporter.generate(
            detections=detections,
            out_dir=tmp_path,
            formats=["markdown"],
        )
        content = result["markdown"].read_text(encoding="utf-8")
        assert str(n_detected) in content, (
            f"Expected detection count '{n_detected}' in Executive Summary."
        )

    def test_mitre_section_present_for_known_patterns(self, tmp_path):
        """MITRE ATT&CK section must appear when any mapped pattern is detected."""
        reporter = ICSIncidentReporter()
        result = reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        content = result["markdown"].read_text(encoding="utf-8")
        assert "MITRE ATT&CK ICS" in content, (
            "MITRE ATT&CK section missing from report with known patterns."
        )

    def test_mitre_technique_id_present(self, tmp_path):
        """At least one ATT&CK technique ID (T0NNN) must appear in the report."""
        reporter = ICSIncidentReporter()
        result = reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        content = result["markdown"].read_text(encoding="utf-8")
        # Check for any T0xxx technique reference
        assert any(f"T0{n:03d}" in content for n in range(800, 900)), (
            "No MITRE ATT&CK technique ID (T08xx) found in report."
        )

    def test_iec62443_section_present(self, tmp_path):
        """IEC 62443 section must appear when a mapped pattern is detected."""
        reporter = ICSIncidentReporter()
        result = reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        content = result["markdown"].read_text(encoding="utf-8")
        assert "IEC 62443" in content, (
            "IEC 62443 section missing from report."
        )

    def test_mitigations_section_present(self, tmp_path):
        """Recommended Mitigations section must appear when attacks are detected."""
        reporter = ICSIncidentReporter()
        result = reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        content = result["markdown"].read_text(encoding="utf-8")
        assert "Mitigations" in content, (
            "Mitigations section missing from report."
        )

    def test_no_detection_produces_clean_summary(self, tmp_path):
        """When no patterns are detected, Executive Summary must say so."""
        reporter = ICSIncidentReporter(incident_id="ICS-CLEAN-001")
        result = reporter.generate(
            detections=_mock_detections_empty(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        content = result["markdown"].read_text(encoding="utf-8")
        assert "No attack patterns detected" in content, (
            "Expected 'No attack patterns detected' in clean-run report."
        )

    def test_no_flows_df_does_not_crash(self, tmp_path):
        """generate() with flows_df=None must not raise and must write the file."""
        reporter = ICSIncidentReporter(incident_id="ICS-TEST-005")
        result = reporter.generate(
            detections=_mock_detections_multi(),
            flows_df=None,
            out_dir=tmp_path,
            formats=["markdown"],
        )
        assert "markdown" in result
        assert result["markdown"].exists()

    def test_flows_df_included_does_not_crash(self, tmp_path):
        """generate() with a real flows_df must not raise."""
        reporter = ICSIncidentReporter(incident_id="ICS-TEST-006")
        result = reporter.generate(
            detections=_mock_detections_multi(),
            flows_df=_mock_flows_df(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        assert "markdown" in result

    def test_custom_incident_id_in_report(self, tmp_path):
        """The incident_id passed to the constructor must appear in the report."""
        incident_id = "ICS-CUSTOM-XYZ-9999"
        reporter = ICSIncidentReporter(incident_id=incident_id)
        result = reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        content = result["markdown"].read_text(encoding="utf-8")
        assert incident_id in content, (
            f"Custom incident_id '{incident_id}' not found in report header."
        )

    def test_output_dir_created_if_absent(self, tmp_path):
        """generate() must create the output directory if it does not already exist."""
        new_dir = tmp_path / "nested" / "deep" / "reports"
        assert not new_dir.exists()
        reporter = ICSIncidentReporter()
        reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=new_dir,
            formats=["markdown"],
        )
        assert new_dir.exists(), "Output directory was not created by generate()."

    def test_markdown_format_only_skips_pdf(self, tmp_path):
        """Requesting formats=['markdown'] must not create a PDF file."""
        reporter = ICSIncidentReporter()
        result = reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        assert "pdf" not in result, (
            "PDF key present in result even though only markdown was requested."
        )
        pdf_files = list(tmp_path.glob("*.pdf"))
        assert not pdf_files, f"Unexpected PDF file created: {pdf_files}"

    def test_cve_appears_in_report_when_provided(self, tmp_path):
        """If a detection includes related_cves, the CVE ID must appear in the report."""
        reporter = ICSIncidentReporter()
        result = reporter.generate(
            detections=_mock_detections_multi(),
            out_dir=tmp_path,
            formats=["markdown"],
        )
        content = result["markdown"].read_text(encoding="utf-8")
        assert "CVE-2022-29952" in content, (
            "CVE ID from detection metadata not found in report."
        )
