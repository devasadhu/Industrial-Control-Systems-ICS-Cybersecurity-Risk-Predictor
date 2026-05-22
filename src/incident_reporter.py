"""
ICS Incident Reporter v1.0.0
==============================
Takes detection output from ICSAttackPatternLibrary.detect_all_patterns()
and generates a structured incident report that answers the three analyst questions:

  1. What happened?    — attack timeline, triggered patterns, affected flows
  2. Why care?         — MITRE ATT&CK ICS mapping, IEC 62443 SL violations, CVEs
  3. What to do?       — per-attack recommended mitigations, Suricata rule refs

Output formats:
  - Markdown (.md)  — always generated, no extra deps
  - PDF (.pdf)      — generated if reportlab is installed

Usage
-----
  from src.incident_reporter import ICSIncidentReporter

  reporter = ICSIncidentReporter()
  reporter.generate(
      detections=library.detect_all_patterns(flows_df),
      flows_df=flows_df,
      out_dir="results/",
  )

  # Standalone demo
  python -m src.incident_reporter --demo --out results/

Dependencies
------------
  reportlab>=3.6.0   optional, for PDF output (pip install reportlab)
  pandas, numpy      already in requirements.txt
"""

from __future__ import annotations

import datetime
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# reportlab — optional
# ---------------------------------------------------------------------------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak,
    )
    _REPORTLAB = True
except ImportError:
    _REPORTLAB = False


# ---------------------------------------------------------------------------
# Static knowledge base
# ---------------------------------------------------------------------------

MITRE_ICS = {
    "modbus_flooding": {
        "technique_id":   "T0814",
        "technique_name": "Denial of Control",
        "tactic":         "Inhibit Response Function",
        "url":            "https://attack.mitre.org/techniques/T0814/",
    },
    "plc_scanning": {
        "technique_id":   "T0846",
        "technique_name": "Remote System Discovery",
        "tactic":         "Discovery",
        "url":            "https://attack.mitre.org/techniques/T0846/",
    },
    "unauthorized_write": {
        "technique_id":   "T0855",
        "technique_name": "Unauthorized Command Message",
        "tactic":         "Impair Process Control",
        "url":            "https://attack.mitre.org/techniques/T0855/",
    },
    "man_in_the_middle": {
        "technique_id":   "T0830",
        "technique_name": "Man in the Middle",
        "tactic":         "Collection",
        "url":            "https://attack.mitre.org/techniques/T0830/",
    },
    "replay_attack": {
        "technique_id":   "T0843",
        "technique_name": "Program Download",
        "tactic":         "Lateral Movement",
        "url":            "https://attack.mitre.org/techniques/T0843/",
    },
}

IEC62443_VIOLATIONS = {
    "modbus_flooding":    ("SR 7.1", "Denial of Service Protection",           "SL-2"),
    "plc_scanning":       ("SR 1.1", "Human User Identification & Auth",       "SL-1"),
    "unauthorized_write": ("SR 2.1", "Authorization Enforcement",              "SL-2"),
    "man_in_the_middle":  ("SR 4.1", "Information Confidentiality",            "SL-2"),
    "replay_attack":      ("SR 3.3", "Security Functionality Verification",    "SL-2"),
}

MITIGATIONS = {
    "modbus_flooding": [
        "Rate-limit Modbus/TCP connections at the network switch (recommended: <100 req/sec per source IP).",
        "Deploy a Modbus-aware industrial firewall (e.g. Tofino, Cisco IE3400) with DoS thresholds.",
        "Enable Suricata rule SID 9000001 from the generated ruleset.",
        "Review PLC connection limits — most PLCs support 2–4 concurrent TCP sessions; confirm configured correctly.",
    ],
    "plc_scanning": [
        "Restrict Modbus/TCP access to known HMI IP addresses via ACL on the OT network switch.",
        "Enable Suricata rule SID 9000002.",
        "Audit OT network for unexpected devices — use passive discovery (e.g. Claroty, Nozomi passive sensors).",
        "Segment engineering workstation network from HMI/PLC network (IEC 62443 Zone/Conduit model).",
    ],
    "unauthorized_write": [
        "Implement application-layer whitelisting: only permit FC01/FC03 reads from HMI; FC06/FC16 writes only from authorized SCADA server.",
        "Enable Suricata rules SID 9000003 and 9000053 (FC06 and FC16 respectively).",
        "Review PLC logic for unexpected register changes — compare current register values against last known-good baseline.",
        "Consider Modbus TCP proxy (e.g. ModShield, Waterfall) for read-only HMI scenarios.",
        "Cross-reference with FrostyGoop (April 2024): this attack pattern matches the Modbus write technique used against Lviv district heating infrastructure.",
    ],
    "man_in_the_middle": [
        "Enable Suricata rule SID 9000004 (TTL anomaly detection).",
        "Deploy ARP inspection and DHCP snooping on OT switches to prevent ARP poisoning.",
        "Verify network topology — any device on PLC segment with TTL=128 is a Windows host (anomalous for Linux PLCs).",
        "Consider encrypted Modbus alternatives (Modbus TLS per RFC 8446 extension) for high-sensitivity segments.",
        "Check for duplicate MAC addresses or unexpected ARP table entries on managed switches.",
    ],
    "replay_attack": [
        "Enable Suricata rule SID 9000005.",
        "Implement sequence number or timestamp validation at the Modbus proxy layer.",
        "Review PLC logs for repeated transaction IDs (MBAP transaction_id field should increment monotonically).",
        "Segment the OT network so captured traffic cannot be replayed from external subnets.",
    ],
}

SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "unknown": 0}
SEVERITY_COLOR_PDF = {
    "critical": colors.HexColor("#c0392b"),
    "high":     colors.HexColor("#e67e22"),
    "medium":   colors.HexColor("#f1c40f"),
    "low":      colors.HexColor("#27ae60"),
    "unknown":  colors.HexColor("#95a5a6"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity(result: dict) -> str:
    return str(result.get("severity", result.get("confidence", "unknown"))).lower()


def _n_flows(result: dict) -> int:
    flows = result.get("flows", [])
    if isinstance(flows, list):
        return len(flows)
    return int(result.get("n_anomalous_flows", result.get("n_flows", 0)))


def _affected_ips(result: dict) -> List[str]:
    ips = set()
    for flow in result.get("flows", []):
        if isinstance(flow, dict):
            for k in ("src_ip", "dst_ip", "source_ip", "destination_ip"):
                v = flow.get(k)
                if v:
                    ips.add(str(v))
    return sorted(ips)


def _now_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

class _MarkdownRenderer:

    def __init__(self, detections: dict, flows_df: Optional[pd.DataFrame], meta: dict):
        self.detections = detections
        self.flows_df = flows_df
        self.meta = meta
        self.detected = {
            k: v for k, v in detections.items()
            if isinstance(v, dict) and v.get("detected", bool(v))
        }

    def render(self) -> str:
        parts = [
            self._header(),
            self._executive_summary(),
            self._attack_timeline(),
            self._detections_detail(),
            self._mitre_mapping(),
            self._iec62443_section(),
            self._mitigations(),
            self._flow_statistics(),
            self._footer(),
        ]
        return "\n\n".join(p for p in parts if p.strip())

    # ---- sections ----------------------------------------------------------

    def _header(self) -> str:
        incident_id = self.meta.get("incident_id", "ICS-001")
        return textwrap.dedent(f"""\
            # ICS Security Incident Report
            **Incident ID:** {incident_id}
            **Generated:** {_today_str()}
            **Classification:** TLP:AMBER — Restricted distribution
            **Tool:** ICS Network Anomaly Detection System v2.0.0
            **Dataset:** {self.meta.get('dataset', 'ICSSIM / Live PCAP')}

            ---
        """)

    def _executive_summary(self) -> str:
        n_detected = len(self.detected)
        if not n_detected:
            return "## Executive Summary\n\nNo attack patterns detected in the analysed traffic window."

        severities = [_severity(v) for v in self.detected.values()]
        max_sev = max(severities, key=lambda s: SEVERITY_RANK.get(s, 0))
        total_flows = sum(_n_flows(v) for v in self.detected.values())
        patterns_str = ", ".join(
            k.replace("_", " ").title() for k in
            sorted(self.detected, key=lambda k: SEVERITY_RANK.get(_severity(self.detected[k]), 0), reverse=True)
        )

        return textwrap.dedent(f"""\
            ## Executive Summary

            **{n_detected} attack pattern(s) detected** across {total_flows} flagged flows.
            Highest severity: **{max_sev.upper()}**

            Detected patterns: {patterns_str}

            Immediate action required for any CRITICAL or HIGH severity finding.
            Refer to the Mitigations section for prioritised response steps.
        """)

    def _attack_timeline(self) -> str:
        if not self.detected or self.flows_df is None or self.flows_df.empty:
            return ""

        lines = ["## Attack Timeline\n"]
        time_col = next((c for c in ("timestamp", "Timestamp", "time", "Time") if c in self.flows_df.columns), None)

        if time_col:
            t_min = self.flows_df[time_col].min()
            t_max = self.flows_df[time_col].max()
            lines.append(f"Analysis window: `{t_min}` → `{t_max}`\n")

        lines.append("| # | Pattern | Severity | Flows | Affected IPs |")
        lines.append("|---|---------|----------|-------|--------------|")

        sorted_patterns = sorted(
            self.detected.items(),
            key=lambda kv: SEVERITY_RANK.get(_severity(kv[1]), 0),
            reverse=True,
        )
        for i, (pattern, result) in enumerate(sorted_patterns, 1):
            sev = _severity(result).upper()
            n = _n_flows(result)
            ips = ", ".join(_affected_ips(result)[:3]) or "N/A"
            name = pattern.replace("_", " ").title()
            lines.append(f"| {i} | {name} | **{sev}** | {n} | `{ips}` |")

        return "\n".join(lines)

    def _detections_detail(self) -> str:
        if not self.detected:
            return ""
        lines = ["## Detection Details\n"]
        for pattern, result in self.detected.items():
            sev = _severity(result).upper()
            n = _n_flows(result)
            conf = result.get("confidence", "N/A")
            source = result.get("detection_source", "ensemble_ml")
            name = pattern.replace("_", " ").title()

            lines.append(f"### {name}")
            lines.append(f"- **Severity:** {sev}")
            lines.append(f"- **Confidence:** {conf}")
            lines.append(f"- **Flagged flows:** {n}")
            lines.append(f"- **Detection source:** `{source}`")

            ips = _affected_ips(result)
            if ips:
                lines.append(f"- **Affected IPs:** {', '.join(f'`{ip}`' for ip in ips[:5])}")

            cves = result.get("related_cves", [])
            if cves:
                lines.append(f"- **Related CVEs:** " + ", ".join(
                    f"[{c['cve_id']}](https://nvd.nist.gov/vuln/detail/{c['cve_id']}) (CVSS {c.get('cvss_score', 'N/A')})"
                    for c in cves[:3]
                ))

            lines.append("")

        return "\n".join(lines)

    def _mitre_mapping(self) -> str:
        if not self.detected:
            return ""
        lines = ["## MITRE ATT&CK ICS Mapping\n"]
        lines.append("| Pattern | Technique ID | Technique Name | Tactic |")
        lines.append("|---------|-------------|----------------|--------|")

        seen = set()
        for pattern in self.detected:
            info = MITRE_ICS.get(pattern)
            if not info:
                continue
            tid = info["technique_id"]
            if tid in seen:
                continue
            seen.add(tid)
            name = pattern.replace("_", " ").title()
            lines.append(
                f"| {name} | [{tid}]({info['url']}) | {info['technique_name']} | {info['tactic']} |"
            )

        lines.append("\nFull matrix: https://attack.mitre.org/matrices/ics/")
        return "\n".join(lines)

    def _iec62443_section(self) -> str:
        relevant = {k: v for k, v in IEC62443_VIOLATIONS.items() if k in self.detected}
        if not relevant:
            return ""
        lines = ["## IEC 62443 Violations\n"]
        lines.append("| Pattern | Security Requirement | Description | Required SL |")
        lines.append("|---------|---------------------|-------------|-------------|")
        for pattern, (sr, desc, sl) in relevant.items():
            name = pattern.replace("_", " ").title()
            lines.append(f"| {name} | {sr} | {desc} | {sl} |")

        lines.append(
            "\nIEC 62443-3-3 defines four Security Levels (SL-1 through SL-4). "
            "Violations indicate the current environment does not meet the minimum SL "
            "required by the standard for the affected component."
        )
        return "\n".join(lines)

    def _mitigations(self) -> str:
        if not self.detected:
            return ""
        lines = ["## Recommended Mitigations\n"]
        sorted_patterns = sorted(
            self.detected.keys(),
            key=lambda k: SEVERITY_RANK.get(_severity(self.detected[k]), 0),
            reverse=True,
        )
        for pattern in sorted_patterns:
            name = pattern.replace("_", " ").title()
            sev = _severity(self.detected[pattern]).upper()
            steps = MITIGATIONS.get(pattern, ["No specific mitigation defined. Review network segmentation."])
            lines.append(f"### {name} [{sev}]\n")
            for i, step in enumerate(steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")
        return "\n".join(lines)

    def _flow_statistics(self) -> str:
        if self.flows_df is None or self.flows_df.empty:
            return ""
        lines = ["## Flow Statistics\n"]
        lines.append(f"- Total flows analysed: **{len(self.flows_df):,}**")

        numeric_cols = self.flows_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in ("packet_rate", "byte_rate", "duration", "src_packet_count"):
            if col in numeric_cols:
                lines.append(
                    f"- `{col}`: mean={self.flows_df[col].mean():.2f}, "
                    f"max={self.flows_df[col].max():.2f}"
                )
        return "\n".join(lines)

    def _footer(self) -> str:
        return textwrap.dedent("""\
            ---
            ## References

            - MITRE ATT&CK ICS: https://attack.mitre.org/matrices/ics/
            - IEC 62443 Standard: https://www.iec.ch/homepage
            - CISA ICS Advisories: https://www.cisa.gov/ics-advisories
            - FrostyGoop Malware Analysis: https://www.dragos.com/blog/frostygoop-ics-malware/
            - NVD CVE Database: https://nvd.nist.gov/

            *Report generated by ICS Network Anomaly Detection System.*
            *For incident response guidance, contact your OT security team.*
        """)


# ---------------------------------------------------------------------------
# PDF renderer
# ---------------------------------------------------------------------------

class _PDFRenderer:

    def __init__(self, detections: dict, flows_df: Optional[pd.DataFrame], meta: dict):
        if not _REPORTLAB:
            raise ImportError("reportlab required: pip install reportlab>=3.6.0")
        self.detections = detections
        self.flows_df = flows_df
        self.meta = meta
        self.detected = {
            k: v for k, v in detections.items()
            if isinstance(v, dict) and v.get("detected", bool(v))
        }
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        _custom_styles = [
            ParagraphStyle(
                name="ReportTitle",
                parent=self.styles["Title"],
                fontSize=20,
                spaceAfter=6,
                textColor=colors.HexColor("#1a252f"),
            ),
            ParagraphStyle(
                name="SectionHeading",
                parent=self.styles["Heading1"],
                fontSize=13,
                spaceBefore=14,
                spaceAfter=6,
                textColor=colors.HexColor("#2c3e50"),
                borderPad=(0, 0, 3, 0),
            ),
            ParagraphStyle(
                name="SubHeading",
                parent=self.styles["Heading2"],
                fontSize=11,
                spaceBefore=8,
                spaceAfter=4,
                textColor=colors.HexColor("#34495e"),
            ),
            ParagraphStyle(
                name="ICSBodyText",
                parent=self.styles["Normal"],
                fontSize=9,
                leading=14,
                spaceAfter=4,
            ),
            ParagraphStyle(
                name="ICSCode",
                parent=self.styles["Normal"],
                fontSize=8,
                backColor=colors.HexColor("#f8f9fa"),
                borderColor=colors.HexColor("#dee2e6"),
                borderWidth=0.5,
                borderPad=4,
                leading=12,
                fontName="Courier",
            ),
            ParagraphStyle(
                name="BulletItem",
                parent=self.styles["Normal"],
                fontSize=9,
                leftIndent=16,
                leading=13,
                spaceAfter=3,
            ),
        ]
        for style in _custom_styles:
            try:
                self.styles.add(style)
            except KeyError:
                # Style already exists in reportlab's default stylesheet — skip.
                pass

    def build(self, out_path: Path) -> None:
        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=2*cm,
            rightMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm,
            title=f"ICS Incident Report {self.meta.get('incident_id', '')}",
            author="ICS Network Anomaly Detection System",
        )
        story = []
        story += self._cover_page()
        story += self._summary_section()
        story += self._timeline_section()
        story += self._detections_section()
        story += self._mitre_section()
        story += self._iec62443_pdf_section()
        story += self._mitigations_section()
        story += self._stats_section()
        story += self._references_section()
        doc.build(story)

    def _p(self, text: str, style: str = "ICSBodyText") -> Paragraph:
        return Paragraph(text, self.styles[style])

    def _hr(self) -> HRFlowable:
        return HRFlowable(width="100%", thickness=0.5,
                          color=colors.HexColor("#bdc3c7"), spaceAfter=6)

    def _cover_page(self) -> list:
        incident_id = self.meta.get("incident_id", "ICS-001")
        story = [
            Spacer(1, 1.5*cm),
            self._p("ICS Security Incident Report", "ReportTitle"),
            self._hr(),
            Spacer(1, 0.4*cm),
            self._p(f"<b>Incident ID:</b> {incident_id}"),
            self._p(f"<b>Generated:</b> {_today_str()}"),
            self._p("<b>Classification:</b> TLP:AMBER — Restricted distribution"),
            self._p(f"<b>Dataset:</b> {self.meta.get('dataset', 'ICSSIM / Live PCAP')}"),
            self._p("<b>Tool:</b> ICS Network Anomaly Detection System v2.0.0"),
            Spacer(1, 0.5*cm),
            self._hr(),
            Spacer(1, 0.3*cm),
        ]
        return story

    def _summary_section(self) -> list:
        story = [self._p("Executive Summary", "SectionHeading")]
        n_detected = len(self.detected)

        if not n_detected:
            story.append(self._p("No attack patterns detected in the analysed traffic window."))
            return story

        severities = [_severity(v) for v in self.detected.values()]
        max_sev = max(severities, key=lambda s: SEVERITY_RANK.get(s, 0))
        total_flows = sum(_n_flows(v) for v in self.detected.values())

        sev_color = SEVERITY_COLOR_PDF.get(max_sev, colors.grey)

        data = [
            ["Metric", "Value"],
            ["Patterns detected", str(n_detected)],
            ["Highest severity", max_sev.upper()],
            ["Total flagged flows", f"{total_flows:,}"],
            ["Flows analysed", f"{len(self.flows_df):,}" if self.flows_df is not None else "N/A"],
        ]
        t = Table(data, colWidths=[6*cm, 10*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3f4")]),
            ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*cm))
        return story

    def _timeline_section(self) -> list:
        story = [self._p("Attack Timeline", "SectionHeading")]
        if not self.detected:
            story.append(self._p("No attacks detected."))
            return story

        data = [["#", "Pattern", "Severity", "Flows", "Affected IPs"]]
        sorted_patterns = sorted(
            self.detected.items(),
            key=lambda kv: SEVERITY_RANK.get(_severity(kv[1]), 0),
            reverse=True,
        )
        for i, (pattern, result) in enumerate(sorted_patterns, 1):
            sev = _severity(result).upper()
            name = pattern.replace("_", " ").title()
            n = _n_flows(result)
            ips = ", ".join(_affected_ips(result)[:2]) or "N/A"
            data.append([str(i), name, sev, str(n), ips])

        col_widths = [1*cm, 5*cm, 2.5*cm, 2*cm, 5.5*cm]
        t = Table(data, colWidths=col_widths)

        sev_styles = []
        for row_i, (_, result) in enumerate(sorted_patterns, 1):
            sev = _severity(result)
            c = SEVERITY_COLOR_PDF.get(sev, colors.grey)
            sev_styles.append(("TEXTCOLOR", (2, row_i), (2, row_i), c))
            sev_styles.append(("FONTNAME",  (2, row_i), (2, row_i), "Helvetica-Bold"))

        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3f4")]),
            ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ] + sev_styles))
        story.append(t)
        story.append(Spacer(1, 0.3*cm))
        return story

    def _detections_section(self) -> list:
        story = [self._p("Detection Details", "SectionHeading")]
        for pattern, result in self.detected.items():
            name = pattern.replace("_", " ").title()
            sev = _severity(result).upper()
            story.append(self._p(f"{name}  [{sev}]", "SubHeading"))
            items = [
                f"<b>Severity:</b> {sev}",
                f"<b>Confidence:</b> {result.get('confidence', 'N/A')}",
                f"<b>Flagged flows:</b> {_n_flows(result):,}",
                f"<b>Detection source:</b> {result.get('detection_source', 'ensemble_ml')}",
            ]
            ips = _affected_ips(result)
            if ips:
                items.append(f"<b>Affected IPs:</b> {', '.join(ips[:4])}")
            cves = result.get("related_cves", [])
            if cves:
                cve_str = ", ".join(f"{c['cve_id']} (CVSS {c.get('cvss_score', 'N/A')})" for c in cves[:3])
                items.append(f"<b>Related CVEs:</b> {cve_str}")
            for item in items:
                story.append(self._p(f"• {item}", "BulletItem"))
            story.append(Spacer(1, 0.15*cm))
        return story

    def _mitre_section(self) -> list:
        relevant = {k: MITRE_ICS[k] for k in self.detected if k in MITRE_ICS}
        if not relevant:
            return []
        story = [self._p("MITRE ATT&CK ICS Mapping", "SectionHeading")]
        data = [["Pattern", "Technique ID", "Technique Name", "Tactic"]]
        seen = set()
        for pattern, info in relevant.items():
            tid = info["technique_id"]
            if tid in seen:
                continue
            seen.add(tid)
            data.append([
                pattern.replace("_", " ").title(),
                tid,
                info["technique_name"],
                info["tactic"],
            ])
        t = Table(data, colWidths=[4.5*cm, 2.5*cm, 5*cm, 4*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3f4")]),
            ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*cm))
        return story

    def _iec62443_pdf_section(self) -> list:
        relevant = {k: v for k, v in IEC62443_VIOLATIONS.items() if k in self.detected}
        if not relevant:
            return []
        story = [self._p("IEC 62443 Violations", "SectionHeading")]
        data = [["Pattern", "Security Requirement", "Description", "Required SL"]]
        for pattern, (sr, desc, sl) in relevant.items():
            data.append([pattern.replace("_", " ").title(), sr, desc, sl])
        t = Table(data, colWidths=[4*cm, 3*cm, 7*cm, 2*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3f4")]),
            ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*cm))
        return story

    def _mitigations_section(self) -> list:
        story = [self._p("Recommended Mitigations", "SectionHeading")]
        sorted_patterns = sorted(
            self.detected.keys(),
            key=lambda k: SEVERITY_RANK.get(_severity(self.detected[k]), 0),
            reverse=True,
        )
        for pattern in sorted_patterns:
            name = pattern.replace("_", " ").title()
            sev = _severity(self.detected[pattern]).upper()
            story.append(self._p(f"{name}  [{sev}]", "SubHeading"))
            steps = MITIGATIONS.get(pattern, ["Review network segmentation and access controls."])
            for i, step in enumerate(steps, 1):
                story.append(self._p(f"{i}. {step}", "BulletItem"))
            story.append(Spacer(1, 0.1*cm))
        return story

    def _stats_section(self) -> list:
        if self.flows_df is None or self.flows_df.empty:
            return []
        story = [self._p("Flow Statistics", "SectionHeading")]
        story.append(self._p(f"Total flows analysed: <b>{len(self.flows_df):,}</b>"))

        stat_rows = [["Feature", "Mean", "Max", "Min"]]
        for col in ("packet_rate", "byte_rate", "duration", "src_packet_count"):
            if col in self.flows_df.columns:
                stat_rows.append([
                    col,
                    f"{self.flows_df[col].mean():.2f}",
                    f"{self.flows_df[col].max():.2f}",
                    f"{self.flows_df[col].min():.2f}",
                ])
        if len(stat_rows) > 1:
            t = Table(stat_rows, colWidths=[5*cm, 3.5*cm, 3.5*cm, 3.5*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3f4")]),
                ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
                ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                ("TOPPADDING",   (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ]))
            story.append(t)
        return story

    def _references_section(self) -> list:
        story = [
            Spacer(1, 0.5*cm),
            self._hr(),
            self._p("References", "SectionHeading"),
        ]
        refs = [
            "MITRE ATT&CK ICS: https://attack.mitre.org/matrices/ics/",
            "IEC 62443 Standard: https://www.iec.ch/homepage",
            "CISA ICS Advisories: https://www.cisa.gov/ics-advisories",
            "FrostyGoop Malware Analysis: https://www.dragos.com/blog/frostygoop-ics-malware/",
            "NVD CVE Database: https://nvd.nist.gov/",
        ]
        for ref in refs:
            story.append(self._p(f"• {ref}", "BulletItem"))
        story.append(Spacer(1, 0.3*cm))
        story.append(self._p(
            "<i>Report generated by ICS Network Anomaly Detection System. "
            "For incident response guidance, contact your OT security team.</i>",
            "ICSBodyText",
        ))
        return story


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ICSIncidentReporter:
    """
    Generates structured incident reports (Markdown + PDF) from detection output.

    Parameters
    ----------
    incident_id : str
        Identifier for this incident (e.g. "ICS-2025-001"). Auto-generated if not given.
    dataset_name : str
        Label for the data source shown in the report header.
    """

    def __init__(
        self,
        incident_id: Optional[str] = None,
        dataset_name: str = "ICSSIM / Live PCAP",
    ):
        self.incident_id = incident_id or f"ICS-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        self.dataset_name = dataset_name

    def generate(
        self,
        detections: Dict[str, Any],
        flows_df: Optional[pd.DataFrame] = None,
        out_dir: str | Path = "results",
        formats: List[str] | None = None,
    ) -> Dict[str, Path]:
        """
        Generate incident report in specified formats.

        Parameters
        ----------
        detections : dict
            Output of ICSAttackPatternLibrary.detect_all_patterns().
        flows_df : pd.DataFrame | None
            Flows DataFrame for statistics and timeline enrichment.
        out_dir : str | Path
            Output directory.
        formats : list[str] | None
            ["markdown", "pdf"] or subset. None = all available.

        Returns
        -------
        dict: {"markdown": Path, "pdf": Path}  (only keys for formats generated)
        """
        if formats is None:
            formats = ["markdown", "pdf"]

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "incident_id": self.incident_id,
            "dataset": self.dataset_name,
        }

        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_paths = {}

        if "markdown" in formats:
            renderer = _MarkdownRenderer(detections, flows_df, meta)
            md_content = renderer.render()
            md_path = out_dir / f"incident_report_{ts}.md"
            md_path.write_text(md_content, encoding="utf-8")
            output_paths["markdown"] = md_path
            print(f"[reporter] Markdown → {md_path}")

        if "pdf" in formats:
            if not _REPORTLAB:
                print("[reporter] WARNING: reportlab not installed — skipping PDF. pip install reportlab")
            else:
                renderer = _PDFRenderer(detections, flows_df, meta)
                pdf_path = out_dir / f"incident_report_{ts}.pdf"
                renderer.build(pdf_path)
                output_paths["pdf"] = pdf_path
                print(f"[reporter] PDF      → {pdf_path}")

        return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _mock_detections() -> dict:
    return {
        "modbus_flooding": {
            "detected": True,
            "severity": "critical",
            "confidence": "high",
            "detection_source": "ensemble_ml",
            "flows": [
                {"src_ip": "10.0.0.52", "dst_ip": "192.168.1.10", "packet_rate": 9800},
            ] * 12,
        },
        "plc_scanning": {
            "detected": True,
            "severity": "high",
            "confidence": "high",
            "detection_source": "dpi_modbus",
            "flows": [
                {"src_ip": "10.0.0.53", "dst_ip": "192.168.1.10"},
            ] * 256,
        },
        "unauthorized_write": {
            "detected": True,
            "severity": "high",
            "confidence": "medium",
            "detection_source": "dpi_modbus",
            "related_cves": [
                {"cve_id": "CVE-2022-29952", "cvss_score": 9.8,
                 "description": "Advantech iView unauthenticated write via Modbus"},
                {"cve_id": "CVE-2021-22763", "cvss_score": 7.5,
                 "description": "Schneider Electric EcoStruxure unauthorized write"},
            ],
            "flows": [
                {"src_ip": "10.0.0.51", "dst_ip": "192.168.1.10"},
            ] * 4,
        },
        "man_in_the_middle": {
            "detected": True,
            "severity": "critical",
            "confidence": "high",
            "detection_source": "dpi_modbus",
            "flows": [
                {"src_ip": "10.0.0.54", "dst_ip": "192.168.1.10"},
            ] * 8,
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ICS Incident Report")
    parser.add_argument("--input",  help="Path to detections JSON")
    parser.add_argument("--out",    default="results", help="Output directory (default: results/)")
    parser.add_argument("--formats", nargs="+", choices=["markdown", "pdf"],
                        default=["markdown", "pdf"])
    parser.add_argument("--demo",   action="store_true",
                        help="Use mock detections (no --input needed)")
    parser.add_argument("--id",     default=None, help="Incident ID string")
    args = parser.parse_args()

    if args.demo or not args.input:
        detections = _mock_detections()
        print("[reporter] Using mock detections (--demo mode)")
    else:
        with open(args.input) as f:
            detections = json.load(f)

    reporter = ICSIncidentReporter(incident_id=args.id)
    paths = reporter.generate(
        detections=detections,
        out_dir=args.out,
        formats=args.formats,
    )
    for fmt, path in paths.items():
        print(f"[reporter] {fmt.upper()} → {path}")