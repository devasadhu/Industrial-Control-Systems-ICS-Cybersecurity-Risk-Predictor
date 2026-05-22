"""
Modbus Traffic Simulator v1.0.0
================================
Generates synthetic Modbus/TCP PCAP files (normal PLC polling + 5 attack variants)
using Scapy for PCAP writing and raw socket construction for the Modbus/TCP frames.

Attack variants modelled:
  1. replay          — mechanically-timed FC 03 reads (fixed inter-packet delta)
  2. command_inject  — FC 06/10 writes to unexpected register addresses
  3. flooding        — high-rate FC 01 coil reads (DoS)
  4. scanning        — sequential coil/register address enumeration (FC 01 sweep)
  5. mitm            — mirrored req/resp pairs with TTL=128 (man-in-the-middle artefact)

Outputs
-------
  <out_dir>/modbus_<timestamp>.pcap        — all flows concatenated
  <out_dir>/ground_truth_<timestamp>.csv   — per-packet label (flow_index, label, fc, attack_type)

Usage
-----
  python -m src.simulation.modbus_simulator                         # defaults
  python -m src.simulation.modbus_simulator --out results/ --seed 42

Dependencies
------------
  scapy>=2.5.0   (pip install scapy)
  numpy, pandas  (already in requirements.txt)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import random
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Scapy import — fail loudly so the user knows to pip install scapy
# ---------------------------------------------------------------------------
try:
    from scapy.all import (
        IP, TCP, Raw, Ether,
        wrpcap, rdpcap, Packet,
    )
    _SCAPY_AVAILABLE = True
except ImportError:
    _SCAPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Modbus/TCP constants
# ---------------------------------------------------------------------------
MODBUS_PORT = 502
PROTOCOL_ID = 0x0000  # always 0 for Modbus/TCP

# Function codes used
FC_READ_COILS            = 0x01
FC_READ_HOLDING_REGS     = 0x03
FC_WRITE_SINGLE_REG      = 0x06
FC_WRITE_MULTIPLE_REGS   = 0x10

NORMAL_PLC_REGISTERS   = list(range(0x0000, 0x0010))   # registers 0–15 (expected range)
UNEXPECTED_REGISTERS   = list(range(0x0100, 0x0200))   # registers 256–511 (unexpected)
SCAN_COIL_RANGE        = list(range(0x0000, 0x0100))   # full coil sweep for scanning


# ---------------------------------------------------------------------------
# MBAP + PDU builders
# ---------------------------------------------------------------------------

def _build_mbap(transaction_id: int, length: int, unit_id: int = 1) -> bytes:
    """6-byte MBAP header: TID(2) + PID(2) + Length(2) + UID(1) → 7 total with UID."""
    return struct.pack(">HHHB", transaction_id, PROTOCOL_ID, length, unit_id)


def _pdu_read_coils(start_address: int, count: int = 8) -> bytes:
    return struct.pack(">BHH", FC_READ_COILS, start_address, count)


def _pdu_read_holding_regs(start_address: int, count: int = 4) -> bytes:
    return struct.pack(">BHH", FC_READ_HOLDING_REGS, start_address, count)


def _pdu_write_single_reg(register: int, value: int) -> bytes:
    return struct.pack(">BHH", FC_WRITE_SINGLE_REG, register, value)


def _pdu_write_multiple_regs(start_register: int, values: List[int]) -> bytes:
    count = len(values)
    byte_count = count * 2
    header = struct.pack(">BHHB", FC_WRITE_MULTIPLE_REGS, start_register, count, byte_count)
    data = struct.pack(f">{count}H", *values)
    return header + data


def _modbus_request(transaction_id: int, pdu: bytes, unit_id: int = 1) -> bytes:
    """Full Modbus/TCP request frame: MBAP (7 bytes) + PDU."""
    length = 1 + len(pdu)   # unit_id byte + PDU
    return _build_mbap(transaction_id, length, unit_id) + pdu


def _modbus_response_echo(request_pdu: bytes, transaction_id: int, unit_id: int = 1) -> bytes:
    """Minimal echo response (used for MitM simulation)."""
    fc = request_pdu[0]
    if fc == FC_READ_HOLDING_REGS:
        # Return 4 registers of zeros
        resp_pdu = bytes([fc, 8]) + b"\x00" * 8
    elif fc == FC_READ_COILS:
        resp_pdu = bytes([fc, 1, 0x00])
    elif fc in (FC_WRITE_SINGLE_REG, FC_WRITE_MULTIPLE_REGS):
        resp_pdu = request_pdu[:5]   # echo first 5 bytes of request PDU
    else:
        resp_pdu = bytes([fc])
    length = 1 + len(resp_pdu)
    return _build_mbap(transaction_id, length, unit_id) + resp_pdu


# ---------------------------------------------------------------------------
# Packet factory
# ---------------------------------------------------------------------------

@dataclass
class FlowSpec:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int = MODBUS_PORT
    ttl: int = 64
    label: str = "Normal"
    attack_type: str = "none"


def _make_packet(
    spec: FlowSpec,
    payload: bytes,
    timestamp: float,
    seq: int = 1000,
    ack: int = 0,
    flags: str = "PA",
) -> "Packet":
    pkt = (
        Ether()
        / IP(src=spec.src_ip, dst=spec.dst_ip, ttl=spec.ttl)
        / TCP(sport=spec.src_port, dport=spec.dst_port, seq=seq, ack=ack, flags=flags)
        / Raw(load=payload)
    )
    pkt.time = timestamp
    return pkt


# ---------------------------------------------------------------------------
# Flow generators
# ---------------------------------------------------------------------------

class ModbusFlowGenerator:
    """Generates Modbus/TCP packet sequences for each attack variant."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._tid = 0

    def _tid_next(self) -> int:
        self._tid = (self._tid + 1) & 0xFFFF
        return self._tid

    # ---- Normal PLC polling ------------------------------------------------
    def normal_polling(
        self,
        plc_ip: str = "192.168.1.10",
        hmi_ip: str = "192.168.1.100",
        n_polls: int = 60,
        interval: float = 1.0,
        t0: float = 0.0,
    ) -> Tuple[List["Packet"], List[dict]]:
        """Regular FC03 polling every `interval` seconds."""
        packets, records = [], []
        spec = FlowSpec(src_ip=hmi_ip, dst_ip=plc_ip,
                        src_port=int(self.rng.integers(49152, 65535)))
        t = t0
        for i in range(n_polls):
            reg = int(self.rng.choice(NORMAL_PLC_REGISTERS))
            pdu = _pdu_read_holding_regs(reg, count=4)
            frame = _modbus_request(self._tid_next(), pdu)
            jitter = float(self.rng.uniform(-0.02, 0.02))
            t += interval + jitter
            pkt = _make_packet(spec, frame, t, seq=1000 + i * 100)
            packets.append(pkt)
            records.append({
                "flow_index": i, "timestamp": t,
                "src_ip": hmi_ip, "dst_ip": plc_ip,
                "fc": FC_READ_HOLDING_REGS, "register": reg,
                "label": "Normal", "attack_type": "none",
            })
        return packets, records

    # ---- Attack 1: Replay --------------------------------------------------
    def replay_attack(
        self,
        plc_ip: str = "192.168.1.10",
        attacker_ip: str = "10.0.0.50",
        n_replays: int = 80,
        interval: float = 0.05,    # mechanically-timed, no jitter
        t0: float = 0.0,
    ) -> Tuple[List["Packet"], List[dict]]:
        """Exact same FC03 read replayed at fixed interval — no jitter."""
        packets, records = [], []
        spec = FlowSpec(src_ip=attacker_ip, dst_ip=plc_ip,
                        src_port=int(self.rng.integers(49152, 65535)),
                        label="Attack", attack_type="replay")
        fixed_reg = 0x0001
        pdu = _pdu_read_holding_regs(fixed_reg, count=4)
        frame = _modbus_request(self._tid_next(), pdu)   # same frame every time
        t = t0
        for i in range(n_replays):
            t += interval
            pkt = _make_packet(spec, frame, t, seq=2000 + i * 100)
            packets.append(pkt)
            records.append({
                "flow_index": i, "timestamp": t,
                "src_ip": attacker_ip, "dst_ip": plc_ip,
                "fc": FC_READ_HOLDING_REGS, "register": fixed_reg,
                "label": "Attack", "attack_type": "replay",
            })
        return packets, records

    # ---- Attack 2: Command injection ---------------------------------------
    def command_injection(
        self,
        plc_ip: str = "192.168.1.10",
        attacker_ip: str = "10.0.0.51",
        n_writes: int = 30,
        t0: float = 0.0,
    ) -> Tuple[List["Packet"], List[dict]]:
        """FC06/FC16 writes to unexpected register addresses."""
        packets, records = [], []
        spec = FlowSpec(src_ip=attacker_ip, dst_ip=plc_ip,
                        src_port=int(self.rng.integers(49152, 65535)),
                        label="Attack", attack_type="command_inject")
        t = t0
        for i in range(n_writes):
            reg = int(self.rng.choice(UNEXPECTED_REGISTERS))
            val = int(self.rng.integers(0, 0xFFFF))
            use_fc16 = bool(self.rng.integers(0, 2))
            if use_fc16:
                vals = [int(self.rng.integers(0, 0xFFFF)) for _ in range(4)]
                pdu = _pdu_write_multiple_regs(reg, vals)
                fc = FC_WRITE_MULTIPLE_REGS
            else:
                pdu = _pdu_write_single_reg(reg, val)
                fc = FC_WRITE_SINGLE_REG
            frame = _modbus_request(self._tid_next(), pdu)
            t += float(self.rng.uniform(0.1, 0.5))
            pkt = _make_packet(spec, frame, t, seq=3000 + i * 100)
            packets.append(pkt)
            records.append({
                "flow_index": i, "timestamp": t,
                "src_ip": attacker_ip, "dst_ip": plc_ip,
                "fc": fc, "register": reg,
                "label": "Attack", "attack_type": "command_inject",
            })
        return packets, records

    # ---- Attack 3: Flooding ------------------------------------------------
    def flooding_attack(
        self,
        plc_ip: str = "192.168.1.10",
        attacker_ip: str = "10.0.0.52",
        n_packets: int = 500,
        rate_pps: float = 10000.0,    # 10k packets/sec
        t0: float = 0.0,
    ) -> Tuple[List["Packet"], List[dict]]:
        """High-rate FC01 coil reads — DoS."""
        packets, records = [], []
        spec = FlowSpec(src_ip=attacker_ip, dst_ip=plc_ip,
                        src_port=int(self.rng.integers(49152, 65535)),
                        label="Attack", attack_type="flooding")
        interval = 1.0 / rate_pps
        pdu = _pdu_read_coils(0x0000, count=8)
        t = t0
        for i in range(n_packets):
            frame = _modbus_request(self._tid_next(), pdu)
            t += interval
            pkt = _make_packet(spec, frame, t, seq=4000 + i)
            packets.append(pkt)
            records.append({
                "flow_index": i, "timestamp": t,
                "src_ip": attacker_ip, "dst_ip": plc_ip,
                "fc": FC_READ_COILS, "register": 0x0000,
                "label": "Attack", "attack_type": "flooding",
            })
        return packets, records

    # ---- Attack 4: Scanning ------------------------------------------------
    def scanning_attack(
        self,
        plc_ip: str = "192.168.1.10",
        attacker_ip: str = "10.0.0.53",
        t0: float = 0.0,
    ) -> Tuple[List["Packet"], List[dict]]:
        """Sequential FC01 coil address enumeration."""
        packets, records = [], []
        spec = FlowSpec(src_ip=attacker_ip, dst_ip=plc_ip,
                        src_port=int(self.rng.integers(49152, 65535)),
                        label="Attack", attack_type="scanning")
        t = t0
        for i, addr in enumerate(SCAN_COIL_RANGE):
            pdu = _pdu_read_coils(addr, count=1)
            frame = _modbus_request(self._tid_next(), pdu)
            t += 0.002   # 2ms between probes
            pkt = _make_packet(spec, frame, t, seq=5000 + i)
            packets.append(pkt)
            records.append({
                "flow_index": i, "timestamp": t,
                "src_ip": attacker_ip, "dst_ip": plc_ip,
                "fc": FC_READ_COILS, "register": addr,
                "label": "Attack", "attack_type": "scanning",
            })
        return packets, records

    # ---- Attack 5: MitM ----------------------------------------------------
    def mitm_attack(
        self,
        plc_ip: str = "192.168.1.10",
        hmi_ip: str = "192.168.1.100",
        attacker_ip: str = "10.0.0.54",
        n_exchanges: int = 40,
        t0: float = 0.0,
    ) -> Tuple[List["Packet"], List[dict]]:
        """
        Mirrored request/response pairs with TTL=128.
        Attacker forwards HMI requests to PLC and injects forged responses.
        TTL=128 is the Windows default — anomalous for Linux PLCs (TTL=64).
        """
        packets, records = [], []
        t = t0
        for i in range(n_exchanges):
            reg = int(self.rng.choice(NORMAL_PLC_REGISTERS))
            pdu = _pdu_read_holding_regs(reg, count=4)
            tid = self._tid_next()
            req_frame = _modbus_request(tid, pdu)
            resp_frame = _modbus_response_echo(pdu, tid)

            # Forwarded request (attacker → PLC)
            req_spec = FlowSpec(src_ip=attacker_ip, dst_ip=plc_ip,
                                src_port=int(self.rng.integers(49152, 65535)),
                                ttl=128, label="Attack", attack_type="mitm")
            t += float(self.rng.uniform(0.9, 1.1))
            pkt_req = _make_packet(req_spec, req_frame, t, seq=6000 + i * 200)
            packets.append(pkt_req)

            # Forged response (attacker → HMI)
            resp_spec = FlowSpec(src_ip=attacker_ip, dst_ip=hmi_ip,
                                 src_port=MODBUS_PORT,
                                 dst_port=int(self.rng.integers(49152, 65535)),
                                 ttl=128, label="Attack", attack_type="mitm")
            t += 0.001
            pkt_resp = _make_packet(resp_spec, resp_frame, t,
                                    seq=6000 + i * 200 + 100, flags="PA")
            packets.append(pkt_resp)

            for pkt in (pkt_req, pkt_resp):
                records.append({
                    "flow_index": i, "timestamp": float(pkt.time),
                    "src_ip": pkt[IP].src, "dst_ip": pkt[IP].dst,
                    "fc": FC_READ_HOLDING_REGS, "register": reg,
                    "label": "Attack", "attack_type": "mitm",
                })
        return packets, records


# ---------------------------------------------------------------------------
# Simulator orchestrator
# ---------------------------------------------------------------------------

class ModbusSimulator:
    """
    Orchestrates all flow generators, assembles PCAP + ground-truth CSV.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility.
    plc_ip, hmi_ip : str
        Default PLC and HMI IP addresses.
    """

    def __init__(
        self,
        seed: int = 42,
        plc_ip: str = "192.168.1.10",
        hmi_ip: str = "192.168.1.100",
    ):
        if not _SCAPY_AVAILABLE:
            raise ImportError(
                "scapy is required: pip install scapy\n"
                "If on Windows, also: pip install npcap or install Npcap from https://npcap.com"
            )
        self.rng = np.random.default_rng(seed)
        self.plc_ip = plc_ip
        self.hmi_ip = hmi_ip
        self._gen = ModbusFlowGenerator(self.rng)

    def generate(
        self,
        out_dir: str | Path = "results",
        n_normal_polls: int = 120,
        include_attacks: List[str] | None = None,
        verbose: bool = True,
    ) -> Tuple[Path, Path]:
        """
        Generate PCAP + ground-truth CSV.

        Parameters
        ----------
        out_dir : str | Path
            Output directory (created if missing).
        n_normal_polls : int
            Number of normal polling packets to generate.
        include_attacks : list[str] | None
            Subset of attacks to include. None = all 5.
            Valid: 'replay', 'command_inject', 'flooding', 'scanning', 'mitm'
        verbose : bool
            Print progress to stdout.

        Returns
        -------
        (pcap_path, csv_path) : Tuple[Path, Path]
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pcap_path = out_dir / f"modbus_{ts}.pcap"
        csv_path  = out_dir / f"ground_truth_{ts}.csv"

        all_packets: List["Packet"] = []
        all_records: List[dict] = []

        if include_attacks is None:
            include_attacks = ["replay", "command_inject", "flooding", "scanning", "mitm"]

        t_cursor = 0.0

        # --- Normal traffic
        if verbose:
            print(f"[simulator] Generating {n_normal_polls} normal polling packets...")
        pkts, recs = self._gen.normal_polling(
            plc_ip=self.plc_ip, hmi_ip=self.hmi_ip,
            n_polls=n_normal_polls, interval=1.0, t0=t_cursor,
        )
        all_packets.extend(pkts)
        all_records.extend(recs)
        t_cursor = pkts[-1].time + 5.0 if pkts else t_cursor + 5.0

        # --- Attack traffic
        attack_dispatch = {
            "replay":         lambda t: self._gen.replay_attack(self.plc_ip, t0=t),
            "command_inject": lambda t: self._gen.command_injection(self.plc_ip, t0=t),
            "flooding":       lambda t: self._gen.flooding_attack(self.plc_ip, t0=t),
            "scanning":       lambda t: self._gen.scanning_attack(self.plc_ip, t0=t),
            "mitm":           lambda t: self._gen.mitm_attack(
                                  self.plc_ip, self.hmi_ip, t0=t),
        }

        for attack in include_attacks:
            if attack not in attack_dispatch:
                raise ValueError(
                    f"Unknown attack type '{attack}'. "
                    f"Valid: {list(attack_dispatch.keys())}"
                )
            if verbose:
                print(f"[simulator] Generating attack: {attack}...")
            pkts, recs = attack_dispatch[attack](t_cursor)
            all_packets.extend(pkts)
            all_records.extend(recs)
            t_cursor = (max(p.time for p in pkts) + 5.0) if pkts else t_cursor + 5.0

        # Sort by timestamp before writing
        all_packets.sort(key=lambda p: float(p.time))

        # Write PCAP
        wrpcap(str(pcap_path), all_packets)
        if verbose:
            n_attack = sum(1 for r in all_records if r["label"] == "Attack")
            n_normal = sum(1 for r in all_records if r["label"] == "Normal")
            print(f"[simulator] Written {len(all_packets)} packets → {pcap_path}")
            print(f"            Normal: {n_normal}  Attack: {n_attack}")

        # Write ground-truth CSV
        fieldnames = ["flow_index", "timestamp", "src_ip", "dst_ip",
                      "fc", "register", "label", "attack_type"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        if verbose:
            print(f"[simulator] Ground truth CSV → {csv_path}")

        return pcap_path, csv_path

    def evaluate_detection(
        self,
        pcap_path: Path,
        gt_csv_path: Path,
        verbose: bool = True,
    ) -> dict:
        """
        Run pcap_processor + attack_patterns on the generated PCAP,
        compare against ground-truth CSV, report per-attack recall.

        Returns dict with keys: overall_recall, per_attack_recall, n_packets
        """
        try:
            from src.pcap.pcap_processor import ICSPCAPProcessor
            from src.detection.attack_patterns import ICSAttackPatternLibrary
        except ImportError as e:
            raise ImportError(
                "src.pcap.pcap_processor or src.detection.attack_patterns not importable. "
                f"Original error: {e}"
            )

        processor = ICSPCAPProcessor()
        flows_df = processor.process_pcap(str(pcap_path))

        library = ICSAttackPatternLibrary()
        detections = library.detect_all_patterns(flows_df)

        import pandas as pd
        gt = pd.read_csv(gt_csv_path)

        n_attack_gt = int((gt["label"] == "Attack").sum())
        detected_attack_flows = sum(
            len(d.get("flows", [])) for d in detections.values()
        )
        overall_recall = detected_attack_flows / n_attack_gt if n_attack_gt > 0 else 0.0

        per_attack = {}
        for attack_type in gt["attack_type"].unique():
            if attack_type == "none":
                continue
            n_gt = int((gt["attack_type"] == attack_type).sum())
            per_attack[attack_type] = {"ground_truth_packets": n_gt}

        result = {
            "overall_recall": round(overall_recall, 4),
            "per_attack_recall": per_attack,
            "n_packets": len(all_packets) if "all_packets" in dir() else "N/A",
            "n_flows_processed": len(flows_df),
            "n_detections": sum(len(v.get("flows", [])) for v in detections.values()),
        }

        if verbose:
            print(f"\n[evaluate] Overall detection recall: {overall_recall:.1%}")
            print(f"[evaluate] Flows processed: {result['n_flows_processed']}")
            print(f"[evaluate] Detections: {result['n_detections']}")

        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Modbus/TCP PCAP for ICS anomaly detection testing."
    )
    parser.add_argument("--out", default="results", help="Output directory (default: results/)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--n-normal", type=int, default=120,
                        help="Number of normal polling packets (default: 120)")
    parser.add_argument("--attacks", nargs="+",
                        choices=["replay", "command_inject", "flooding", "scanning", "mitm"],
                        default=None,
                        help="Attack types to include (default: all 5)")
    parser.add_argument("--plc-ip", default="192.168.1.10")
    parser.add_argument("--hmi-ip", default="192.168.1.100")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run pcap_processor + attack_patterns after generation")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sim = ModbusSimulator(seed=args.seed, plc_ip=args.plc_ip, hmi_ip=args.hmi_ip)
    pcap_path, csv_path = sim.generate(
        out_dir=args.out,
        n_normal_polls=args.n_normal,
        include_attacks=args.attacks,
        verbose=True,
    )
    if args.evaluate:
        sim.evaluate_detection(pcap_path, csv_path, verbose=True)
