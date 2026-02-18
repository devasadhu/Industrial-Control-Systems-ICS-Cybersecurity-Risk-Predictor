"""
ICS Protocol Deep Packet Inspection
Analyzes Modbus, DNP3, EtherNet/IP, S7comm, OPC UA

Author: Sadhana Devarajan
Version: 1.0.0
"""

from scapy.all import *
from typing import Dict, List, Optional
import logging
import struct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICSProtocolAnalyzer:
    """Deep packet inspection for industrial control system protocols."""
    
    CRITICAL_PORTS = {
        502: "Modbus/TCP",
        20000: "DNP3",
        44818: "EtherNet/IP",
        102: "S7comm (Siemens)",
        4840: "OPC UA",
        47808: "BACnet",
        1911: "Niagara Fox",
        789: "Red Lion Crimson",
        2222: "EtherNet/IP"
    }
    
    MODBUS_FUNCTIONS = {
        0x01: "Read Coils",
        0x02: "Read Discrete Inputs",
        0x03: "Read Holding Registers",
        0x04: "Read Input Registers",
        0x05: "Write Single Coil",
        0x06: "Write Single Register",
        0x0F: "Write Multiple Coils",
        0x10: "Write Multiple Registers",
        0x14: "Read File Record",
        0x15: "Write File Record",
        0x16: "Mask Write Register",
        0x17: "Read/Write Multiple Registers"
    }
    
    # Critical Modbus registers (example for Schneider Electric PLCs)
    CRITICAL_REGISTERS = {
        range(0, 100): "System Configuration",
        range(100, 200): "Safety Interlocks",
        range(200, 300): "Process Control",
        range(1000, 2000): "Alarm Settings"
    }
    
    def __init__(self):
        """Initialize protocol analyzer."""
        self.alerts = []
        self.protocol_stats = {proto: 0 for proto in self.CRITICAL_PORTS.values()}
    
    def analyze_packet(self, packet) -> Optional[Dict]:
        """
        Analyze packet for any ICS protocol.
        
        Returns detailed analysis or None.
        """
        if not packet.haslayer(TCP):
            return None
        
        dst_port = packet[TCP].dport
        
        # Check if it's a known ICS port
        if dst_port not in self.CRITICAL_PORTS:
            return None
        
        # Route to specific analyzer
        if dst_port == 502:
            return self.analyze_modbus(packet)
        elif dst_port == 20000:
            return self.analyze_dnp3(packet)
        elif dst_port == 44818 or dst_port == 2222:
            return self.analyze_ethernetip(packet)
        elif dst_port == 102:
            return self.analyze_s7comm(packet)
        elif dst_port == 4840:
            return self.analyze_opcua(packet)
        else:
            return {
                'protocol': self.CRITICAL_PORTS[dst_port],
                'port': dst_port,
                'severity': 'MEDIUM',
                'alert': f'{self.CRITICAL_PORTS[dst_port]} detected on port {dst_port}',
                'details': {}
            }
    
    def analyze_modbus(self, packet) -> Optional[Dict]:
        """
        Deep analysis of Modbus/TCP packet.
        
        Modbus TCP/IP Frame Structure:
        - Transaction ID (2 bytes)
        - Protocol ID (2 bytes) - always 0x0000
        - Length (2 bytes)
        - Unit ID (1 byte)
        - Function Code (1 byte)
        - Data (variable)
        """
        if not packet.haslayer(TCP) or packet[TCP].dport != 502:
            return None
        
        payload = bytes(packet[TCP].payload)
        if len(payload) < 8:
            return None
        
        try:
            # Parse Modbus TCP/IP header
            transaction_id = struct.unpack('>H', payload[0:2])[0]
            protocol_id = struct.unpack('>H', payload[2:4])[0]
            length = struct.unpack('>H', payload[4:6])[0]
            unit_id = payload[6]
            function_code = payload[7]
            
            # Validate protocol ID
            if protocol_id != 0:
                return {
                    'protocol': 'Modbus/TCP',
                    'severity': 'CRITICAL',
                    'alert': f'Invalid Modbus protocol ID: 0x{protocol_id:04X} (should be 0x0000)',
                    'details': {
                        'transaction_id': transaction_id,
                        'protocol_id': protocol_id,
                        'is_malformed': True
                    }
                }
            
            analysis = {
                'protocol': 'Modbus/TCP',
                'transaction_id': transaction_id,
                'unit_id': unit_id,
                'function_code': function_code,
                'function_name': self.MODBUS_FUNCTIONS.get(function_code, f'Unknown (0x{function_code:02X})'),
                'is_write': function_code in [0x05, 0x06, 0x0F, 0x10, 0x15, 0x16],
                'is_read': function_code in [0x01, 0x02, 0x03, 0x04, 0x14],
                'severity': 'LOW',
                'details': {}
            }
            
            # Parse data based on function code
            if len(payload) > 8:
                data = payload[8:]
                
                # Read Holding Registers (0x03)
                if function_code == 0x03 and len(data) >= 4:
                    start_addr = struct.unpack('>H', data[0:2])[0]
                    num_regs = struct.unpack('>H', data[2:4])[0]
                    
                    analysis['details']['start_address'] = start_addr
                    analysis['details']['register_count'] = num_regs
                    
                    # Check if accessing critical registers
                    if self._is_critical_register(start_addr):
                        analysis['severity'] = 'HIGH'
                        analysis['alert'] = f'Read access to critical registers: {start_addr}-{start_addr + num_regs}'
                
                # Write Single Register (0x06)
                elif function_code == 0x06 and len(data) >= 4:
                    reg_addr = struct.unpack('>H', data[0:2])[0]
                    reg_value = struct.unpack('>H', data[2:4])[0]
                    
                    analysis['details']['register_address'] = reg_addr
                    analysis['details']['register_value'] = reg_value
                    analysis['severity'] = 'HIGH'
                    analysis['alert'] = f'Modbus WRITE to register {reg_addr} = {reg_value}'
                    
                    if self._is_critical_register(reg_addr):
                        analysis['severity'] = 'CRITICAL'
                        analysis['alert'] = f'CRITICAL: Write to protected register {reg_addr}'
                
                # Write Multiple Registers (0x10)
                elif function_code == 0x10 and len(data) >= 5:
                    start_addr = struct.unpack('>H', data[0:2])[0]
                    num_regs = struct.unpack('>H', data[2:4])[0]
                    byte_count = data[4]
                    
                    analysis['details']['start_address'] = start_addr
                    analysis['details']['register_count'] = num_regs
                    analysis['details']['byte_count'] = byte_count
                    analysis['severity'] = 'CRITICAL'
                    analysis['alert'] = f'Multiple register write: {num_regs} registers starting at {start_addr}'
            
            # Flag unusual function codes
            if function_code > 0x18 or function_code not in self.MODBUS_FUNCTIONS:
                analysis['severity'] = 'CRITICAL'
                analysis['alert'] = f'Unusual/malicious Modbus function code: 0x{function_code:02X}'
                analysis['details']['suspicious'] = True
            
            # Default alert for writes
            if analysis['is_write'] and 'alert' not in analysis:
                analysis['alert'] = f"Modbus {analysis['function_name']}"
            elif analysis['is_read'] and 'alert' not in analysis:
                analysis['alert'] = f"Modbus {analysis['function_name']}"
            
            self.protocol_stats['Modbus/TCP'] += 1
            return analysis
            
        except Exception as e:
            logger.error(f"Modbus parsing error: {e}")
            return {
                'protocol': 'Modbus/TCP',
                'severity': 'MEDIUM',
                'alert': 'Malformed Modbus packet',
                'details': {'error': str(e)}
            }
    
    def analyze_dnp3(self, packet) -> Optional[Dict]:
        """
        Analyze DNP3 packet (common in electrical utilities).
        
        DNP3 is used in SCADA systems for power grids, water treatment.
        """
        if not packet.haslayer(TCP) or packet[TCP].dport != 20000:
            return None
        
        payload = bytes(packet[TCP].payload)
        if len(payload) < 10:
            return None
        
        # DNP3 frame starts with 0x0564
        start_bytes = payload[0:2]
        
        if start_bytes != b'\x05\x64':
            return None
        
        # Parse DNP3 header
        length = payload[2]
        control = payload[3]
        dest_addr = struct.unpack('<H', payload[4:6])[0]
        src_addr = struct.unpack('<H', payload[6:8])[0]
        
        analysis = {
            'protocol': 'DNP3',
            'severity': 'HIGH',
            'alert': 'DNP3 SCADA protocol detected',
            'details': {
                'destination_address': dest_addr,
                'source_address': src_addr,
                'control': control,
                'frame_length': length
            }
        }
        
        # DNP3 in non-utility networks is suspicious
        if dest_addr > 65000 or src_addr > 65000:
            analysis['severity'] = 'CRITICAL'
            analysis['alert'] = 'DNP3 with unusual addressing detected'
        
        self.protocol_stats['DNP3'] += 1
        return analysis
    
    def analyze_ethernetip(self, packet) -> Optional[Dict]:
        """
        Analyze EtherNet/IP (Rockwell Automation, Allen-Bradley).
        
        Used in manufacturing automation, conveyor systems.
        """
        dst_port = packet[TCP].dport
        
        if dst_port not in [44818, 2222]:
            return None
        
        payload = bytes(packet[TCP].payload)
        
        analysis = {
            'protocol': 'EtherNet/IP',
            'port': dst_port,
            'severity': 'MEDIUM',
            'alert': 'EtherNet/IP communication (Rockwell PLC)',
            'details': {
                'vendor': 'Rockwell Automation / Allen-Bradley',
                'typical_use': 'Industrial automation, PLCs'
            }
        }
        
        # Check for encapsulation header
        if len(payload) >= 24:
            # EtherNet/IP encapsulation format
            command = struct.unpack('<H', payload[0:2])[0]
            
            # Common commands
            commands = {
                0x0004: 'ListServices',
                0x0063: 'ListIdentity',
                0x0064: 'ListInterfaces',
                0x0065: 'RegisterSession',
                0x0066: 'UnRegisterSession',
                0x006F: 'SendRRData',
                0x0070: 'SendUnitData'
            }
            
            analysis['details']['command'] = commands.get(command, f'Unknown (0x{command:04X})')
            
            # Write commands are high risk
            if command in [0x006F, 0x0070]:
                analysis['severity'] = 'HIGH'
                analysis['alert'] = f'EtherNet/IP write command: {analysis["details"]["command"]}'
        
        self.protocol_stats['EtherNet/IP'] += 1
        return analysis
    
    def analyze_s7comm(self, packet) -> Optional[Dict]:
        """
        Analyze S7comm (Siemens PLCs).
        
        S7-300, S7-400, S7-1200, S7-1500 series.
        """
        if not packet.haslayer(TCP) or packet[TCP].dport != 102:
            return None
        
        payload = bytes(packet[TCP].payload)
        
        analysis = {
            'protocol': 'S7comm',
            'vendor': 'Siemens',
            'severity': 'HIGH',
            'alert': 'S7comm detected (Siemens PLC protocol)',
            'details': {
                'typical_devices': 'S7-300, S7-400, S7-1200, S7-1500',
                'risk': 'Direct PLC access'
            }
        }
        
        # S7comm uses COTP (ISO 8073)
        # Check for TPKT header (RFC 1006)
        if len(payload) >= 4:
            version = payload[0]
            if version == 0x03:  # TPKT version 3
                length = struct.unpack('>H', payload[2:4])[0]
                analysis['details']['tpkt_length'] = length
                analysis['severity'] = 'CRITICAL'
                analysis['alert'] = 'Active S7comm session (PLC programming/control)'
        
        self.protocol_stats['S7comm (Siemens)'] += 1
        return analysis
    
    def analyze_opcua(self, packet) -> Optional[Dict]:
        """
        Analyze OPC UA (OPC Unified Architecture).
        
        Modern industrial communication standard.
        """
        if not packet.haslayer(TCP) or packet[TCP].dport != 4840:
            return None
        
        analysis = {
            'protocol': 'OPC UA',
            'port': 4840,
            'severity': 'MEDIUM',
            'alert': 'OPC UA communication detected',
            'details': {
                'description': 'Industrial automation standard',
                'typical_use': 'Data exchange between industrial systems'
            }
        }
        
        self.protocol_stats['OPC UA'] += 1
        return analysis
    
    def _is_critical_register(self, address: int) -> bool:
        """Check if Modbus register is in critical range."""
        for reg_range, description in self.CRITICAL_REGISTERS.items():
            if address in reg_range:
                return True
        return False
    
    def get_statistics(self) -> Dict:
        """Get protocol statistics."""
        return {
            'protocols_detected': {k: v for k, v in self.protocol_stats.items() if v > 0},
            'total_packets': sum(self.protocol_stats.values()),
            'alerts': len(self.alerts)
        }
    
    def analyze_pcap_file(self, pcap_path: str) -> List[Dict]:
        """
        Analyze entire PCAP file for ICS protocols.
        
        Args:
            pcap_path: Path to PCAP file
            
        Returns:
            List of analysis results
        """
        logger.info(f"Analyzing PCAP: {pcap_path}")
        
        try:
            packets = rdpcap(pcap_path)
            results = []
            
            for packet in packets:
                analysis = self.analyze_packet(packet)
                if analysis:
                    results.append(analysis)
                    
                    # Track high-severity alerts
                    if analysis['severity'] in ['HIGH', 'CRITICAL']:
                        self.alerts.append(analysis)
            
            logger.info(f"✅ Analysis complete")
            logger.info(f"   Packets analyzed: {len(packets)}")
            logger.info(f"   ICS packets found: {len(results)}")
            logger.info(f"   High-severity alerts: {len(self.alerts)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze PCAP: {e}")
            return []


def demo_protocol_analyzer():
    """Demo the protocol analyzer."""
    print("="*80)
    print("ICS PROTOCOL DEEP PACKET INSPECTION")
    print("="*80)
    
    analyzer = ICSProtocolAnalyzer()
    
    # Check for PCAP files
    from pathlib import Path
    
    pcap_files = list(Path("./data").glob("*.pcap"))
    
    if not pcap_files:
        print("\n⚠️  No PCAP files found in data/")
        print("Generate test traffic with: python generate_test_pcap.py")
        return
    
    # Analyze each PCAP
    for pcap_file in pcap_files:
        print(f"\n📦 Analyzing: {pcap_file.name}")
        print("="*80)
        
        results = analyzer.analyze_pcap_file(str(pcap_file))
        
        # Show findings
        if results:
            print(f"\n🔍 Found {len(results)} ICS protocol packets:\n")
            
            for i, analysis in enumerate(results[:10], 1):  # Show first 10
                severity_icon = {
                    'LOW': '🟢',
                    'MEDIUM': '🟡',
                    'HIGH': '🟠',
                    'CRITICAL': '🔴'
                }[analysis['severity']]
                
                print(f"{i}. {severity_icon} {analysis['protocol']} - {analysis['severity']}")
                print(f"   Alert: {analysis['alert']}")
                if analysis.get('details'):
                    for key, value in list(analysis['details'].items())[:3]:
                        print(f"   • {key}: {value}")
                print()
        else:
            print("   No ICS protocols detected")
    
    # Statistics
    stats = analyzer.get_statistics()
    print("\n" + "="*80)
    print("📊 PROTOCOL STATISTICS")
    print("="*80)
    
    if stats['protocols_detected']:
        for protocol, count in stats['protocols_detected'].items():
            print(f"  • {protocol}: {count} packets")
    
    print(f"\n⚠️  High-severity alerts: {stats['alerts']}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    demo_protocol_analyzer()