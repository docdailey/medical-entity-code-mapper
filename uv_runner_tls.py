#!/usr/bin/env python3
"""
UV Runner for TLS-Secured Medical Entity Code Mapper
====================================================

Production-ready launcher for running TLS-secured medical coding servers using UV.
UV is a fast Python package manager and runner written in Rust.

This script orchestrates the startup of all TLS-secured medical coding servers:
- ICD-10 diagnosis codes (port 8911)
- SNOMED CT clinical terms (port 8912)
- LOINC laboratory codes (port 8913)
- RxNorm medication codes (port 8914)
- HCPCS device/supply codes (port 8915)
- Entity mapper service (port 8920)
- Web interface (port 8921)

Requirements:
    - UV installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
    - Python 3.10+
    - All dependencies in pyproject.toml
    - TLS certificates in certs/ directory

Usage:
    uv run uv_runner_tls.py [options]

Options:
    --servers     Comma-separated list of servers to start (default: all)
    --host        Host to bind to (default: localhost)
    --cert-dir    Directory containing TLS certificates (default: certs)
    --log-level   Logging level (default: INFO)
    --no-web      Skip starting the web interface
    --verify-cert Enable certificate verification for inter-server communication

Examples:
    # Start all servers with TLS
    uv run uv_runner_tls.py
    
    # Start only ICD-10 and SNOMED servers
    uv run uv_runner_tls.py --servers icd10,snomed
    
    # Start with custom certificate directory
    uv run uv_runner_tls.py --cert-dir /path/to/certs

Security Notes:
    - All servers use TLS 1.2+ encryption
    - Strong cipher suites are enforced
    - Audit logging is enabled for compliance
    - Input validation prevents injection attacks
    - Self-signed certificates are generated for development

Author: Medical Entity Code Mapper Team
License: Modified MIT License (Non-commercial use)
"""

import argparse
import subprocess
import sys
import os
import time
import signal
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import socket
from datetime import datetime

# Configure logging with production settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/uv_runner_tls.log', mode='a')
    ]
)
logger = logging.getLogger('uv_runner_tls')


class TLSServerManager:
    """
    Manages lifecycle of TLS-secured medical coding servers.
    
    This class handles:
    - Server process management
    - Health checking
    - Graceful shutdown
    - Certificate validation
    - Resource monitoring
    """
    
    # Server configurations with their respective ports and scripts
    SERVERS = {
        'icd10': {
            'script': 'src/servers/icd10_server_tls.py',
            'port': 8911,
            'name': 'ICD-10 TLS Server',
            'health_check': 'tcp',
            'startup_time': 15
        },
        'snomed': {
            'script': 'src/servers/snomed_server_tls.py',
            'port': 8912,
            'name': 'SNOMED CT TLS Server',
            'health_check': 'tcp',
            'startup_time': 15
        },
        'loinc': {
            'script': 'src/servers/loinc_server_tls.py',
            'port': 8913,
            'name': 'LOINC TLS Server',
            'health_check': 'tcp',
            'startup_time': 15
        },
        'rxnorm': {
            'script': 'src/servers/rxnorm_server_tls.py',
            'port': 8914,
            'name': 'RxNorm TLS Server',
            'health_check': 'tcp',
            'startup_time': 15
        },
        'hcpcs': {
            'script': 'src/servers/hcpcs_server_tls.py',
            'port': 8915,
            'name': 'HCPCS TLS Server',
            'health_check': 'tcp',
            'startup_time': 10
        },
        'mapper': {
            'script': 'src/entity_mapper_server_tls.py',
            'port': 8920,
            'name': 'Entity Mapper TLS Server',
            'health_check': 'tcp',
            'startup_time': 15,
            'depends_on': ['icd10', 'snomed', 'loinc', 'rxnorm', 'hcpcs']
        },
        'web': {
            'script': 'src/web_interface_tls.py',
            'port': 8921,
            'name': 'Web Interface TLS Server',
            'health_check': 'https',
            'startup_time': 5,
            'depends_on': ['mapper']
        }
    }
    
    def __init__(self, host: str = 'localhost', cert_dir: str = 'certs', 
                 verify_cert: bool = False):
        """
        Initialize TLS Server Manager.
        
        Args:
            host: Host to bind servers to
            cert_dir: Directory containing TLS certificates
            verify_cert: Whether to verify certificates for inter-server communication
        """
        self.host = host
        self.cert_dir = Path(cert_dir)
        self.verify_cert = verify_cert
        self.processes: Dict[str, subprocess.Popen] = {}
        self.start_time = datetime.now()
        
        # Ensure required directories exist
        self._ensure_directories()
        
        # Check for UV installation
        self._check_uv_installation()
        
        # Validate certificates
        self._validate_certificates()
    
    def _ensure_directories(self):
        """Ensure required directories exist with proper permissions."""
        directories = ['logs', 'audit_logs', self.cert_dir]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True, mode=0o755)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def _check_uv_installation(self):
        """Verify UV is installed and accessible."""
        try:
            result = subprocess.run(['uv', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"UV version: {result.stdout.strip()}")
            else:
                raise RuntimeError("UV not found")
        except FileNotFoundError:
            logger.error("UV is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
            sys.exit(1)
    
    def _validate_certificates(self):
        """Validate TLS certificates exist or create self-signed for development."""
        cert_file = self.cert_dir / 'server.crt'
        key_file = self.cert_dir / 'server.key'
        
        if not cert_file.exists() or not key_file.exists():
            logger.warning("TLS certificates not found. Creating self-signed certificates...")
            self._create_self_signed_certificates()
        else:
            logger.info(f"Using existing certificates from {self.cert_dir}")
    
    def _create_self_signed_certificates(self):
        """Create self-signed certificates for development/testing."""
        try:
            # Generate private key
            subprocess.run([
                'openssl', 'genrsa', '-out', 
                str(self.cert_dir / 'server.key'), '2048'
            ], check=True, capture_output=True)
            
            # Generate certificate
            subprocess.run([
                'openssl', 'req', '-new', '-x509', '-key',
                str(self.cert_dir / 'server.key'),
                '-out', str(self.cert_dir / 'server.crt'),
                '-days', '365',
                '-subj', f'/C=US/ST=State/L=City/O=MedicalEntityMapper/CN={self.host}'
            ], check=True, capture_output=True)
            
            # Set secure permissions on private key
            os.chmod(str(self.cert_dir / 'server.key'), 0o600)
            
            logger.info("Created self-signed certificates successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create certificates: {e}")
            sys.exit(1)
    
    def start_server(self, server_name: str) -> bool:
        """
        Start a specific TLS server.
        
        Args:
            server_name: Name of the server to start
            
        Returns:
            True if server started successfully, False otherwise
        """
        if server_name not in self.SERVERS:
            logger.error(f"Unknown server: {server_name}")
            return False
        
        server_config = self.SERVERS[server_name]
        
        # Check dependencies
        if 'depends_on' in server_config:
            for dep in server_config['depends_on']:
                if dep not in self.processes or self.processes[dep].poll() is not None:
                    logger.warning(f"Dependency {dep} not running for {server_name}")
                    return False
        
        # Set environment variables for TLS
        env = os.environ.copy()
        env['TLS_CERT_FILE'] = str(self.cert_dir / 'server.crt')
        env['TLS_KEY_FILE'] = str(self.cert_dir / 'server.key')
        env['TLS_VERIFY'] = str(self.verify_cert).lower()
        env['PYTHONUNBUFFERED'] = '1'  # Ensure real-time logging
        
        # Start the server using UV
        cmd = ['uv', 'run', server_config['script']]
        
        try:
            logger.info(f"Starting {server_config['name']}...")
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[server_name] = process
            
            # Wait for server to be ready
            time.sleep(server_config.get('startup_time', 10))
            
            # Health check
            if self._health_check(server_name):
                logger.info(f"✓ {server_config['name']} started successfully on port {server_config['port']}")
                return True
            else:
                logger.error(f"✗ {server_config['name']} failed health check")
                self.stop_server(server_name)
                return False
                
        except Exception as e:
            logger.error(f"Failed to start {server_name}: {e}")
            return False
    
    def _health_check(self, server_name: str) -> bool:
        """
        Perform health check on a server.
        
        Args:
            server_name: Name of the server to check
            
        Returns:
            True if server is healthy, False otherwise
        """
        server_config = self.SERVERS[server_name]
        port = server_config['port']
        check_type = server_config.get('health_check', 'tcp')
        
        if check_type == 'tcp':
            return self._tcp_health_check(self.host, port)
        elif check_type == 'https':
            return self._https_health_check(self.host, port)
        else:
            return True
    
    def _tcp_health_check(self, host: str, port: int, timeout: int = 5) -> bool:
        """Check if TCP port is open and accepting connections."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.debug(f"TCP health check failed for {host}:{port}: {e}")
            return False
    
    def _https_health_check(self, host: str, port: int) -> bool:
        """Check if HTTPS endpoint is responding."""
        try:
            import requests
            response = requests.get(
                f"https://{host}:{port}/",
                verify=False,  # Self-signed cert
                timeout=5
            )
            return response.status_code < 500
        except Exception as e:
            logger.debug(f"HTTPS health check failed for {host}:{port}: {e}")
            return False
    
    def stop_server(self, server_name: str):
        """
        Stop a specific server gracefully.
        
        Args:
            server_name: Name of the server to stop
        """
        if server_name in self.processes:
            process = self.processes[server_name]
            if process.poll() is None:
                logger.info(f"Stopping {self.SERVERS[server_name]['name']}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {server_name}")
                    process.kill()
                    process.wait()
            del self.processes[server_name]
    
    def start_all(self, servers: Optional[List[str]] = None):
        """
        Start all specified servers in dependency order.
        
        Args:
            servers: List of servers to start (None for all)
        """
        if servers is None:
            servers = list(self.SERVERS.keys())
        
        # Start servers in dependency order
        started = set()
        max_iterations = len(servers) * 2
        iterations = 0
        
        while len(started) < len(servers) and iterations < max_iterations:
            iterations += 1
            for server in servers:
                if server in started:
                    continue
                
                # Check dependencies
                deps = self.SERVERS[server].get('depends_on', [])
                if all(dep in started for dep in deps):
                    if self.start_server(server):
                        started.add(server)
                    else:
                        logger.error(f"Failed to start {server}")
                        self.stop_all()
                        sys.exit(1)
        
        if len(started) < len(servers):
            logger.error("Failed to start all servers due to dependency issues")
            self.stop_all()
            sys.exit(1)
        
        logger.info(f"All {len(started)} servers started successfully")
    
    def stop_all(self):
        """Stop all running servers in reverse dependency order."""
        # Stop in reverse order to handle dependencies
        for server_name in reversed(list(self.processes.keys())):
            self.stop_server(server_name)
        logger.info("All servers stopped")
    
    def monitor(self):
        """
        Monitor running servers and restart if needed.
        
        This method runs continuously, checking server health
        and restarting failed servers.
        """
        logger.info("Starting server monitoring...")
        
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                for server_name, process in list(self.processes.items()):
                    # Check if process is still running
                    if process.poll() is not None:
                        logger.warning(f"{server_name} has crashed, restarting...")
                        self.start_server(server_name)
                    
                    # Perform health check
                    elif not self._health_check(server_name):
                        logger.warning(f"{server_name} failed health check, restarting...")
                        self.stop_server(server_name)
                        self.start_server(server_name)
                
                # Log uptime
                uptime = datetime.now() - self.start_time
                logger.debug(f"System uptime: {uptime}")
                
            except KeyboardInterrupt:
                logger.info("Monitoring interrupted")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def get_status(self) -> Dict[str, Dict[str, any]]:
        """
        Get status of all servers.
        
        Returns:
            Dictionary with server status information
        """
        status = {}
        for server_name, server_config in self.SERVERS.items():
            is_running = (server_name in self.processes and 
                         self.processes[server_name].poll() is None)
            is_healthy = self._health_check(server_name) if is_running else False
            
            status[server_name] = {
                'name': server_config['name'],
                'port': server_config['port'],
                'running': is_running,
                'healthy': is_healthy,
                'pid': self.processes[server_name].pid if is_running else None
            }
        
        return status


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    if 'manager' in globals():
        manager.stop_all()
    sys.exit(0)


def main():
    """Main entry point for UV TLS runner."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='UV Runner for TLS-Secured Medical Entity Code Mapper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--servers',
        help='Comma-separated list of servers to start (default: all)',
        default=None
    )
    parser.add_argument(
        '--host',
        help='Host to bind to (default: localhost)',
        default='localhost'
    )
    parser.add_argument(
        '--cert-dir',
        help='Directory containing TLS certificates (default: certs)',
        default='certs'
    )
    parser.add_argument(
        '--log-level',
        help='Logging level (default: INFO)',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO'
    )
    parser.add_argument(
        '--no-web',
        help='Skip starting the web interface',
        action='store_true'
    )
    parser.add_argument(
        '--verify-cert',
        help='Enable certificate verification for inter-server communication',
        action='store_true'
    )
    parser.add_argument(
        '--status',
        help='Show server status and exit',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create server manager
    global manager
    manager = TLSServerManager(
        host=args.host,
        cert_dir=args.cert_dir,
        verify_cert=args.verify_cert
    )
    
    # Handle status request
    if args.status:
        status = manager.get_status()
        print("\nServer Status:")
        print("-" * 60)
        for server_name, info in status.items():
            status_icon = "✓" if info['healthy'] else "✗" if info['running'] else "○"
            print(f"{status_icon} {info['name']:<30} Port: {info['port']:<6} PID: {info['pid'] or 'N/A'}")
        sys.exit(0)
    
    # Parse server list
    servers = None
    if args.servers:
        servers = [s.strip() for s in args.servers.split(',')]
    elif args.no_web:
        servers = [s for s in TLSServerManager.SERVERS.keys() if s != 'web']
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start servers
        logger.info("Starting TLS-secured medical coding servers...")
        manager.start_all(servers)
        
        # Show status
        print("\n" + "="*60)
        print("TLS-Secured Medical Entity Code Mapper")
        print("="*60)
        print(f"Host: {args.host}")
        print(f"Certificate Directory: {args.cert_dir}")
        print(f"Certificate Verification: {'Enabled' if args.verify_cert else 'Disabled'}")
        print("\nRunning Servers:")
        
        status = manager.get_status()
        for server_name, info in status.items():
            if info['running']:
                print(f"  - {info['name']}: https://{args.host}:{info['port']}")
        
        print("\nPress Ctrl+C to stop all servers")
        print("="*60 + "\n")
        
        # Monitor servers
        manager.monitor()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        manager.stop_all()
        sys.exit(1)


if __name__ == '__main__':
    main()