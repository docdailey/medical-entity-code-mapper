#!/usr/bin/env python3
"""
Start all TLS-secured medical coding servers
Ensures certificates exist and starts servers with encryption
"""

import subprocess
import time
import socket
import sys
import os
from pathlib import Path

# Server configurations with TLS ports
TLS_SERVERS = [
    ("ICD-10 TLS", "src/servers/icd10_server_tls.py", 8911),
    ("SNOMED TLS", "src/servers/snomed_server_tls.py", 8912),
    ("LOINC TLS", "src/servers/loinc_server_tls.py", 8913),
    ("RxNorm TLS", "src/servers/rxnorm_server_tls.py", 8914),
    ("HCPCS TLS", "src/servers/hcpcs_server_tls.py", 8915),
]

def check_certificates():
    """Check if TLS certificates exist"""
    cert_file = os.getenv('TLS_CERT_FILE', 'certs/server.crt')
    key_file = os.getenv('TLS_KEY_FILE', 'certs/server.key')
    
    if not Path(cert_file).exists() or not Path(key_file).exists():
        print("⚠️  TLS certificates not found!")
        print("Generating self-signed certificates for development...")
        
        # Generate certificates
        subprocess.run([
            sys.executable, 'scripts/generate_certificates.py', 
            '--type', 'self-signed'
        ], check=True)
        
        print("✓ Certificates generated")
        return True
    
    print("✓ TLS certificates found")
    return True

def check_port(port):
    """Check if a port is in use"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except:
        return False

def kill_existing_servers():
    """Kill any existing server processes"""
    print("Stopping existing servers...")
    subprocess.run(["pkill", "-f", "python.*server.*tls"], stderr=subprocess.DEVNULL)
    time.sleep(2)

def start_servers():
    """Start all TLS servers"""
    processes = []
    
    # Set certificate environment variables
    cert_file = os.getenv('TLS_CERT_FILE', 'certs/server.crt')
    key_file = os.getenv('TLS_KEY_FILE', 'certs/server.key')
    
    env = os.environ.copy()
    env['TLS_CERT_FILE'] = cert_file
    env['TLS_KEY_FILE'] = key_file
    
    for name, script, port in TLS_SERVERS:
        if check_port(port):
            print(f"✗ {name} server already running on port {port}")
            continue
            
        print(f"Starting {name} server on port {port}...")
        log_file = f"{name.lower().replace(' ', '_')}.log"
        
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                [sys.executable, script],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env
            )
            processes.append((name, process, port))
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if it started successfully
        if check_port(port):
            print(f"✓ {name} server started successfully (TLS-secured)")
        else:
            print(f"✗ {name} server failed to start")
    
    return processes

def main():
    print("🔒 Starting TLS-Secured Medical Coding Servers")
    print("=" * 50)
    
    # Check certificates
    if not check_certificates():
        print("❌ Certificate check failed")
        return
    
    # Kill existing servers
    kill_existing_servers()
    
    # Start new servers
    processes = start_servers()
    
    print("\n" + "=" * 50)
    print("✅ TLS servers started successfully")
    print("\n🔒 Security Information:")
    print(f"- Certificate: {os.getenv('TLS_CERT_FILE', 'certs/server.crt')}")
    print(f"- Private Key: {os.getenv('TLS_KEY_FILE', 'certs/server.key')}")
    print("- Protocol: TLS 1.2+")
    print("- Ciphers: Strong ciphers only (ECDHE+AESGCM, etc.)")
    
    print("\nTLS Server Ports:")
    for name, _, port in TLS_SERVERS:
        if check_port(port):
            print(f"  - {name}: localhost:{port} (encrypted)")
    
    print("\n⚠️  Development Notice:")
    print("- Using self-signed certificates (not for production)")
    print("- For production, use CA-signed certificates")
    print("- Enable mutual TLS for additional security")
    
    print("\nTo test TLS connection:")
    print("python examples/tls_client_example.py")
    
    print("\nTo stop all servers: pkill -f 'python.*server.*tls'")

if __name__ == "__main__":
    main()