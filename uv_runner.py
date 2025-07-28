#!/usr/bin/env python
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "torch==2.7.1",
#   "transformers>=4.30.0",
#   "sentence-transformers>=2.2.2",
#   "faiss-cpu>=1.7.4",
#   "numpy>=1.24.0",
#   "scikit-learn>=1.3.0",
#   "pandas>=2.0.0",
#   "tqdm>=4.65.0",
#   "python-dotenv>=1.0.0",
#   "requests",
# ]
# ///
"""
UV Runner for Medical Entity Code Mapper
Run with: uv run uv_runner.py
"""

import os
import sys
import time
import socket
import subprocess
from pathlib import Path
import json

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color

def print_status(message, status="info"):
    """Print colored status messages"""
    if status == "success":
        print(f"{GREEN}✓ {message}{NC}")
    elif status == "error":
        print(f"{RED}✗ {message}{NC}")
    elif status == "warning":
        print(f"{YELLOW}⚠ {message}{NC}")
    else:
        print(f"  {message}")

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

def check_models():
    """Check if models are downloaded"""
    models_dir = Path("data/models")
    required_models = ["clinical_ner", "disease_ner", "bge_m3"]
    
    print(f"\n{YELLOW}Checking models...{NC}")
    
    all_present = True
    for model in required_models:
        model_path = models_dir / model
        if model_path.exists() and any(model_path.iterdir()):
            print_status(f"Model '{model}' found", "success")
        else:
            print_status(f"Model '{model}' missing", "warning")
            all_present = False
    
    return all_present

def download_models():
    """Download models using UV run"""
    print(f"\n{YELLOW}Downloading models...{NC}")
    # Use subprocess to run UV command
    result = subprocess.run(
        ["uv", "run", "scripts/download_models.py", "--skip-optional"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print_status("Models downloaded successfully", "success")
        return True
    else:
        print_status(f"Failed to download models: {result.stderr}", "error")
        return False

def start_servers_parallel():
    """Start all servers in parallel"""
    print(f"\n{YELLOW}Starting servers...{NC}")
    
    servers = [
        ("ICD-10", "src/servers/icd10_server.py", 8901),
        ("SNOMED", "src/servers/snomed_server.py", 8902),
        ("LOINC", "src/servers/loinc_server.py", 8903),
        ("RxNorm", "src/servers/rxnorm_server.py", 8904),
        ("HCPCS", "src/servers/hcpcs_server.py", 8905),
    ]
    
    servers_to_start = []
    
    # Check which servers need to be started
    for name, script, port in servers:
        if check_port(port):
            print_status(f"{name} server already running on port {port}", "success")
        else:
            print_status(f"{name} server needs to be started", "warning")
            servers_to_start.append((name, script, port))
    
    # Start all servers that need starting
    if servers_to_start:
        print(f"\n{YELLOW}Starting {len(servers_to_start)} servers in parallel...{NC}")
        
        for name, script, port in servers_to_start:
            log_file = f"{name.lower().replace('-', '')}.log"
            with open(log_file, 'w') as log:
                subprocess.Popen(
                    ["uv", "run", "python", script],
                    stdout=log,
                    stderr=subprocess.STDOUT
                )
            print_status(f"Started {name} server process", "info")
        
        # Wait 20 seconds for all servers to start
        print(f"\n{YELLOW}Waiting 20 seconds for servers to initialize...{NC}")
        for i in range(20):
            sys.stdout.write(f"\r  Progress: {i+1}/20 seconds")
            sys.stdout.flush()
            time.sleep(1)
        print()  # New line
        
        # Check if all servers are now running
        print(f"\n{YELLOW}Checking server status...{NC}")
        all_running = True
        for name, script, port in servers:
            if check_port(port):
                print_status(f"{name} server is running on port {port}", "success")
            else:
                print_status(f"{name} server failed to start on port {port}", "error")
                all_running = False
        
        return all_running
    else:
        print_status("All servers already running", "success")
        return True

def test_servers():
    """Test all servers"""
    print(f"\n{YELLOW}Testing servers...{NC}")
    
    tests = [
        ("ICD-10", 8901, "1,chest pain"),
        ("SNOMED", 8902, "1,hypertension"),
        ("LOINC", 8903, "1,blood glucose"),
        ("RxNorm", 8904, "1,aspirin"),
        ("HCPCS", 8905, "1,urinary catheter"),
    ]
    
    all_passed = True
    
    for name, port, query in tests:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(('localhost', port))
            sock.send(f"{query}\n".encode())
            response = sock.recv(1024).decode().strip()
            sock.close()
            print_status(f"{name}: '{query}' → '{response}'", "success")
        except Exception as e:
            print_status(f"{name}: {e}", "error")
            all_passed = False
    
    return all_passed

def test_clinical_note():
    """Test with a full clinical note"""
    print(f"\n{YELLOW}Testing with clinical note...{NC}")
    
    # Sample clinical note
    clinical_note = """
    Chief Complaint: Chest pain and shortness of breath
    
    HPI: 68-year-old male presents with acute onset chest pain radiating to left arm. 
    Patient reports dyspnea on exertion. History of hypertension and diabetes mellitus type 2.
    Currently taking metformin 1000mg twice daily and lisinopril 10mg daily.
    
    Vitals: BP 165/95, HR 92, RR 22, O2 sat 94% on room air
    
    Labs: Troponin I elevated at 0.8 ng/mL, glucose 185 mg/dL, HbA1c 8.2%
    
    Assessment: Acute coronary syndrome, uncontrolled diabetes
    
    Plan: Admit for cardiac catheterization, start aspirin 325mg, adjust diabetes medications
    """
    
    print(f"\n{GREEN}Clinical Note:{NC}")
    print("-" * 60)
    print(clinical_note.strip())
    print("-" * 60)
    
    # Use subprocess to run the medical entity mapper
    print(f"\n{YELLOW}Processing note through Medical Entity Code Mapper...{NC}")
    
    result = subprocess.run(
        ["uv", "run", "python", "src/mappers/medical_entity_mapper.py"],
        input=clinical_note,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"\n{GREEN}Extracted Entities and Codes:{NC}")
        print("-" * 60)
        print(result.stdout)
        return True
    else:
        print_status(f"Failed to process note: {result.stderr}", "error")
        return False

def main():
    """Main execution flow"""
    print(f"{GREEN}🏥 Medical Entity Code Mapper - UV Runner{NC}")
    print("=" * 50)
    
    # Step 1: Check and download models if needed
    if not check_models():
        if not download_models():
            return 1
    else:
        print_status("All models already present", "success")
    
    # Step 2: Start servers in parallel
    if not start_servers_parallel():
        print_status("Failed to start all servers", "error")
        return 1
    
    # Step 3: Test servers
    if not test_servers():
        print_status("Some tests failed", "error")
        return 1
    
    # Step 4: Test with clinical note
    if test_clinical_note():
        print(f"\n{GREEN}✅ All tests passed! System is ready.{NC}")
        print("\nServers are running on:")
        print("  - ICD-10:  localhost:8901")
        print("  - SNOMED:  localhost:8902")
        print("  - LOINC:   localhost:8903")
        print("  - RxNorm:  localhost:8904")
        print("  - HCPCS:   localhost:8905")
        print(f"\n{YELLOW}Servers are running in background.{NC}")
        print(f"{YELLOW}To stop: pkill -f 'python.*server'{NC}")
    else:
        print_status("Clinical note test failed", "error")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())