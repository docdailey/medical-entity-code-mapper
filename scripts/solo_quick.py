#!/usr/bin/env python3
"""Quick solo benchmark for each server"""

import time
import socket
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

def query_server(host, port, query, seq_num=1):
    """Send query to server and measure response time"""
    try:
        start = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((host, port))
        
        message = f"{seq_num},{query}\n"
        sock.send(message.encode('utf-8'))
        
        response = sock.recv(4096).decode('utf-8').strip()
        sock.close()
        
        latency = (time.time() - start) * 1000  # milliseconds
        success = 'ERROR' not in response and 'TIMEOUT' not in response
        
        return {'latency': latency, 'success': success}
    except Exception:
        return {'latency': 5000, 'success': False}

def test_server_solo(name, port, queries):
    """Test server individually"""
    print(f"\n🎯 Testing {name} Solo (200 requests, 20 threads)")
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for i in range(200):
            query = queries[i % len(queries)]
            future = executor.submit(query_server, 'localhost', port, query, i+1)
            futures.append(future)
        
        for future in as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    latencies = [r['latency'] for r in results if r['success']]
    
    if latencies:
        latencies_sorted = sorted(latencies)
        qps = 200 / total_time
        avg_latency = statistics.mean(latencies)
        p95_latency = latencies_sorted[int(len(latencies)*0.95)]
        
        print(f"  QPS: {qps:.1f}")
        print(f"  Avg: {avg_latency:.1f}ms")
        print(f"  P95: {p95_latency:.1f}ms")
        return qps
    return 0

def main():
    servers = [
        ('ICD-10', 8901, ["chest pain", "hypertension", "diabetes", "pneumonia", "sepsis"]),
        ('SNOMED', 8902, ["heart disease", "infection", "fracture", "asthma", "stroke"]),
        ('LOINC', 8903, ["glucose", "hemoglobin", "creatinine", "troponin", "cholesterol"]),
        ('RxNorm', 8904, ["aspirin", "metformin", "lisinopril", "insulin", "warfarin"]),
        ('HCPCS', 8905, ["wheelchair", "catheter", "walker", "syringe", "oxygen"])
    ]
    
    print("🏥 Solo Server Performance Test")
    print("=" * 40)
    
    # Kill all servers first
    print("Stopping all servers...")
    import os
    os.system("pkill -f 'python.*server' 2>/dev/null")
    time.sleep(2)
    
    solo_results = []
    
    for name, port, queries in servers:
        print(f"\nStarting {name} server alone...")
        
        # Start only this server
        if name == 'ICD-10':
            os.system(f"cd /Volumes/My4TBDrive/icd10/medical-entity-code-mapper && python src/servers/icd10_server.py > /dev/null 2>&1 &")
        elif name == 'SNOMED':
            os.system(f"cd /Volumes/My4TBDrive/icd10/medical-entity-code-mapper && python src/servers/snomed_server.py > /dev/null 2>&1 &")
        elif name == 'LOINC':
            os.system(f"cd /Volumes/My4TBDrive/icd10/medical-entity-code-mapper && python src/servers/loinc_server.py > /dev/null 2>&1 &")
        elif name == 'RxNorm':
            os.system(f"cd /Volumes/My4TBDrive/icd10/medical-entity-code-mapper && python src/servers/rxnorm_server.py > /dev/null 2>&1 &")
        elif name == 'HCPCS':
            os.system(f"cd /Volumes/My4TBDrive/icd10/medical-entity-code-mapper && python src/servers/hcpcs_server.py > /dev/null 2>&1 &")
        
        # Wait for server to start
        time.sleep(10)
        
        # Test it
        qps = test_server_solo(name, port, queries)
        solo_results.append((name, qps))
        
        # Kill it
        os.system("pkill -f 'python.*server' 2>/dev/null")
        time.sleep(2)
    
    print("\n" + "=" * 40)
    print("📊 SOLO PERFORMANCE SUMMARY")
    print("=" * 40)
    print("\nQPS when running alone:")
    for name, qps in solo_results:
        print(f"  {name}: {qps:.1f} QPS")

if __name__ == "__main__":
    main()