#!/usr/bin/env python3
"""Quick benchmark to get performance metrics"""

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
    except:
        return {'latency': 5000, 'success': False}

def benchmark_server(name, port, queries, num_requests=100, threads=10):
    """Benchmark a single server"""
    print(f"\n📊 Benchmarking {name} (port {port})...")
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for i in range(num_requests):
            query = queries[i % len(queries)]
            future = executor.submit(query_server, 'localhost', port, query, i+1)
            futures.append(future)
        
        for future in as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    latencies = [r['latency'] for r in results if r['success']]
    
    if latencies:
        print(f"  ✓ QPS: {num_requests / total_time:.1f}")
        print(f"  ✓ Avg latency: {statistics.mean(latencies):.1f}ms")
        print(f"  ✓ P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
        print(f"  ✓ Success rate: {len(latencies)/len(results)*100:.1f}%")
        return num_requests / total_time
    else:
        print(f"  ✗ Server not responding")
        return 0

def main():
    servers = {
        'ICD-10': (8901, ["chest pain", "hypertension", "diabetes", "pneumonia", "sepsis"]),
        'SNOMED': (8902, ["heart disease", "infection", "fracture", "asthma", "stroke"]),
        'LOINC': (8903, ["glucose", "hemoglobin", "creatinine", "troponin", "cholesterol"]),
        'RxNorm': (8904, ["aspirin", "metformin", "lisinopril", "insulin", "warfarin"]),
        'HCPCS': (8905, ["wheelchair", "catheter", "walker", "syringe", "oxygen"])
    }
    
    print("🏥 Medical Entity Code Mapper - Quick Benchmark")
    print("=" * 50)
    
    total_qps = 0
    active_servers = 0
    
    for name, (port, queries) in servers.items():
        qps = benchmark_server(name, port, queries)
        if qps > 0:
            total_qps += qps
            active_servers += 1
    
    if active_servers > 0:
        print(f"\n📊 SUMMARY:")
        print(f"  Active servers: {active_servers}/5")
        print(f"  Combined QPS: {total_qps:.1f}")
        print(f"  Average QPS per server: {total_qps/active_servers:.1f}")
    else:
        print("\n❌ No servers running! Start with: uv run uv_runner.py")

if __name__ == "__main__":
    main()