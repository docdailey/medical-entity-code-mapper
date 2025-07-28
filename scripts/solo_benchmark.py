#!/usr/bin/env python3
"""Solo benchmark - test each server individually with all system resources"""

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

def benchmark_server_solo(name, port, queries, num_requests=500, threads=50):
    """Benchmark a single server with more aggressive parameters"""
    print(f"\n{'='*60}")
    print(f"🎯 SOLO BENCHMARK: {name} (port {port})")
    print(f"{'='*60}")
    print(f"Requests: {num_requests}, Threads: {threads}")
    
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
        latencies_sorted = sorted(latencies)
        print(f"\n📊 Results:")
        print(f"  ✓ QPS: {num_requests / total_time:.1f}")
        print(f"  ✓ Total time: {total_time:.2f}s")
        print(f"  ✓ Success rate: {len(latencies)/len(results)*100:.1f}%")
        print(f"\n⏱️  Latency (ms):")
        print(f"  ✓ Min: {min(latencies):.1f}")
        print(f"  ✓ Avg: {statistics.mean(latencies):.1f}")
        print(f"  ✓ Median: {statistics.median(latencies):.1f}")
        print(f"  ✓ P90: {latencies_sorted[int(len(latencies)*0.90)]:.1f}")
        print(f"  ✓ P95: {latencies_sorted[int(len(latencies)*0.95)]:.1f}")
        print(f"  ✓ P99: {latencies_sorted[int(len(latencies)*0.99)]:.1f}")
        print(f"  ✓ Max: {max(latencies):.1f}")
        
        return {
            'server': name,
            'qps': num_requests / total_time,
            'avg_latency': statistics.mean(latencies),
            'p95_latency': latencies_sorted[int(len(latencies)*0.95)],
            'p99_latency': latencies_sorted[int(len(latencies)*0.99)]
        }
    else:
        print(f"  ✗ Server not responding")
        return None

def main():
    servers = {
        'ICD-10': (8901, ["chest pain", "hypertension", "diabetes mellitus", "pneumonia", "sepsis", 
                          "heart failure", "asthma", "COPD", "stroke", "depression"]),
        'SNOMED': (8902, ["myocardial infarction", "coronary disease", "renal failure", "hepatitis", 
                          "bronchitis", "arthritis", "anemia", "thyroid disorder", "obesity", "anxiety"]),
        'LOINC': (8903, ["glucose", "hemoglobin A1c", "creatinine", "troponin", "cholesterol",
                         "triglycerides", "sodium", "potassium", "white blood cell", "platelet count"]),
        'RxNorm': (8904, ["aspirin 81mg", "metformin 1000mg", "lisinopril 10mg", "atorvastatin 40mg", 
                          "insulin glargine", "albuterol inhaler", "warfarin 5mg", "metoprolol 50mg",
                          "omeprazole 20mg", "gabapentin 300mg"]),
        'HCPCS': (8905, ["foley catheter", "standard wheelchair", "hospital bed", "walker with wheels", 
                         "insulin syringe", "blood glucose test strips", "nebulizer machine", 
                         "oxygen concentrator", "CPAP device", "compression stockings"])
    }
    
    print("🏥 Medical Entity Code Mapper - Solo Performance Benchmark")
    print("Testing each server individually with full system resources")
    
    # Check which servers are running
    running_servers = []
    for name, (port, queries) in servers.items():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result == 0:
                running_servers.append((name, port, queries))
        except:
            pass
    
    if not running_servers:
        print("\n❌ No servers running! Start with: uv run uv_runner.py")
        return
    
    print(f"\nFound {len(running_servers)} running servers")
    print("\nStarting solo benchmarks...")
    
    # Run solo benchmarks
    results = []
    for name, port, queries in running_servers:
        # Test with increasing load
        for num_requests, threads in [(100, 10), (500, 50), (1000, 100)]:
            print(f"\n{'='*60}")
            print(f"Testing {name} with {num_requests} requests, {threads} threads")
            result = benchmark_server_solo(name, port, queries, num_requests, threads)
            if result:
                result['requests'] = num_requests
                result['threads'] = threads
                results.append(result)
            time.sleep(2)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SOLO BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    # Group by server
    by_server = {}
    for r in results:
        if r['server'] not in by_server:
            by_server[r['server']] = []
        by_server[r['server']].append(r)
    
    print("\nBest QPS achieved (1000 requests, 100 threads):")
    for server, server_results in by_server.items():
        best = max(server_results, key=lambda x: x['qps'])
        print(f"  {server}: {best['qps']:.1f} QPS")
    
    print("\nScaling comparison (QPS by load):")
    print("Server    | 100 req  | 500 req  | 1000 req |")
    print("----------|----------|----------|----------|")
    for server in ['HCPCS', 'LOINC', 'RxNorm', 'SNOMED', 'ICD-10']:
        if server in by_server:
            line = f"{server:<9} |"
            for reqs in [100, 500, 1000]:
                result = next((r for r in by_server[server] if r['requests'] == reqs), None)
                if result:
                    line += f" {result['qps']:>7.1f}  |"
                else:
                    line += "    -     |"
            print(line)

if __name__ == "__main__":
    main()