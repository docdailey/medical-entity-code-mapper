#!/usr/bin/env python3
"""
Benchmark optimized dynamic batching servers
Tests throughput, latency, and compares with baseline
"""

import time
import socket
import statistics
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np

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
        
        return {'latency': latency, 'success': success, 'response': response}
    except Exception as e:
        return {'latency': 5000, 'success': False, 'error': str(e)}

def benchmark_server(name, port, queries, num_requests=1000, threads=50):
    """Comprehensive benchmark of a server"""
    print(f"\n{'='*60}")
    print(f"🎯 Benchmarking {name} (Optimized with Dynamic Batching)")
    print(f"{'='*60}")
    print(f"Configuration: {num_requests} requests, {threads} concurrent threads")
    
    results = []
    start_time = time.time()
    
    # Run benchmark
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for i in range(num_requests):
            query = queries[i % len(queries)]
            future = executor.submit(query_server, 'localhost', port, query, i+1)
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    successful_results = [r for r in results if r['success']]
    latencies = [r['latency'] for r in successful_results]
    
    if not latencies:
        print(f"  ✗ No successful requests")
        return None
    
    # Sort for percentiles
    latencies_sorted = sorted(latencies)
    
    metrics = {
        'server': name,
        'total_requests': num_requests,
        'successful_requests': len(successful_results),
        'failed_requests': num_requests - len(successful_results),
        'success_rate': len(successful_results) / num_requests * 100,
        'total_time': total_time,
        'qps': num_requests / total_time,
        'latency': {
            'min': min(latencies),
            'avg': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'p90': latencies_sorted[int(len(latencies) * 0.90)],
            'p95': latencies_sorted[int(len(latencies) * 0.95)],
            'p99': latencies_sorted[int(len(latencies) * 0.99)],
            'max': max(latencies)
        }
    }
    
    # Print results
    print(f"\n📊 Results:")
    print(f"  ✓ QPS: {metrics['qps']:.1f} queries/second")
    print(f"  ✓ Success Rate: {metrics['success_rate']:.1f}%")
    print(f"  ✓ Total Time: {metrics['total_time']:.2f}s")
    
    print(f"\n⏱️  Latency Distribution (ms):")
    print(f"  ✓ Min:    {metrics['latency']['min']:.1f}")
    print(f"  ✓ Avg:    {metrics['latency']['avg']:.1f}")
    print(f"  ✓ Median: {metrics['latency']['median']:.1f}")
    print(f"  ✓ P90:    {metrics['latency']['p90']:.1f}")
    print(f"  ✓ P95:    {metrics['latency']['p95']:.1f}")
    print(f"  ✓ P99:    {metrics['latency']['p99']:.1f}")
    print(f"  ✓ Max:    {metrics['latency']['max']:.1f}")
    
    return metrics

def main():
    # Server configurations with test queries
    servers = {
        'ICD-10': {
            'port': 8901,
            'queries': [
                "chest pain radiating to left arm",
                "type 2 diabetes mellitus with complications",
                "acute myocardial infarction",
                "essential hypertension",
                "bacterial pneumonia",
                "chronic obstructive pulmonary disease",
                "congestive heart failure",
                "acute kidney injury",
                "major depressive disorder",
                "generalized anxiety disorder"
            ]
        },
        'SNOMED': {
            'port': 8902,
            'queries': [
                "coronary artery disease",
                "diabetes type 2",
                "myocardial infarction",
                "essential hypertension",
                "community acquired pneumonia",
                "chronic bronchitis",
                "heart failure",
                "renal failure acute",
                "depression",
                "anxiety disorder"
            ]
        },
        'LOINC': {
            'port': 8903,
            'queries': [
                "blood glucose level",
                "hemoglobin A1c",
                "complete blood count with differential",
                "comprehensive metabolic panel",
                "cardiac troponin I",
                "brain natriuretic peptide",
                "thyroid stimulating hormone",
                "lipid panel",
                "urinalysis complete",
                "serum creatinine"
            ]
        },
        'RxNorm': {
            'port': 8904,
            'queries': [
                "aspirin 81 mg oral tablet",
                "metformin 1000 mg extended release",
                "lisinopril 10 mg tablet",
                "atorvastatin 40 mg oral tablet",
                "insulin glargine 100 units/ml",
                "albuterol sulfate inhaler",
                "warfarin sodium 5 mg",
                "metoprolol tartrate 50 mg",
                "omeprazole 20 mg capsule",
                "gabapentin 300 mg capsule"
            ]
        },
        'HCPCS': {
            'port': 8905,
            'queries': [
                "indwelling foley catheter",
                "standard wheelchair",
                "hospital bed electric",
                "walker with wheels",
                "insulin syringe with needle",
                "blood glucose test strips",
                "nebulizer machine portable",
                "oxygen concentrator",
                "continuous positive airway pressure device",
                "compression stockings knee high"
            ]
        }
    }
    
    print("🚀 Medical Entity Code Mapper - Optimized Server Benchmark")
    print("=" * 60)
    print("Testing servers with dynamic batching optimization")
    print("Target: 300+ QPS (based on 326.2 QPS achieved in /icd10)")
    
    # Check which servers are running
    running_servers = []
    for name, config in servers.items():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', config['port']))
            sock.close()
            if result == 0:
                running_servers.append(name)
                print(f"✓ {name} server detected on port {config['port']}")
            else:
                print(f"✗ {name} server not running on port {config['port']}")
        except Exception:
            print(f"✗ {name} server not running on port {config['port']}")
    
    if not running_servers:
        print("\n❌ No servers running!")
        print("Start servers with: python scripts/start_optimized_servers.py")
        return
    
    # Warm-up phase
    print("\n🔥 Warming up servers...")
    for name in running_servers:
        config = servers[name]
        for _ in range(10):
            query_server('localhost', config['port'], config['queries'][0])
    
    # Run benchmarks
    all_results = []
    
    print("\n📊 Starting benchmark tests...")
    
    # Test with increasing load
    test_configs = [
        (100, 10, "Light Load"),
        (500, 50, "Medium Load"),
        (1000, 100, "Heavy Load")
    ]
    
    for num_requests, threads, load_name in test_configs:
        print(f"\n{'='*60}")
        print(f"🔄 Test Configuration: {load_name}")
        print(f"{'='*60}")
        
        for name in running_servers:
            config = servers[name]
            metrics = benchmark_server(
                name,
                config['port'],
                config['queries'],
                num_requests=num_requests,
                threads=threads
            )
            if metrics:
                metrics['load'] = load_name
                all_results.append(metrics)
        
        # Brief pause between test configs
        time.sleep(2)
    
    # Summary report
    print(f"\n{'='*60}")
    print("📈 BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    # Group by server
    by_server = {}
    for result in all_results:
        server = result['server']
        if server not in by_server:
            by_server[server] = []
        by_server[server].append(result)
    
    print("\n🏆 Best QPS Achieved (Heavy Load - 1000 requests, 100 threads):")
    print("-" * 40)
    
    heavy_load_results = [r for r in all_results if r['load'] == 'Heavy Load']
    heavy_load_results.sort(key=lambda x: x['qps'], reverse=True)
    
    for result in heavy_load_results:
        improvement = (result['qps'] / 50) * 100  # Assuming 50 QPS baseline
        print(f"{result['server']:10s}: {result['qps']:>6.1f} QPS "
              f"(~{improvement:.0f}% of baseline)")
    
    print("\n📊 Performance Scaling by Load:")
    print("-" * 60)
    print(f"{'Server':<10} | {'Light (100)':<12} | {'Medium (500)':<12} | {'Heavy (1000)':<12}")
    print("-" * 60)
    
    for server in ['ICD-10', 'SNOMED', 'LOINC', 'RxNorm', 'HCPCS']:
        if server in by_server:
            line = f"{server:<10} |"
            for load in ['Light Load', 'Medium Load', 'Heavy Load']:
                result = next((r for r in by_server[server] if r['load'] == load), None)
                if result:
                    line += f" {result['qps']:>6.1f} QPS  |"
                else:
                    line += " " * 13 + "|"
            print(line)
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"benchmark_optimized_results_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    
    # Check if we hit the target
    max_qps = max((r['qps'] for r in heavy_load_results), default=0)
    if max_qps >= 300:
        print(f"\n✅ TARGET ACHIEVED! Peak QPS: {max_qps:.1f} (Target: 300+)")
    else:
        print(f"\n⚠️  Peak QPS: {max_qps:.1f} (Target: 300+)")
        print("   Consider adjusting batch sizes or adding more inference workers")

if __name__ == "__main__":
    main()