#!/usr/bin/env python3
"""
LOINC TCP Socket Server with Dynamic Batching
Port: 8903
Protocol: sequence_number,description\n -> sequence_number,loinc_code\n
"""

import socket
import threading
import time
import pickle
import numpy as np
import torch
import faiss
import re
import queue
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
import signal
import sys
import gc

class LOINCFAISSServer:
    def __init__(self, host='0.0.0.0', port=8903):
        self.host = host
        self.port = port
        # Get project root directory
        base_dir = Path(__file__).parent.parent.parent
        self.index_dir = base_dir / "data" / "indices" / "loinc_bge_m3"
        
        # Dynamic batching components
        self.request_queue = queue.Queue()
        self.response_queues = {}
        self.queue_lock = threading.Lock()
        
        # Adaptive batching parameters
        self.min_batch_size = 2
        self.max_batch_size = 25
        self.max_wait_time = 0.01  # 10ms
        self.last_batch_time = time.time()
        
        # Performance monitoring
        self.total_requests = 0
        self.total_time = 0
        self.batch_count = 0
        
        # Initialize model and index
        self._load_resources()
        
        # Start inference workers
        self.workers = []
        for i in range(3):  # 3 inference workers
            worker = threading.Thread(target=self._inference_worker, daemon=True)
            worker.start()
            self.workers.append(worker)
            
        # Performance monitor thread
        monitor = threading.Thread(target=self._monitor_performance, daemon=True)
        monitor.start()
        
    def _load_resources(self):
        """Load BGE-M3 model and FAISS index"""
        print("Loading BGE-M3 model...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = SentenceTransformer('BAAI/bge-m3', device=str(self.device))
        self.model.eval()
        
        # Set max sequence length
        self.model.max_seq_length = 512
        
        # Warmup
        with torch.no_grad():
            _ = self.model.encode(["warmup"], convert_to_tensor=True, normalize_embeddings=True)
        
        print("Loading FAISS index...")
        index_path = self.index_dir / "faiss.index"
        metadata_path = self.index_dir / "metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"FAISS index or metadata not found in {self.index_dir}")
        
        self.index = faiss.read_index(str(index_path))
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded LOINC index with {self.index.ntotal} vectors")
        print(f"Loaded metadata with {len(self.metadata)} entries")
        
    def normalize_description(self, desc):
        """Normalize description - remove all punctuation, lowercase, single spaces"""
        if not desc:
            return desc
        desc = desc.lower()
        desc = re.sub(r'[^a-z0-9\s]', ' ', desc)
        desc = ' '.join(desc.split())
        return desc.strip()
    
    def _inference_worker(self):
        """Worker thread for batch inference"""
        while True:
            try:
                # Collect batch
                batch = []
                batch_start = time.time()
                
                # Wait for first request
                try:
                    first_item = self.request_queue.get(timeout=1.0)
                    batch.append(first_item)
                except queue.Empty:
                    continue
                
                # Collect more requests up to max_batch_size
                deadline = time.time() + self.max_wait_time
                while len(batch) < self.max_batch_size and time.time() < deadline:
                    try:
                        timeout = deadline - time.time()
                        if timeout > 0:
                            item = self.request_queue.get(timeout=timeout)
                            batch.append(item)
                    except queue.Empty:
                        break
                
                if batch:
                    self._process_batch(batch)
                    
            except Exception as e:
                print(f"Inference worker error: {e}")
                import traceback
                traceback.print_exc()
    
    def _process_batch(self, batch):
        """Process a batch of queries"""
        start_time = time.time()
        
        # Extract queries
        queries = [self.normalize_description(item[1]) for item in batch]
        
        # Encode queries
        with torch.no_grad():
            query_embeddings = self.model.encode(
                queries,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            query_embeddings = query_embeddings.cpu().numpy()
        
        # Search FAISS index
        k = 5  # Top 5 results
        distances, indices = self.index.search(query_embeddings, k)
        
        # Process results
        for i, (seq_num, query, response_queue) in enumerate(batch):
            try:
                # Get best match
                best_idx = indices[i, 0]
                if best_idx >= 0 and best_idx < len(self.metadata):
                    loinc_code = self.metadata[best_idx].get('code', 'NO_CODE')
                    score = float(distances[i, 0])
                else:
                    loinc_code = 'NO_MATCH'
                    score = 0.0
                
                response_queue.put((seq_num, loinc_code, score))
                
            except Exception as e:
                print(f"Error processing result: {e}")
                response_queue.put((seq_num, 'ERROR', 0.0))
        
        # Update performance metrics
        batch_time = time.time() - start_time
        with self.queue_lock:
            self.total_requests += len(batch)
            self.total_time += batch_time
            self.batch_count += 1
        
        # Adaptive batch size adjustment
        avg_latency = batch_time / len(batch) * 1000  # ms per request
        if avg_latency > 50 and self.min_batch_size > 2:
            self.min_batch_size -= 1
        elif avg_latency < 20 and self.min_batch_size < 10:
            self.min_batch_size += 1
            
    def _monitor_performance(self):
        """Monitor and report performance metrics"""
        while True:
            time.sleep(30)  # Report every 30 seconds
            with self.queue_lock:
                if self.total_requests > 0:
                    avg_latency = (self.total_time / self.total_requests) * 1000
                    throughput = self.total_requests / max(self.total_time, 0.001)
                    print(f"\n=== Performance Report ===")
                    print(f"Total requests: {self.total_requests}")
                    print(f"Average latency: {avg_latency:.2f} ms")
                    print(f"Throughput: {throughput:.2f} QPS")
                    print(f"Batch count: {self.batch_count}")
                    print(f"Avg batch size: {self.total_requests/max(self.batch_count,1):.2f}")
                    print(f"Current min batch size: {self.min_batch_size}")
                    print(f"Queue size: {self.request_queue.qsize()}")
                    print("========================\n")
    
    def handle_client(self, client_socket, address):
        """Handle individual client connections"""
        try:
            client_socket.settimeout(30.0)
            
            while True:
                # Receive data
                data = client_socket.recv(4096).decode('utf-8').strip()
                if not data:
                    break
                
                # Parse request
                lines = data.split('\n')
                for line in lines:
                    if not line:
                        continue
                        
                    try:
                        parts = line.split(',', 1)
                        if len(parts) != 2:
                            continue
                            
                        seq_num = parts[0]
                        description = parts[1]
                        
                        # Create response queue
                        response_queue = queue.Queue()
                        
                        # Submit to processing queue
                        self.request_queue.put((seq_num, description, response_queue))
                        
                        # Wait for response
                        try:
                            seq_num_resp, loinc_code, score = response_queue.get(timeout=5.0)
                            response = f"{seq_num_resp},{loinc_code}\n"
                            client_socket.send(response.encode('utf-8'))
                        except queue.Empty:
                            response = f"{seq_num},TIMEOUT\n"
                            client_socket.send(response.encode('utf-8'))
                            
                    except Exception as e:
                        print(f"Error processing request: {e}")
                        response = f"{seq_num},ERROR\n"
                        client_socket.send(response.encode('utf-8'))
                        
        except socket.timeout:
            print(f"Client {address} timed out")
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
    
    def start(self):
        """Start the TCP server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Set socket options for better performance
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(128)  # Increased backlog
        
        print(f"LOINC TCP Server listening on {self.host}:{self.port}")
        print("Protocol: sequence_number,description -> sequence_number,loinc_code")
        print(f"Using device: {self.device}")
        print("Dynamic batching enabled")
        
        try:
            while True:
                client_socket, address = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.server_socket.close()

def signal_handler(sig, frame):
    print('\nShutting down gracefully...')
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    server = LOINCFAISSServer()
    server.start()