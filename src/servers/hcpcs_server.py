#!/usr/bin/env python3
"""
TCP Server for HCPCS code assignment using FAISS similarity search
Port: 8905
Protocol: sequence_number,description\n -> sequence_number,hcpcs_code\n
"""

import socket
import threading
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import time
from pathlib import Path
import signal
import sys
import queue
from collections import deque
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HCPCSFAISSServer:
    def __init__(self, host='127.0.0.1', port=8905):  # Loopback only for security
        self.host = host
        self.port = port
        self.running = False
        
        # FAISS index path
        # Get project root directory
        base_dir = Path(__file__).parent.parent.parent
        self.index_dir = base_dir / "data" / "indices" / "hcpcs_bge_m3"
        
        # Load FAISS index and metadata
        logging.info("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        
        logging.info("Loading metadata...")
        with open(self.index_dir / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load BGE-M3 model
        logging.info("Loading BGE-M3 model...")
        
        # Determine device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS (Metal Performance Shaders)")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU")
        
        self.model = SentenceTransformer('BAAI/bge-m3', device=device)
        
        # Dynamic batching components
        self.request_queue = queue.Queue()
        self.min_batch_size = 2
        self.max_batch_size = 25
        self.max_wait_time = 0.01  # 10ms
        self.batch_timeout = 0.05   # 50ms max wait
        
        # Performance metrics
        self.total_requests = 0
        self.total_time = 0
        self.recent_latencies = deque(maxlen=100)
        
        # Start inference workers
        self.num_workers = 3
        self.inference_threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._inference_worker, daemon=True)
            t.start()
            self.inference_threads.append(t)
        
        logging.info(f"HCPCS server initialized with {len(self.metadata)} entries")
    
    def _inference_worker(self):
        """Worker thread that processes batches of requests"""
        while True:
            batch = []
            results = []
            start_time = time.time()
            
            # Collect requests for batch
            while len(batch) < self.max_batch_size:
                try:
                    timeout = self.max_wait_time if batch else None
                    request = self.request_queue.get(timeout=timeout)
                    batch.append(request)
                    
                    # Check if we should process early
                    if len(batch) >= self.min_batch_size:
                        wait_time = time.time() - start_time
                        if wait_time >= self.max_wait_time:
                            break
                    
                    # Absolute timeout check
                    if time.time() - start_time >= self.batch_timeout:
                        break
                        
                except queue.Empty:
                    if batch:  # Process what we have
                        break
                    continue
            
            if not batch:
                continue
            
            # Process batch
            try:
                # Extract queries
                queries = [req['query'] for req in batch]
                
                # Encode all queries at once
                query_embeddings = self.model.encode(
                    queries,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # Search for each query
                for i, (query_embedding, request) in enumerate(zip(query_embeddings, batch)):
                    # Search in FAISS
                    k = 1  # Get top match
                    distances, indices = self.index.search(
                        query_embedding.reshape(1, -1).astype('float32'), 
                        k
                    )
                    
                    # Get best match
                    if indices[0][0] < len(self.metadata):
                        best_match = self.metadata[indices[0][0]]
                        hcpcs_code = best_match.get('hcpcs_code', best_match.get('code', 'UNKNOWN'))
                        confidence = float(distances[0][0])
                        
                        # Log match details for medical devices
                        if any(term in request['query'].lower() for term in ['catheter', 'wheelchair', 'walker', 'syringe']):
                            logging.info(f"Match: '{request['query']}' -> {hcpcs_code}: {best_match.get('description', '')[:60]}...")
                    else:
                        hcpcs_code = "NO_MATCH"
                        confidence = 0.0
                    
                    # Send response
                    request['callback'](hcpcs_code, confidence)
                    
                    # Update metrics
                    latency = time.time() - request['start_time']
                    self.recent_latencies.append(latency)
                
            except Exception as e:
                logging.error(f"Batch processing error: {e}")
                # Send error responses
                for request in batch:
                    request['callback']("ERROR", 0.0)
    
    def handle_client(self, client_socket, address):
        """Handle individual client connection"""
        logging.info(f"New connection from {address}")
        
        try:
            while self.running:
                data = client_socket.recv(4096).decode('utf-8').strip()
                if not data:
                    break
                
                # Parse request: sequence_number,description
                parts = data.split(',', 1)
                if len(parts) != 2:
                    client_socket.send(b"ERROR,Invalid format\n")
                    continue
                
                seq_num, description = parts
                start_time = time.time()
                
                # Create response callback
                response_event = threading.Event()
                response_data = {}
                
                def callback(hcpcs_code, confidence):
                    response_data['code'] = hcpcs_code
                    response_data['confidence'] = confidence
                    response_event.set()
                
                # Queue request
                self.request_queue.put({
                    'query': description,
                    'callback': callback,
                    'start_time': start_time
                })
                
                # Wait for response
                if response_event.wait(timeout=5.0):
                    response = f"{seq_num},{response_data['code']}\n"
                    client_socket.send(response.encode('utf-8'))
                    
                    # Update metrics
                    self.total_requests += 1
                    self.total_time += time.time() - start_time
                else:
                    client_socket.send(f"{seq_num},TIMEOUT\n".encode('utf-8'))
                
        except Exception as e:
            logging.error(f"Client handler error: {e}")
        finally:
            client_socket.close()
            logging.info(f"Connection closed from {address}")
    
    def start(self):
        """Start the TCP server"""
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        logging.info(f"HCPCS server listening on {self.host}:{self.port}")
        logging.info(f"Protocol: sequence_number,description -> sequence_number,hcpcs_code")
        
        # Start metrics reporter
        metrics_thread = threading.Thread(target=self._report_metrics, daemon=True)
        metrics_thread.start()
        
        try:
            while self.running:
                client_socket, address = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
        except KeyboardInterrupt:
            logging.info("Server interrupted")
        finally:
            self.stop()
    
    def _report_metrics(self):
        """Report performance metrics periodically"""
        while self.running:
            time.sleep(30)  # Report every 30 seconds
            
            if self.total_requests > 0:
                avg_latency = self.total_time / self.total_requests * 1000
                recent_avg = np.mean(self.recent_latencies) * 1000 if self.recent_latencies else 0
                qps = self.total_requests / max(self.total_time, 1)
                
                logging.info(f"Metrics - Total requests: {self.total_requests}, "
                           f"Avg latency: {avg_latency:.1f}ms, "
                           f"Recent avg: {recent_avg:.1f}ms, "
                           f"QPS: {qps:.1f}")
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if hasattr(self, 'server_socket'):
            self.server_socket.close()
        logging.info("HCPCS server stopped")

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logging.info("Shutdown signal received")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server = HCPCSFAISSServer()
    server.start()