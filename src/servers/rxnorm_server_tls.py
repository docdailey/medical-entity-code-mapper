#!/usr/bin/env python3
"""
TLS-secured RxNorm TCP Socket Server with Dynamic Batching
Port: 8914
Protocol: sequence_number,description\n -> sequence_number,rxnorm_code\n
"""

import socket
import ssl
import threading
import time
import pickle
import numpy as np
import torch
import faiss
import re
import queue
import os
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
import signal
import sys
import gc
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# TLS Configuration
TLS_CERT_FILE = os.getenv('TLS_CERT_FILE', 'certs/server.crt')
TLS_KEY_FILE = os.getenv('TLS_KEY_FILE', 'certs/server.key')

class RxNormFAISSTLSServer:
    def __init__(self, host='0.0.0.0', port=8914):
        self.host = host
        self.port = port
        # Get project root directory
        base_dir = Path(__file__).parent.parent.parent
        self.index_dir = base_dir / "data" / "indices" / "rxnorm_bge_m3"
        
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
        
        # Create SSL context
        self.ssl_context = self.create_ssl_context()
        
    def create_ssl_context(self):
        """Create SSL context for TLS server"""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
        # Load certificate and key
        try:
            context.load_cert_chain(certfile=TLS_CERT_FILE, keyfile=TLS_KEY_FILE)
            logging.info(f"Loaded TLS certificate from {TLS_CERT_FILE}")
        except Exception as e:
            logging.error(f"Failed to load TLS certificate: {e}")
            sys.exit(1)
        
        # Set secure options
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        return context
        
    def _load_resources(self):
        """Load model and FAISS index"""
        logging.info("Loading RxNorm FAISS index...")
        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        
        logging.info("Loading metadata...")
        with open(self.index_dir / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        logging.info("Loading BGE-M3 model...")
        
        # Check for MPS support (Apple Silicon)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logging.info("Using MPS (Metal Performance Shaders)")
        else:
            self.device = torch.device("cpu")
            logging.info("Using CPU")
        
        self.model = SentenceTransformer('BAAI/bge-m3', device=self.device)
        self.model.eval()
        
        # Warm up
        logging.info("Warming up model...")
        for size in [1, 4, 8, 16]:
            _ = self.model.encode(["test"] * size, show_progress_bar=False)
        
        logging.info(f"RxNorm server ready with {self.index.ntotal} medications")
        
    def _should_process_batch(self):
        """Determine if we should process the current batch"""
        current_size = self.request_queue.qsize()
        time_since_last = time.time() - self.last_batch_time
        
        if current_size >= self.max_batch_size:
            return True
        elif current_size >= self.min_batch_size and time_since_last >= self.max_wait_time:
            return True
        elif current_size > 0 and time_since_last >= self.max_wait_time * 2:
            return True
        
        return False
        
    def _batch_processor(self):
        """Process requests in adaptive batches"""
        while True:
            if self._should_process_batch():
                batch_start = time.time()
                batch = []
                
                # Collect batch
                while len(batch) < self.max_batch_size and not self.request_queue.empty():
                    try:
                        batch.append(self.request_queue.get_nowait())
                    except queue.Empty:
                        break
                
                if batch:
                    # Process batch
                    texts = [f"Represent this sentence for searching relevant passages: {item[1]}" 
                            for item in batch]
                    
                    embeddings = self.model.encode(
                        texts, 
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    
                    distances, indices = self.index.search(embeddings.astype('float32'), 1)
                    
                    # Send responses
                    for i, (seq_num, text, response_queue) in enumerate(batch):
                        if indices[i][0] >= 0:
                            rxnorm_code = self.metadata[indices[i][0]]['rxnorm_code']
                        else:
                            rxnorm_code = "NO_MATCH"
                        
                        response_queue.put(f"{seq_num},{rxnorm_code}\n")
                    
                    # Update stats
                    batch_time = time.time() - batch_start
                    self.total_requests += len(batch)
                    self.total_time += batch_time
                    self.batch_count += 1
                    self.last_batch_time = time.time()
                    
                    # Adaptive tuning
                    if self.batch_count % 10 == 0:
                        avg_batch_size = self.total_requests / self.batch_count
                        avg_latency = (self.total_time / self.batch_count) * 1000
                        
                        if avg_latency > 20 and self.min_batch_size > 2:
                            self.min_batch_size -= 1
                        elif avg_latency < 10 and avg_batch_size > 10:
                            self.min_batch_size = min(8, self.min_batch_size + 1)
                            
            else:
                time.sleep(0.001)  # 1ms sleep
                
    def handle_client(self, client_socket, address):
        """Handle individual client connections"""
        response_queue = queue.Queue()
        client_id = f"{address[0]}:{address[1]}"
        
        with self.queue_lock:
            self.response_queues[client_id] = response_queue
        
        try:
            buffer = ""
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                
                buffer += data.decode('utf-8')
                lines = buffer.split('\n')
                buffer = lines[-1]
                
                for line in lines[:-1]:
                    if line.strip():
                        parts = line.split(',', 1)
                        if len(parts) == 2:
                            seq_num, text = parts
                            self.request_queue.put((seq_num, text.strip(), response_queue))
                
                # Send any ready responses
                while not response_queue.empty():
                    response = response_queue.get()
                    client_socket.sendall(response.encode('utf-8'))
                    
        except Exception as e:
            logging.error(f"Client error: {e}")
        finally:
            with self.queue_lock:
                del self.response_queues[client_id]
            client_socket.close()
            
    def start(self):
        """Start the TLS server"""
        # Start batch processor
        processor_thread = threading.Thread(target=self._batch_processor, daemon=True)
        processor_thread.start()
        
        # Create server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(128)
        
        logging.info(f"TLS-secured RxNorm server listening on {self.host}:{self.port}")
        logging.info(f"Certificate: {TLS_CERT_FILE}")
        logging.info(f"Adaptive batching: min={self.min_batch_size}, max={self.max_batch_size}")
        
        try:
            while True:
                client_socket, address = server_socket.accept()
                # Wrap client socket with SSL
                try:
                    ssl_client_socket = self.ssl_context.wrap_socket(client_socket, server_side=True)
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(ssl_client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                except ssl.SSLError as e:
                    logging.warning(f"SSL handshake failed: {e}")
                    client_socket.close()
                    
        except KeyboardInterrupt:
            logging.info("Shutting down RxNorm TLS server...")
        finally:
            server_socket.close()
            
            # Print final stats
            if self.batch_count > 0:
                avg_batch_size = self.total_requests / self.batch_count
                avg_latency = (self.total_time / self.batch_count) * 1000
                throughput = self.total_requests / self.total_time if self.total_time > 0 else 0
                
                logging.info(f"Final stats:")
                logging.info(f"  Total requests: {self.total_requests}")
                logging.info(f"  Total batches: {self.batch_count}")
                logging.info(f"  Avg batch size: {avg_batch_size:.1f}")
                logging.info(f"  Avg latency: {avg_latency:.1f}ms")
                logging.info(f"  Throughput: {throughput:.1f} req/s")

def signal_handler(sig, frame):
    logging.info("Shutdown signal received")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server = RxNormFAISSTLSServer()
    server.start()