#!/usr/bin/env python3
"""
Base class for dynamic batching medical coding servers
Implements the high-performance architecture from /icd10 that achieved 326.2 QPS
"""

import socket
import threading
import queue
import time
import logging
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional, Any
import signal
import sys
from dataclasses import dataclass
from collections import defaultdict, deque
from abc import ABC, abstractmethod

# Configuration
NUM_CLIENT_THREADS = 50
NUM_INFERENCE_WORKERS = 3

# Dynamic batching parameters
MIN_BATCH_SIZE = 2      # Start small for better latency
MAX_BATCH_SIZE = 25     # Cap for memory efficiency
MAX_WAIT_TIME_MS = 10   # Maximum wait time
ADAPTIVE_WINDOW = 100   # Number of batches for adaptation

@dataclass
class Request:
    id: str
    seq_num: str
    text: str
    client_socket: socket.socket
    arrival_time: float

@dataclass
class BatchMetrics:
    batch_size: int
    wait_time: float
    inference_time: float
    total_latency: float
    throughput: float

class DynamicBatcher:
    """Adaptive dynamic batching system"""
    
    def __init__(self, min_batch_size=2, max_batch_size=25, max_wait_time_ms=10):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        
        self.pending_requests = deque()
        self.metrics_history = deque(maxlen=ADAPTIVE_WINDOW)
        self.lock = threading.Lock()
        
        # Adaptive parameters
        self.current_min_batch = min_batch_size
        self.current_max_wait = max_wait_time_ms
        
    def add_request(self, request: Request):
        """Add a request to the batcher"""
        with self.lock:
            self.pending_requests.append(request)
    
    def should_process_batch(self) -> bool:
        """Determine if current batch should be processed"""
        if not self.pending_requests:
            return False
            
        batch_size = len(self.pending_requests)
        
        # Check max batch size
        if batch_size >= self.max_batch_size:
            return True
        
        # Check min batch size
        if batch_size >= self.current_min_batch:
            # Check wait time for oldest request
            oldest_request = self.pending_requests[0]
            wait_time_ms = (time.time() - oldest_request.arrival_time) * 1000
            
            if wait_time_ms >= self.current_max_wait:
                return True
                
        # Single request with timeout
        if batch_size > 0:
            oldest_request = self.pending_requests[0]
            wait_time_ms = (time.time() - oldest_request.arrival_time) * 1000
            if wait_time_ms >= self.max_wait_time_ms * 5:  # 5x timeout for single requests
                return True
            
        return False
    
    def get_batch(self) -> Optional[List[Request]]:
        """Get a batch if ready"""
        with self.lock:
            if not self.should_process_batch():
                return None
                
            # Take all pending requests up to max_batch_size
            batch_size = min(len(self.pending_requests), self.max_batch_size)
            batch = []
            
            for _ in range(batch_size):
                batch.append(self.pending_requests.popleft())
                
            return batch
    
    def record_metrics(self, metrics: BatchMetrics):
        """Record batch metrics and adapt parameters"""
        self.metrics_history.append(metrics)
        
        if len(self.metrics_history) >= 10:  # Need some history
            self._adapt_parameters()
    
    def _adapt_parameters(self):
        """Adapt batching parameters based on metrics"""
        recent_metrics = list(self.metrics_history)
        
        # Calculate averages
        avg_latency = sum(m.total_latency for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_batch_size = sum(m.batch_size for m in recent_metrics) / len(recent_metrics)
        
        # Adapt based on performance
        if avg_latency > 50:  # Latency too high (50ms threshold)
            # Reduce batch requirements to improve latency
            self.current_min_batch = max(2, self.current_min_batch - 1)
            self.current_max_wait = max(5, self.current_max_wait - 1)
        elif avg_throughput < 100 and avg_batch_size < 10:  # Throughput too low
            # Increase batching to improve throughput
            self.current_min_batch = min(10, self.current_min_batch + 1)
            self.current_max_wait = min(20, self.current_max_wait + 2)

class BaseDynamicMedicalServer(ABC):
    """Base class for dynamic batching medical coding servers"""
    
    def __init__(self, host='127.0.0.1', port=8901, server_name="Medical"):  # Loopback only for security
        self.host = host
        self.port = port
        self.server_name = server_name
        self.running = True
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"{server_name}Server")
        
        # Resources
        self.model = None
        self.device = None
        self.faiss_index = None
        self.metadata = None
        self.batcher = DynamicBatcher(
            min_batch_size=MIN_BATCH_SIZE,
            max_batch_size=MAX_BATCH_SIZE,
            max_wait_time_ms=MAX_WAIT_TIME_MS
        )
        
        # Statistics
        self.stats = defaultdict(int)
        self.stats_lock = threading.Lock()
        self.request_counter = 0
        self.request_counter_lock = threading.Lock()
        
        # Thread pool
        self.client_thread_pool = ThreadPoolExecutor(max_workers=NUM_CLIENT_THREADS)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received")
        self.running = False
        sys.exit(0)
    
    @abstractmethod
    def get_index_path(self) -> Path:
        """Return path to FAISS index directory"""
        pass
    
    @abstractmethod
    def get_metadata_key(self) -> str:
        """Return the key in metadata dict for the code (e.g., 'icd10_code', 'snomed_code')"""
        pass
    
    @abstractmethod
    def get_default_code(self) -> str:
        """Return default code when no match found"""
        pass
    
    def load_resources(self):
        """Load model and FAISS index"""
        self.logger.info("Loading BGE-M3 model...")
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.logger.info("Using MPS (Metal Performance Shaders)")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU")
        
        # Load model
        self.model = SentenceTransformer('BAAI/bge-m3', device=self.device)
        self.model.eval()
        
        # Warm up
        self.logger.info("Warming up model...")
        for size in [1, 2, 4, 8, 16]:
            _ = self.model.encode(["warmup"] * size, batch_size=size, show_progress_bar=False)
        self.logger.info("Model ready")
        
        # Load FAISS index
        index_path = self.get_index_path()
        self.faiss_index = faiss.read_index(str(index_path / "faiss.index"))
        self.logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        
        # Load metadata
        with open(index_path / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        self.logger.info(f"Loaded metadata for {len(self.metadata)} codes")
    
    def inference_worker(self):
        """Process batches with dynamic batching"""
        while self.running:
            try:
                # Get a batch if ready
                batch = self.batcher.get_batch()
                
                if batch is None:
                    time.sleep(0.001)  # 1ms sleep
                    continue
                
                start_time = time.time()
                batch_size = len(batch)
                
                # Calculate wait time
                wait_times = [(start_time - req.arrival_time) for req in batch]
                avg_wait_time = sum(wait_times) / len(wait_times)
                
                # Prepare texts with BGE-M3 prefix
                texts = [f"Represent this sentence for searching relevant passages: {req.text}" 
                        for req in batch]
                
                # Model inference
                inference_start = time.time()
                with torch.no_grad():
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                inference_time = time.time() - inference_start
                
                # FAISS search
                search_start = time.time()
                embeddings_float32 = embeddings.astype('float32')
                distances, indices = self.faiss_index.search(embeddings_float32, 1)
                search_time = time.time() - search_start
                
                # Send responses
                response_count = 0
                code_key = self.get_metadata_key()
                default_code = self.get_default_code()
                
                for i, req in enumerate(batch):
                    try:
                        if indices[i][0] >= 0:
                            idx = indices[i][0]
                            meta = self.metadata[idx]
                            code = meta.get(code_key, default_code)
                        else:
                            code = default_code
                        
                        response = f"{req.seq_num},{code}\n"
                        req.client_socket.sendall(response.encode('utf-8'))
                        response_count += 1
                    except Exception as e:
                        self.logger.error(f"Error sending response: {e}")
                
                # Record metrics
                total_time = time.time() - start_time
                throughput = batch_size / total_time if total_time > 0 else 0
                
                metrics = BatchMetrics(
                    batch_size=batch_size,
                    wait_time=avg_wait_time * 1000,
                    inference_time=inference_time * 1000,
                    total_latency=total_time * 1000,
                    throughput=throughput
                )
                self.batcher.record_metrics(metrics)
                
                # Update stats
                with self.stats_lock:
                    self.stats['total_requests'] += batch_size
                    self.stats['total_batches'] += 1
                    self.stats['total_time'] += total_time
                
                self.logger.debug(f"Batch processed: size={batch_size}, "
                                f"latency={total_time*1000:.1f}ms, "
                                f"throughput={throughput:.1f} req/s")
                
            except Exception as e:
                self.logger.error(f"Inference worker error: {e}")
    
    def handle_client(self, client_socket, client_address):
        """Handle client connection"""
        try:
            while self.running:
                data = client_socket.recv(1024)
                if not data:
                    break
                
                text = data.decode('utf-8').strip()
                if not text:
                    continue
                
                # Parse request
                parts = text.split(',', 1)
                if len(parts) != 2:
                    client_socket.sendall(b"ERROR,Invalid format\n")
                    continue
                
                seq_num, description = parts
                
                # Create request
                with self.request_counter_lock:
                    self.request_counter += 1
                    request_id = f"req_{self.request_counter}"
                
                request = Request(
                    id=request_id,
                    seq_num=seq_num,
                    text=description,
                    client_socket=client_socket,
                    arrival_time=time.time()
                )
                
                # Add to batcher
                self.batcher.add_request(request)
                
        except Exception as e:
            self.logger.error(f"Client handler error: {e}")
        finally:
            client_socket.close()
    
    def report_stats(self):
        """Report statistics periodically"""
        while self.running:
            time.sleep(10)
            
            with self.stats_lock:
                if self.stats['total_requests'] > 0:
                    avg_time = self.stats['total_time'] / self.stats['total_batches']
                    avg_batch_size = self.stats['total_requests'] / self.stats['total_batches']
                    qps = self.stats['total_requests'] / self.stats['total_time'] if self.stats['total_time'] > 0 else 0
                    
                    self.logger.info(f"Stats: {self.stats['total_requests']} total requests, "
                                   f"{self.stats['total_batches']} batches "
                                   f"(avg size: {avg_batch_size:.1f}), "
                                   f"QPS: {qps:.1f}")
    
    def start(self):
        """Start the server"""
        # Load resources
        self.load_resources()
        
        # Start inference workers
        inference_threads = []
        for i in range(NUM_INFERENCE_WORKERS):
            t = threading.Thread(target=self.inference_worker, name=f"InferenceWorker-{i}")
            t.start()
            inference_threads.append(t)
        
        # Start stats reporter
        stats_thread = threading.Thread(target=self.report_stats, name="StatsReporter")
        stats_thread.start()
        
        # Start TCP server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(128)
        
        self.logger.info(f"{self.server_name} dynamic batching server listening on {self.host}:{self.port}")
        self.logger.info(f"Config: min_batch={MIN_BATCH_SIZE}, max_batch={MAX_BATCH_SIZE}, "
                        f"max_wait={MAX_WAIT_TIME_MS}ms, workers={NUM_INFERENCE_WORKERS}")
        
        try:
            while self.running:
                client_socket, client_address = server_socket.accept()
                self.client_thread_pool.submit(self.handle_client, client_socket, client_address)
        except KeyboardInterrupt:
            self.logger.info("Server interrupted")
        finally:
            self.running = False
            server_socket.close()
            self.client_thread_pool.shutdown(wait=True)