#!/usr/bin/env python3
"""
Dynamic batching BGE-M3 TCP server
Optimizes the balance between latency and throughput
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
from typing import List, Dict, Tuple, Optional
import signal
import sys
from dataclasses import dataclass
from collections import defaultdict, deque

# Configuration
HOST = '127.0.0.1'  # Loopback only for security
PORT = 8901
NUM_CLIENT_THREADS = 50

# Dynamic batching parameters
MIN_BATCH_SIZE = 8      # Minimum batch size before processing
MAX_BATCH_SIZE = 64     # Maximum batch size
MAX_WAIT_TIME_MS = 5    # Maximum time to wait for batch to fill (ms)
ADAPTIVE_WINDOW = 100   # Number of batches to consider for adaptation

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    
    def __init__(self, min_batch_size=8, max_batch_size=64, max_wait_time_ms=5):
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
        
        # Check min batch size and time
        if batch_size >= self.current_min_batch:
            return True
            
        # Check wait time for oldest request
        oldest_request = self.pending_requests[0]
        wait_time_ms = (time.time() - oldest_request.arrival_time) * 1000
        
        if wait_time_ms >= self.current_max_wait:
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
            self.current_min_batch = max(4, self.current_min_batch - 1)
            self.current_max_wait = max(2, self.current_max_wait - 0.5)
        elif avg_throughput < 200 and avg_batch_size < 20:  # Throughput too low
            # Increase batching to improve throughput
            self.current_min_batch = min(16, self.current_min_batch + 1)
            self.current_max_wait = min(10, self.current_max_wait + 0.5)
        
        logger.debug(f"Adapted parameters: min_batch={self.current_min_batch}, "
                    f"max_wait={self.current_max_wait}ms")

# Global resources
model = None
device = None
faiss_index = None
icd_metadata = None
batcher = None
running = True

# Statistics
stats = defaultdict(int)
stats_lock = threading.Lock()

def load_resources():
    """Load model and FAISS index"""
    global model, device, faiss_index, icd_metadata
    
    logger.info("Loading BGE-M3 model...")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Load model
    model = SentenceTransformer('BAAI/bge-m3', device=device)
    model.eval()
    
    # Warm up with various batch sizes
    logger.info("Warming up model...")
    for size in [1, 8, 16, 32, 64]:
        _ = model.encode(["warmup"] * size, batch_size=size, show_progress_bar=False)
    logger.info("Model ready")
    
    # Load FAISS index
    base_dir = Path(__file__).parent.parent.parent  # Get to project root
    index_dir = base_dir / "data" / "indices" / "icd10_bge_m3"
    faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
    # Note: This is IndexFlatIP, not HNSW, so no efSearch parameter
    logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors")
    
    # Load metadata
    with open(index_dir / "metadata.pkl", 'rb') as f:
        icd_metadata = pickle.load(f)
    logger.info(f"Loaded metadata for {len(icd_metadata)} ICD-10 codes")

def inference_worker():
    """Process batches with dynamic batching"""
    global batcher
    
    while running:
        try:
            # Get a batch if ready
            batch = batcher.get_batch()
            
            if batch is None:
                time.sleep(0.001)  # 1ms sleep to avoid busy waiting
                continue
            
            start_time = time.time()
            batch_size = len(batch)
            
            # Calculate wait time for metrics
            wait_times = [(start_time - req.arrival_time) for req in batch]
            avg_wait_time = sum(wait_times) / len(wait_times)
            
            # Prepare texts
            texts = [f"Represent this sentence for searching relevant passages: {req.text}" 
                    for req in batch]
            
            # Model inference
            inference_start = time.time()
            with torch.no_grad():
                embeddings = model.encode(
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
            distances, indices = faiss_index.search(embeddings_float32, 1)
            search_time = time.time() - search_start
            
            # Send responses
            response_count = 0
            for i, req in enumerate(batch):
                try:
                    if indices[i][0] >= 0:
                        idx = indices[i][0]
                        meta = icd_metadata[idx]
                        icd_code = meta['icd10_code']
                    else:
                        icd_code = 'R69'
                    
                    response = f"{req.seq_num},{icd_code}\n"
                    req.client_socket.sendall(response.encode('utf-8'))
                    response_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to send response: {e}")
            
            # Calculate metrics
            total_time = time.time() - start_time
            throughput = batch_size / total_time if total_time > 0 else 0
            avg_latency = (total_time / batch_size) * 1000  # ms
            
            # Record metrics
            metrics = BatchMetrics(
                batch_size=batch_size,
                wait_time=avg_wait_time * 1000,  # ms
                inference_time=inference_time * 1000,  # ms
                total_latency=avg_latency,
                throughput=throughput
            )
            batcher.record_metrics(metrics)
            
            # Update stats
            with stats_lock:
                stats['total_requests'] += batch_size
                stats['total_batches'] += 1
                stats['total_inference_time'] += inference_time
            
            # Log performance
            logger.info(f"Batch processed: size={batch_size}, "
                       f"wait={avg_wait_time*1000:.1f}ms, "
                       f"inference={inference_time*1000:.1f}ms, "
                       f"total={total_time*1000:.1f}ms, "
                       f"throughput={throughput:.1f} req/s")
            
        except Exception as e:
            logger.error(f"Inference worker error: {e}")
            import traceback
            traceback.print_exc()

def handle_client(client_socket: socket.socket, client_address: tuple):
    """Handle client connection"""
    request_count = 0
    
    try:
        logger.info(f"New connection from {client_address}")
        
        # Configure socket
        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client_socket.settimeout(30.0)
        
        buffer = ""
        
        while running:
            try:
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line == "HEALTH":
                        client_socket.sendall(b"OK\n")
                        continue
                    
                    if ',' not in line:
                        continue
                    
                    # Parse request
                    seq_num, text = line.split(',', 1)
                    
                    # Create request and add to batcher
                    request = Request(
                        id=f"{client_address}:{request_count}",
                        seq_num=seq_num,
                        text=text.strip(),
                        client_socket=client_socket,
                        arrival_time=time.time()
                    )
                    
                    batcher.add_request(request)
                    request_count += 1
                    
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Client receive error: {e}")
                break
        
        logger.info(f"Connection from {client_address} closed after {request_count} requests")
        
    except Exception as e:
        logger.error(f"Client handler error: {e}")
    finally:
        client_socket.close()

def stats_reporter():
    """Report statistics periodically"""
    last_requests = 0
    last_time = time.time()
    
    while running:
        time.sleep(10)
        
        with stats_lock:
            total_requests = stats['total_requests']
            total_batches = stats['total_batches']
            total_inference_time = stats['total_inference_time']
        
        current_time = time.time()
        period_time = current_time - last_time
        period_requests = total_requests - last_requests
        
        if period_time > 0:
            qps = period_requests / period_time
            
            if total_batches > 0:
                avg_batch_size = total_requests / total_batches
            else:
                avg_batch_size = 0
            
            logger.info(f"Stats: {total_requests} total requests, "
                       f"{total_batches} batches (avg size: {avg_batch_size:.1f}), "
                       f"QPS: {qps:.1f}")
        
        last_requests = total_requests
        last_time = current_time

def start_server():
    """Start the dynamic batching server"""
    global running, batcher
    
    # Initialize batcher
    batcher = DynamicBatcher(
        min_batch_size=MIN_BATCH_SIZE,
        max_batch_size=MAX_BATCH_SIZE,
        max_wait_time_ms=MAX_WAIT_TIME_MS
    )
    
    # Load resources
    load_resources()
    
    # Start threads
    threads = []
    
    # Start multiple inference workers
    num_inference_workers = 3
    for i in range(num_inference_workers):
        worker = threading.Thread(target=inference_worker, name=f"InferenceWorker-{i}")
        worker.start()
        threads.append(worker)
    
    # Stats reporter
    stats_thread = threading.Thread(target=stats_reporter, name="StatsReporter", daemon=True)
    stats_thread.start()
    
    # Set up server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(128)
    server_socket.settimeout(1.0)
    
    logger.info(f"Dynamic batching server listening on {HOST}:{PORT}")
    logger.info(f"Config: min_batch={MIN_BATCH_SIZE}, max_batch={MAX_BATCH_SIZE}, "
               f"max_wait={MAX_WAIT_TIME_MS}ms, workers={num_inference_workers}")
    
    # Client handler pool
    with ThreadPoolExecutor(max_workers=NUM_CLIENT_THREADS) as executor:
        try:
            while running:
                try:
                    client_socket, client_address = server_socket.accept()
                    executor.submit(handle_client, client_socket, client_address)
                except socket.timeout:
                    continue
                except Exception as e:
                    if running:
                        logger.error(f"Accept error: {e}")
                        
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            running = False
            
            # Wait for threads
            for thread in threads:
                thread.join(timeout=5)
            
            server_socket.close()
            
            # Final stats
            with stats_lock:
                total_requests = stats['total_requests']
                total_batches = stats['total_batches']
                if total_batches > 0:
                    avg_batch_size = total_requests / total_batches
                    logger.info(f"Final stats: {total_requests} requests in {total_batches} batches "
                               f"(avg size: {avg_batch_size:.1f})")

def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received")
    running = False

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    start_server()