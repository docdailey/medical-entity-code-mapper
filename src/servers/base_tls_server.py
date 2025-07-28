"""
Base TLS server with dynamic batching for medical coding services
Provides secure encrypted communication for healthcare data
"""

import socket
import ssl
import threading
import queue
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

class TLSConfig:
    """TLS configuration for secure servers"""
    def __init__(self, certfile: str = None, keyfile: str = None, 
                 cafile: str = None, require_client_cert: bool = False):
        # Use environment variables or defaults
        self.certfile = certfile or os.getenv('TLS_CERT_FILE', 'certs/server.crt')
        self.keyfile = keyfile or os.getenv('TLS_KEY_FILE', 'certs/server.key')
        self.cafile = cafile or os.getenv('TLS_CA_FILE', None)
        self.require_client_cert = require_client_cert
        
        # Create certs directory if it doesn't exist
        cert_dir = Path(self.certfile).parent
        cert_dir.mkdir(exist_ok=True)
        
        # Check if certificates exist, if not, create self-signed for development
        if not Path(self.certfile).exists() or not Path(self.keyfile).exists():
            self._create_self_signed_cert()
    
    def _create_self_signed_cert(self):
        """Create self-signed certificate for development/testing"""
        try:
            import subprocess
            logging.warning("Creating self-signed certificate for development use")
            
            # Generate private key
            subprocess.run([
                'openssl', 'genrsa', '-out', self.keyfile, '2048'
            ], check=True, capture_output=True)
            
            # Generate certificate
            subprocess.run([
                'openssl', 'req', '-new', '-x509', '-key', self.keyfile,
                '-out', self.certfile, '-days', '365',
                '-subj', '/C=US/ST=State/L=City/O=MedicalEntityMapper/CN=localhost'
            ], check=True, capture_output=True)
            
            logging.info(f"Created self-signed certificate: {self.certfile}")
        except Exception as e:
            logging.error(f"Failed to create self-signed certificate: {e}")
            raise

class BaseTLSDynamicServer:
    """Base class for TLS-secured dynamic batching TCP servers"""
    
    def __init__(self, host: str, port: int, server_name: str,
                 min_batch_size: int = 2, max_batch_size: int = 25,
                 max_wait_time_ms: int = 10, num_workers: int = 3,
                 tls_config: Optional[TLSConfig] = None):
        self.host = host
        self.port = port
        self.server_name = server_name
        self.tls_config = tls_config or TLSConfig()
        
        # Dynamic batching parameters
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.num_workers = num_workers
        
        # Queues and threading
        self.request_queue = queue.Queue()
        self.running = False
        
        # Logging
        self.logger = logging.getLogger(server_name)
        self.logger.setLevel(logging.INFO)
        
        # Performance tracking
        self.total_requests = 0
        self.total_batches = 0
        self.last_log_time = time.time()
        
        # Thread pools
        self.inference_pool = ThreadPoolExecutor(max_workers=num_workers)
        self.client_thread_pool = ThreadPoolExecutor(max_workers=100)
        
    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for secure communication"""
        # Use TLS 1.2 or higher for security
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # Load certificate and key
        context.load_cert_chain(
            certfile=self.tls_config.certfile,
            keyfile=self.tls_config.keyfile
        )
        
        # Optional: Require client certificates for mutual TLS
        if self.tls_config.require_client_cert:
            context.verify_mode = ssl.CERT_REQUIRED
            if self.tls_config.cafile:
                context.load_verify_locations(cafile=self.tls_config.cafile)
        
        # Set strong ciphers only
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        return context
    
    def load_model(self):
        """Override this to load your specific model"""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def load_index(self):
        """Override this to load your specific FAISS index"""
        raise NotImplementedError("Subclasses must implement load_index")
    
    def process_batch(self, queries: List[str]) -> List[str]:
        """Override this to implement batch processing logic"""
        raise NotImplementedError("Subclasses must implement process_batch")
    
    def inference_worker(self):
        """Worker thread for processing batches"""
        while self.running:
            batch = []
            batch_start_time = time.time()
            
            try:
                # Wait for first item
                first_item = self.request_queue.get(timeout=1.0)
                batch.append(first_item)
            except queue.Empty:
                continue
            
            # Collect more items up to max_batch_size or max_wait_time
            deadline = batch_start_time + (self.max_wait_time_ms / 1000.0)
            
            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    remaining_time = deadline - time.time()
                    if remaining_time > 0:
                        item = self.request_queue.get(timeout=remaining_time)
                        batch.append(item)
                except queue.Empty:
                    break
            
            # Process batch if we have enough items or timeout reached
            if len(batch) >= self.min_batch_size or \
               (len(batch) > 0 and time.time() >= deadline):
                try:
                    queries = [item[1] for item in batch]
                    results = self.process_batch(queries)
                    
                    # Send results back
                    for (result_queue, _), result in zip(batch, results):
                        result_queue.put(result)
                    
                    # Update stats
                    self.total_batches += 1
                    self.total_requests += len(batch)
                    
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
                    # Send error responses
                    for (result_queue, _) in batch:
                        result_queue.put(f"ERROR: {str(e)}")
            else:
                # Put items back if batch too small
                for item in batch:
                    self.request_queue.put(item)
    
    def handle_client(self, client_socket: ssl.SSLSocket, client_address: tuple):
        """Handle individual client connections with TLS"""
        try:
            self.logger.debug(f"Secure connection from {client_address}")
            client_socket.settimeout(30.0)
            
            buffer = ""
            while True:
                try:
                    data = client_socket.recv(4096).decode('utf-8')
                    if not data:
                        break
                    
                    buffer += data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if not line:
                            continue
                        
                        parts = line.split(',', 1)
                        if len(parts) != 2:
                            response = "ERROR: Invalid format. Use: sequence_number,query\\n"
                            client_socket.send(response.encode('utf-8'))
                            continue
                        
                        seq_num, query = parts
                        
                        # Submit to processing queue
                        result_queue = queue.Queue()
                        self.request_queue.put((result_queue, query))
                        
                        # Wait for result
                        try:
                            result = result_queue.get(timeout=5.0)
                            response = f"{seq_num},{result}\n"
                        except queue.Empty:
                            response = f"{seq_num},TIMEOUT\n"
                        
                        client_socket.send(response.encode('utf-8'))
                        
                except socket.timeout:
                    continue
                except ssl.SSLError as e:
                    self.logger.error(f"SSL error: {e}")
                    break
                except Exception as e:
                    self.logger.error(f"Client receive error: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Client handler error: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def log_stats(self):
        """Log performance statistics"""
        current_time = time.time()
        if current_time - self.last_log_time >= 10:
            elapsed = current_time - self.last_log_time
            if self.total_batches > 0:
                avg_batch_size = self.total_requests / self.total_batches
                qps = self.total_requests / elapsed
                self.logger.info(
                    f"Stats: {self.total_requests} total requests, "
                    f"{self.total_batches} batches (avg size: {avg_batch_size:.1f}), "
                    f"QPS: {qps:.1f}"
                )
            self.last_log_time = current_time
    
    def run(self):
        """Run the TLS server"""
        # Load model and index
        self.logger.info(f"Loading model for {self.server_name}...")
        self.load_model()
        
        self.logger.info("Loading index...")
        self.load_index()
        
        # Start inference workers
        self.running = True
        for i in range(self.num_workers):
            self.inference_pool.submit(self.inference_worker)
        
        # Create SSL context
        ssl_context = self.create_ssl_context()
        
        # Create server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(128)
        
        # Wrap with SSL
        server_socket = ssl_context.wrap_socket(server_socket, server_side=True)
        
        self.logger.info(
            f"{self.server_name} TLS server listening on {self.host}:{self.port} "
            f"(Certificate: {self.tls_config.certfile})"
        )
        self.logger.info(
            f"Config: min_batch={self.min_batch_size}, max_batch={self.max_batch_size}, "
            f"max_wait={self.max_wait_time_ms}ms, workers={self.num_workers}"
        )
        
        # Stats thread
        stats_thread = threading.Thread(target=self._stats_logger, daemon=True)
        stats_thread.start()
        
        try:
            while True:
                client_socket, client_address = server_socket.accept()
                self.client_thread_pool.submit(self.handle_client, client_socket, client_address)
        except KeyboardInterrupt:
            self.logger.info("Server interrupted")
        finally:
            self.running = False
            server_socket.close()
            self.inference_pool.shutdown(wait=True)
            self.client_thread_pool.shutdown(wait=True)
    
    def _stats_logger(self):
        """Background thread for logging stats"""
        while self.running:
            self.log_stats()
            time.sleep(10)