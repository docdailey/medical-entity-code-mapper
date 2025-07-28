#!/usr/bin/env python3
"""
TLS-secured TCP Server for SNOMED CT code assignment using FAISS similarity search
Port: 8912
Protocol: sequence_number,description\n -> sequence_number,snomed_code\n
"""

import socket
import ssl
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
import os
import queue
from collections import deque
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# TLS Configuration
TLS_CERT_FILE = os.getenv('TLS_CERT_FILE', 'certs/server.crt')
TLS_KEY_FILE = os.getenv('TLS_KEY_FILE', 'certs/server.key')

class SNOMEDFAISSTLSServer:
    def __init__(self, host='0.0.0.0', port=8912):
        self.host = host
        self.port = port
        self.running = False
        
        # FAISS index path
        # Get project root directory
        base_dir = Path(__file__).parent.parent.parent
        self.index_dir = base_dir / "data" / "indices" / "snomed_bge_m3"
        
        # Load FAISS index and metadata
        logging.info("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        
        logging.info("Loading metadata...")
        with open(self.index_dir / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load BGE-M3 model
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
        
        # Warm up model
        logging.info("Warming up model...")
        _ = self.model.encode(["warmup"], show_progress_bar=False)
        
        logging.info(f"SNOMED server ready with {self.index.ntotal} concepts")
        
        # Request batching
        self.request_queue = queue.Queue()
        self.batch_size = 16
        self.batch_timeout = 0.01  # 10ms
        
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
        
    def batch_processor(self):
        """Process requests in batches"""
        batch = []
        last_process_time = time.time()
        
        while self.running:
            try:
                # Try to get a request with timeout
                timeout = max(0.001, self.batch_timeout - (time.time() - last_process_time))
                request = self.request_queue.get(timeout=timeout)
                batch.append(request)
                
                # Process if batch is full
                if len(batch) >= self.batch_size:
                    self.process_batch(batch)
                    batch = []
                    last_process_time = time.time()
                    
            except queue.Empty:
                # Process partial batch on timeout
                if batch and time.time() - last_process_time >= self.batch_timeout:
                    self.process_batch(batch)
                    batch = []
                    last_process_time = time.time()
                    
    def process_batch(self, batch):
        """Process a batch of requests"""
        texts = []
        for seq_num, text, client_socket in batch:
            # Add prefix for better performance with BGE-M3
            texts.append(f"Represent this sentence for searching relevant passages: {text}")
        
        # Encode texts
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        
        # Search in FAISS
        distances, indices = self.index.search(embeddings.astype('float32'), 1)
        
        # Send responses
        for i, (seq_num, text, client_socket) in enumerate(batch):
            try:
                if indices[i][0] >= 0:
                    snomed_code = self.metadata[indices[i][0]]['snomed_code']
                else:
                    snomed_code = "NO_MATCH"
                
                response = f"{seq_num},{snomed_code}\n"
                client_socket.sendall(response.encode('utf-8'))
            except Exception as e:
                logging.error(f"Error sending response: {e}")
                
    def handle_client(self, client_socket, client_address):
        """Handle a single client connection"""
        try:
            client_socket.settimeout(30.0)
            
            while self.running:
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    
                    # Process each request
                    requests = data.decode('utf-8').strip().split('\n')
                    for request in requests:
                        if not request:
                            continue
                            
                        parts = request.split(',', 1)
                        if len(parts) != 2:
                            response = f"{request},INVALID_FORMAT\n"
                            client_socket.sendall(response.encode('utf-8'))
                            continue
                        
                        seq_num, text = parts
                        self.request_queue.put((seq_num, text.strip(), client_socket))
                        
                except socket.timeout:
                    break
                except Exception as e:
                    logging.error(f"Error handling client request: {e}")
                    break
                    
        except Exception as e:
            logging.error(f"Client handler error: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
                
    def start(self):
        """Start the TLS server"""
        self.running = True
        
        # Start batch processor thread
        batch_thread = threading.Thread(target=self.batch_processor, daemon=True)
        batch_thread.start()
        
        # Create server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(128)
        server_socket.settimeout(1.0)
        
        logging.info(f"TLS-secured SNOMED server listening on {self.host}:{self.port}")
        logging.info(f"Certificate: {TLS_CERT_FILE}")
        
        try:
            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()
                    # Wrap client socket with SSL
                    try:
                        ssl_client_socket = self.ssl_context.wrap_socket(client_socket, server_side=True)
                        client_thread = threading.Thread(
                            target=self.handle_client, 
                            args=(ssl_client_socket, client_address),
                            daemon=True
                        )
                        client_thread.start()
                    except ssl.SSLError as e:
                        logging.warning(f"SSL handshake failed: {e}")
                        client_socket.close()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logging.error(f"Error accepting connection: {e}")
                        
        except KeyboardInterrupt:
            logging.info("Shutdown requested")
        finally:
            self.stop()
            server_socket.close()
            
    def stop(self):
        """Stop the server"""
        self.running = False
        logging.info("SNOMED TLS server stopped")

def signal_handler(sig, frame):
    logging.info("Shutdown signal received")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server = SNOMEDFAISSTLSServer()
    server.start()