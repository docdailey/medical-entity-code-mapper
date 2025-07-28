#!/usr/bin/env python3
"""
TLS-secured ICD-10 server with dynamic batching for high-performance medical coding
Provides encrypted communication for sensitive healthcare data
"""

import numpy as np
import torch
import faiss
import pickle
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List
import os

from base_tls_server import BaseTLSDynamicServer, TLSConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ICD10TLSServer(BaseTLSDynamicServer):
    """TLS-secured ICD-10 server with dynamic batching"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8901,
                 index_path: str = 'indices/icd10_bge_m3',
                 tls_config: TLSConfig = None):
        super().__init__(
            host=host,
            port=port,
            server_name="ICD-10 TLS",
            min_batch_size=2,
            max_batch_size=25,
            max_wait_time_ms=10,
            num_workers=3,
            tls_config=tls_config
        )
        self.index_path = Path(index_path)
        self.model = None
        self.index = None
        self.metadata = None
        self.device = None
        
    def load_model(self):
        """Load BGE-M3 model with MPS support"""
        # Check for MPS availability (Apple Silicon)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.logger.info("Using MPS (Metal Performance Shaders)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info("Using CUDA")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU")
        
        # Load model
        self.model = SentenceTransformer('BAAI/bge-m3')
        
        # Move model to device
        if self.device.type != 'cpu':
            self.model = self.model.to(self.device)
        
        # Warm up model
        self.logger.info("Warming up model...")
        _ = self.model.encode(["test"], convert_to_tensor=True)
        self.logger.info("Model ready")
        
    def load_index(self):
        """Load FAISS index and metadata"""
        # Load FAISS index
        index_file = self.index_path / 'faiss.index'
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        self.logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load metadata
        metadata_file = self.index_path / 'metadata.pkl'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        self.logger.info(f"Loaded metadata for {len(self.metadata['codes'])} codes")
        
    def process_batch(self, queries: List[str]) -> List[str]:
        """Process a batch of queries and return ICD-10 codes"""
        # Add prefix for better performance
        prefixed_queries = [f"Represent this sentence for searching relevant passages: {q}" 
                           for q in queries]
        
        # Encode queries
        query_embeddings = self.model.encode(
            prefixed_queries,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Convert to numpy for FAISS
        if self.device.type != 'cpu':
            query_embeddings = query_embeddings.cpu()
        query_embeddings = query_embeddings.numpy()
        
        # Search in FAISS
        k = 1  # Get top result
        distances, indices = self.index.search(query_embeddings, k)
        
        # Get codes for each query
        results = []
        for idx_list in indices:
            if idx_list[0] >= 0:  # Valid result
                code = self.metadata['codes'][idx_list[0]]
                results.append(code)
            else:
                results.append("NO_MATCH")
        
        return results

def main():
    """Run the TLS-secured ICD-10 server"""
    # Get configuration from environment or use defaults
    host = os.getenv('SERVER_HOST', '0.0.0.0')
    port = int(os.getenv('ICD10_TLS_PORT', '8911'))  # Different port for TLS
    index_path = os.getenv('ICD10_INDEX_PATH', 'indices/icd10_bge_m3')
    
    # TLS configuration
    tls_config = TLSConfig(
        certfile=os.getenv('TLS_CERT_FILE'),
        keyfile=os.getenv('TLS_KEY_FILE'),
        require_client_cert=os.getenv('TLS_REQUIRE_CLIENT_CERT', 'false').lower() == 'true'
    )
    
    # Create and run server
    server = ICD10TLSServer(host, port, index_path, tls_config)
    server.run()

if __name__ == "__main__":
    main()