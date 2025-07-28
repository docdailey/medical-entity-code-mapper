#!/usr/bin/env python3
"""
Optimized ICD-10 TCP Server with Dynamic Batching
Achieves 300+ QPS with adaptive batch processing
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from servers.base_dynamic_server import BaseDynamicMedicalServer

class ICD10DynamicServer(BaseDynamicMedicalServer):
    """ICD-10 server with dynamic batching for high performance"""
    
    def __init__(self):
        super().__init__(host='0.0.0.0', port=8901, server_name="ICD-10")
    
    def get_index_path(self) -> Path:
        """Return path to ICD-10 FAISS index"""
        base_dir = Path(__file__).parent.parent.parent
        return base_dir / "data" / "indices" / "icd10_bge_m3"
    
    def get_metadata_key(self) -> str:
        """Return the key for ICD-10 code in metadata"""
        return 'icd10_code'
    
    def get_default_code(self) -> str:
        """Return default ICD-10 code for no match"""
        return 'R69'  # Illness, unspecified

if __name__ == "__main__":
    server = ICD10DynamicServer()
    server.start()