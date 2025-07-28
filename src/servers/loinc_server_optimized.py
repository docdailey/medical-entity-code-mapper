#!/usr/bin/env python3
"""
Optimized LOINC TCP Server with Dynamic Batching
Achieves 300+ QPS with adaptive batch processing
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from servers.base_dynamic_server import BaseDynamicMedicalServer

class LOINCDynamicServer(BaseDynamicMedicalServer):
    """LOINC server with dynamic batching for high performance"""
    
    def __init__(self):
        super().__init__(host='0.0.0.0', port=8903, server_name="LOINC")
    
    def get_index_path(self) -> Path:
        """Return path to LOINC FAISS index"""
        base_dir = Path(__file__).parent.parent.parent
        return base_dir / "data" / "indices" / "loinc_bge_m3"
    
    def get_metadata_key(self) -> str:
        """Return the key for LOINC code in metadata"""
        return 'loinc_code'
    
    def get_default_code(self) -> str:
        """Return default LOINC code for no match"""
        return 'UNKNOWN'

if __name__ == "__main__":
    server = LOINCDynamicServer()
    server.start()