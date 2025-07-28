#!/usr/bin/env python3
"""
Optimized SNOMED CT TCP Server with Dynamic Batching
Achieves 300+ QPS with adaptive batch processing
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from servers.base_dynamic_server import BaseDynamicMedicalServer

class SNOMEDDynamicServer(BaseDynamicMedicalServer):
    """SNOMED CT server with dynamic batching for high performance"""
    
    def __init__(self):
        super().__init__(host='0.0.0.0', port=8902, server_name="SNOMED")
    
    def get_index_path(self) -> Path:
        """Return path to SNOMED FAISS index"""
        base_dir = Path(__file__).parent.parent.parent
        return base_dir / "data" / "indices" / "snomed_bge_m3"
    
    def get_metadata_key(self) -> str:
        """Return the key for SNOMED code in metadata"""
        return 'snomed_code'
    
    def get_default_code(self) -> str:
        """Return default SNOMED code for no match"""
        return '261665006'  # Unknown

if __name__ == "__main__":
    server = SNOMEDDynamicServer()
    server.start()