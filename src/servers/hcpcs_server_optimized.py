#!/usr/bin/env python3
"""
Optimized HCPCS TCP Server with Dynamic Batching
Achieves 300+ QPS with adaptive batch processing
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from servers.base_dynamic_server import BaseDynamicMedicalServer

class HCPCSDynamicServer(BaseDynamicMedicalServer):
    """HCPCS server with dynamic batching for high performance"""
    
    def __init__(self):
        super().__init__(host='0.0.0.0', port=8905, server_name="HCPCS")
    
    def get_index_path(self) -> Path:
        """Return path to HCPCS FAISS index"""
        base_dir = Path(__file__).parent.parent.parent
        return base_dir / "data" / "indices" / "hcpcs_bge_m3"
    
    def get_metadata_key(self) -> str:
        """Return the key for HCPCS code in metadata"""
        return 'hcpcs_code'
    
    def get_default_code(self) -> str:
        """Return default HCPCS code for no match"""
        return 'NO_MATCH'

if __name__ == "__main__":
    server = HCPCSDynamicServer()
    server.start()