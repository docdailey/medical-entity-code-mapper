#!/usr/bin/env python3
"""
Example TLS client for secure medical coding servers
Demonstrates how to connect securely to TLS-enabled servers
"""

import socket
import ssl
import time
import sys
from pathlib import Path

def query_tls_server(host: str, port: int, query: str, seq_num: int = 1,
                     verify_cert: bool = False, ca_cert: str = None) -> str:
    """
    Query a TLS-secured medical coding server
    
    Args:
        host: Server hostname
        port: Server port
        query: Medical text to code
        seq_num: Sequence number for the query
        verify_cert: Whether to verify server certificate (False for self-signed)
        ca_cert: Path to CA certificate for verification
    
    Returns:
        Server response (code or error)
    """
    try:
        # Create SSL context
        if verify_cert and ca_cert:
            # Production mode - verify certificate
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            context.load_verify_locations(cafile=ca_cert)
        else:
            # Development mode - accept self-signed certificates
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        
        # Create socket and wrap with SSL
        with socket.create_connection((host, port), timeout=5.0) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                # Print connection info
                print(f"Connected to {host}:{port}")
                print(f"Using cipher: {ssock.cipher()}")
                print(f"Server certificate: {ssock.getpeercert()}")
                
                # Send query
                message = f"{seq_num},{query}\n"
                ssock.send(message.encode('utf-8'))
                
                # Receive response
                response = ssock.recv(4096).decode('utf-8').strip()
                
                # Parse response
                parts = response.split(',', 1)
                if len(parts) == 2:
                    _, code = parts
                    return code
                else:
                    return f"Invalid response: {response}"
                    
    except ssl.SSLError as e:
        return f"SSL Error: {e}"
    except socket.timeout:
        return "Connection timeout"
    except Exception as e:
        return f"Error: {e}"

def main():
    """Example usage of TLS client"""
    # Configuration
    host = 'localhost'
    port = 8911  # TLS port for ICD-10
    
    # Example medical texts
    test_queries = [
        "chest pain radiating to left arm",
        "type 2 diabetes mellitus with complications",
        "acute myocardial infarction",
        "essential hypertension",
        "bacterial pneumonia"
    ]
    
    print("TLS Medical Coding Client Example")
    print("=" * 50)
    print(f"Connecting to TLS server at {host}:{port}")
    print("Note: Using self-signed certificate (development mode)")
    print("=" * 50)
    
    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        start_time = time.time()
        
        result = query_tls_server(host, port, query, seq_num=i, verify_cert=False)
        
        latency = (time.time() - start_time) * 1000
        print(f"Result: {result}")
        print(f"Latency: {latency:.1f}ms")
    
    print("\n" + "=" * 50)
    print("TLS Security Notes:")
    print("- All communication is encrypted using TLS 1.2+")
    print("- For production, use proper certificates from a CA")
    print("- Enable certificate verification in production")
    print("- Consider mutual TLS for additional security")

if __name__ == "__main__":
    main()