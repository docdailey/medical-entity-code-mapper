#!/usr/bin/env python3
"""
Generate TLS certificates for medical coding servers
Includes options for self-signed (development) and CSR generation (production)
"""

import subprocess
import os
from pathlib import Path
import argparse

def generate_self_signed_cert(domain: str = "localhost", 
                             cert_dir: str = "certs",
                             days: int = 365):
    """Generate self-signed certificate for development"""
    cert_path = Path(cert_dir)
    cert_path.mkdir(exist_ok=True)
    
    key_file = cert_path / "server.key"
    cert_file = cert_path / "server.crt"
    
    print(f"Generating self-signed certificate for {domain}...")
    
    # Generate private key
    subprocess.run([
        'openssl', 'genrsa', '-out', str(key_file), '4096'
    ], check=True)
    print(f"✓ Generated private key: {key_file}")
    
    # Generate certificate
    subprocess.run([
        'openssl', 'req', '-new', '-x509', '-key', str(key_file),
        '-out', str(cert_file), '-days', str(days),
        '-subj', f'/C=US/ST=State/L=City/O=Healthcare/OU=MedicalCoding/CN={domain}',
        '-addext', f'subjectAltName=DNS:{domain},DNS:*.{domain},IP:127.0.0.1'
    ], check=True)
    print(f"✓ Generated certificate: {cert_file}")
    
    # Set secure permissions
    os.chmod(key_file, 0o600)
    print("✓ Set secure permissions on private key")
    
    return cert_file, key_file

def generate_csr(domain: str, cert_dir: str = "certs"):
    """Generate Certificate Signing Request for production"""
    cert_path = Path(cert_dir)
    cert_path.mkdir(exist_ok=True)
    
    key_file = cert_path / "server.key"
    csr_file = cert_path / "server.csr"
    
    print(f"Generating CSR for {domain}...")
    
    # Generate private key
    subprocess.run([
        'openssl', 'genrsa', '-out', str(key_file), '4096'
    ], check=True)
    print(f"✓ Generated private key: {key_file}")
    
    # Generate CSR
    subprocess.run([
        'openssl', 'req', '-new', '-key', str(key_file),
        '-out', str(csr_file),
        '-subj', f'/C=US/ST=State/L=City/O=Healthcare/OU=MedicalCoding/CN={domain}'
    ], check=True)
    print(f"✓ Generated CSR: {csr_file}")
    
    # Set secure permissions
    os.chmod(key_file, 0o600)
    
    print("\nNext steps for production:")
    print("1. Submit the CSR to your Certificate Authority")
    print("2. Save the signed certificate as 'certs/server.crt'")
    print("3. If provided, save the CA chain as 'certs/ca-chain.crt'")
    
    return csr_file, key_file

def create_client_cert(client_name: str, cert_dir: str = "certs"):
    """Generate client certificate for mutual TLS"""
    cert_path = Path(cert_dir)
    cert_path.mkdir(exist_ok=True)
    
    key_file = cert_path / f"client-{client_name}.key"
    cert_file = cert_path / f"client-{client_name}.crt"
    
    print(f"Generating client certificate for {client_name}...")
    
    # Generate private key
    subprocess.run([
        'openssl', 'genrsa', '-out', str(key_file), '4096'
    ], check=True)
    
    # Generate certificate
    subprocess.run([
        'openssl', 'req', '-new', '-x509', '-key', str(key_file),
        '-out', str(cert_file), '-days', 365,
        '-subj', f'/C=US/ST=State/L=City/O=Healthcare/OU=Client/CN={client_name}'
    ], check=True)
    
    # Set secure permissions
    os.chmod(key_file, 0o600)
    
    print(f"✓ Generated client certificate: {cert_file}")
    print(f"✓ Generated client key: {key_file}")
    
    return cert_file, key_file

def main():
    parser = argparse.ArgumentParser(description="Generate TLS certificates")
    parser.add_argument('--type', choices=['self-signed', 'csr', 'client'],
                       default='self-signed',
                       help='Type of certificate to generate')
    parser.add_argument('--domain', default='localhost',
                       help='Domain name for the certificate')
    parser.add_argument('--client-name', default='medical-client',
                       help='Client name for mutual TLS')
    parser.add_argument('--cert-dir', default='certs',
                       help='Directory to store certificates')
    parser.add_argument('--days', type=int, default=365,
                       help='Certificate validity in days')
    
    args = parser.parse_args()
    
    print("Medical Entity Code Mapper - TLS Certificate Generator")
    print("=" * 60)
    
    if args.type == 'self-signed':
        cert_file, key_file = generate_self_signed_cert(
            args.domain, args.cert_dir, args.days
        )
        print(f"\n✓ Self-signed certificate generated successfully!")
        print(f"\nTo use with servers, set environment variables:")
        print(f"export TLS_CERT_FILE={cert_file}")
        print(f"export TLS_KEY_FILE={key_file}")
        
    elif args.type == 'csr':
        csr_file, key_file = generate_csr(args.domain, args.cert_dir)
        print(f"\n✓ CSR generated successfully!")
        
    elif args.type == 'client':
        cert_file, key_file = create_client_cert(args.client_name, args.cert_dir)
        print(f"\n✓ Client certificate generated successfully!")
        print(f"\nFor mutual TLS, add to server configuration:")
        print(f"export TLS_CA_FILE={cert_file}")
        print(f"export TLS_REQUIRE_CLIENT_CERT=true")
    
    print("\n⚠️  Security Recommendations:")
    print("- Use self-signed certificates for development only")
    print("- For production, use certificates from a trusted CA")
    print("- Keep private keys secure and never commit to git")
    print("- Rotate certificates regularly (e.g., every 90 days)")
    print("- Use mutual TLS for additional security in production")

if __name__ == "__main__":
    main()