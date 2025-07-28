#!/usr/bin/env python3
"""
TLS-secured Entity Mapper TCP Server
Port: 8920
Protocol: sequence_number,clinical_text\n -> sequence_number,json_results\n
"""

import socket
import ssl
import threading
import json
import time
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from transformers import pipeline
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.input_validator import get_input_validator

# Configuration
HOST = '127.0.0.1'  # Loopback only for security
PORT = 8920  # TLS port for entity mapper
NUM_CLIENT_THREADS = 50

# TLS Configuration
TLS_CERT_FILE = os.getenv('TLS_CERT_FILE', 'certs/server.crt')
TLS_KEY_FILE = os.getenv('TLS_KEY_FILE', 'certs/server.key')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EntityMapperTLSServer:
    def __init__(self):
        # Input validator
        self.validator = get_input_validator()
        
        # Server configurations for TLS servers
        self.servers = {
            'diagnosis': {
                'icd10': ('localhost', 8911),   # TLS port
                'snomed': ('localhost', 8912)   # TLS port
            },
            'lab': {
                'loinc': ('localhost', 8913)    # TLS port
            },
            'medication': {
                'rxnorm': ('localhost', 8914)   # TLS port
            },
            'device': {
                'hcpcs': ('localhost', 8915)    # TLS port
            }
        }
        
        # SSL context for connecting to TLS servers
        self.client_ssl_context = ssl.create_default_context()
        self.client_ssl_context.check_hostname = False
        self.client_ssl_context.verify_mode = ssl.CERT_NONE
        
        # Initialize NER pipelines with proper aggregation
        logger.info("Loading NER models with improved configuration...")
        
        # Clinical NER with aggregation to handle subword tokens
        self.clinical_ner = pipeline(
            "ner", 
            model="samrawal/bert-base-uncased_clinical-ner",
            aggregation_strategy="max",  # Properly merge subword tokens
            device=-1
        )
        
        # Try biomedical NER for better medication recognition
        try:
            self.bio_ner = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="max",
                device=-1
            )
            self.use_bio_ner = True
            logger.info("Loaded biomedical NER for enhanced medication detection")
        except Exception:
            self.bio_ner = None
            self.use_bio_ner = False
            logger.info("Biomedical NER not available, using clinical NER only")
        
        # Disease NER with aggregation
        self.disease_ner = pipeline(
            "ner",
            model="alvaroalon2/biobert_diseases_ner",
            aggregation_strategy="max",
            device=-1
        )
        
        # Updated entity type mappings
        self.entity_type_map = {
            # Clinical NER labels
            'problem': 'diagnosis',
            'test': 'lab',
            'treatment': 'medication',
            
            # Disease NER labels
            'DISEASE': 'diagnosis',
            'DRUG': 'medication',
            'CHEMICAL': 'medication',
            
            # Biomedical NER labels (if available)
            'Disease_disorder': 'diagnosis',
            'Sign_symptom': 'diagnosis',
            'Medication': 'medication',
            'Diagnostic_procedure': 'lab',
            'Lab_value': 'lab',
            'Therapeutic_procedure': 'treatment'
        }
        
        # Medical device keywords for HCPCS routing
        self.device_keywords = {
            'catheter', 'wheelchair', 'walker', 'crutches', 'cane',
            'monitor', 'pump', 'syringe', 'needle', 'tube', 'tray',
            'bed', 'lift', 'brace', 'splint', 'prosthetic', 'orthotic',
            'cpap', 'bipap', 'ventilator', 'nebulizer', 'oxygen',
            'glucose meter', 'test strips', 'lancets', 'dressing',
            'bandage', 'gauze', 'tape', 'suture', 'staple',
            'implant', 'pacemaker', 'defibrillator', 'stent',
            'iv', 'infusion', 'picc', 'port', 'drain', 'bag',
            'commode', 'toilet', 'urinal', 'ostomy', 'colostomy',
            'tracheostomy', 'feeding tube', 'ng tube', 'peg tube'
        }
        
        # Context rules for better code selection
        self.context_rules = {
            'hypertension': {
                'adult_keywords': ['adult', 'elderly', 'male', 'female', 'year-old'],
                'pediatric_keywords': ['infant', 'child', 'neonate', 'newborn'],
                'adult_code': 'I10',
                'pediatric_code': 'P29.2'
            },
            'sepsis': {
                'device_keywords': ['catheter', 'line', 'device', 'implant'],
                'general_code': 'A41.9',
                'device_code': 'T85.735'
            }
        }
        
        logger.info("Entity Mapper TLS Server initialized")
        
    def _is_medical_device(self, entity_text: str) -> bool:
        """Check if entity is likely a medical device"""
        entity_lower = entity_text.lower()
        for device_keyword in self.device_keywords:
            if device_keyword in entity_lower:
                return True
        return False
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract entities using multiple NER models"""
        entities = []
        
        # Clinical NER extraction
        clinical_entities = self.clinical_ner(text)
        for entity in clinical_entities:
            entity_type = entity.get('entity_group', '').replace('B-', '').replace('I-', '')
            category = self.entity_type_map.get(entity_type)
            if category:
                # Check if it's a medical device
                if self._is_medical_device(entity['word']) and category == 'medication':
                    category = 'device'
                
                entities.append({
                    'entity': entity['word'],
                    'extracted_text': entity['word'],
                    'category': category,
                    'label': entity_type,
                    'score': entity['score'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'model': 'clinical'
                })
        
        # Biomedical NER extraction (if available)
        if self.use_bio_ner:
            bio_entities = self.bio_ner(text)
            for entity in bio_entities:
                entity_type = entity.get('entity_group', '')
                category = self.entity_type_map.get(entity_type)
                if category:
                    # Check if already found by clinical NER
                    duplicate = False
                    for existing in entities:
                        if (existing['start'] <= entity['start'] <= existing['end'] or
                            existing['start'] <= entity['end'] <= existing['end']):
                            duplicate = True
                            break
                    
                    if not duplicate:
                        # Check if it's a medical device
                        if self._is_medical_device(entity['word']) and category == 'medication':
                            category = 'device'
                        
                        entities.append({
                            'entity': entity['word'],
                            'extracted_text': entity['word'],
                            'category': category,
                            'label': entity_type,
                            'score': entity['score'],
                            'start': entity['start'],
                            'end': entity['end'],
                            'model': 'biomedical'
                        })
        
        # Disease NER extraction
        disease_entities = self.disease_ner(text)
        for entity in disease_entities:
            entity_type = entity.get('entity_group', '').replace('B-', '').replace('I-', '')
            category = self.entity_type_map.get(entity_type)
            if category:
                # Check for duplicates
                duplicate = False
                for existing in entities:
                    if (existing['start'] <= entity['start'] <= existing['end'] or
                        existing['start'] <= entity['end'] <= existing['end']):
                        duplicate = True
                        break
                
                if not duplicate:
                    # Check if it's a medical device
                    if self._is_medical_device(entity['word']) and category == 'medication':
                        category = 'device'
                    
                    entities.append({
                        'entity': entity['word'],
                        'extracted_text': entity['word'],
                        'category': category,
                        'label': entity_type,
                        'score': entity['score'],
                        'start': entity['start'],
                        'end': entity['end'],
                        'model': 'disease'
                    })
        
        # Sort by position and remove low-confidence entities
        entities = [e for e in entities if e['score'] > 0.8]
        entities.sort(key=lambda x: x['start'])
        
        return entities
    
    def _apply_context_rules(self, entity: Dict, text: str) -> Optional[str]:
        """Apply context-aware rules for better code selection"""
        entity_lower = entity['entity'].lower()
        text_lower = text.lower()
        
        # Check hypertension context
        if 'hypertension' in entity_lower:
            if any(keyword in text_lower for keyword in self.context_rules['hypertension']['pediatric_keywords']):
                return self.context_rules['hypertension']['pediatric_code']
            else:
                return self.context_rules['hypertension']['adult_code']
        
        # Check sepsis context
        if 'sepsis' in entity_lower:
            if any(keyword in text_lower for keyword in self.context_rules['sepsis']['device_keywords']):
                return self.context_rules['sepsis']['device_code']
            else:
                return self.context_rules['sepsis']['general_code']
        
        return None
    
    def _query_server(self, host: str, port: int, query: str, seq_num: int = 1) -> Optional[str]:
        """Query a TLS TCP server for code lookup"""
        try:
            # Validate host and port
            if not self.validator.validate_host_port(host, port):
                logger.warning(f"Invalid host/port: {host}:{port}")
                return None
                
            # Validate and sanitize query
            sanitized_query = self.validator.sanitize_query(query)
            if not sanitized_query:
                return None
                
            # Validate sequence number
            seq_num_str = str(seq_num)
            if not self.validator.validate_sequence_number(seq_num_str):
                seq_num_str = "1"
                
            # Create socket and wrap with SSL
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            
            with self.client_ssl_context.wrap_socket(sock, server_hostname=host) as ssock:
                ssock.connect((host, port))
                
                message = f"{seq_num_str},{sanitized_query}\n"
                ssock.send(message.encode('utf-8'))
                
                response = ssock.recv(4096).decode('utf-8').strip()
                
                if response:
                    parts = response.split(',')
                    if len(parts) >= 2:
                        code = parts[1].strip()
                        # Validate the returned code
                        if code and code not in ['NO_MATCH', 'NO_CODE', 'ERROR', 'TIMEOUT']:
                            # Additional validation to ensure code is safe
                            if len(code) <= self.validator.MAX_CODE_LENGTH and code.replace('.', '').replace('-', '').isalnum():
                                return code
                        
        except Exception as e:
            logger.error(f"Error querying {host}:{port}: {e}")
            
        return None
    
    def _lookup_codes(self, entity: Dict, text: str) -> Dict[str, Optional[str]]:
        """Lookup codes for an entity"""
        codes = {}
        
        # Check for context-specific code first
        if entity['category'] == 'diagnosis':
            context_code = self._apply_context_rules(entity, text)
            if context_code:
                codes['icd10'] = context_code
                # Still query SNOMED
                servers = self.servers.get(entity['category'], {})
                if 'snomed' in servers:
                    host, port = servers['snomed']
                    code = self._query_server(host, port, entity['entity'])
                    if code:
                        codes['snomed'] = code
                return codes
        
        # Normal server queries
        servers = self.servers.get(entity['category'], {})
        for system, (host, port) in servers.items():
            code = self._query_server(host, port, entity['entity'])
            codes[system] = code
                
        return codes
    
    def process_text(self, text: str) -> List[Dict]:
        """Process text to extract entities and map to codes"""
        # Validate and sanitize input
        if not text:
            return []
            
        sanitized_text = self.validator.sanitize_query(text)
        if not sanitized_text:
            logger.warning("Invalid or malicious input detected")
            return []
            
        # Extract entities
        entities = self._extract_entities(sanitized_text)
        
        # Process each entity
        results = []
        for entity in entities:
            # Get codes
            codes = self._lookup_codes(entity, text)
            
            # Skip if no codes found
            if not codes or all(code in [None, 'NO_MATCH', 'NO_CODE', 'ERROR', 'TIMEOUT'] for code in codes.values()):
                continue
            
            # Build result
            result = {
                'input_text': text,
                'category': entity['category'],
                'entity_name': entity['entity'],
                'extracted_text': entity['extracted_text'],
                'position': {
                    'start': entity['start'],
                    'end': entity['end']
                },
                'ner_confidence': round(float(entity['score']), 3),
                'codes': codes,
                'extraction_model': entity['model']
            }
            
            results.append(result)
            
        return results

# Global mapper instance
mapper = None
running = True

def create_ssl_context():
    """Create SSL context for TLS server"""
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    
    # Load certificate and key
    try:
        context.load_cert_chain(certfile=TLS_CERT_FILE, keyfile=TLS_KEY_FILE)
        logger.info(f"Loaded TLS certificate from {TLS_CERT_FILE}")
    except Exception as e:
        logger.error(f"Failed to load TLS certificate: {e}")
        sys.exit(1)
    
    # Set secure options
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
    
    return context

def handle_client(client_socket: socket.socket, client_address: tuple):
    """Handle a single client connection"""
    try:
        client_socket.settimeout(30.0)
        
        while running:
            try:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                
                # Process request
                request = data.decode('utf-8').strip()
                if not request:
                    continue
                    
                parts = request.split(',', 1)
                if len(parts) != 2:
                    response = f"{request},ERROR:INVALID_FORMAT\n"
                    client_socket.sendall(response.encode('utf-8'))
                    continue
                
                seq_num, text = parts
                
                # Process text with mapper
                results = mapper.process_text(text.strip())
                
                # Convert results to JSON
                json_results = json.dumps(results, separators=(',', ':'))
                
                # Send response
                response = f"{seq_num},{json_results}\n"
                client_socket.sendall(response.encode('utf-8'))
                
            except socket.timeout:
                break
            except Exception as e:
                logger.error(f"Error handling client request: {e}")
                break
                
    except Exception as e:
        logger.error(f"Client handler error: {e}")
    finally:
        try:
            client_socket.close()
        except:
            pass

def start_server():
    """Start the TLS-secured entity mapper server"""
    global mapper, running
    
    # Initialize mapper
    mapper = EntityMapperTLSServer()
    
    # Create SSL context
    ssl_context = create_ssl_context()
    
    # Set up server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(128)
    server_socket.settimeout(1.0)
    
    logger.info(f"TLS-secured Entity Mapper server listening on {HOST}:{PORT}")
    logger.info(f"Certificate: {TLS_CERT_FILE}")
    
    # Client handler pool
    with threading.ThreadPoolExecutor(max_workers=NUM_CLIENT_THREADS) as executor:
        try:
            while running:
                try:
                    client_socket, client_address = server_socket.accept()
                    # Wrap client socket with SSL
                    try:
                        ssl_client_socket = ssl_context.wrap_socket(client_socket, server_side=True)
                        executor.submit(handle_client, ssl_client_socket, client_address)
                    except ssl.SSLError as e:
                        logger.warning(f"SSL handshake failed: {e}")
                        client_socket.close()
                except socket.timeout:
                    continue
                except Exception as e:
                    if running:
                        logger.error(f"Accept error: {e}")
                        
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            running = False
            server_socket.close()

def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received")
    running = False

if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    start_server()