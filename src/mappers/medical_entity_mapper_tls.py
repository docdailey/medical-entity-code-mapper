#!/usr/bin/env python3
"""
TLS-Secured Medical Entity Code Mapper
Connects to TLS-enabled coding servers for secure healthcare data processing
"""

import json
import socket
import ssl
import time
from typing import List, Dict, Optional, Tuple
from transformers import pipeline
from collections import defaultdict
import os
from pathlib import Path

class TLSMedicalEntityCodeMapper:
    def __init__(self, verify_cert: bool = False, ca_cert: str = None):
        # TLS server configurations (different ports)
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
        
        # TLS configuration
        self.verify_cert = verify_cert
        self.ca_cert = ca_cert
        self._setup_ssl_context()
        
        # Initialize NER pipelines with proper aggregation
        print("Loading NER models for TLS entity mapper...")
        
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
            print("Loaded biomedical NER for enhanced medication detection")
        except Exception:
            self.bio_ner = None
            print("Biomedical NER not available, using clinical NER only")
        
        # Disease-specific NER
        self.disease_ner = pipeline(
            "ner",
            model="alvaroalon2/biobert_diseases_ner", 
            aggregation_strategy="max",
            device=-1
        )
        
        # Device keywords for better device detection
        self.device_keywords = {
            'catheter', 'wheelchair', 'walker', 'cane', 'crutch',
            'syringe', 'needle', 'tube', 'pump', 'monitor',
            'prosthetic', 'orthotic', 'brace', 'splint',
            'ventilator', 'nebulizer', 'cpap', 'bipap',
            'glucose meter', 'oximeter', 'thermometer',
            'bed', 'lift', 'commode', 'shower chair',
            'dressing', 'bandage', 'gauze', 'tape'
        }
        
        print("TLS Medical Entity Code Mapper initialized")
        print("🔒 Using TLS encryption for all server connections")
    
    def _setup_ssl_context(self):
        """Setup SSL context for TLS connections"""
        if self.verify_cert and self.ca_cert:
            # Production mode - verify certificates
            self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            self.ssl_context.load_verify_locations(cafile=self.ca_cert)
        else:
            # Development mode - accept self-signed certificates
            self.ssl_context = ssl.create_default_context()
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE
    
    def _is_medical_device(self, entity_text: str) -> bool:
        """Check if entity is likely a medical device"""
        entity_lower = entity_text.lower()
        for device_keyword in self.device_keywords:
            if device_keyword in entity_lower:
                return True
        return False
    
    def query_tls_server(self, host: str, port: int, text: str) -> Optional[str]:
        """Query a TLS-secured coding server"""
        try:
            # Create socket and wrap with SSL
            with socket.create_connection((host, port), timeout=5.0) as sock:
                with self.ssl_context.wrap_socket(sock, server_hostname=host) as ssock:
                    # Send query
                    message = f"1,{text}\n"
                    ssock.send(message.encode('utf-8'))
                    
                    # Receive response
                    response = ssock.recv(4096).decode('utf-8').strip()
                    
                    # Parse response
                    if ',' in response:
                        _, code = response.split(',', 1)
                        if code and code != 'NO_MATCH' and code != 'ERROR' and code != 'TIMEOUT':
                            return code
                    return None
                    
        except ssl.SSLError as e:
            print(f"SSL Error querying {host}:{port}: {e}")
            return None
        except Exception as e:
            print(f"Error querying {host}:{port}: {e}")
            return None
    
    def categorize_entities(self, clinical_entities: List[Dict], 
                          disease_entities: List[Dict],
                          bio_entities: Optional[List[Dict]] = None) -> Dict[str, List[Dict]]:
        """
        Categorize entities into diagnosis, lab, medication, or device
        """
        categorized = defaultdict(list)
        seen = set()  # Track unique entities
        
        # First pass: Check for devices (override medication categorization)
        all_entities = []
        
        # Process clinical entities
        for ent in clinical_entities:
            if ent['entity_group'] == 'TREATMENT' and self._is_medical_device(ent['word']):
                ent['category'] = 'device'
            else:
                ent['category'] = self._map_clinical_label(ent['entity_group'])
            all_entities.append(ent)
        
        # Process disease entities
        for ent in disease_entities:
            ent['category'] = 'diagnosis'  # Disease NER is diagnosis-focused
            all_entities.append(ent)
        
        # Process biomedical entities if available
        if bio_entities:
            for ent in bio_entities:
                # Check if it's a device first
                if self._is_medical_device(ent['word']):
                    ent['category'] = 'device'
                else:
                    ent['category'] = self._map_bio_label(ent['entity_group'])
                all_entities.append(ent)
        
        # Deduplicate and categorize
        for ent in all_entities:
            key = (ent['word'].lower(), ent['category'])
            if key not in seen:
                seen.add(key)
                categorized[ent['category']].append(ent)
        
        return dict(categorized)
    
    def _map_clinical_label(self, label: str) -> str:
        """Map clinical NER labels to our categories"""
        mapping = {
            'PROBLEM': 'diagnosis',
            'TEST': 'lab',
            'TREATMENT': 'medication'  # Will be overridden for devices
        }
        return mapping.get(label, 'other')
    
    def _map_bio_label(self, label: str) -> str:
        """Map biomedical NER labels to our categories"""
        # Biomedical NER has many detailed labels
        label_lower = label.lower()
        
        if any(term in label_lower for term in ['disease', 'symptom', 'diagnosis', 'disorder']):
            return 'diagnosis'
        elif any(term in label_lower for term in ['drug', 'medication', 'chemical']):
            return 'medication'
        elif any(term in label_lower for term in ['test', 'lab', 'assay']):
            return 'lab'
        elif any(term in label_lower for term in ['device', 'equipment']):
            return 'device'
        else:
            return 'other'
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract medical entities using all available NER models"""
        all_entities = []
        
        # Clinical NER
        clinical_entities = self.clinical_ner(text)
        
        # Disease NER
        disease_entities = self.disease_ner(text)
        
        # Biomedical NER (if available)
        bio_entities = None
        if self.bio_ner:
            bio_entities = self.bio_ner(text)
        
        # Categorize entities
        categorized = self.categorize_entities(clinical_entities, disease_entities, bio_entities)
        
        # Map entities to codes
        for category, entities in categorized.items():
            if category == 'other':
                continue
                
            servers = self.servers.get(category, {})
            
            for entity in entities:
                entity_info = {
                    'entity': entity['word'],
                    'category': category,
                    'confidence': entity['score'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'codes': {}
                }
                
                # Query relevant servers for this category
                for system, (host, port) in servers.items():
                    code = self.query_tls_server(host, port, entity['word'])
                    if code:
                        entity_info['codes'][system] = code
                
                if entity_info['codes']:  # Only add if we found codes
                    all_entities.append(entity_info)
        
        return all_entities
    
    def process_text(self, text: str) -> Dict:
        """Process clinical text and return structured results"""
        start_time = time.time()
        
        entities = self.extract_entities(text)
        
        processing_time = time.time() - start_time
        
        return {
            'text': text,
            'entities': entities,
            'processing_time': processing_time,
            'entity_count': len(entities),
            'tls_secured': True
        }

def main():
    """Example usage of TLS entity mapper"""
    # Initialize mapper (development mode with self-signed certs)
    mapper = TLSMedicalEntityCodeMapper(verify_cert=False)
    
    # Example clinical texts
    test_texts = [
        "Patient with hypertension prescribed lisinopril, CBC shows elevated WBC.",
        "75-year-old male with diabetes on insulin, requires glucose meter and test strips.",
        "Post-op patient with urinary catheter, started on ciprofloxacin for UTI.",
        "COPD exacerbation, started on prednisone and albuterol nebulizer treatments."
    ]
    
    print("\n🔒 TLS-Secured Medical Entity Code Mapping")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\n📝 Input: {text}")
        result = mapper.process_text(text)
        
        print(f"⏱️  Processing time: {result['processing_time']:.3f}s")
        print(f"🔍 Found {result['entity_count']} entities:")
        
        for entity in result['entities']:
            print(f"\n  • {entity['entity']} ({entity['category']})")
            print(f"    Confidence: {entity['confidence']:.3f}")
            print(f"    Position: {entity['start']}-{entity['end']}")
            print(f"    Codes: {entity['codes']}")
    
    print("\n✅ All connections were TLS-encrypted")

if __name__ == "__main__":
    main()