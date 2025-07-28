#!/usr/bin/env python3
"""
Improved Medical Entity Code Mapper
Fixes tokenization issues and improves entity extraction
"""

import json
import socket
import time
from typing import List, Dict, Optional, Tuple
from transformers import pipeline
from collections import defaultdict

class ImprovedMedicalEntityCodeMapper:
    def __init__(self):
        # Server configurations
        self.servers = {
            'diagnosis': {
                'icd10': ('localhost', 8901),
                'snomed': ('localhost', 8902)
            },
            'lab': {
                'loinc': ('localhost', 8903)
            },
            'medication': {
                'rxnorm': ('localhost', 8904)
            },
            'device': {
                'hcpcs': ('localhost', 8905)
            }
        }
        
        # Initialize NER pipelines with proper aggregation
        print("Loading NER models with improved configuration...")
        
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
            print("Loaded biomedical NER for enhanced medication detection")
        except:
            self.bio_ner = None
            self.use_bio_ner = False
            print("Biomedical NER not available, using clinical NER only")
        
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
        """Query a TCP server for code lookup"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((host, port))
            
            message = f"{seq_num},{query}\n"
            sock.send(message.encode('utf-8'))
            
            response = sock.recv(4096).decode('utf-8').strip()
            sock.close()
            
            if response:
                parts = response.split(',')
                if len(parts) >= 2:
                    code = parts[1].strip()
                    if code and code not in ['NO_MATCH', 'NO_CODE', 'ERROR', 'TIMEOUT']:
                        return code
                        
        except Exception as e:
            print(f"Error querying {host}:{port}: {e}")
            
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
        # Extract entities
        entities = self._extract_entities(text)
        
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

def main():
    """Test the improved mapper"""
    mapper = ImprovedMedicalEntityCodeMapper()
    
    # Test cases focusing on tokenization issues
    test_cases = [
        "Patient on Donepezil 10 mg daily, Tamsulosin 0.4 mg daily, and Cefepime for sepsis.",
        "75-year-old male with hypertension and sepsis secondary to urinary catheter.",
        "Child with hypertension, prescribed lisinopril."
    ]
    
    all_results = []
    
    for i, text in enumerate(test_cases):
        print(f"\nTest case {i+1}: {text[:50]}...")
        entities = mapper.process_text(text)
        print(f"Found {len(entities)} entities")
        
        for entity in entities:
            codes_str = ', '.join([f"{k}: {v}" for k, v in entity['codes'].items() if v])
            print(f"  - {entity['entity_name']} ({entity['category']}) [{codes_str}] via {entity['extraction_model']}")
        
        all_results.extend(entities)
    
    # Save results
    with open('improved_mapper_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to improved_mapper_test_results.json")

if __name__ == "__main__":
    main()