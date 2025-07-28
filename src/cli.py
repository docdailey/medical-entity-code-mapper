#!/usr/bin/env python3
"""
Command-line interface for Medical Entity Code Mapper
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mappers.medical_entity_mapper import MedicalEntityMapper

def extract(args):
    """Extract entities from text or file"""
    # Initialize mapper
    mapper = MedicalEntityMapper()
    
    # Get text to process
    if args.file:
        with open(args.file, 'r') as f:
            text = f.read()
    else:
        text = args.text
    
    # Process text
    entities = mapper.process_text(text)
    
    # Output results
    if args.format == 'json':
        print(json.dumps(entities, indent=2))
    else:  # human-readable
        if entities:
            for i, entity in enumerate(entities, 1):
                print(f"\n{i}. {entity['entity_name']}")
                print(f"   Text: '{entity['extracted_text']}'")
                print(f"   Category: {entity['category']}")
                print(f"   Codes:")
                
                for ontology, codes in entity['codes'].items():
                    if codes and codes[0]['code'] != 'NO_CODE':
                        code = codes[0]
                        print(f"     - {ontology.upper()}: {code['code']} - {code['description']}")
        else:
            print("No entities found")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Medical Entity Code Mapper - Extract and code medical entities"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-t", "--text", help="Text to process")
    input_group.add_argument("-f", "--file", help="File to process")
    
    # Output options
    parser.add_argument(
        "-o", "--format", 
        choices=["json", "human"], 
        default="human",
        help="Output format (default: human)"
    )
    
    args = parser.parse_args()
    extract(args)

if __name__ == "__main__":
    main()