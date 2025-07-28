#!/usr/bin/env python3
"""
Quick start example for Medical Entity Code Mapper
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mappers.medical_entity_mapper import ImprovedMedicalEntityCodeMapper as MedicalEntityMapper

def main():
    """Run example"""
    print("🏥 Medical Entity Code Mapper - Quick Start Example")
    print("=" * 50)
    
    # Example clinical texts
    test_texts = [
        "Patient with hypertension prescribed lisinopril 10mg daily. CBC shows elevated WBC.",
        "Chest pain radiating to left arm, EKG shows ST elevation. Started on aspirin.",
        "Type 2 diabetes mellitus, HbA1c 8.5%. Started metformin 500mg twice daily.",
        "75yo male with CHF, placed foley catheter for urinary retention.",
        "Post-op day 2, walker provided for ambulation. Pain controlled with morphine."
    ]
    
    # Initialize mapper
    print("\nInitializing mapper (downloading models if needed)...")
    mapper = MedicalEntityMapper()
    
    # Process each text
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*50}")
        print(f"Example {i}: {text[:60]}...")
        print("-" * 50)
        
        # Extract entities and map to codes
        entities = mapper.process_text(text)
        
        if entities:
            for entity in entities:
                print(f"\n📍 Entity: {entity['entity_name']}")
                print(f"   Text: '{entity['extracted_text']}'")
                print(f"   Category: {entity['category']}")
                print(f"   Codes:")
                
                for ontology, code in entity['codes'].items():
                    if code and code not in ['NO_CODE', 'NO_MATCH', None]:
                        print(f"     - {ontology.upper()}: {code}")
        else:
            print("   No entities found")
    
    print("\n" + "="*50)
    print("✅ Example completed!")
    print("\nNote: Make sure TCP servers are running:")
    print("  ./scripts/start_servers.sh")

if __name__ == "__main__":
    main()