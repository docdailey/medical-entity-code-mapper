#!/usr/bin/env python3
"""
Quick start example for Medical Entity Code Mapper with UV
Run with: uv run python examples/quickstart_uv.py
"""

import socket
import sys

def test_server(name, port, query):
    """Test a single server"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('localhost', port))
        sock.send(f"{query}\n".encode())
        response = sock.recv(1024).decode().strip()
        sock.close()
        return response
    except Exception as e:
        return f"Error: {e}"

def main():
    """Run quick start tests"""
    print("🏥 Medical Entity Code Mapper - Quick Start")
    print("=" * 50)
    print("\n⚠️  Make sure servers are running: uv run uv_runner.py")
    print()
    
    # Test cases
    tests = [
        ("ICD-10", 8901, [
            "1,chest pain",
            "2,hypertension",
            "3,diabetes mellitus type 2",
            "4,acute myocardial infarction"
        ]),
        ("SNOMED", 8902, [
            "1,hypertension",
            "2,diabetes",
            "3,pneumonia",
            "4,covid-19"
        ]),
        ("LOINC", 8903, [
            "1,blood glucose",
            "2,hemoglobin a1c",
            "3,white blood cell count",
            "4,troponin I"
        ]),
        ("RxNorm", 8904, [
            "1,aspirin",
            "2,lisinopril",
            "3,metformin",
            "4,azithromycin"
        ])
    ]
    
    # Run tests
    for name, port, queries in tests:
        print(f"\n{name} Server (Port {port})")
        print("-" * 30)
        
        for query in queries:
            response = test_server(name, port, query)
            seq, desc = query.split(',', 1)
            
            if response.startswith(seq + ","):
                code = response.split(',', 1)[1]
                print(f"✅ {desc} → {code}")
            else:
                print(f"❌ {desc} → {response}")

    # Clinical note example
    print(f"\n{'='*50}")
    print("Full Clinical Note Processing")
    print("=" * 50)
    print("\nTo process a full clinical note with entity extraction:")
    print("uv run python src/mappers/medical_entity_mapper.py")
    
    print("\nExample clinical note:")
    print("""
    Chief Complaint: Chest pain and shortness of breath
    
    HPI: 68-year-old male with acute onset chest pain radiating to left arm. 
    History of hypertension and diabetes mellitus type 2.
    Currently taking metformin 1000mg twice daily and lisinopril 10mg daily.
    
    Assessment: Acute coronary syndrome, uncontrolled diabetes
    """)

if __name__ == "__main__":
    main()