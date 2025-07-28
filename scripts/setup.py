#!/usr/bin/env python3
"""
Complete setup script for Medical Entity Code Mapper

This script:
1. Downloads required models
2. Validates FAISS indices
3. Sets up configuration
4. Runs validation tests
"""

import sys
import os
from pathlib import Path
import subprocess
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"🏥 {title}")
    print("=" * 60)

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    print("✅ Python version OK")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = [
        "torch",
        "transformers",
        "sentence-transformers",
        "faiss-cpu",  # or faiss-gpu
        "numpy",
        "tqdm"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def download_models():
    """Download required models"""
    print_header("Downloading Models")
    
    # Check which models are already available
    model_status = Config.check_models_available()
    missing_required = Config.get_missing_required_models()
    
    if not missing_required:
        print("✅ All required models already downloaded!")
        return True
    
    print(f"Need to download: {', '.join(missing_required)}")
    
    # Run download script
    download_script = Path(__file__).parent / "download_models.py"
    result = subprocess.run([sys.executable, str(download_script)], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Model download failed!")
        print(result.stderr)
        return False
    
    print("✅ Models downloaded successfully!")
    return True

def validate_indices():
    """Validate FAISS indices"""
    print_header("Validating FAISS Indices")
    
    index_status = Config.validate_indices()
    
    all_valid = True
    for ontology, is_valid in index_status.items():
        if is_valid:
            print(f"✅ {ontology.upper()} index found")
        else:
            print(f"❌ {ontology.upper()} index missing")
            all_valid = False
    
    if not all_valid:
        print("\n⚠️  Some indices are missing!")
        print("   Indices should be placed in: data/indices/")
        print("   Expected structure:")
        print("   - data/indices/icd10_bge_m3/faiss.index")
        print("   - data/indices/snomed_bge_m3/faiss.index")
        print("   - data/indices/loinc_bge_m3/faiss.index")
        print("   - data/indices/rxnorm_bge_m3/faiss.index")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directory Structure")
    
    directories = [
        Config.DATA_DIR,
        Config.MODELS_DIR,
        Config.INDICES_DIR,
        Config.DATA_DIR / "logs",
        Config.DATA_DIR / "cache"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")
    
    return True

def run_validation_test():
    """Run a simple validation test"""
    print_header("Running Validation Test")
    
    try:
        # Try to import and create mapper
        from src.mappers.medical_entity_mapper import MedicalEntityMapper
        
        print("Creating mapper instance...")
        mapper = MedicalEntityMapper()
        
        print("Testing entity extraction...")
        test_text = "Patient with diabetes on metformin"
        entities = mapper.process_text(test_text)
        
        if entities:
            print(f"✅ Found {len(entities)} entities!")
            for entity in entities:
                print(f"   - {entity['entity_name']} ({entity['category']})")
        else:
            print("⚠️  No entities found in test")
            
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def create_example_env():
    """Create example .env file"""
    env_example = Config.BASE_DIR / ".env.example"
    
    if not env_example.exists():
        content = """# Device configuration
DEVICE=cpu  # Options: cpu, cuda, mps

# Server configuration
SERVER_HOST=0.0.0.0
ICD10_PORT=8901
SNOMED_PORT=8902
LOINC_PORT=8903
RXNORM_PORT=8904

# Performance settings
BATCH_SIZE=32
MAX_WORKERS=3
TIMEOUT=5.0
"""
        with open(env_example, 'w') as f:
            f.write(content)
        print(f"✅ Created {env_example}")

def main():
    """Main setup function"""
    print("\n🏥 Medical Entity Code Mapper - Setup Script")
    print("=" * 60)
    
    # Run setup steps
    steps = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", create_directories),
        ("Models", download_models),
        ("Indices", validate_indices)
    ]
    
    all_passed = True
    for step_name, step_func in steps:
        if not step_func():
            all_passed = False
            print(f"\n❌ Setup failed at: {step_name}")
            break
    
    if all_passed:
        # Create example files
        create_example_env()
        
        # Save configuration
        Config.save_config()
        
        # Optional validation test
        print("\nRun validation test? (y/n): ", end="")
        if input().lower() == 'y':
            run_validation_test()
        
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start servers: ./scripts/start_servers.sh")
        print("2. Run examples: python examples/quickstart.py")
        print("3. Read docs: docs/ARCHITECTURE.md")
    else:
        print("\n❌ Setup incomplete. Please fix issues and run again.")
        sys.exit(1)

if __name__ == "__main__":
    main()