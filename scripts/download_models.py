#!/usr/bin/env python
"""
Download required models for Medical Entity Code Mapper
"""
import os
import sys
from pathlib import Path
import json
import argparse

# Set environment variable before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def download_models(skip_optional=False):
    """Download all required models"""
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Import after setting path
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    
    models_dir = project_root / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download FAISS indices first
    print("📥 Downloading FAISS indices...")
    indices_dir = project_root / "data" / "indices"
    if not indices_dir.exists() or len(list(indices_dir.glob("*/faiss.index"))) < 5:
        print("Downloading from Hugging Face Hub...")
        snapshot_download(
            repo_id="docdailey/medical-entity-code-mapper-indices",
            repo_type="dataset",
            local_dir=str(indices_dir),
            local_dir_use_symlinks=False
        )
        print("✓ FAISS indices downloaded")
    else:
        print("✓ FAISS indices already exist")
    
    print("\n📥 Downloading models...")
    
    # Download clinical NER model
    print("\n1/3 Downloading clinical NER model...")
    model_name = "samrawal/bert-base-uncased_clinical-ner"
    save_path = models_dir / "clinical_ner"
    if not save_path.exists() or not list(save_path.iterdir()):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print("✓ Clinical NER downloaded")
    else:
        print("✓ Clinical NER already exists")
    
    # Download disease NER model
    print("\n2/3 Downloading disease NER model...")
    model_name = "alvaroalon2/biobert_diseases_ner"
    save_path = models_dir / "disease_ner"
    if not save_path.exists() or not list(save_path.iterdir()):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print("✓ Disease NER downloaded")
    else:
        print("✓ Disease NER already exists")
    
    # Download BGE-M3 model
    print("\n3/3 Downloading BGE-M3 model...")
    model_name = "BAAI/bge-m3"
    save_path = models_dir / "bge_m3"
    if not save_path.exists() or not list(save_path.iterdir()):
        model = SentenceTransformer(model_name)
        model.save(str(save_path))
        print("✓ BGE-M3 downloaded")
    else:
        print("✓ BGE-M3 already exists")
    
    # Save model info
    for model_key, model_name in [
        ("clinical_ner", "samrawal/bert-base-uncased_clinical-ner"),
        ("disease_ner", "alvaroalon2/biobert_diseases_ner"),
        ("bge_m3", "BAAI/bge-m3")
    ]:
        info_path = models_dir / model_key / "model_info.json"
        info = {
            "model_key": model_key,
            "model_name": model_name,
            "downloaded": True
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    print("\n✅ All models downloaded successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download models for Medical Entity Code Mapper")
    parser.add_argument("--skip-optional", action="store_true", help="Skip optional models")
    args = parser.parse_args()
    
    try:
        download_models(args.skip_optional)
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()