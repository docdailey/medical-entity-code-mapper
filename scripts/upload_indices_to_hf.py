#!/usr/bin/env python3
"""
Upload FAISS indices to Hugging Face Hub
"""
import os
from pathlib import Path
from huggingface_hub import HfApi
from tqdm import tqdm

def main():
    print("🚀 Uploading FAISS indices to Hugging Face Hub...")
    
    api = HfApi()
    repo_id = "docdailey/medical-entity-code-mapper-indices"
    indices_dir = Path(__file__).parent.parent / "data" / "indices"
    
    # Create dataset card
    dataset_card = """---
license: mit
tags:
- medical
- healthcare
- faiss
- embeddings
- icd10
- snomed
- loinc
- rxnorm
size_categories:
- 1M<n<10M
---

# Medical Entity Code Mapper - FAISS Indices

This dataset contains pre-built FAISS indices for medical entity code mapping using BGE-M3 embeddings.

## Contents

- **ICD-10-CM**: 559,940 diagnostic codes with UMLS synonyms
- **SNOMED CT**: 350,000+ clinical concepts  
- **LOINC**: 104,000+ laboratory codes
- **RxNorm**: 270,000+ medication concepts

## Usage

These indices are used by the [Medical Entity Code Mapper](https://github.com/docdailey/medical-entity-code-mapper) project.

The indices will be automatically downloaded when you run:
```bash
uv run uv_runner.py
```

Or manually with:
```bash
uv run python scripts/download_models.py
```

## Index Details

Each index contains:
- `faiss.index`: HNSW index for fast similarity search
- `metadata.pkl`: Associated medical codes and descriptions
- `*_embeddings.npy`: Original BGE-M3 embeddings (for some indices)

## License

MIT License with Commercial Restriction - see the main project repository for details.

## Citation

If you use these indices, please cite:
```
William Dailey, MD, MSEng, MSMI (2024). 
Medical Entity Code Mapper FAISS Indices. 
Hugging Face Datasets. https://huggingface.co/datasets/docdailey/medical-entity-code-mapper-indices
```
"""
    
    # Upload dataset card
    print("📝 Creating dataset card...")
    api.upload_file(
        path_or_fileobj=dataset_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    # Upload indices directory
    print(f"\n📦 Uploading indices from {indices_dir}...")
    
    # Get total size for progress
    total_size = 0
    for root, dirs, files in os.walk(indices_dir):
        for file in files:
            if not file.startswith('.'):
                total_size += os.path.getsize(os.path.join(root, file))
    
    print(f"Total size: {total_size / 1024**3:.1f} GB")
    
    # Upload with progress
    api.upload_folder(
        folder_path=str(indices_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload FAISS indices for medical entity code mapping"
    )
    
    print(f"\n✅ Successfully uploaded indices to: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()