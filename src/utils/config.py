"""
Configuration utilities for Medical Entity Code Mapper
"""

import os
from pathlib import Path
import json
from typing import Dict, List, Optional

class Config:
    """Configuration management for the application"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent  # Project root
    SRC_DIR = BASE_DIR / "src"
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = DATA_DIR / "models"
    INDICES_DIR = DATA_DIR / "indices"
    
    # Index paths
    ICD10_INDEX_DIR = INDICES_DIR / "icd10_bge_m3"
    SNOMED_INDEX_DIR = INDICES_DIR / "snomed_bge_m3"
    LOINC_INDEX_DIR = INDICES_DIR / "loinc_bge_m3"
    RXNORM_INDEX_DIR = INDICES_DIR / "rxnorm_bge_m3"
    
    # Model paths
    BGE_MODEL_DIR = MODELS_DIR / "bge_m3"
    CLINICAL_NER_DIR = MODELS_DIR / "clinical_ner"
    DISEASE_NER_DIR = MODELS_DIR / "disease_ner"
    
    # Server configuration
    DEFAULT_HOST = "0.0.0.0"
    ICD10_PORT = 8901
    SNOMED_PORT = 8902
    LOINC_PORT = 8903
    RXNORM_PORT = 8904
    
    @classmethod
    def get_index_path(cls, ontology: str) -> Path:
        """Get the path to a specific ontology index"""
        paths = {
            "icd10": cls.ICD10_INDEX_DIR,
            "snomed": cls.SNOMED_INDEX_DIR,
            "loinc": cls.LOINC_INDEX_DIR,
            "rxnorm": cls.RXNORM_INDEX_DIR
        }
        return paths.get(ontology.lower())
    
    @classmethod
    def get_faiss_index_file(cls, ontology: str) -> Path:
        """Get the FAISS index file path"""
        index_dir = cls.get_index_path(ontology)
        if index_dir:
            return index_dir / "faiss.index"
        return None
    
    @classmethod
    def get_metadata_file(cls, ontology: str) -> Path:
        """Get the metadata file path"""
        index_dir = cls.get_index_path(ontology)
        if index_dir:
            return index_dir / "metadata.pkl"
        return None
    
    @classmethod
    def check_models_available(cls) -> Dict[str, bool]:
        """Check which models are available"""
        models = {
            "bge_m3": cls.BGE_MODEL_DIR.exists(),
            "clinical_ner": cls.CLINICAL_NER_DIR.exists(),
            "disease_ner": cls.DISEASE_NER_DIR.exists()
        }
        return models
    
    @classmethod
    def get_missing_required_models(cls) -> List[str]:
        """Get list of missing required models"""
        model_status = cls.check_models_available()
        missing = []
        
        # All models are required for full functionality
        for model_name, is_available in model_status.items():
            if not is_available:
                missing.append(model_name)
        
        return missing
    
    @classmethod
    def validate_indices(cls) -> Dict[str, bool]:
        """Validate that all required indices exist"""
        indices = {}
        
        for ontology in ["icd10", "snomed", "loinc", "rxnorm"]:
            index_file = cls.get_faiss_index_file(ontology)
            metadata_file = cls.get_metadata_file(ontology)
            
            indices[ontology] = (
                index_file is not None and 
                index_file.exists() and 
                metadata_file is not None and 
                metadata_file.exists()
            )
        
        return indices
    
    @classmethod
    def get_env_var(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable with optional default"""
        return os.environ.get(key, default)
    
    @classmethod
    def get_device(cls) -> str:
        """Get the device to use for computation"""
        device = cls.get_env_var("DEVICE", "cpu").lower()
        
        # Validate device
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("CUDA requested but not available, falling back to CPU")
                return "cpu"
        elif device == "mps":
            import torch
            if not torch.backends.mps.is_available():
                print("MPS requested but not available, falling back to CPU")
                return "cpu"
        
        return device
    
    @classmethod
    def save_config(cls):
        """Save current configuration to file"""
        config = {
            "base_dir": str(cls.BASE_DIR),
            "models_dir": str(cls.MODELS_DIR),
            "indices_dir": str(cls.INDICES_DIR),
            "models_available": cls.check_models_available(),
            "indices_available": cls.validate_indices(),
            "device": cls.get_device()
        }
        
        config_file = cls.DATA_DIR / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)