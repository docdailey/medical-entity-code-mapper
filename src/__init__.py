"""
Medical Entity Code Mapper

A high-performance system for extracting medical entities from clinical text
and mapping them to standard medical ontologies (ICD-10, SNOMED CT, LOINC, RxNorm).
"""

__version__ = "1.0.0"
__author__ = "William Dailey, MD, MSEng, MSMI"
__email__ = "docdailey@gmail.com"

from .mappers.medical_entity_mapper import ImprovedMedicalEntityCodeMapper as MedicalEntityMapper

__all__ = ["MedicalEntityMapper"]