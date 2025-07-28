"""
Unit tests for Medical Entity Code Mapper
Tests entity extraction, categorization, and code mapping
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from mappers.medical_entity_mapper import ImprovedMedicalEntityCodeMapper


class TestMedicalEntityMapper(unittest.TestCase):
    """Test cases for the medical entity mapper"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the NER pipelines to avoid loading models during tests
        with patch('mappers.medical_entity_mapper.pipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            self.mapper = ImprovedMedicalEntityCodeMapper()
    
    def test_device_detection(self):
        """Test medical device keyword detection"""
        # Test positive cases
        device_terms = [
            "urinary catheter",
            "wheelchair",
            "walker",
            "CPAP machine",
            "glucose meter",
            "insulin syringe"
        ]
        
        for term in device_terms:
            with self.subTest(term=term):
                self.assertTrue(
                    self.mapper._is_medical_device(term),
                    f"Failed to detect '{term}' as medical device"
                )
        
        # Test negative cases
        non_device_terms = [
            "aspirin",
            "lisinopril",
            "blood test",
            "hypertension"
        ]
        
        for term in non_device_terms:
            with self.subTest(term=term):
                self.assertFalse(
                    self.mapper._is_medical_device(term),
                    f"Incorrectly detected '{term}' as medical device"
                )
    
    def test_clinical_label_mapping(self):
        """Test mapping of clinical NER labels to categories"""
        mappings = {
            'PROBLEM': 'diagnosis',
            'TEST': 'lab',
            'TREATMENT': 'medication'
        }
        
        for label, expected_category in mappings.items():
            result = self.mapper._map_clinical_label(label)
            self.assertEqual(
                result, expected_category,
                f"Label '{label}' mapped to '{result}', expected '{expected_category}'"
            )
        
        # Test unknown label
        self.assertEqual(self.mapper._map_clinical_label('UNKNOWN'), 'other')
    
    def test_bio_label_mapping(self):
        """Test mapping of biomedical NER labels to categories"""
        test_cases = [
            ('disease', 'diagnosis'),
            ('symptom', 'diagnosis'),
            ('drug', 'medication'),
            ('medication', 'medication'),
            ('test', 'lab'),
            ('lab_test', 'lab'),
            ('device', 'device'),
            ('unknown_label', 'other')
        ]
        
        for label, expected_category in test_cases:
            result = self.mapper._map_bio_label(label)
            self.assertEqual(
                result, expected_category,
                f"Bio label '{label}' mapped to '{result}', expected '{expected_category}'"
            )
    
    @patch('socket.socket')
    def test_query_server_success(self, mock_socket):
        """Test successful server query"""
        # Mock socket behavior
        mock_conn = MagicMock()
        mock_conn.recv.return_value = b"1,R07.9\n"
        mock_socket.return_value.__enter__.return_value = mock_conn
        
        result = self.mapper.query_server('localhost', 8901, 'chest pain')
        self.assertEqual(result, 'R07.9')
    
    @patch('socket.socket')
    def test_query_server_no_match(self, mock_socket):
        """Test server query with no match"""
        # Mock socket behavior
        mock_conn = MagicMock()
        mock_conn.recv.return_value = b"1,NO_MATCH\n"
        mock_socket.return_value.__enter__.return_value = mock_conn
        
        result = self.mapper.query_server('localhost', 8901, 'unknown condition')
        self.assertIsNone(result)
    
    @patch('socket.socket')
    def test_query_server_timeout(self, mock_socket):
        """Test server query timeout handling"""
        # Mock socket timeout
        mock_socket.side_effect = TimeoutError()
        
        result = self.mapper.query_server('localhost', 8901, 'test query')
        self.assertIsNone(result)
    
    def test_categorize_entities(self):
        """Test entity categorization logic"""
        # Mock entity data
        clinical_entities = [
            {'entity_group': 'PROBLEM', 'word': 'hypertension', 'score': 0.99, 'start': 0, 'end': 12},
            {'entity_group': 'TREATMENT', 'word': 'lisinopril', 'score': 0.98, 'start': 20, 'end': 30},
            {'entity_group': 'TREATMENT', 'word': 'urinary catheter', 'score': 0.97, 'start': 40, 'end': 56}
        ]
        
        disease_entities = [
            {'entity_group': 'DISEASE', 'word': 'diabetes', 'score': 0.96, 'start': 60, 'end': 68}
        ]
        
        result = self.mapper.categorize_entities(clinical_entities, disease_entities)
        
        # Check categorization
        self.assertIn('diagnosis', result)
        self.assertIn('medication', result)
        self.assertIn('device', result)
        
        # Check device override for catheter
        device_entities = [e for e in result['device'] if e['word'] == 'urinary catheter']
        self.assertEqual(len(device_entities), 1, "Catheter should be categorized as device")
        
        # Check medication
        med_entities = [e for e in result['medication'] if e['word'] == 'lisinopril']
        self.assertEqual(len(med_entities), 1, "Lisinopril should be categorized as medication")
    
    def test_deduplicate_entities(self):
        """Test that duplicate entities are removed"""
        clinical_entities = [
            {'entity_group': 'PROBLEM', 'word': 'Hypertension', 'score': 0.99, 'start': 0, 'end': 12},
            {'entity_group': 'PROBLEM', 'word': 'hypertension', 'score': 0.98, 'start': 50, 'end': 62}
        ]
        
        disease_entities = []
        
        result = self.mapper.categorize_entities(clinical_entities, disease_entities)
        
        # Should only have one hypertension entity (case-insensitive deduplication)
        self.assertEqual(len(result['diagnosis']), 1)


class TestEntityMapperIntegration(unittest.TestCase):
    """Integration tests for end-to-end functionality"""
    
    @patch('mappers.medical_entity_mapper.pipeline')
    @patch('socket.socket')
    def test_process_clinical_text(self, mock_socket, mock_pipeline):
        """Test full text processing pipeline"""
        # Mock NER results
        mock_clinical_ner = MagicMock()
        mock_clinical_ner.return_value = [
            {'entity_group': 'PROBLEM', 'word': 'chest pain', 'score': 0.99, 'start': 0, 'end': 10}
        ]
        
        mock_disease_ner = MagicMock()
        mock_disease_ner.return_value = []
        
        # Configure pipeline mocks
        mock_pipeline.side_effect = [mock_clinical_ner, None, mock_disease_ner]
        
        # Mock socket responses
        mock_conn = MagicMock()
        mock_conn.recv.return_value = b"1,R07.9\n"
        mock_socket.return_value.__enter__.return_value = mock_conn
        
        # Create mapper and process text
        mapper = ImprovedMedicalEntityCodeMapper()
        result = mapper.process_text("Patient with chest pain")
        
        # Verify results
        self.assertIn('entities', result)
        self.assertIn('processing_time', result)
        self.assertGreater(len(result['entities']), 0)
        
        # Check entity details
        entity = result['entities'][0]
        self.assertEqual(entity['entity'], 'chest pain')
        self.assertEqual(entity['category'], 'diagnosis')
        self.assertIn('codes', entity)


class TestInputValidation(unittest.TestCase):
    """Test input validation and sanitization"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('mappers.medical_entity_mapper.pipeline'):
            self.mapper = ImprovedMedicalEntityCodeMapper()
    
    def test_empty_input(self):
        """Test handling of empty input"""
        result = self.mapper.process_text("")
        self.assertEqual(len(result['entities']), 0)
    
    def test_none_input(self):
        """Test handling of None input"""
        with self.assertRaises(AttributeError):
            self.mapper.process_text(None)
    
    def test_very_long_input(self):
        """Test handling of very long input"""
        # Create a very long text
        long_text = "Patient with hypertension. " * 1000
        
        # Should handle without error
        result = self.mapper.process_text(long_text)
        self.assertIn('entities', result)
    
    def test_special_characters(self):
        """Test handling of special characters"""
        special_text = "Patient with <script>alert('xss')</script> and SQL'; DROP TABLE patients;--"
        
        # Should process without executing any code
        result = self.mapper.process_text(special_text)
        self.assertIn('entities', result)
    
    def test_unicode_input(self):
        """Test handling of unicode characters"""
        unicode_text = "Patient with 高血压 (hypertension) and café-au-lait spots"
        
        # Should handle unicode without error
        result = self.mapper.process_text(unicode_text)
        self.assertIn('entities', result)


if __name__ == '__main__':
    unittest.main()