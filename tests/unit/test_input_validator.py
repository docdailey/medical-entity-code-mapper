"""
Unit tests for input validation and sanitization
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from utils.input_validator import InputValidator


class TestInputValidator(unittest.TestCase):
    """Test input validation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = InputValidator()
    
    def test_sanitize_query_valid(self):
        """Test sanitization of valid queries"""
        valid_queries = [
            "Patient with hypertension",
            "CBC and metabolic panel ordered",
            "Type 2 diabetes mellitus",
            "Urinary catheter insertion",
            "Chest pain radiating to left arm"
        ]
        
        for query in valid_queries:
            result = self.validator.sanitize_query(query)
            self.assertIsNotNone(result)
            self.assertEqual(result.strip(), query)
    
    def test_sanitize_query_sql_injection(self):
        """Test SQL injection detection"""
        sql_injections = [
            "'; DROP TABLE patients; --",
            "1' OR '1'='1",
            "admin'--",
            "UNION SELECT * FROM users",
            "'; DELETE FROM medical_records WHERE '1'='1"
        ]
        
        for injection in sql_injections:
            result = self.validator.sanitize_query(injection)
            self.assertIsNone(result, f"SQL injection not blocked: {injection}")
    
    def test_sanitize_query_xss(self):
        """Test XSS attack detection"""
        xss_attacks = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='evil.com'></iframe>",
            "<object data='malicious.swf'></object>"
        ]
        
        for xss in xss_attacks:
            result = self.validator.sanitize_query(xss)
            self.assertIsNone(result, f"XSS attack not blocked: {xss}")
    
    def test_sanitize_query_command_injection(self):
        """Test command injection detection"""
        command_injections = [
            "test; cat /etc/passwd",
            "test | rm -rf /",
            "test`whoami`",
            "test && malicious_command",
            "test || shutdown -h now"
        ]
        
        for cmd in command_injections:
            result = self.validator.sanitize_query(cmd)
            self.assertIsNone(result, f"Command injection not blocked: {cmd}")
    
    def test_sanitize_query_path_traversal(self):
        """Test path traversal detection"""
        path_traversals = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "%2e%2e%2f%2e%2e%2f",
            "....//....//etc/passwd"
        ]
        
        for path in path_traversals:
            result = self.validator.sanitize_query(path)
            self.assertIsNone(result, f"Path traversal not blocked: {path}")
    
    def test_sanitize_query_length_limit(self):
        """Test query length limits"""
        # Test query at limit
        max_query = "a" * InputValidator.MAX_QUERY_LENGTH
        result = self.validator.sanitize_query(max_query)
        self.assertIsNotNone(result)
        
        # Test query over limit
        long_query = "a" * (InputValidator.MAX_QUERY_LENGTH + 1)
        result = self.validator.sanitize_query(long_query)
        self.assertIsNone(result)
    
    def test_validate_sequence_number(self):
        """Test sequence number validation"""
        # Valid sequence numbers
        valid_seq_nums = ["1", "123", "abc123", "seq_001", "test-456"]
        for seq in valid_seq_nums:
            result = self.validator.validate_sequence_number(seq)
            self.assertEqual(result, seq)
        
        # Invalid sequence numbers
        invalid_seq_nums = ["", "seq;rm -rf", "1' OR '1'='1", "a" * 100, "seq#123"]
        for seq in invalid_seq_nums:
            result = self.validator.validate_sequence_number(seq)
            self.assertIsNone(result)
    
    def test_validate_medical_code(self):
        """Test medical code validation"""
        # Valid codes
        test_cases = [
            ("I10", "icd10", "I10"),
            ("E11.9", "icd10", "E11.9"),
            ("38341003", "snomed", "38341003"),
            ("2345-7", "loinc", "2345-7"),
            ("29046", "rxnorm", "29046"),
            ("A4338", "hcpcs", "A4338")
        ]
        
        for code, code_type, expected in test_cases:
            result = self.validator.validate_medical_code(code, code_type)
            self.assertEqual(result, expected)
        
        # Invalid codes
        invalid_cases = [
            ("ZZ99", "icd10"),  # Invalid ICD-10 format (double letter)
            ("abc", "snomed"),  # Non-numeric SNOMED
            ("12345", "loinc"),  # Missing dash in LOINC
            ("XX123", "hcpcs"),  # Invalid HCPCS format (double letter)
        ]
        
        for code, code_type in invalid_cases:
            result = self.validator.validate_medical_code(code, code_type)
            self.assertIsNone(result)
    
    def test_sanitize_tcp_message(self):
        """Test TCP message sanitization"""
        # Valid messages
        valid_messages = [
            "1,chest pain\n",
            "seq_001,hypertension\n",
            "100,type 2 diabetes mellitus\n"
        ]
        
        for msg in valid_messages:
            result = self.validator.sanitize_tcp_message(msg)
            self.assertIsNotNone(result)
            self.assertTrue(result.endswith('\n'))
        
        # Invalid messages
        invalid_messages = [
            "no_comma_here\n",
            "1,2,3,too,many,commas\n",
            "1,'; DROP TABLE patients; --\n",
            "",
            "1\n",  # Missing query part
        ]
        
        for msg in invalid_messages:
            result = self.validator.sanitize_tcp_message(msg)
            self.assertIsNone(result)
    
    def test_validate_entity_category(self):
        """Test entity category validation"""
        valid_categories = ['diagnosis', 'medication', 'lab', 'device', 'other']
        for cat in valid_categories:
            self.assertTrue(self.validator.validate_entity_category(cat))
            self.assertTrue(self.validator.validate_entity_category(cat.upper()))
        
        invalid_categories = ['invalid', 'test', '', 'diagnosis; DROP TABLE']
        for cat in invalid_categories:
            self.assertFalse(self.validator.validate_entity_category(cat))
    
    def test_sanitize_json_response(self):
        """Test JSON response sanitization"""
        # Test with various data types
        test_data = {
            'text': '<script>alert("xss")</script>',
            'number': 123,
            'float': 45.67,
            'bool': True,
            'nested': {
                'inner': '<img src=x onerror=alert(1)>'
            },
            'list': ['item1', '<script>bad</script>', 123],
            'long_string': 'a' * 2000
        }
        
        result = self.validator.sanitize_json_response(test_data)
        
        # Check sanitization
        self.assertNotIn('<script>', result['text'])
        self.assertNotIn('<img', result['nested']['inner'])
        self.assertNotIn('<script>', str(result['list']))
        self.assertLessEqual(len(result['long_string']), 1000)
        self.assertEqual(result['number'], 123)
        self.assertEqual(result['bool'], True)
    
    def test_validate_host_port(self):
        """Test host and port validation"""
        # Valid combinations
        valid_cases = [
            ('localhost', 8080),
            ('127.0.0.1', 8901),
            ('0.0.0.0', 443)
        ]
        
        for host, port in valid_cases:
            self.assertTrue(self.validator.validate_host_port(host, port))
        
        # Invalid combinations
        invalid_cases = [
            ('external.com', 8080),  # External host
            ('localhost', 0),        # Invalid port
            ('localhost', 70000),    # Port too high
            ('', 8080),             # Empty host
            ('localhost', 'abc'),   # Non-numeric port
        ]
        
        for host, port in invalid_cases:
            self.assertFalse(self.validator.validate_host_port(host, port))
    
    def test_unicode_normalization(self):
        """Test unicode normalization"""
        # Test various unicode inputs
        unicode_inputs = [
            "café",  # Latin small letter e with acute
            "naïve",  # Latin small letter i with diaeresis
            "你好",   # Chinese characters
            "🏥",    # Hospital emoji
        ]
        
        for text in unicode_inputs:
            result = self.validator.sanitize_query(text)
            self.assertIsNotNone(result)
    
    def test_null_byte_removal(self):
        """Test null byte removal"""
        text_with_null = "test\0data"
        result = self.validator.sanitize_query(text_with_null)
        self.assertIsNotNone(result)
        self.assertNotIn('\0', result)


if __name__ == '__main__':
    unittest.main()