"""
Input validation and sanitization for medical entity mapper
Provides security against injection attacks and malformed inputs
"""

import re
import html
import unicodedata
from typing import Optional, Dict, Any
import logging
from urllib.parse import quote

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Comprehensive input validation and sanitization
    
    Features:
    - SQL injection prevention
    - XSS attack prevention
    - Command injection prevention
    - Path traversal prevention
    - Size limits enforcement
    - Character encoding validation
    - Medical term validation
    """
    
    # Maximum input lengths
    MAX_QUERY_LENGTH = 5000
    MAX_SEQUENCE_NUMBER_LENGTH = 20
    MAX_CODE_LENGTH = 50
    
    # Dangerous patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|\#|\/\*|\*\/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(';|\";\s*(DROP|DELETE|UPDATE|INSERT))",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<img[^>]*src[^>]*>",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",
        r"\$\(",
        r"\n|\r",
        r"&&|\|\|",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e",
        r"\.\.",
    ]
    
    # Valid medical code patterns
    VALID_CODE_PATTERNS = {
        'icd10': r'^[A-Z]\d{2}(\.\d{1,4})?$',  # Letter + 2 digits + optional decimal and up to 4 more digits
        'snomed': r'^\d{6,18}$',
        'loinc': r'^\d{1,5}-\d{1}$',
        'rxnorm': r'^\d{1,7}$',
        'hcpcs': r'^[A-Z]\d{4}$',
    }
    
    @staticmethod
    def sanitize_query(query: str) -> Optional[str]:
        """
        Sanitize a medical query text
        
        Args:
            query: Raw query text
            
        Returns:
            Sanitized query or None if invalid
        """
        if not query:
            return None
            
        # Check length
        if len(query) > InputValidator.MAX_QUERY_LENGTH:
            logger.warning(f"Query too long: {len(query)} characters")
            return None
        
        # Remove null bytes
        query = query.replace('\0', '')
        
        # Normalize unicode
        query = unicodedata.normalize('NFKC', query)
        
        # HTML escape
        query = html.escape(query)
        
        # Check for dangerous patterns
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"SQL injection pattern detected: {pattern}")
                return None
                
        for pattern in InputValidator.XSS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"XSS pattern detected: {pattern}")
                return None
                
        for pattern in InputValidator.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, query):
                logger.warning(f"Command injection pattern detected: {pattern}")
                return None
                
        for pattern in InputValidator.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Path traversal pattern detected: {pattern}")
                return None
        
        # Remove multiple spaces and trim
        query = ' '.join(query.split())
        query = query.strip()
        
        return query
    
    @staticmethod
    def validate_sequence_number(seq_num: str) -> Optional[str]:
        """
        Validate a sequence number
        
        Args:
            seq_num: Raw sequence number
            
        Returns:
            Validated sequence number or None if invalid
        """
        if not seq_num:
            return None
            
        # Check length
        if len(seq_num) > InputValidator.MAX_SEQUENCE_NUMBER_LENGTH:
            return None
            
        # Must be alphanumeric with optional dashes/underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', seq_num):
            return None
            
        return seq_num
    
    @staticmethod
    def validate_medical_code(code: str, code_type: str) -> Optional[str]:
        """
        Validate a medical code
        
        Args:
            code: Medical code to validate
            code_type: Type of code (icd10, snomed, etc.)
            
        Returns:
            Validated code or None if invalid
        """
        if not code or not code_type:
            return None
            
        # Check length
        if len(code) > InputValidator.MAX_CODE_LENGTH:
            return None
            
        # Check against known patterns
        if code_type.lower() in InputValidator.VALID_CODE_PATTERNS:
            pattern = InputValidator.VALID_CODE_PATTERNS[code_type.lower()]
            if not re.match(pattern, code.upper()):
                logger.warning(f"Invalid {code_type} code format: {code}")
                return None
                
        return code.upper()
    
    @staticmethod
    def sanitize_tcp_message(message: str) -> Optional[str]:
        """
        Sanitize a complete TCP message
        
        Args:
            message: Raw TCP message
            
        Returns:
            Sanitized message or None if invalid
        """
        if not message:
            return None
            
        # Remove trailing whitespace but keep newline
        message = message.rstrip()
        
        # Must end with newline
        if not message.endswith('\n'):
            message += '\n'
            
        # Check for exactly one comma
        if message.count(',') != 1:
            logger.warning(f"Invalid message format: wrong number of commas")
            return None
            
        # Split and validate parts
        parts = message.strip().split(',', 1)
        if len(parts) != 2:
            return None
            
        seq_num = InputValidator.validate_sequence_number(parts[0])
        if not seq_num:
            return None
            
        query = InputValidator.sanitize_query(parts[1])
        if not query:
            return None
            
        return f"{seq_num},{query}\n"
    
    @staticmethod
    def validate_entity_category(category: str) -> bool:
        """
        Validate entity category
        
        Args:
            category: Entity category
            
        Returns:
            True if valid, False otherwise
        """
        valid_categories = {
            'diagnosis', 'medication', 'lab', 'device', 'other'
        }
        return category.lower() in valid_categories
    
    @staticmethod
    def sanitize_json_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize JSON response data
        
        Args:
            data: Response data dictionary
            
        Returns:
            Sanitized data
        """
        if not isinstance(data, dict):
            return {}
            
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize keys
            if not isinstance(key, str) or len(key) > 100:
                continue
                
            # Sanitize values based on type
            if isinstance(value, str):
                # HTML escape strings
                sanitized[key] = html.escape(value[:1000])  # Limit length
            elif isinstance(value, (int, float)):
                sanitized[key] = value
            elif isinstance(value, bool):
                sanitized[key] = value
            elif isinstance(value, dict):
                # Recursive sanitization
                sanitized[key] = InputValidator.sanitize_json_response(value)
            elif isinstance(value, list):
                # Sanitize list items
                sanitized[key] = [
                    InputValidator.sanitize_json_response(item) if isinstance(item, dict)
                    else html.escape(str(item)[:100]) if isinstance(item, str)
                    else item
                    for item in value[:100]  # Limit list size
                ]
            else:
                # Convert other types to string and escape
                sanitized[key] = html.escape(str(value)[:100])
                
        return sanitized
    
    @staticmethod
    def validate_host_port(host: str, port: int) -> bool:
        """
        Validate host and port for server connections
        
        Args:
            host: Hostname or IP address
            port: Port number
            
        Returns:
            True if valid, False otherwise
        """
        # Validate host
        if not host or len(host) > 255:
            return False
            
        # Check for localhost/private IPs only (security)
        allowed_hosts = {'localhost', '127.0.0.1', '0.0.0.0'}
        if host not in allowed_hosts:
            # Could add regex for private IP ranges if needed
            logger.warning(f"Non-local host attempted: {host}")
            return False
            
        # Validate port
        if not isinstance(port, int) or port < 1 or port > 65535:
            return False
            
        # Check for privileged ports
        if port < 1024:
            logger.warning(f"Privileged port attempted: {port}")
            
        return True


def get_input_validator() -> InputValidator:
    """Get input validator instance"""
    return InputValidator()