#!/usr/bin/env python3
"""
TLS-Secured Web Interface for Medical Entity Code Mapper
Provides HTTPS access to the entity extraction and coding system
"""

from flask import Flask, request, jsonify, render_template_string
import ssl
import os
from pathlib import Path
from medical_entity_mapper_tls import TLSMedicalEntityCodeMapper
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.input_validator import get_input_validator
from utils.audit_logger import get_audit_logger

app = Flask(__name__)

# Initialize TLS entity mapper
mapper = TLSMedicalEntityCodeMapper(verify_cert=False)

# Initialize validators and loggers
validator = get_input_validator()
audit_logger = get_audit_logger()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Medical Entity Code Mapper - TLS Secured</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .security-badge {
            background-color: #27ae60;
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            display: inline-block;
            margin-left: 10px;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            min-height: 150px;
        }
        button {
            background-color: #3498db;
            color: white;
            padding: 10px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .results {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .entity {
            background-color: #ecf0f1;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .entity-header {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .codes {
            margin-top: 10px;
        }
        .code-item {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            margin: 2px;
            font-size: 14px;
        }
        .category {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 14px;
            margin-left: 10px;
        }
        .category-diagnosis { background-color: #e74c3c; color: white; }
        .category-medication { background-color: #f39c12; color: white; }
        .category-lab { background-color: #9b59b6; color: white; }
        .category-device { background-color: #1abc9c; color: white; }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .error {
            background-color: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            display: none;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Medical Entity Code Mapper <span class="security-badge">🔒 TLS Secured</span></h1>
        <p>Extract medical entities and map to ICD-10, SNOMED CT, LOINC, RxNorm, and HCPCS codes</p>
    </div>
    
    <div>
        <h3>Enter Clinical Text:</h3>
        <textarea id="clinical-text" placeholder="Example: Patient with hypertension prescribed lisinopril. CBC shows elevated WBC. Requires walker for ambulation."></textarea>
        <button onclick="processText()">Extract Entities & Map Codes</button>
    </div>
    
    <div class="loading" id="loading">
        <p>🔄 Processing text securely over TLS...</p>
    </div>
    
    <div class="error" id="error"></div>
    
    <div class="results" id="results" style="display: none;">
        <h3>Extracted Entities:</h3>
        <div id="entities-list"></div>
        <div style="margin-top: 20px; color: #7f8c8d;">
            <small>Processing time: <span id="processing-time"></span> seconds</small><br>
            <small>🔒 All server connections are TLS-encrypted</small>
        </div>
    </div>
    
    <div class="footer">
        <p>© 2024 Medical Entity Code Mapper | TLS Version | All data encrypted in transit</p>
    </div>
    
    <script>
        function processText() {
            const text = document.getElementById('clinical-text').value;
            if (!text.trim()) {
                showError('Please enter some clinical text to process.');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            
            fetch('/api/extract', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                displayResults(data);
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                showError('Error processing text: ' + error.message);
            });
        }
        
        function displayResults(data) {
            document.getElementById('results').style.display = 'block';
            document.getElementById('processing-time').textContent = data.processing_time.toFixed(3);
            
            const entitiesList = document.getElementById('entities-list');
            entitiesList.innerHTML = '';
            
            if (data.entities.length === 0) {
                entitiesList.innerHTML = '<p>No medical entities found in the text.</p>';
                return;
            }
            
            data.entities.forEach(entity => {
                const entityDiv = document.createElement('div');
                entityDiv.className = 'entity';
                
                const categoryClass = 'category-' + entity.category;
                
                let html = `
                    <div class="entity-header">
                        ${entity.entity}
                        <span class="category ${categoryClass}">${entity.category}</span>
                    </div>
                    <div>Confidence: ${(entity.confidence * 100).toFixed(1)}%</div>
                    <div>Position: ${entity.start}-${entity.end}</div>
                `;
                
                if (Object.keys(entity.codes).length > 0) {
                    html += '<div class="codes">Codes: ';
                    for (const [system, code] of Object.entries(entity.codes)) {
                        html += `<span class="code-item">${system.toUpperCase()}: ${code}</span>`;
                    }
                    html += '</div>';
                } else {
                    html += '<div class="codes"><em>No codes found</em></div>';
                }
                
                entityDiv.innerHTML = html;
                entitiesList.appendChild(entityDiv);
            });
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = '❌ ' + message;
            errorDiv.style.display = 'block';
        }
        
        // Enable Enter key to submit
        document.getElementById('clinical-text').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                processText();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/extract', methods=['POST'])
def extract_entities():
    """API endpoint for entity extraction"""
    client_ip = request.remote_addr
    
    try:
        # Log API access
        audit_logger.log_access(
            user_id=None,
            resource="web_api",
            action="EXTRACT_ENTITIES",
            result="ATTEMPT",
            client_ip=client_ip
        )
        
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            audit_logger.log_access(
                user_id=None,
                resource="web_api",
                action="EXTRACT_ENTITIES",
                result="FAILED",
                client_ip=client_ip
            )
            return jsonify({'error': 'No text provided'}), 400
        
        # Validate and sanitize input
        sanitized_text = validator.sanitize_query(text)
        if not sanitized_text:
            # Log potential security event
            audit_logger.log_security_event(
                event_type="MALICIOUS_INPUT_BLOCKED",
                severity="WARNING",
                details={"input_length": len(text), "endpoint": "/api/extract"},
                client_ip=client_ip
            )
            return jsonify({'error': 'Invalid input detected'}), 400
        
        # Process text using TLS entity mapper
        result = mapper.process_text(sanitized_text)
        
        # Sanitize response
        sanitized_result = validator.sanitize_json_response({
            'entities': result,
            'status': 'success'
        })
        
        # Log successful extraction
        audit_logger.log_query(
            query_type="WEB_API",
            query_text=sanitized_text[:100],  # Log only first 100 chars
            server="web_interface",
            response_code="SUCCESS",
            latency_ms=0,  # Could add timing if needed
            client_ip=client_ip
        )
        
        return jsonify(sanitized_result)
    
    except Exception as e:
        # Log error
        audit_logger.log_error(
            error_type="API_ERROR",
            error_message=str(e),
            context={"endpoint": "/api/extract", "client_ip": client_ip}
        )
        return jsonify({'error': 'Internal server error'}), 500

def create_ssl_context():
    """Create SSL context for HTTPS"""
    cert_file = os.getenv('TLS_CERT_FILE', 'certs/server.crt')
    key_file = os.getenv('TLS_KEY_FILE', 'certs/server.key')
    
    # Check if certificates exist
    if not Path(cert_file).exists() or not Path(key_file).exists():
        print("⚠️  TLS certificates not found!")
        print("Run: python scripts/generate_certificates.py --type self-signed")
        return None
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    
    # Use strong ciphers
    context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
    
    return context

def main():
    """Run the HTTPS web interface"""
    # Create SSL context
    ssl_context = create_ssl_context()
    if not ssl_context:
        return
    
    print("🔒 Starting TLS-Secured Medical Entity Code Mapper Web Interface")
    print("=" * 60)
    print("Access the interface at: https://localhost:5443")
    print("Note: You may see a certificate warning with self-signed certificates")
    print("=" * 60)
    
    # Add Flask configuration
    app.config['ENV'] = 'production'
    
    # Run with HTTPS
    app.run(
        host='127.0.0.1',  # Loopback only for security
        port=5443,
        ssl_context=ssl_context,
        debug=False
    )

if __name__ == '__main__':
    main()