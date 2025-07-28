# Medical Entity Code Mapper

[![License: Custom](https://img.shields.io/badge/License-Custom-yellow.svg)](https://github.com/docdailey/medical-entity-code-mapper/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![GitHub stars](https://img.shields.io/github/stars/docdailey/medical-entity-code-mapper)
![GitHub forks](https://img.shields.io/github/forks/docdailey/medical-entity-code-mapper)
![GitHub issues](https://img.shields.io/github/issues/docdailey/medical-entity-code-mapper)
![GitHub last commit](https://img.shields.io/github/last-commit/docdailey/medical-entity-code-mapper)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=docdailey.medical-entity-code-mapper)

A high-performance medical entity extraction and coding system that identifies clinical entities in text and maps them to standard medical ontologies (ICD-10, SNOMED CT, LOINC, RxNorm).

## 💻 System Requirements

- **Operating System**: macOS (Apple Silicon/Intel), Linux
- **Memory**: 32GB RAM recommended (16GB minimum)
- **Storage**: ~10GB free space for models and indices
- **Python**: 3.10 - 3.13
- **Package Manager**: [UV](https://github.com/astral-sh/uv) (Rust-based Python package manager)

*Tested on Mac Studio M2 Max with 32GB RAM*

## 🎯 Purpose

This system addresses the critical need for automated medical coding by:
- Extracting medical entities from clinical text using state-of-the-art NER models
- Mapping entities to multiple medical coding systems simultaneously
- Providing context-aware code selection (e.g., adult vs pediatric codes)
- Offering high-performance TCP socket servers for production deployment
- **NEW**: TLS encryption for secure handling of sensitive healthcare data

## 🚀 Features

- **Multi-Ontology Support**: ICD-10-CM, SNOMED CT, LOINC, RxNorm, HCPCS
- **Advanced NER**: Multiple BERT-based models for comprehensive entity extraction
- **Context-Aware Coding**: Intelligent code selection based on clinical context
- **High Performance**: Vector similarity search using FAISS with BGE-M3 embeddings
- **Production Ready**: TCP socket servers with dynamic batching and MPS acceleration
- **Comprehensive Coverage**: Enhanced with UMLS Metathesaurus synonyms
- **Automatic Model Management**: Models downloaded automatically on first run
- **Medical Device Recognition**: HCPCS codes for devices, supplies, and equipment

## 📦 Data Storage

The system automatically downloads:
- **NER Models** (~2GB): Clinical entity recognition models
- **BGE-M3 Model** (~2GB): Embedding model for similarity search  
- **FAISS Indices** (~9GB): Pre-built indices hosted on [Hugging Face](https://huggingface.co/datasets/docdailey/medical-entity-code-mapper-indices)

All data is downloaded on first run and cached locally.

## 🏥 Supported Medical Ontologies

1. **ICD-10-CM** (Port 8901)
   - 559,940 diagnostic codes with UMLS synonyms
   - Context-aware selection (adult vs pediatric)
   
2. **SNOMED CT** (Port 8902)
   - 350,000+ clinical concepts
   - Comprehensive medical terminology
   
3. **LOINC** (Port 8903)
   - 104,000+ laboratory and clinical codes
   - Test results and observations
   
4. **RxNorm** (Port 8904)
   - 270,000+ medication concepts
   - Drug names, ingredients, and formulations
   
5. **HCPCS** (Port 8905)
   - 8,725 codes for medical devices and supplies
   - Durable medical equipment (DME)
   - Distinguishes devices from medications
   
### Medical Device Recognition

The system automatically detects medical devices in clinical text and routes them to HCPCS instead of RxNorm:

```python
# Example: "Patient has a urinary catheter" 
# → Detected as device (not medication)
# → Coded as HCPCS C1758 (Catheter, ureteral)
```

Supported device categories:
- Catheters (urinary, IV, central line, PICC)
- Mobility aids (wheelchairs, walkers, crutches, canes)
- Respiratory equipment (CPAP, BiPAP, nebulizers, oxygen)
- Monitoring devices (glucose meters, blood pressure monitors)
- Surgical supplies (syringes, needles, dressings, sutures)
- DME (hospital beds, lifts, commodes)
- Prosthetics and orthotics

## 🔒 Security Features

### TLS Encryption (NEW)
Protect sensitive healthcare data with TLS-encrypted communication:

```bash
# Generate certificates
python scripts/generate_certificates.py --type self-signed  # Development
python scripts/generate_certificates.py --type csr --domain yourdomain.com  # Production

# Start ALL TLS-enabled servers at once
python scripts/start_tls_servers.py

# Or run individual TLS servers
export TLS_CERT_FILE=certs/server.crt
export TLS_KEY_FILE=certs/server.key
python src/servers/icd10_server_tls.py   # Port 8911 (ICD-10 TLS)
python src/servers/snomed_server_tls.py  # Port 8912 (SNOMED TLS)
python src/servers/loinc_server_tls.py   # Port 8913 (LOINC TLS)
python src/servers/rxnorm_server_tls.py  # Port 8914 (RxNorm TLS)
python src/servers/hcpcs_server_tls.py   # Port 8915 (HCPCS TLS)

# Connect with TLS client
python examples/tls_client_example.py
```

**TLS Server Ports:**
- ICD-10 TLS: 8911 (standard: 8901)
- SNOMED TLS: 8912 (standard: 8902)
- LOINC TLS: 8913 (standard: 8903)
- RxNorm TLS: 8914 (standard: 8904)
- HCPCS TLS: 8915 (standard: 8905)

**Security Notes:**
- All TLS servers use TLS 1.2+ with strong ciphers
- Self-signed certificates for development only
- Production requires proper CA-signed certificates
- Optional mutual TLS for client authentication
- Never commit certificates or keys to git

## 🔧 Quick Start

### 1. Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### 2. Clone and Run

```bash
# Clone repository
git clone https://github.com/docdailey/medical-entity-code-mapper.git
cd medical-entity-code-mapper

# Run the complete system (downloads models, starts servers, runs tests)
uv run uv_runner.py
```

That's it! UV handles all dependencies, downloads models if needed, starts all servers in parallel, and runs integration tests.

### 3. Manual Server Control (Optional)

```bash
# Start individual servers
uv run python src/servers/icd10_server.py   # Port 8901
uv run python src/servers/snomed_server.py  # Port 8902
uv run python src/servers/loinc_server.py   # Port 8903
uv run python src/servers/rxnorm_server.py  # Port 8904

# Stop all servers
pkill -f 'python.*server'
```

## 📦 What's Included vs Downloaded

### Included in Repository:
- ✅ Source code and scripts
- ✅ FAISS indices (pre-built for all ontologies)
- ✅ Documentation and examples
- ✅ Test suite

### Downloaded Automatically:
- 📥 Clinical NER model (samrawal/bert-base-uncased_clinical-ner)
- 📥 Disease NER model (alvaroalon2/biobert_diseases_ner)
- 📥 BGE-M3 embedding model (BAAI/bge-m3)
- 📥 Optional: Biomedical NER model (d4data/biomedical-ner-all)

**Note**: Models are downloaded once and cached locally (~8-10 GB total).

## 💻 Usage

### Basic Entity Extraction

```bash
# Run the medical entity mapper directly
uv run python src/mappers/medical_entity_mapper.py

# Or use in your Python code
uv run python your_script.py
```

```python
from src.mappers.medical_entity_mapper import process_clinical_text

# Process clinical text
text = "Patient with hypertension prescribed lisinopril. CBC shows elevated WBC."
entities = process_clinical_text(text)

# Results include entity extraction and code mapping
for entity in entities:
    print(f"{entity['entity_name']} ({entity['category']}) -> {entity['codes']}")
```

### TCP Socket Protocol

```
Request:  sequence_number,description\n
Response: sequence_number,code\n

Example:
Request:  1,chest pain\n
Response: 1,R07.9\n
```

## 🏗️ Architecture

### System Components

1. **NER Models** (auto-downloaded)
   - Clinical NER: problem, treatment, test entities
   - Disease NER: disease, drug, chemical entities
   - Optional Biomedical NER: 107 entity types

2. **FAISS Indices** (included)
   - ICD-10: 146,000+ diagnosis codes
   - SNOMED CT: 350,000+ clinical concepts
   - LOINC: 104,000+ laboratory codes
   - RxNorm: 270,000+ medication concepts

3. **TCP Servers**
   - High-performance socket servers
   - Dynamic batching for throughput
   - MPS/CUDA acceleration support

## 📊 Performance Benchmarks

### 🚀 Optimized Dynamic Batching Performance

Using advanced optimization techniques from production deployments, all servers now include:
- **Dynamic batching** with adaptive batch sizes (2-25)
- **Multiple inference workers** for parallel processing
- **MPS acceleration** on Apple Silicon
- **Intelligent request queuing** with millisecond-precision timing

### Benchmarks on Mac Studio M2 Max (32GB RAM)

#### Solo Server Performance (Heavy Load: 1000 requests, 100 concurrent)

| Server   | QPS  | Avg Latency | P95 Latency | Index Size | Entries    | Target Met |
|----------|------|-------------|-------------|------------|------------|------------|
| HCPCS    | 397.6| 244.8ms     | 326.5ms     | 74MB       | 8,725      | ✅ YES     |
| LOINC    | 304.1| 320.3ms     | 416.6ms     | 896MB      | 104,000+   | ✅ YES     |
| RxNorm   | 202.8| 480.6ms     | 587.9ms     | 2.2GB      | 270,000+   | ❌ NO      |
| SNOMED   | 188.8| 516.3ms     | 612.8ms     | 3.1GB      | 350,000+   | ❌ NO      |
| ICD-10   | 160.4| 607.4ms     | 686.8ms     | 4.4GB      | 559,940    | ❌ NO      |

#### All Servers Running Together

| Server   | QPS  | Success Rate | Notes                                    |
|----------|------|--------------|------------------------------------------|
| HCPCS    | 386.2| 100%         | Exceeded 300 QPS target even with load  |
| LOINC    | 299.3| 100%         | Near target with resource contention     |
| RxNorm   | 241.6| 100%         | Good performance for large index         |
| ICD-10   | 71.6 | 100%         | Largest index, most affected by sharing  |
| SNOMED   | 64.4 | 100%         | Complex ontology impacts performance     |

### 📈 Estimated Performance on M3 Ultra (512GB RAM)

Based on architectural improvements and memory bandwidth scaling:

| Server   | Est. QPS | Est. Latency | Performance Gain | Notes                           |
|----------|----------|--------------|------------------|---------------------------------|
| HCPCS    | 750-850  | <150ms       | 2.0x             | Small index fits in cache       |
| LOINC    | 600-700  | <200ms       | 2.0x             | Benefits from faster memory     |
| RxNorm   | 400-500  | <300ms       | 2.0x             | Large index gains from RAM      |
| SNOMED   | 350-450  | <350ms       | 2.0x             | Complex queries scale well      |
| ICD-10   | 300-400  | <400ms       | 2.0x             | 512GB eliminates memory pressure|

**M3 Ultra Advantages:**
- **16x memory**: 512GB vs 32GB eliminates swapping for large indices
- **2x GPU cores**: 76-core vs 38-core GPU doubles MPS throughput
- **2x memory bandwidth**: 800GB/s vs 400GB/s accelerates FAISS operations
- **All indices in RAM**: Zero disk I/O during operation
- **Larger batch sizes**: Can increase max_batch from 25 to 50-100

### Running Optimized Servers

```bash
# Start all optimized servers
python scripts/start_optimized_servers.py

# Benchmark optimized servers
python scripts/benchmark_optimized.py

# Run specific heavy load test
python scripts/benchmark_optimized.py --load heavy --requests 1000 --threads 100
```

### Key Performance Features

- **MPS Acceleration**: Leverages Apple Silicon GPU for embedding computation
- **Dynamic Batching**: Automatically batches requests for optimal throughput
- **FAISS HNSW Index**: O(log n) search complexity for fast similarity matching
- **Persistent Connections**: Supports connection reuse for reduced latency
- **Zero-copy Protocol**: Efficient TCP socket communication

### Performance Tuning Tips

1. **For M2 Max (32GB)**: Keep batch sizes at 25, use 3 workers
2. **For M3 Ultra (512GB)**: Increase batch sizes to 50-100, use 6-8 workers
3. **Memory-constrained systems**: Reduce max_batch_size and workers
4. **CPU-bound workloads**: Increase inference workers to CPU core count
5. **Network optimization**: Use Unix sockets for local deployments

## 🛠️ Advanced Configuration

### Environment Variables (.env)

```bash
# Device configuration
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
```

### Manual Model Management

```bash
# Download only required models
uv run python scripts/download_models.py

# Include optional biomedical NER
uv run python scripts/download_models.py --skip-optional=false

# Verify models
uv run python scripts/download_models.py --verify-only

# Use custom model directory
uv run python scripts/download_models.py --models-dir=/path/to/models
```

### UV Configuration

The project uses `pyproject.toml` for dependency management. UV automatically:
- Creates isolated virtual environments
- Installs all dependencies including PyTorch with correct platform builds
- Manages Python version compatibility
- Provides reproducible builds with `uv.lock`

## 📄 License

This project is licensed under a Modified MIT License - **Non-Commercial Use Only**.

**Commercial use is prohibited without a license.** Any use in commercial products, services, or revenue-generating activities requires a commercial license.

Key points:
- ✅ Free for personal use, academic research, and education
- ✅ Free for non-profit organizations
- ✅ Free for healthcare organizations providing direct patient care
- ✅ Free for government agencies and public health departments
- ✅ Free for clinical research and trials
- ✅ Free for medical education and quality improvement
- ❌ Cannot be sold as a commercial product or service without a license
- ❌ Cannot be included in commercial software packages without a license
- ❌ Cannot be used to provide paid services to third parties without a license

For commercial licensing inquiries, contact: docdailey@gmail.com

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

- NER models from HuggingFace community
- UMLS Metathesaurus for medical synonyms
- i2b2/n2c2 datasets for clinical NLP
- FAISS for efficient similarity search
- BGE-M3 for medical embeddings

## ⚠️ Disclaimer

This software is intended for research and development purposes. Always validate results with qualified healthcare professionals before clinical use. The authors are not responsible for any clinical decisions made using this software.

## 📧 Contact

For questions, issues, or commercial licensing inquiries:
- GitHub Issues: [Create an issue](https://github.com/docdailey/medical-entity-code-mapper/issues)
- Email: docdailey@gmail.com