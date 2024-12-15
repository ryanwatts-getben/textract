import pickle
"""
Configuration settings for the RAG Disease application.
"""

import os
from pathlib import Path
from typing import Set, List

#####################################
# APP CONFIGURATIONS
#####################################

# Environment Configuration
ENV_PATH = Path('/var/www/medchron-api/.env')

# AWS Configuration
AWS_UPLOAD_BUCKET_NAME = "generate-input-f5bef08a-9228-4f8c-a550-56d842b94088"

# CORS Configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://dev.everycase.ai",
    "http://localhost:5001"
]

CORS_CONFIG = {
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": [
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "Accept",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers"
        ],
        "supports_credentials": True,
        "expose_headers": ["Content-Type", "Authorization"]
    }
}

# File Processing Configuration
SUPPORTED_EXTENSIONS: Set[str] = {'.txt', '.pdf', '.json', '.xml', '.xsd', '.doc', '.docx', '.csv', '.rrf'}

# Index Configuration
DEFAULT_USER_ID = '00000'
DEFAULT_PROJECT_ID = '22222'
MAX_WORKERS = 10

# GPU Configuration
CUDA_MEMORY_FRACTION = 0.5
CUDA_DEVICE = 0

# LLM Configuration
LLM_MODEL = "claude-3-5-sonnet-20240620"

# Query Engine Configuration
QUERY_ENGINE_CONFIG = {
    "response_mode": "compact",
    "similarity_top_k": 50,
    "verbose": True
}

# Logging Configuration
LOG_CONFIG = {
    "level": "INFO",
    "format": '%(asctime)s - %(levelname)s - %(message)s'
}

# Server Configuration
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 5001
}

# Document Processing Configuration
class DocumentConfig:
    # PDF Processing
    PDF_MIN_LINE_LENGTH = 3
    PDF_BATCH_SIZE = 10

    # XML Processing
    XML_VALIDATION_ENABLED = True
    
    # Index Processing
    INDEX_BATCH_SIZE = 100000
    CHUNK_SIZE = 512
    MAX_NODES_PER_BATCH = 500
    PROCESSING_TIMEOUT = 300  # 5 minutes
    
    # Chunking Configuration
    CHUNK_OVERLAP = 20
    EMBED_BATCH_SIZE = 32

# Error Messages
ERROR_MESSAGES = {
    "missing_disease_names": "Invalid request. Expected {\"diseaseNames\": [\"disease1\", \"disease2\", ...]}",
    "no_valid_files": "No valid documents found to index",
    "no_processed_docs": "No valid documents were successfully processed",
    "missing_query_fields": "Missing required fields: {fields}"
}

# File Paths
TEMP_DIR_PREFIX = "index_storage"
INDEX_CACHE_FILENAME = "index_cache.pkl"
INDEX_METADATA_FILENAME = "index_metadata.pkl"

#####################################
# RAG CONFIGURATIONS
#####################################

# Embedding Model Configuration
EMBEDDING_MODEL_CONFIG = {
    "model_name": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "embed_batch_size": 32,
    "normalize": True
}

# Document Processing Configuration
class RAGDocumentConfig:
    # Chunk Processing
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks for memory efficiency
    
    # Index Manager Settings
    BATCH_SIZE = 100000
    CHUNK_SIZE_TOKENS = 512
    MAX_NODES_PER_BATCH = 500
    PROCESSING_TIMEOUT = 300  # 5 minutes
    
    # Index Categories
    VALID_CATEGORIES = ['icd', 'der', 'sct', 'cmo', 'other']
    
    # Parallel Processing
    MAX_PARALLEL_PROCESSES = 10

# Vector Store Settings
VECTOR_STORE_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 20,
    "embed_batch_size": 32,
    "show_progress": True
}

# Fallback Processing Configuration
FALLBACK_CONFIG = {
    "use_async": True,
    "show_progress": False,
    "batch_size": 1  # Process one document at a time in fallback mode
}

# Storage Configuration
STORAGE_CONFIG = {
    "pickle_protocol": pickle.HIGHEST_PROTOCOL,
    "cache_filename": "batch_index.pkl",
    "metadata_filename": "index_metadata.pkl"
}

# Error Messages for RAG
RAG_ERROR_MESSAGES = {
    "unsupported_file": "Unsupported file type: {extension}",
    "no_text_content": "No text content extracted from {file_path}",
    "processing_error": "Error processing document {file_path}: {error}",
    "index_creation_error": "Error creating index: {error}",
    "fallback_error": "Fallback processing failed: {error}"
}

# Category Prefixes for Document Classification
CATEGORY_PREFIXES = {
    'icd': 'ICD-10 Codes',
    'der': 'Derived Terms',
    'sct': 'SNOMED CT',
    'cmo': 'Clinical Modifications',
    'other': 'Other Documents'
}
