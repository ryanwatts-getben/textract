# Standard library imports
import json
import logging
import os
import subprocess
import tempfile
import pickle
import re
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import  Optional, Tuple, Dict, Any
import shutil
from pypdf import PdfReader
import csv
import torch
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# Third-party imports
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    Document,
    load_index_from_storage,
)
from llama_index.llms.anthropic import Anthropic

# Local imports
import db
from examplequery import get_mass_tort_data, get_disease_project_by_status, update_disease_project_status , get_mass_torts_by_user_id, get_disease_project, create_disease_project
from rag import create_index, preprocess_document
from ragindex import process_user_projects
from disease_definition_generator import generate_multiple_definitions, get_embedding_model
from app_rag_disease_config import (
    ENV_PATH,
    AWS_UPLOAD_BUCKET_NAME,
    ALLOWED_ORIGINS,
    CORS_CONFIG,
    SUPPORTED_EXTENSIONS,
    DEFAULT_USER_ID,
    DEFAULT_PROJECT_ID,
    MAX_WORKERS,
    CUDA_MEMORY_FRACTION,
    CUDA_DEVICE,
    LLM_MODEL,
    QUERY_ENGINE_CONFIG,
    LOG_CONFIG,
    SERVER_CONFIG,
    DocumentConfig,
    ERROR_MESSAGES,
    TEMP_DIR_PREFIX,
    INDEX_CACHE_FILENAME,
    INDEX_METADATA_FILENAME
)
from scrape import scrape_medline_plus
from scan import scan_documents, ScanInput

# Load environment variables from .env file
load_dotenv(dotenv_path=ENV_PATH)

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Update CORS configuration
CORS(app, resources=CORS_CONFIG)

s3_client = boto3.client('s3')  # Initialize S3 client

if __name__ == '__main__':
    # Initialize multiprocessing support
    import multiprocessing
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('fork')
        except RuntimeError:
            pass  # Method already set
    multiprocessing.freeze_support()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_CONFIG["level"]),
        format=LOG_CONFIG["format"]
    )
    logger = logging.getLogger(__name__)
    
    # Initialize the embedding model before starting the server
    try:
        from disease_definition_generator import DiseaseDefinitionEngine
        engine = DiseaseDefinitionEngine()
        logger.info("[app] Successfully initialized DiseaseDefinitionEngine")
    except Exception as e:
        logger.error(f"[app] Error initializing DiseaseDefinitionEngine: {str(e)}")
        raise
    
    # Start the Flask app
    app.run(
        host=SERVER_CONFIG['host'],
        port=SERVER_CONFIG['port'],
        debug=False  # Set to False to avoid multiprocessing issues
    )