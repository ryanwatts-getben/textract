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
import time
import datetime  # Add global import of datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
import shutil
from pypdf import PdfReader
import csv
import torch
from typing import Dict, List
# Third-party imports
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import sys
import requests
import base64

load_dotenv()

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    Document,
)
from llama_index.llms.anthropic import Anthropic

# Local imports
from examplequery import get_mass_tort_data, get_disease_project_by_status, update_disease_project_status, get_mass_torts_by_user_id, get_disease_project, create_disease_project, update_disease_project_status_by_user, get_all_disease_project_by_status
from rag import create_index
from ragindex import process_user_projects
from disease_definition_generator import generate_multiple_definitions, get_embedding_model
from app_rag_disease_config import (
    ENV_PATH,
    AWS_UPLOAD_BUCKET_NAME,
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
    INDEX_CACHE_FILENAME
)
from scrape import scrape_medline_plus
from scan import scan_documents, load_index, check_user_projects

# Add import for Salesforce context gathering script
from salesforce_get_all_context_by_matter import initialize_salesforce, organize_matter_context, set_salesforce_auth_globals

# Add import for Salesforce new client script
from salesforce_create_new_client import process_nulaw_response_and_create_project

# Add import for salesforce_refresh_token
from salesforce_refresh_token import refresh_salesforce_token, verify_credentials, get_salesforce_headers, get_cli_credentials, update_env_file, authenticate_with_username_password

# Load environment variables from .env file
load_dotenv(dotenv_path=ENV_PATH)

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources=CORS_CONFIG)

# Initialize S3 client
s3_client = boto3.client('s3')

def configure_cuda_memory_limit(fraction=CUDA_MEMORY_FRACTION, device=CUDA_DEVICE):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device=device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        logger.info(f"[rag] Initial GPU Memory Allocated: {initial_memory / 1e9:.2f} GB / {total_memory / 1e9:.2f} GB")
    else:
        logger.info("CUDA is not available.")

# Update CORS configuration
CORS(app, resources=CORS_CONFIG)

s3_client = boto3.client('s3')  # Initialize S3 client

def process_file(bucket_name, file_path, script_name):
    user_id, case_id = file_path.split('/')[:2]
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    message_body = {
        'file_name': file_name_without_extension,
        'file_path': file_path,
        'bucket': bucket_name,
        'case_id': case_id,
        'user_id': user_id,
    }
    logger.info(f"[app] Processing: {message_body}")
    # Construct the command to run the appropriate split script
    command = [
        "python", script_name,
        json.dumps(message_body)
    ]
    try:
        subprocess.run(command, check=True)
        logger.info(f"[app] Successfully processed {file_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"[app] Error processing {file_path}: {e}")
        raise

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        bucket_name = data['bucket']
        records = data.get('records', [])
        bills = data.get('bills', [])
        accident_reports = data.get('reports', [])
        total_records = len(records)
        total_bills = len(bills)
        logger.info(f'[app] Total records: {total_records}, Total bills: {total_bills}')
        
        for accident_report in accident_reports:
            process_file(bucket_name, accident_report['file_path'], 'split_reports.py')
        for record in records:
            process_file(bucket_name, record['file_path'], 'split_records.py')
        for bill in bills:
            process_file(bucket_name, bill['file_path'], 'split_bills.py')
       
        logger.info(f"[app] Successfully processed {total_records} records and {total_bills} bills")
        return jsonify({"status": "success", "message": f"Successfully processed {total_records} records and {total_bills} bills"}), 200
    except Exception as e:
        logger.error(f"[app] Error occurred while processing records and bills: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
def query_index(index: VectorStoreIndex, query_text: str) -> str:
    """Query the index with given text using Claude."""
    try:
        # Initialize Claude with the latest model
        llm = Anthropic(
            model=LLM_MODEL,
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )

        # Create query engine with detailed configurations
        query_engine = index.as_query_engine(
            llm=llm,
            **QUERY_ENGINE_CONFIG
        )
        
        # Log the query for debugging
        logger.debug(f'[rag] Query Text: {query_text}')
        
        # Execute the query and get response
        response = query_engine.query(query_text)
        
        # Log the response for debugging
        logger.debug(f'[rag] Response Text: {response}')
        
        return response.response
        
    except Exception as e:
        logger.error(f"[app] Error querying index: {str(e)}")
        raise

# def extract_text_from_file(content: bytes, file_extension: str) -> str:
#     """
#     Extract text from various file types.
    
#     Args:
#         content (bytes): The binary content of the file
#         file_extension (str): The file extension (including the dot)
        
#     Returns:
#         str: The extracted text
#     """
#     try:
#         if file_extension == '.txt':
#             return content.decode('utf-8', errors='ignore')
            
#         elif file_extension == '.pdf':
#             with io.BytesIO(content) as pdf_buffer:
#                 reader = PdfReader(pdf_buffer)
#                 text = []
#                 for page in reader.pages:
#                     text.append(page.extract_text() or '')
#                 return '\n'.join(text)
                
#         elif file_extension == '.docx':
#             with io.BytesIO(content) as docx_buffer:
#                 doc = Document(docx_buffer)
#                 return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
                
#         elif file_extension == '.csv':
#             text_data = content.decode('utf-8', errors='ignore')
#             csv_reader = csv.reader(text_data.splitlines())
#             return '\n'.join(','.join(row) for row in csv_reader)
            
#         elif file_extension in {'.xml', '.xsd'}:
#             try:
#                 root = DefusedET.fromstring(content.decode('utf-8', errors='ignore'))
#                 text_content = []
                
#                 def extract_text_from_element(element, path=""):
#                     current_path = f"{path}/{element.tag}" if path else element.tag
                    
#                     # Add element text if present
#                     if element.text and element.text.strip():
#                         text_content.append(f"{current_path}: {element.text.strip()}")
                    
#                     # Process attributes
#                     for key, value in element.attrib.items():
#                         text_content.append(f"{current_path}[@{key}]: {value}")
                    
#                     # Process child elements
#                     for child in element:
#                         extract_text_from_element(child, current_path)
                
#                 extract_text_from_element(root)
#                 return '\n'.join(text_content)
                
#             except Exception as xml_e:
#                 logger.error(f"[app] Error parsing XML/XSD content: {str(xml_e)}")
#                 return content.decode('utf-8', errors='ignore')
                
#         elif file_extension == '.rrf':
#             # RRF (Rich Release Format) files are typically pipe-delimited
#             text_data = content.decode('utf-8', errors='ignore')
#             return '\n'.join(line for line in text_data.splitlines())
            
#         elif file_extension == '.json':
#             try:
#                 json_data = json.loads(content.decode('utf-8', errors='ignore'))
#                 if isinstance(json_data, dict):
#                     return '\n'.join(f"{k}: {v}" for k, v in json_data.items())
#                 elif isinstance(json_data, list):
#                     return '\n'.join(str(item) for item in json_data)
#                 else:
#                     return str(json_data)
#             except json.JSONDecodeError:
#                 return content.decode('utf-8', errors='ignore')
        
#         else:
#             logger.warning(f"[app] Unsupported file extension: {file_extension}")
#             return content.decode('utf-8', errors='ignore')
            
#     except Exception as e:
#         logger.error(f"[app] Error extracting text from {file_extension} file: {str(e)}")
#         return ""

def process_single_document(s3_client, key: str) -> Optional[Document]:
    """Process a single document from S3."""
    try:
        logger.info(f"[app] Processing document: {key}")
        
        # Get the file from S3
        response = s3_client.get_object(Bucket=AWS_UPLOAD_BUCKET_NAME, Key=key)
        content = response['Body'].read()
        
        # Get file extension
        file_extension = os.path.splitext(key)[1].lower()
        logger.info(f"[app] File extension: {file_extension}")
        
        if file_extension not in SUPPORTED_EXTENSIONS:
            logger.warning(f"[app] Unsupported file extension: {file_extension}")
            return None
            
        # Extract text based on file type
        if file_extension == '.pdf':
            text = extract_text_from_pdf(content)
        # elif file_extension == '.docx':
        #     text = extract_text_from_docx(content)
        elif file_extension == '.json':
            text = process_json_content(json.loads(content.decode('utf-8')))
        # elif file_extension == '.xml' or file_extension == '.xsd':
        #     text = extract_text_from_xml(content)
        elif file_extension == '.csv':
            text = extract_text_from_csv(content)
        else:  # .txt and other text files
            text = content.decode('utf-8', errors='ignore')
            
        if not text:
            logger.warning(f"[app] No text content extracted from {key}")
            return None
            
        # Create document with metadata
        document = Document(
            text=text,
            metadata={
                'source': key,
                'file_type': file_extension,
                'timestamp': str(response.get('LastModified', '')),
                'size': response.get('ContentLength', 0)
            }
        )
        
        logger.info(f"[app] Successfully processed document: {key} (size: {len(text)} chars)")
        return document
        
    except Exception as e:
        logger.error(f"[app] Error processing document {key}: {str(e)}")
        return None

def create_or_update_index(user_id: str = DEFAULT_USER_ID, project_id: str = DEFAULT_PROJECT_ID, force_refresh: bool = False, max_workers: int = MAX_WORKERS) -> Tuple[VectorStoreIndex, str]:
    """
    Create or update an index for the specified user and project.
    """
    s3_client = boto3.client('s3')

    # Define the S3 paths
    input_prefix = f"{user_id}/{project_id}/input/"
    index_key = f"{user_id}/{project_id}/index.pkl"

    logger.info(f"[app] Looking for documents in s3://{AWS_UPLOAD_BUCKET_NAME}/{input_prefix}")

    temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    index_cache_path = os.path.join(temp_dir, INDEX_CACHE_FILENAME)

    try:
        # Check if index exists in S3
        try:
            s3_client.head_object(Bucket=AWS_UPLOAD_BUCKET_NAME, Key=index_key)
            index_exists = True
            logger.info(f"[app] Index found at s3://{AWS_UPLOAD_BUCKET_NAME}/{index_key}")
        except ClientError as e:
            # Index not found
            index_exists = False
            logger.info(f"[app] No existing index found at s3://{AWS_UPLOAD_BUCKET_NAME}/{index_key}")

        if index_exists and not force_refresh:
            # Load existing index from S3
            logger.info(f"[app] Loading existing index from s3://{AWS_UPLOAD_BUCKET_NAME}/{index_key}")
            with open(index_cache_path, 'wb') as f:
                s3_client.download_fileobj(AWS_UPLOAD_BUCKET_NAME, index_key, f)
            with open(index_cache_path, 'rb') as f:
                index = pickle.load(f)
            return index, 'Existing index loaded successfully'

        # Create new index or refresh existing one
        logger.info('[app] Creating new index')
        documents = []

        # List all objects in the input prefix
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=AWS_UPLOAD_BUCKET_NAME, Prefix=input_prefix)
            
            # Collect all valid files
            files_to_process = set()  # Using set to avoid duplicates
            all_files = []
            
            for page in pages:
                if 'Contents' not in page:
                    logger.warning(f"[app] No contents found in page for prefix: {input_prefix}")
                    continue
                    
                for obj in page.get('Contents', []):
                    key = obj.get('Key')
                    if not key:
                        continue
                        
                    all_files.append(key)
                    file_extension = os.path.splitext(key)[1].lower()
                    
                    # Skip duplicate files (e.g., icd10cm vs icd-10-cm)
                    base_name = os.path.basename(key).lower()
                    if any(x in base_name for x in ['icd10cm-', 'icd10-']):
                        if any(x in files_to_process for x in [key.replace('icd10cm-', 'icd-10-cm-'), 
                                                             key.replace('icd10-', 'icd-10-cm-')]):
                            logger.info(f"[app] Skipping duplicate file: {key}")
                            continue
                    
                    if file_extension in SUPPORTED_EXTENSIONS:
                        files_to_process.add(key)
                        logger.info(f"[app] Found valid file to process: {key}")
                    else:
                        logger.warning(f"[app] Skipping unsupported file: {key} (extension: {file_extension})")
            
            logger.info(f"[app] Found {len(all_files)} total files, {len(files_to_process)} valid files to process")
            logger.debug(f"[app] All files found: {all_files}")
            
            if not files_to_process:
                logger.error(f"[app] No valid files found in prefix: {input_prefix}")
                logger.error(f"[app] Supported extensions are: {SUPPORTED_EXTENSIONS}")
                raise ValueError(ERROR_MESSAGES["no_valid_files"])

        except ClientError as e:
            logger.error(f"[app] Error listing objects in bucket: {str(e)}")
            raise ValueError(f"Error accessing S3: {str(e)}")

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {
                executor.submit(process_single_document, s3_client, key): key
                for key in files_to_process
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    doc = future.result()
                    if doc:
                        documents.append(doc)
                        logger.info(f'[app] Added document from {os.path.basename(key)}')
                    else:
                        logger.warning(f'[app] No content extracted from {key}')
                except Exception as e:
                    logger.error(f'[app] Error processing file {key}: {e}')

        if not documents:
            logger.error('[app] No documents were successfully processed')
            raise ValueError(ERROR_MESSAGES["no_processed_docs"])

        # Create index
        logger.info(f'[app] Creating new index with {len(documents)} documents')
        embed_model, _ = get_embedding_model()
        index = create_index(
            documents=documents,
            embed_model=embed_model,
            batch_size=DocumentConfig.INDEX_BATCH_SIZE,
            chunk_size=DocumentConfig.CHUNK_SIZE
        )

        # Cache the index locally
        logger.info(f"[app] Saving index to local cache: {index_cache_path}")
        with open(index_cache_path, 'wb') as f:
            pickle.dump(index, f)

        # Upload the index to S3
        logger.info(f"[app] Uploading index to S3: s3://{AWS_UPLOAD_BUCKET_NAME}/{index_key}")
        s3_client.upload_file(index_cache_path, AWS_UPLOAD_BUCKET_NAME, index_key)
        logger.info(f"[app] Index upload complete. Local copy at: {index_cache_path}, S3 copy at: s3://{AWS_UPLOAD_BUCKET_NAME}/{index_key}")

        return index, 'New index created and uploaded successfully'

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"[app] Cleaned up temporary directory: {temp_dir}")

class ProgressPercentage:
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self.progress = 0

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            self.progress = (self._seen_so_far / self._size) * 100
            logger.info(f'[app] Upload progress: {self.progress:.2f}%')

@app.route('/index_files', methods=['POST'])
def index_route():
    """
    Endpoint to create or update indices for all projects of a user.
    Expects a JSON payload with just userId: {"userId": "user_id_here"}
    """
    try:
        # Extract userId from the request
        data = request.get_json()
        if not data or 'userId' not in data:
            logger.error("[app] Missing userId in request payload")
            return jsonify({
                'status': 'error',
                'message': 'userId is required'
            }), 400
            
        user_id = data['userId']
        logger.info(f"[app] Received index request for user_id: {user_id}")
        
        # Configure CUDA memory limit
        configure_cuda_memory_limit(0.5, device=0)
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Process all projects for the user
        report = process_user_projects(
            s3_client=s3_client,
            bucket_name=AWS_UPLOAD_BUCKET_NAME,
            user_id=user_id,
            force_refresh=True
        )
        
        # Return a success response with the processing report
        return jsonify({
            'status': 'success',
            'message': 'Indexing completed',
            'report': report
        }), 200
        
    except Exception as e:
        logger.error(f"[app] Error in /index route: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        force_refresh = data.get('force_refresh', False)
        
        # Handle both naming conventions
        query_text = data.get('query_text') or data.get('query', '')
        project_id = data.get('project_id') or data.get('projectId', '')
        user_id = data.get('user_id') or data.get('userId', '')
        
        if not all([query_text, project_id, user_id]):
            missing = []
            if not query_text: missing.append('query text')
            if not project_id: missing.append('project ID')
            if not user_id: missing.append('user ID')
            return jsonify({
                'status': 'error',
                'message': ERROR_MESSAGES["missing_query_fields"].format(fields=", ".join(missing))
            }), 400

        # Get or create index
        try:
            index, _ = create_or_update_index(user_id, project_id, force_refresh)
            
            # Query the index
            response_text = query_index(index, query_text)
            if not response_text or response_text == "Empty Response":
                return jsonify({
                    'status': 'error',
                    'message': 'No relevant information found for your query.'
                }), 404

            return jsonify({'status': 'success', 'response': response_text}), 200
            
        except ValueError as ve:
            return jsonify({
                'status': 'error',
                'message': str(ve)
            }), 404
            
    except Exception as e:
        logger.exception(f'[app] Error processing query: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

def process_pdf_page(page, page_num, total_pages):
    """Process a single PDF page."""
    try:
        page_text = page.extract_text()
        if page_text:
            text = page_text.strip()
            logger.info(f'[app] Successfully extracted page {page_num}/{total_pages} with {len(text)} characters')
            return text
        else:
            logger.warning(f'[app] Empty text extracted from page {page_num}/{total_pages}')
            return ""
    except Exception as e:
        logger.error(f'[app] Error extracting text from page {page_num}: {str(e)}')
        return ""

def read_pdf_with_fallbacks(file_path):
    """
    Attempt to read PDF text using multiple methods with fallbacks.
    Returns the best result or None if all methods fail.
    """
    text_results = []
    
    # Method 1: PyPDF
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            num_pages = len(pdf_reader.pages)
            logger.info(f'[app] PDF has {num_pages} pages')
            
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text.strip() + '\n'
                        logger.info(f'[app] Successfully extracted page {i+1}/{num_pages} with {len(page_text)} characters')
                except Exception as e:
                    logger.warning(f'[app] PyPDF failed on page {i+1}: {str(e)}')
                    
        if text.strip():
            text_results.append(('pypdf', text))
    except Exception as e:
        logger.error(f'[app] PyPDF extraction failed: {str(e)}')

    # # Method 2: pdfplumber
    # try:
    #     text = ""
    #     with pdfplumber.open(file_path) as pdf:
    #         for i, page in enumerate(pdf.pages):
    #             try:
    #                 page_text = page.extract_text()
    #                 if page_text:
    #                     text += page_text.strip() + '\n'
    #                     logger.info(f'[app] pdfplumber: Extracted page {i+1} with {len(page_text)} characters')
    #             except Exception as e:
    #                 logger.warning(f'[app] pdfplumber failed on page {i+1}: {str(e)}')
                    
    #     if text.strip():
    #         text_results.append(('pdfplumber', text))
    # except Exception as e:
    #     logger.error(f'[app] pdfplumber extraction failed: {str(e)}')

    # # Method 3: pdfminer
    # try:
    #     text = pdfminer_extract_text(file_path)
    #     if text.strip():
    #         text_results.append(('pdfminer', text))
    #         logger.info(f'[app] pdfminer: Extracted {len(text)} characters')
    # except Exception as e:
    #     logger.error(f'[app] pdfminer extraction failed: {str(e)}')

    # # Choose the best result
    # if text_results:
    #     # Sort by text length (assuming longer text is better)
    #     text_results.sort(key=lambda x: len(x[1]), reverse=True)
    #     method, text = text_results[0]
    #     logger.info(f'[app] Using {method} result with {len(text)} characters')
        return clean_text(text)
    
    return None

def clean_text(text):
    """Clean and normalize extracted text."""
    if not text:
        return None
        
    # Remove null bytes and control characters
    text = ''.join(char if ord(char) >= 32 or char in '\n\t' else ' ' for char in text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove any remaining problematic characters
    text = text.replace('\x00', '')
    
    # Remove very short lines (likely garbage)
    lines = [line for line in text.split('\n') if len(line.strip()) > DocumentConfig.PDF_MIN_LINE_LENGTH]
    text = '\n'.join(lines)
    
    return text if text.strip() else None

# def read_file_content(file_path):
#     """Reads and validates file content."""
#     try:
#         file_lower = file_path.lower()
        
#         if file_lower.endswith('.pdf'):
#             logger.info(f'[app] Reading PDF file: {file_path}')
#             text = read_pdf_with_fallbacks(file_path)
            
#         elif file_lower.endswith('.json'):
#             logger.info(f'[app] Reading JSON file: {file_path}')
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 data = json.load(file)
#                 text = process_json_content(data)
                
#         elif file_lower.endswith('.xml'):
#             logger.info(f'[app] Reading XML file: {file_path}')
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 xml_content = file.read()
#             xsd_content = None
#             # Check for corresponding XSD file
#             xsd_file_path = os.path.splitext(file_path)[0] + '.xsd'
#             if os.path.exists(xsd_file_path):
#                 logger.info(f'[app] Found XSD schema: {xsd_file_path}')
#                 with open(xsd_file_path, 'r', encoding='utf-8') as xsd_file:
#                     xsd_content = xsd_file.read()
#             text = parse_xml_content(xml_content, xsd_content)
            
#         elif file_lower.endswith('.xsd'):
#             logger.info(f'[app] Reading XSD file: {file_path} (treating as XML)')
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 xml_content = file.read()
#             text = parse_xml_content(xml_content)
            
#         else:  # Text files
#             logger.info(f'[app] Reading text file: {file_path}')
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 text = file.read()

#         # Validate and clean the text
#         text = clean_text(text)
        
#         if not text:
#             logger.warning(f'[app] No valid content extracted from {file_path}')
#             return None
            
#         logger.info(f'[app] Successfully extracted {len(text)} characters from {file_path}')
#         return text
        
#     except Exception as e:
#         logger.error(f"[app] Error reading file {file_path}: {e}")
#         return None

def sanitize_filename(filename):
    """
    Sanitizes the filename to remove or replace characters not allowed in file systems.
    """
    # Replace invalid characters with an underscore
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Ensure that the index is initialized or loaded
# You might need to adjust this depending on your application flow
index = None  # Placeholder for the index variable

# def parse_xml_content(xml_content: str, xsd_content: Optional[str] = None) -> str:
#     """
#     Parse XML content and optionally validate against XSD schema.
#     """
#     try:
#         # Parse XML securely using DefusedET
#         root = DefusedET.fromstring(xml_content)
        
#         # If XSD content is provided, validate XML
#         if xsd_content:
#             logger.info("[app] Validating XML against XSD schema")
#             # Parse XSD content securely
#             xsd_root = DefusedET.fromstring(xsd_content)
#             xml_schema = ET.XMLSchema(xsd_root)
#             if not xml_schema.validate(root):
#                 logger.warning(f"[app] XML validation failed against XSD schema")
#                 # Optionally, raise an error or log details
#                 # xml_schema.assertValid(root)
        
#         # Rest of the function remains the same
#         text_content = []

#         def process_element(element, path=""):
#             """Recursively process XML elements."""
#             current_path = f"{path}/{element.tag}" if path else element.tag

#             # Add element text if present
#             if element.text and element.text.strip():
#                 text_content.append(f"{current_path}: {element.text.strip()}")

#             # Process attributes
#             for key, value in element.attrib.items():
#                 text_content.append(f"{current_path}[@{key}]: {value}")

#             # Process child elements
#             for child in element:
#                 process_element(child, current_path)

#         process_element(root)
#         return '\n'.join(text_content)

#     except ET.ParseError as e:
#         logger.error(f"[app] XML parsing error: {str(e)}")
#         raise
#     except Exception as e:
#         logger.error(f"[app] Error processing XML content: {str(e)}")
#         raise

def process_json_content(data):
    """Process JSON data and convert it to text."""
    if isinstance(data, dict):
        return '\n'.join(f"{k}: {v}" for k, v in data.items() if v)
    elif isinstance(data, list):
        return '\n'.join(str(item) for item in data if item)
    else:
        return str(data)

@app.route('/generate-multiple-disease-definitions', methods=['POST'])
def generate_definitions():
    """
    Endpoint to generate multiple disease definitions.
    Expects JSON payload: {"diseaseNames": ["disease1", "disease2", ...]}
    """
    try:
        logger.info("[app] Received request to generate disease definitions")
        # Validate request
        request_data = request.get_json()
        logger.info(f"[app] Request data received: {request_data}")
        
        if not request_data or 'diseaseNames' not in request_data:
            logger.warning("[app] Invalid request: missing diseaseNames")
            return jsonify({
                'error': ERROR_MESSAGES["missing_disease_names"]
            }), 400

        disease_names = request_data['diseaseNames']
        
        # Generate definitions using the imported function
        results = generate_multiple_definitions(disease_names)
        
        logger.info("[app] Definitions generated successfully")
        return jsonify(results)

    except ValueError as ve:
        logger.error(f"[app] Validation error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"[app] Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"[app] 404 Not Found: {request.path}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"[app] 500 Internal Server Error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF content."""
    try:
        with io.BytesIO(content) as pdf_file:
            reader = PdfReader(pdf_file)
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or '')
            return '\n'.join(text)
    except Exception as e:
        logger.error(f"[app] Error extracting text from PDF: {str(e)}")
        return ""

# def extract_text_from_docx(content: bytes) -> str:
#     """Extract text from DOCX content."""
#     try:
#         with io.BytesIO(content) as docx_file:
#             doc = docx.Document(docx_file)
#             return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
#     except Exception as e:
#         logger.error(f"[app] Error extracting text from DOCX: {str(e)}")
#         return ""

# def extract_text_from_xml(content: bytes) -> str:
#     """Extract text from XML content."""
#     try:
#         root = DefusedET.fromstring(content.decode('utf-8', errors='ignore'))
#         text_content = []
        
#         def extract_text_from_element(element, path=""):
#             current_path = f"{path}/{element.tag}" if path else element.tag
            
#             # Add element text if present
#             if element.text and element.text.strip():
#                 text_content.append(f"{current_path}: {element.text.strip()}")
            
#             # Process attributes
#             for key, value in element.attrib.items():
#                 text_content.append(f"{current_path}[@{key}]: {value}")
            
#             # Process child elements
#             for child in element:
#                 extract_text_from_element(child, current_path)
        
#         extract_text_from_element(root)
#         return '\n'.join(text_content)
        
#     except Exception as e:
#         logger.error(f"[app] Error extracting text from XML: {str(e)}")
#         return ""

def extract_text_from_csv(content: bytes) -> str:
    """Extract text from CSV content."""
    try:
        text_data = content.decode('utf-8', errors='ignore')
        csv_reader = csv.reader(text_data.splitlines())
        return '\n'.join(','.join(str(cell) for cell in row) for row in csv_reader)
    except Exception as e:
        logger.error(f"[app] Error extracting text from CSV: {str(e)}")
        return ""

@app.route('/scrape/', methods=['POST'])
def scrape_disease_info():
    """
    Endpoint to scrape disease information from MedlinePlus.
    
    Accepts two formats:
    1. Array format:
    {
        "diseaseUrl": [
            ["http://www.nlm.nih.gov/medlineplus/abdominalpain.html", "Abdominal Pain"]
        ]
    }
    
    2. Single URL format:
    {
        "diseaseUrl": "http://www.nlm.nih.gov/medlineplus/abdominalpain.html"
    }
    
    Returns:
        JSON object with scraped content
    """
    try:
        # Log the raw request data for debugging
        raw_data = request.get_data()
        logger.info(f"[app] Raw request data: {raw_data}")
        
        # Get JSON data and log it
        data = request.get_json(force=True)
        logger.info(f"[app] Parsed request data: {data}")
        
        if not data:
            logger.error("[app] No JSON data in request body")
            return jsonify({
                'error': 'No JSON data in request body'
            }), 400
            
        if 'diseaseUrl' not in data:
            logger.error("[app] Missing diseaseUrl in request body")
            return jsonify({
                'error': 'Missing diseaseUrl in request body'
            }), 400
            
        disease_url_data = data['diseaseUrl']
        
        # Handle single URL format
        if isinstance(disease_url_data, str):
            logger.info(f"[app] Processing single URL: {disease_url_data}")
            result = scrape_medline_plus(disease_url_data)
            if result is not None:
                return jsonify(result)
            else:
                return jsonify({
                    'error': 'Failed to scrape content from the provided URL'
                }), 500
        
        # Handle array format
        elif isinstance(disease_url_data, list):
            results = {}
            for url_pair in disease_url_data:
                if not isinstance(url_pair, list) or len(url_pair) != 2:
                    logger.error(f"[app] Invalid URL pair format: {url_pair}")
                    continue
                    
                url, disease_name = url_pair
                logger.info(f"[app] Scraping data for {disease_name} from {url}")
                
                result = scrape_medline_plus(url)
                
                if result is not None:
                    results[disease_name] = result
                else:
                    logger.error(f"[app] Failed to scrape content for {disease_name} from {url}")
                    results[disease_name] = {'error': f'Failed to scrape content for {disease_name}'}
            
            if not results:
                return jsonify({
                    'error': 'Failed to scrape content from any of the provided URLs'
                }), 500
                
            return jsonify(results)
        
        else:
            logger.error("[app] Invalid diseaseUrl format")
            return jsonify({
                'error': 'diseaseUrl must be either a string URL or an array of [URL, name] pairs'
            }), 400
        
    except Exception as e:
        logger.error(f"[app] Error processing scrape request: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/beginScan', methods=['GET'])
def begin_scan():
    """
    Endpoint to begin scanning documents for mass tort analysis.
    Only processes projects that have been marked as READY by /prepareScan.
    """
    try:
        logger.info("[app] Received scan request")

        # Check for already processing projects
        processing_disease_project = get_disease_project_by_status('PROCESSING')
        if processing_disease_project:
            logger.info(f"[app] Found processing project {processing_disease_project['id']}")
            return jsonify({
                'status': 'success',
                'message': 'found existing record in PROCESSING state'
            }), 200

        # Get the first READY project
        disease_project = get_disease_project_by_status('READY')
        if not disease_project:
            logger.info("[app] No READY projects found")
            return jsonify({
                'status': 'success',
                'message': 'No READY projects found'
            }), 200
        
        logger.info(f"[app] Found READY project {disease_project['userId']}")

        # Update status to PROCESSING
        update_disease_project_status_by_user(
            project_id=disease_project['projectId'],
            user_id=disease_project['userId'],
            status='PROCESSING'
        )

        # Get mass tort data
        mass_tort_data = get_mass_torts_by_user_id(user_id=disease_project['userId'])
        if not mass_tort_data:
            logger.error(f"[app] No mass torts found for user {disease_project['userId']}")
            update_disease_project_status_by_user(
                project_id=disease_project['projectId'],
                user_id=disease_project['userId'],
                status='ERROR'
            )
            return jsonify({
                'status': 'error',
                'message': 'No mass torts found for user'
            }), 400

        logger.info(f"[app] Found {len(mass_tort_data)} mass torts for user {disease_project['userId']}")

        data = get_mass_tort_data(
            user_id=disease_project['userId'],
            project_id=disease_project['projectId'],
            mass_tort_ids=[disease_project['massTortId']]
        )

        if not data:
            logger.error("[app] No JSON data provided")
            update_disease_project_status_by_user(
                project_id=disease_project['projectId'],
                user_id=disease_project['userId'],
                status='ERROR'
            )
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400

        # Validate required fields
        required_fields = ['userId', 'projectId', 'massTorts']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"[app] Missing required fields: {missing_fields}")
            update_disease_project_status_by_user(
                project_id=disease_project['projectId'],
                user_id=disease_project['userId'],
                status='ERROR'
            )
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        logger.info(f"[app] Starting scan for project {data['projectId']}")
        logger.info(f"[app] Processing {len(data['massTorts'])} mass torts")
        
        # Process the scan with validated data
        try:
            result = scan_documents(data)
            
            if result.get('status') == 'error':
                logger.error(f"[app] Scan failed: {result.get('message')}")
                update_disease_project_status_by_user(
                    project_id=disease_project['projectId'],
                    user_id=disease_project['userId'],
                    status='ERROR'
                )
                return jsonify(result), 500
                
            logger.info("[app] Scan completed successfully")
            process_scanned_results(result, mass_tort_data, project_id=data['projectId'], user_id=disease_project['userId'])
            return jsonify(result), 200
            
        except Exception as scan_error:
            error_msg = f"Error during document scan: {str(scan_error)}"
            update_disease_project_status_by_user(
                project_id=disease_project['projectId'],
                user_id=disease_project['userId'],
                status='ERROR'
            )
            logger.error(f"[app] {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 500

    except Exception as e:
        error_msg = f"Error processing scan request: {str(e)}"
        logger.error(f"[app] {error_msg}")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

def check_if_index_or_input_exists(disease_projects: List[Dict]) -> Dict:
    for project in disease_projects:
        user_id = project['userId']
        project_id = project['projectId']
        mass_tort_id = project['massTortId']

        if not user_id or not project_id:
            raise ValueError(f"Missing required userId or projectId for project: {project}")

        logger.info(f"[scan] Starting scan for user {user_id}")
        
        # Check for active projects first
        projects_check = check_user_projects(user_id, mass_tort_id)
        if projects_check['status'] == 'error':
            raise ValueError(projects_check['message'])
        
        # Load index using the existing load_index function
        logger.info("[scan] Loading index from S3")
        load = load_index(user_id, project_id)
        
        if not load:
            logger.error(f"[scan] Failed to load index for user {user_id}, project {project_id}")
            update_disease_project_status_by_user(
                project_id=project_id,
                user_id=user_id,
                status='ERROR'
            )

    # Return all projects with a specific status
    return get_disease_project_by_status()

@app.route('/api/disease-scanner/reports/generate-report', methods=['POST'])
def generate_report():
    """Generate a report based on the provided data."""
    try:
        data = request.json
        logger.info(f"[app] Generating report with data: {json.dumps(data, indent=2)}")
        
        # Extract required fields
        user_id = data.get('userId')
        project_id = data.get('projectId')
        mass_tort_id = data.get('massTortId')
        diseases = data.get('diseases', [])
        
        if not all([user_id, project_id, mass_tort_id, diseases]):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: userId, projectId, massTortId, or diseases'
            }), 400
            
        # Call scan_documents to process the diseases
        scan_input = {
            'userId': user_id,
            'projectId': project_id,
            'massTorts': [{
                'id': mass_tort_id,
                'diseases': diseases
            }]
        }
        
        scan_results = scan_documents(scan_input)
        
        if scan_results.get('status') == 'success':
            return jsonify(scan_results), 200
        else:
            return jsonify({
                'status': 'error',
                'message': scan_results.get('message', 'Failed to generate report')
            }), 500
            
    except Exception as e:
        logger.error(f"[app] Error generating report: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
def update_disease_project_errorstatus(data):
    # Extract the required fields from data
    user_id = data.get('userId')
    project_id = data.get('projectId')
    mass_tort_id = data['massTorts'][0]['id'] if data.get('massTorts') else None

    if all([user_id, project_id, mass_tort_id]) and data.get('massTorts'):
        # Iterate over each disease in the mass torts
        for disease in data['massTorts'][0]['diseases']:
            disease_id = disease['id']
            # Call the update function with the correct disease_id
            update_disease_project_status(
                project_id=project_id,
                mass_tort_id=mass_tort_id,
                disease_id=disease_id,
                status='ERROR'
            )
            logger.info(
                "[app] Updated disease project status to ERROR for project %s, disease %s, mass tort %s",
                project_id, disease_id, mass_tort_id
            )
        logger.info(
            "[app] Updated disease project statuses to ERROR for project %s, mass tort %s",
            project_id, mass_tort_id
        )
    else:
        logger.error(
            "[app] Could not update disease project status: missing required fields or diseases"
        )

def process_scanned_results(data, mass_tort_data, project_id, user_id):
    """
    Process scanned results and update disease projects accordingly.

    Args:
        data (dict): The scanned results data.
        mass_tort_data (list): List of mass tort data.
        project_id (str): The project ID.
        user_id (str): The user ID.

    Returns:
        None
    """
    logger.info("[app] Processing scanned results: %s", json.dumps(mass_tort_data, indent=2))
    
    try:
        if data.get('status') == 'success' and data.get('results'):
            # Process each mass tort result
            for mass_tort_result in data['results']:
                try:
                    # Find the matching mass tort by official name
                    mass_tort = next(
                        (mt for mt in mass_tort_data if mt['officialName'] == mass_tort_result['mass_tort_name']),
                        None
                    )
                    if not mass_tort:
                        print('[scan.route] Mass tort not found:', mass_tort_result['mass_tort_name'])
                        continue
                    
                    # Process each disease result
                    for disease_result in mass_tort_result['diseases']:
                        try:
                            # Find the matching disease by name
                            mass_tort_disease = next(
                                (d for d in mass_tort['massTortDiseases'] if d['disease']['name'] == disease_result['disease_name']),
                                None
                            )
                            if not mass_tort_disease:
                                print('[scan.route] Disease not found:', disease_result['disease_name'])
                                continue

                            # Check for existing project
                            existing_project = get_disease_project(
                                project_id=project_id,
                                mass_tort_id=mass_tort['id'],
                                disease_id=mass_tort_disease['disease']['id']
                            )

                            if existing_project:
                                # Update existing project
                                update_disease_project_status(
                                    project_id=project_id,
                                    mass_tort_id=mass_tort['id'],
                                    disease_id=mass_tort_disease['disease']['id'],
                                    status='COMPLETED'
                                )
                                logger.info(
                                    "[app] Updated disease project: %s",
                                    existing_project['id']
                                )
                            else:
                                # Create new project
                                new_project = create_disease_project(
                                    name=mass_tort_disease['disease']['name'],
                                    project_id=project_id,
                                    disease_id=mass_tort_disease['disease']['id'],
                                    mass_tort_id=mass_tort['id'],
                                    user_id=user_id,
                                    status='COMPLETED',
                                    confidence=0.9,
                                    match_count=0,
                                )
                                logger.info(
                                    "[app] Created new disease project: %s",
                                    new_project['id']
                                )
                        except Exception as e:
                            logger.error(
                                "[app] Error processing disease '%s' for mass tort '%s': %s",
                                disease_result.get('disease_name', 'Unknown'), mass_tort_result['mass_tort_name'], str(e)
                            )
                except Exception as e:
                    logger.error(
                        "[app] Error processing mass tort '%s': %s",
                        mass_tort_result.get('mass_tort_name', 'Unknown'), str(e)
                    )
            logger.info("[app] Scanned results processed successfully")
        else:
            logger.error("[app] Error processing scanned results: %s", data.get('message'))
    
    except Exception as e:
        logger.error("[app] Exception in process_scanned_results: %s", str(e))
        raise

def check_files_for_project(project: Dict) -> Tuple[Dict, bool]:
    """
    Check if index.pkl or input folder exists for a single project.
    
    Args:
        project: Dictionary containing project information
        
    Returns:
        Tuple[Dict, bool]: Project dict and boolean indicating if files exist
    """
    user_id = project.get('userId')
    project_id = project.get('projectId')

    if not user_id or not project_id:
        logger.error(f"[app] Missing required userId or projectId for project: {project}")
        return project, False

    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Check for index.pkl
        index_key = f"{user_id}/{project_id}/index.pkl"
        has_index = False
        try:
            s3_client.head_object(Bucket=AWS_UPLOAD_BUCKET_NAME, Key=index_key)
            has_index = True
            logger.info(f"[app] Found index.pkl at s3://{AWS_UPLOAD_BUCKET_NAME}/{index_key}")
        except ClientError:
            logger.info(f"[app] No index.pkl found at s3://{AWS_UPLOAD_BUCKET_NAME}/{index_key}")
        
        # Check for input folder
        input_prefix = f"{user_id}/{project_id}/input/"
        has_input = False
        try:
            response = s3_client.list_objects_v2(
                Bucket=AWS_UPLOAD_BUCKET_NAME,
                Prefix=input_prefix,
                MaxKeys=1
            )
            has_input = response.get('KeyCount', 0) > 0
            if has_input:
                logger.info(f"[app] Found input folder at s3://{AWS_UPLOAD_BUCKET_NAME}/{input_prefix}")
            else:
                logger.info(f"[app] Empty or no input folder at s3://{AWS_UPLOAD_BUCKET_NAME}/{input_prefix}")
        except ClientError as e:
            logger.error(f"[app] Error checking input folder: {str(e)}")
        
        return project, (has_index or has_input)
        
    except Exception as e:
        logger.error(f"[app] Error checking files for project {project_id}: {str(e)}")
        return project, False

def prepare_projects_for_scan(disease_projects: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Check all projects concurrently and separate them into valid and invalid projects.
    
    Args:
        disease_projects: List of disease project dictionaries
        
    Returns:
        Tuple[List[Dict], List[Dict]]: Lists of valid and invalid projects
    """
    if not disease_projects:
        return [], []
    
    MAX_WORKERS = 5
    valid_projects = []
    invalid_projects = []
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all projects for checking
            future_to_project = {
                executor.submit(check_files_for_project, project): project 
                for project in disease_projects
            }
            
            # Process results as they complete
            for future in as_completed(future_to_project):
                project = future_to_project[future]
                try:
                    project, files_exist = future.result()
                    
                    if files_exist:
                        valid_projects.append(project)
                    else:
                        invalid_projects.append(project)
                        
                except Exception as e:
                    logger.error(f"[app] Error processing project {project.get('projectId')}: {str(e)}")
                    invalid_projects.append(project)
        
        return valid_projects, invalid_projects
        
    except Exception as e:
        logger.error(f"[app] Error in concurrent project checking: {str(e)}")
        return [], disease_projects

@app.route('/prepareScan', methods=['GET'])
def prepare_scan():
    """
    Endpoint to check all pending projects and prepare them for scanning.
    Only performs the file existence check and updates project statuses.
    """
    try:
        logger.info("[app] Received prepare scan request")

        # Get all disease projects
        disease_projects = get_all_disease_project_by_status()
        if not disease_projects:
            logger.info("[app] No disease projects found")
            return jsonify({
                'status': 'success',
                'message': 'No disease projects found',
                'validProjects': [],
                'invalidProjects': []
            }), 200

        # Check all projects concurrently
        valid_projects, invalid_projects = prepare_projects_for_scan(disease_projects)
        
        # Update statuses for all projects
        for project in valid_projects:
            update_disease_project_status_by_user(
                project_id=project['projectId'],
                user_id=project['userId'],
                status='READY'  # New status to indicate project is ready for scanning
            )
            
        for project in invalid_projects:
            update_disease_project_status_by_user(
                project_id=project['projectId'],
                user_id=project['userId'],
                status='ERROR'
            )
        
        # Return results
        return jsonify({
            'status': 'success',
            'message': f'Found {len(valid_projects)} valid and {len(invalid_projects)} invalid projects',
            'validProjects': valid_projects,
            'invalidProjects': invalid_projects
        }), 200
        
    except Exception as e:
        error_msg = f"Error preparing scan: {str(e)}"
        logger.error(f"[app] {error_msg}")
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'validProjects': [],
            'invalidProjects': []
        }), 500

# Create a simple rate limiter for the Salesforce API
class RateLimiter:
    def __init__(self, max_calls=5, period=60):
        """
        Initialize a simple rate limiter
        
        Args:
            max_calls (int): Maximum number of calls allowed in the period
            period (int): Period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()
        
    def is_allowed(self):
        """
        Check if a call is allowed under the current rate limit
        
        Returns:
            bool: True if the call is allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            # Remove expired calls
            self.calls = [t for t in self.calls if now - t < self.period]
            # Check if we're under the limit
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            return False

# Initialize rate limiter for Salesforce API
sf_rate_limiter = RateLimiter(max_calls=5, period=60)  # 5 calls per minute

# Helper function to find the Salesforce CLI executable
def find_salesforce_cli():
    """
    Attempt to locate the Salesforce CLI executable
    
    Returns:
        str: Path to the sf executable if found, None otherwise
    """
    try:
        # Check if running on Windows
        is_windows = sys.platform.startswith('win')
        
        # List of potential CLI locations based on platform
        potential_paths = []
        
        if is_windows:
            # Windows-specific paths with proper extensions
            potential_paths = [
                os.path.expanduser("~\\AppData\\Local\\sf\\bin\\sf.cmd"),
                os.path.expanduser("~\\AppData\\Local\\sf\\sf.cmd"),
                os.path.expanduser("~\\AppData\\Local\\sfdx\\bin\\sf.cmd"),
                "C:\\Program Files\\sf\\bin\\sf.cmd",
                os.path.expanduser("~\\AppData\\Roaming\\npm\\sf.cmd"),
                "C:\\Program Files\\sfdx\\bin\\sf.cmd",
                # With .CMD extension (case sensitivity matters on some systems)
                os.path.expanduser("~\\AppData\\Local\\sf\\bin\\sf.CMD"),
                os.path.expanduser("~\\AppData\\Local\\sf\\sf.CMD"),
                "C:\\Program Files\\sf\\bin\\sf.CMD",
                # With .exe extension
                os.path.expanduser("~\\AppData\\Local\\sf\\bin\\sf.exe"),
                "C:\\Program Files\\sf\\bin\\sf.exe",
            ]
        else:
            # Unix-like paths
            potential_paths = [
                os.path.expanduser("~/.npm/bin/sf"),
                "/usr/local/bin/sf",
                "/usr/bin/sf",
                os.path.expanduser("~/.local/bin/sf"),
                shutil.which("sf")
            ]
        
        # Add generic PATH search as fallback
        if not is_windows:
            potential_paths.append(shutil.which("sf"))
        else:
            potential_paths.append(shutil.which("sf.cmd"))
            potential_paths.append(shutil.which("sf.CMD"))
            potential_paths.append(shutil.which("sf"))
        
        # Try to find where sf is installed by checking each path
        for path in potential_paths:
            if path and os.path.exists(path):
                logger.info(f"[app] Found Salesforce CLI at: {path}")
                return path
                
        # Try running platform-specific commands to find the CLI
        try:
            if is_windows:
                result = subprocess.run(['where.exe', 'sf'], 
                                      capture_output=True, 
                                      text=True, 
                                      check=True)
            else:
                result = subprocess.run(['which', 'sf'], 
                                      capture_output=True, 
                                      text=True, 
                                      check=True)
                                      
            paths = result.stdout.strip().split('\n')
            if paths and os.path.exists(paths[0]):
                logger.info(f"[app] Found Salesforce CLI using {('where' if is_windows else 'which')} command: {paths[0]}")
                return paths[0]
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning(f"[app] Could not locate sf using {('where' if is_windows else 'which')} command")
        
        # Final attempt: try executable names without full paths
        if is_windows:
            # For Windows, return just the command name and let subprocess find it
            logger.warning("[app] Could not find full path to Salesforce CLI, defaulting to command name")
            return "sf.cmd"
        else:
            logger.warning("[app] Could not locate Salesforce CLI automatically")
            return "sf"  # Return just the command name as last resort
            
    except Exception as e:
        logger.error(f"[app] Error finding Salesforce CLI: {str(e)}")
        # Return a sensible default even on error
        return "sf.cmd" if sys.platform.startswith('win') else "sf"

@app.route('/nulaw', methods=['POST', 'GET'])
def get_matter_context():
    """
    Endpoint to retrieve context for a Matter from Salesforce.
    
    Expects:
        POST with JSON body containing:
        - matter_id: The Salesforce Matter ID to retrieve context for
        - sf_path: (optional) Path to the Salesforce CLI executable
        - download_files: (optional, defaults to false) Whether to download files
        
    Returns:
        JSON response with matter context or error message
        
    For GET requests (AWS health checks), simply returns a 200 status code
    """
    # For health check requests (GET), return simple 200 response
    if request.method == 'GET':
        logger.info("[app] Health check GET request to /nulaw")
        return jsonify({"status": "healthy"}), 200
        
    try:
        # Get request parameters from JSON body
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({"error": "No request data provided"}), 400
            
        # Extract matter_id from request
        matter_id = request_data.get('matter_id')
        
        if not matter_id:
            return jsonify({"error": "matter_id is required"}), 400
            
        # Get sf_path from request (optional)
        sf_path = request_data.get('sf_path')
        
        # Get download_files from request (optional, defaults to False)
        download_files = request_data.get('download_files', False)

        logger.info(f"[app] Fetching context for Matter ID: {matter_id}, download_files={download_files}")
        
        # Try to use cached context if available (similar to nulawdocs endpoint)
        context = None
        context_cache_path = os.path.join('cache', 'matter_contexts', f"{matter_id}.json")
        
        # Check if we have a cached context and download_files is False
        # (we only use cache for non-download requests to ensure files are fresh when needed)
        if not download_files and os.path.exists(context_cache_path):
            try:
                with open(context_cache_path, 'r') as f:
                    context = json.load(f)
                logger.info(f"[app] Loaded cached matter context from {context_cache_path}")
                
                # Return cached context directly
                result = {
                    'status': 'success',
                    'matter_id': matter_id,
                    'download_files': download_files,
                    'context': context,
                    'source': 'cache'
                }
                
                # Create a project from the matter context
                logger.info(f"[app] Creating project from cached matter context")
                from salesforce_create_new_client import process_nulaw_response_and_create_project
                project_result = process_nulaw_response_and_create_project(result, False, download_files)
                
                if project_result:
                    # Add project information to the response
                    result['project_id'] = project_result.get('id')
                    result['project_name'] = project_result.get('name')
                    result['project_created'] = True
                    
                return jsonify(result), 200
            except Exception as cache_error:
                logger.warning(f"[app] Error loading cached context: {str(cache_error)}")
                # Continue to try getting from Salesforce
                
        # If we need to download files or cache didn't work, try Salesforce
        # Step 1: Try to get .env credentials directly
        token = os.getenv("SALESFORCE_ACCESS_TOKEN")
        instance_url = os.getenv("SALESFORCE_INSTANCE_URL")
        headers = None
        
        if token and instance_url:
            logger.info("[app] Found stored Salesforce credentials in .env")
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        
        # Step 2: If no valid credentials in .env or they're expired, run the refresh token script
        if not headers or not token or not instance_url:
            logger.info("[app] No stored credentials or credentials are incomplete, running token refresh script")
            # Execute script directly with subprocess
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salesforce_refresh_token.py")
            logger.info(f"[app] Executing script: {script_path}")
            
            try:
                # Run the script and capture output
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    check=False  # Don't raise an exception on non-zero return code
                )
                
                # Log the script output for debugging
                logger.info(f"[app] Script stdout: {result.stdout}")
                if result.stderr:
                    logger.warning(f"[app] Script stderr: {result.stderr}")
                    
                # Extract token and instance_url from script output
                if "Token refreshed successfully" in result.stdout:
                    logger.info("[app] Token refreshed successfully by script")
                    # Reload .env to get the updated token
                    load_dotenv(force=True)
                    token = os.getenv("SALESFORCE_ACCESS_TOKEN")
                    instance_url = os.getenv("SALESFORCE_INSTANCE_URL")
                    
                    if token and instance_url:
                        logger.info("[app] Using refreshed token from .env")
                        headers = {
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json"
                        }
                    else:
                        # Try to extract token directly from script output
                        logger.info("[app] Trying to extract token directly from script output")
                        token_match = re.search(r'New token: ([a-zA-Z0-9]{10})\.+', result.stdout)
                        instance_match = re.search(r'Instance URL: (https://[^\s]+)', result.stdout)
                        
                        if token_match and instance_match:
                            # We can't get the full token from the output (it's masked), but we know it worked
                            # so let's rerun the get_salesforce_headers function
                            from salesforce_refresh_token import get_salesforce_headers
                            headers, instance_url = get_salesforce_headers(force_refresh=False)
                            if headers and instance_url:
                                logger.info("[app] Successfully extracted credentials from script result")
            except Exception as script_error:
                logger.error(f"[app] Error running refresh token script: {str(script_error)}")
        
        # Step 3: Process the context request using the standard REST API via SOQL queries
        sf_result = None
        if headers and instance_url:
            logger.info("[app] Using Salesforce credentials")
            try:
                # Use the salesforce_get_all_context_by_matter.py implementation directly
                from salesforce_get_all_context_by_matter import set_salesforce_auth_globals, organize_matter_context
                
                # Set the credentials in the global variables
                set_salesforce_auth_globals(headers, instance_url)
                
                # Get the context using SOQL queries
                context = organize_matter_context(matter_id, download_files)
                
                if isinstance(context, dict) and "error" in context:
                    logger.error(f"[app] Error from organize_matter_context: {context['error']}")
                    return jsonify({
                        'status': 'error',
                        'error': context['error'],
                        'matter_id': matter_id
                    }), 404
                
                # Cache the context for future use
                os.makedirs(os.path.dirname(context_cache_path), exist_ok=True)
                with open(context_cache_path, 'w') as f:
                    json.dump(context, f)
                logger.info(f"[app] Cached matter context to {context_cache_path}")
                
                # Create the result object
                result = {
                    'status': 'success',
                    'matter_id': matter_id,
                    'download_files': download_files,
                    'context': context,
                    'source': 'soql_query'
                }
                
                # Create a project from the matter context
                logger.info(f"[app] Creating project from matter context (download_files={download_files})")
                from salesforce_create_new_client import process_nulaw_response_and_create_project
                project_result = process_nulaw_response_and_create_project(result, False, download_files)
                
                if project_result:
                    # Add project information to the response
                    result['project_id'] = project_result.get('id')
                    result['project_name'] = project_result.get('name')
                    result['project_created'] = True
                
                return jsonify(result), 200
                
            except Exception as process_error:
                logger.error(f"[app] Error processing matter context with credentials: {str(process_error)}")
        
        # If we get here, we couldn't get context from Salesforce - create a minimal context
        logger.warning("[app] Failed to get matter context from Salesforce, creating minimal context")
        
        # Print diagnostics to help debug why authentication failed
        logger.info("[app] Running diagnostics to debug Salesforce authentication issues...")
        try:
            from salesforce_refresh_token import print_diagnostics
            print_diagnostics()
        except Exception as diag_error:
            logger.error(f"[app] Error running diagnostics: {str(diag_error)}")
        
        # Create minimal context with the matter ID
        minimal_context = {
            "matter": {
                "Id": matter_id,
                "Name": f"Matter {matter_id}",
                "RecordTypeId": "012C90430000FAKE",  # Placeholder record type
                "Sharepoint_Folder__c": matter_id
            },
            "matter_team": [],
            "files": [],
            "related_records": {}
        }
        
        result = {
            'status': 'partial_success',
            'matter_id': matter_id,
            'download_files': download_files,
            'context': minimal_context,
            'source': 'minimal',
            'warning': 'Generated from minimal context due to Salesforce authentication issues'
        }
        
        # Create a project from the minimal matter context
        logger.info(f"[app] Creating project from minimal matter context")
        from salesforce_create_new_client import process_nulaw_response_and_create_project
        project_result = process_nulaw_response_and_create_project(result, False, download_files)
        
        if project_result:
            # Add project information to the response
            result['project_id'] = project_result.get('id')
            result['project_name'] = project_result.get('name')
            result['project_created'] = True
            
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"[app] Error retrieving matter context: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/nulawdocs/', methods=['POST'])
@app.route('/nulawdocs', methods=['POST'])  # Add route without trailing slash to prevent 308 redirects
def get_nulaw_documents():
    """
    Endpoint to retrieve documents for a Matter from Salesforce and SharePoint.
    
    Expects:
        POST with JSON body containing:
        - matter_id: The Salesforce Matter ID
        - projectId: The project ID for tracking/logging
        
    Returns:
        JSON array of document objects with format:
        [
            {
                "filename": "filename.pdf",
                "data": "base64EncodedData...",
                "contentType": "application/pdf",
                "folder": "Medical Bills"  // Optional field containing the parent folder name
            },
            ...
        ]
    """
    try:
        # Get request parameters
        request_data = request.get_json()
        
        if not request_data:
            logger.error("[app] No request data provided in /nulawdocs")
            return jsonify({"error": "No request data provided"}), 400
            
        matter_id = request_data.get('matter_id')
        project_id = request_data.get('projectId')
        
        # Validate matter_id
        if not matter_id:
            logger.error("[app] matter_id is required in /nulawdocs")
            return jsonify({"error": "matter_id is required"}), 400
            
        # Basic validation of matter_id format (Salesforce IDs are typically 15 or 18 chars)
        # This helps prevent obvious SQL injection attempts
        if not isinstance(matter_id, str) or len(matter_id) not in [15, 18] or not matter_id.isalnum():
            logger.error(f"[app] Invalid matter_id format: {matter_id}")
            return jsonify({"error": "Invalid matter_id format"}), 400
        
        logger.info(f"[app] Retrieving documents for Matter ID: {matter_id}, Project ID: {project_id or 'Not provided'}")
        
        # Set SF_CLI_PATH environment variable if possible
        sf_path = find_salesforce_cli()
        if sf_path:
            os.environ['SF_CLI_PATH'] = sf_path
            logger.info(f"[app] Using Salesforce CLI path: {sf_path}")
        
        # Initialize empty context - we'll either get it from a file or mock it
        context = None
        
        # First, check if we have a cached context file for this matter
        context_cache_path = os.path.join('cache', 'matter_contexts', f"{matter_id}.json")
        if os.path.exists(context_cache_path):
            try:
                with open(context_cache_path, 'r') as f:
                    context = json.load(f)
                logger.info(f"[app] Loaded cached matter context from {context_cache_path}")
            except Exception as cache_error:
                logger.warning(f"[app] Error loading cached context: {str(cache_error)}")
        
        # If we still don't have context, try to get it from Salesforce
        if not context:
            # ============================
            # Salesforce Authentication - try but don't require success
            # ============================
            sf_initialized = False
            try:
                # Try to initialize Salesforce connection but don't require it to succeed
                sf_initialized = initialize_salesforce()
                if not sf_initialized:
                    logger.warning("[app] Failed to initialize Salesforce connection, attempting token refresh...")
                    token, instance_url = refresh_salesforce_token(update_env=True)
                    
                    if token and instance_url:
                        sf_initialized = initialize_salesforce()
                
                # If we have a Salesforce connection, try to get the matter context
                if sf_initialized:
                    logger.info("[app] Successfully connected to Salesforce, retrieving matter context")
                    try:
                        context = organize_matter_context(matter_id, download_files=False)
                        
                        # Cache the context for future use
                        os.makedirs(os.path.dirname(context_cache_path), exist_ok=True)
                        with open(context_cache_path, 'w') as f:
                            json.dump(context, f)
                        logger.info(f"[app] Cached matter context to {context_cache_path}")
                    except Exception as context_error:
                        logger.error(f"[app] Error retrieving matter context: {str(context_error)}")
            except Exception as sf_error:
                logger.warning(f"[app] Error during Salesforce authentication: {str(sf_error)}")
                # We'll continue even without Salesforce auth
        
        # If we still don't have context, create a minimal mock context with the matter ID
        if not context:
            logger.warning("[app] Creating minimal context for SharePoint access")
            context = {
                "matter": {
                    "Id": matter_id,
                    "Name": f"Matter {matter_id}",
                    # Add minimal fields needed for SharePoint access
                    "Sharepoint_Folder__c": matter_id
                }
            }
            
        # Initialize result documents list
        result_documents = []
        
        # Step 2: Get SharePoint documents recursively
        sharepoint_documents = []
        try:
            logger.info("[app] Retrieving documents from SharePoint")
            from share_point_get_documents import get_sharepoint_documents_for_matter, download_file, get_sharepoint_site_id, get_drive_id
            from share_point_refresh_token import get_bearer_token, get_auth_headers, get_reliable_token
            
            # Force refresh SharePoint token to ensure we have valid authentication
            # Use get_reliable_token instead of get_bearer_token
            sharepoint_token = get_reliable_token(max_attempts=3)
            if not sharepoint_token:
                logger.error("[app] Failed to get SharePoint authentication token after multiple attempts")
                return jsonify({"error": "Failed to authenticate with SharePoint after multiple attempts"}), 401
            else:
                logger.info("[app] Successfully authenticated with SharePoint using reliable token")
            
            # Get site and drive IDs for downloading files
            site_id = get_sharepoint_site_id()
            drive_id = None
            if site_id:
                drive_id = get_drive_id(site_id)
                if not drive_id:
                    logger.warning("[app] Could not get SharePoint drive ID")
            
            # Get documents from SharePoint
            sharepoint_documents = get_sharepoint_documents_for_matter(
                context, 
                download_docs=False,
                recursive=True  # Get documents from all subfolders
            )
            
            if sharepoint_documents:
                logger.info(f"[app] Retrieved {len(sharepoint_documents)} documents from SharePoint")
            else:
                logger.warning("[app] No SharePoint documents found")
                sharepoint_documents = []
                
        except Exception as sp_error:
            logger.error(f"[app] Error retrieving SharePoint documents: {str(sp_error)}")
            sharepoint_documents = []
        
        # Process SharePoint documents
        if isinstance(sharepoint_documents, list):
            for doc in sharepoint_documents:
                if isinstance(doc, dict):
                    # Get file details
                    file_name = doc.get("name", "")
                    if not file_name and "originalName" in doc:
                        file_name = doc["originalName"]
                    if not file_name and "parentReference" in doc and "name" in doc["parentReference"]:
                        file_name = doc["parentReference"]["name"]
                    
                    if not file_name:
                        file_name = "Unnamed Document"
                        logger.warning(f"[app] Document has no name: {doc.get('id', 'unknown ID')}")
                    
                    # Determine content type based on file extension
                    content_type = "application/octet-stream"
                    if "fileType" in doc and doc["fileType"]:
                        content_type = f"application/{doc['fileType']}"
                    elif "file" in doc and "mimeType" in doc["file"]:
                        content_type = doc["file"]["mimeType"]
                    elif file_name:
                        _, file_ext = os.path.splitext(file_name)
                        if file_ext.lower() in ['.pdf']:
                            content_type = "application/pdf"
                        elif file_ext.lower() in ['.doc', '.docx']:
                            content_type = "application/msword"
                        elif file_ext.lower() in ['.xls', '.xlsx']:
                            content_type = "application/vnd.ms-excel"
                    
                    # Try to download using the helper functions first
                    if "id" in doc and drive_id:
                        try:
                            logger.info(f"[app] Downloading SharePoint file: {file_name} using helper function")
                            file_result = download_file(drive_id, doc["id"])
                            if file_result:
                                filename, file_content = file_result
                                
                                # Encode to base64
                                base64_content = base64.b64encode(file_content).decode('utf-8')
                                
                                # Extract folder name for debug and better reporting
                                folder_name = None
                                parent_ref = doc.get("parentReference", {})
                                if parent_ref:
                                    folder_name = parent_ref.get("name")
                                    logger.debug(f"[app] Document {filename} parentReference: {parent_ref}")
                                # Add to result list
                                result_documents.append({
                                    "filename": filename,
                                    "data": base64_content,
                                    "contentType": content_type,
                                    "folder": folder_name
                                })
                                logger.info(f"[app] Added SharePoint document: {filename}")
                                continue  # Skip to next document
                        except Exception as download_error:
                            logger.error(f"[app] Error downloading SharePoint file with helper: {str(download_error)}")
                            # Will fall back to other methods
                    
                    # If helper method failed, try using the download URL
                    download_url = None
                    if "@microsoft.graph.downloadUrl" in doc:
                        download_url = doc.get("@microsoft.graph.downloadUrl")
                    elif "webUrl" in doc:
                        download_url = doc.get("webUrl")
                    
                    if download_url:
                        try:
                            logger.info(f"[app] Downloading SharePoint file from URL: {file_name}")
                            
                            # Use the SharePoint authentication headers
                            headers = get_auth_headers()
                            
                            # Download file with timeout
                            response = requests.get(download_url, headers=headers, timeout=30)
                            response.raise_for_status()
                            file_content = response.content
                            
                            # Encode to base64
                            base64_content = base64.b64encode(file_content).decode('utf-8')
                            
                            # Add to result list
                            result_documents.append({
                                "filename": file_name,
                                "data": base64_content,
                                "contentType": content_type,
                               "folder": folder_name
                            })
                            logger.info(f"[app] Added SharePoint document: {file_name}")
                        except Exception as download_error:
                            logger.error(f"[app] Error downloading SharePoint file from URL: {str(download_error)}")
                            logger.warning(f"[app] Failed to download SharePoint document: {file_name}")
        
        # If needed, we can also get Salesforce documents, but that seems unnecessary based on the error
        # and the fact that the documents are stored in SharePoint
                
        # Log summary
        logger.info(f"[app] Successfully processed {len(result_documents)} documents out of {len(sharepoint_documents)} available")
        
        if not result_documents:
            logger.warning(f"[app] No documents could be downloaded for Matter ID: {matter_id}")
            
        # Check if the total size of the response is large
        total_size = sum(len(doc.get("data", "")) for doc in result_documents)
        if total_size > 10 * 1024 * 1024:  # If total size is greater than 10MB
            logger.info(f"[app] Large response detected ({total_size/1024/1024:.2f}MB), using chunked response")
            # Process larger responses in chunks to avoid content-length mismatch
            response = make_response(json.dumps(result_documents))
            response.headers['Content-Type'] = 'application/json'
            # Disable content-length to use chunked transfer encoding
            response.headers.pop('Content-Length', None)
            return response
        
        # For smaller responses, use standard jsonify
        return jsonify(result_documents)
        
    except Exception as e:
        logger.error(f"[app] Error retrieving documents: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
    
    # Create directories for caching
    os.makedirs('cache', exist_ok=True)
    os.makedirs('cache/documents', exist_ok=True)
    os.makedirs('cache/sharepoint', exist_ok=True)
    os.makedirs('cache/matter_contexts', exist_ok=True)  # Add this line for matter context caching

    # Configure rate limiter for Salesforce requests
    sf_rate_limiter = RateLimiter(max_calls=3, period=60)  # 3 calls per minute
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5001)