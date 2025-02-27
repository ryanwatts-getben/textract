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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
import shutil
from pypdf import PdfReader
import csv
import torch
from typing import Dict, List
# Third-party imports
from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

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
from salesforce_get_all_context_by_matter import initialize_salesforce, organize_matter_context

# Add import for Salesforce new client script
from salesforce_create_new_client import process_nulaw_response_and_create_project

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
    Attempt to locate the Salesforce CLI executable on Windows
    
    Returns:
        str: Path to the sf.cmd file if found, None otherwise
    """
    try:
        # Common locations for Salesforce CLI on Windows
        potential_paths = [
            os.path.expanduser("~\\AppData\\Local\\sf\\sf.cmd"),
            os.path.expanduser("~\\AppData\\Local\\sfdx\\bin\\sf.cmd"),
            "C:\\Program Files\\sfdx\\bin\\sf.cmd",
            # Search in PATH
            shutil.which("sf"),
            shutil.which("sf.cmd"),
        ]
        
        # Try to find where sf is installed
        for path in potential_paths:
            if path and os.path.exists(path):
                logger.info(f"[app] Found Salesforce CLI at: {path}")
                return path
                
        # Try running 'where sf' command on Windows
        try:
            result = subprocess.run(['where.exe', 'sf'], 
                                   capture_output=True, 
                                   text=True, 
                                   check=True)
            paths = result.stdout.strip().split('\n')
            if paths and os.path.exists(paths[0]):
                logger.info(f"[app] Found Salesforce CLI using 'where' command: {paths[0]}")
                return paths[0]
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("[app] Could not locate sf using 'where' command")
            
        logger.warning("[app] Could not locate Salesforce CLI automatically")
        return None
        
    except Exception as e:
        logger.error(f"[app] Error finding Salesforce CLI: {str(e)}")
        return None

# Add new endpoint for Salesforce Matter context with JSON request body
@app.route('/nulaw', methods=['POST'])
def get_matter_context():
    """
    Endpoint to fetch Salesforce Matter context by MATTER_ID and create a project
    
    Expected request format:
    {
        "matter_id": "a0OUR000004DwOr2AK",
        "sf_path": "C:\\path\\to\\sf.cmd" (optional),
        "download_files": true (optional, defaults to true)
    }
    """
    try:
        logger.info("[app] Received Salesforce Matter context request (POST)")
        
        # Get request data with enhanced error handling
        try:
            if not request.is_json:
                logger.error("[app] Request is not JSON format")
                return jsonify({
                    'error': 'Request must be in JSON format with Content-Type: application/json'
                }), 400
                
            request_data = request.get_json()
            logger.info(f"[app] Request data: {request_data}")
            
            if not request_data:
                logger.error("[app] Empty request body")
                return jsonify({
                    'error': 'Request body cannot be empty'
                }), 400
                
            if 'matter_id' not in request_data:
                logger.error("[app] Missing matter_id in request")
                return jsonify({
                    'error': 'matter_id is required in the request body'
                }), 400
                
        except Exception as req_error:
            logger.error(f"[app] Error parsing request: {str(req_error)}")
            return jsonify({
                'error': f'Error parsing request: {str(req_error)}'
            }), 400
            
        matter_id = request_data['matter_id']
        sf_path = request_data.get('sf_path')  # Optional parameter
        download_files = request_data.get('download_files', True)  # Optional parameter, defaults to True
        
        # Process the matter context
        result = _process_matter_context(matter_id, sf_path, download_files)
        
        # If response is successful, create a project
        if isinstance(result, tuple) and len(result) == 2 and result[1] == 200:
            try:
                response_data = result[0].get_json()
                logger.info(f"[app] Creating project from matter context (download_files={download_files})")
                
                # Process the nulaw response and create a project
                project_result = process_nulaw_response_and_create_project(response_data, False, download_files)
                
                if project_result and project_result.get('status') == 'success':
                    # Add project creation result to the response
                    response_data['project_creation'] = project_result
                    return jsonify(response_data), 200
                else:
                    # Add project creation error to the response
                    response_data['project_creation'] = {
                        'status': 'error',
                        'message': 'Failed to create project from matter context'
                    }
                    return jsonify(response_data), 200
            except Exception as proj_error:
                logger.error(f"[app] Error creating project from matter context: {str(proj_error)}")
                # Still return the original result even if project creation fails
                return result
        
        # Return the original result if there was an error in processing matter context
        return result
        
    except Exception as e:
        logger.error(f"[app] Error processing Salesforce Matter context request: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

# Update the GET endpoint for /nulaw
@app.route('/nulaw/<matter_id>', methods=['GET'])
def get_matter_context_by_url(matter_id):
    """
    Endpoint to fetch Salesforce Matter context by MATTER_ID via URL parameter and create a project
    
    Example: GET /nulaw/a0OUR000004DwOr2AK?sf_path=C:\\path\\to\\sf.cmd&download_files=false
    """
    try:
        logger.info(f"[app] Received Salesforce Matter context request (GET) for ID: {matter_id}")
        
        if not matter_id:
            logger.error("[app] Missing matter_id in URL")
            return jsonify({
                'error': 'matter_id is required in the URL'
            }), 400
        
        # Get sf_path from query parameters if provided
        sf_path = request.args.get('sf_path')
        
        # Get download_files from query parameters if provided (defaults to True)
        download_files_param = request.args.get('download_files', 'true').lower()
        download_files = download_files_param not in ['false', '0', 'no']
        
        # Process the matter context
        result = _process_matter_context(matter_id, sf_path, download_files)
        
        # If response is successful, create a project
        if isinstance(result, tuple) and len(result) == 2 and result[1] == 200:
            try:
                response_data = result[0].get_json()
                logger.info(f"[app] Creating project from matter context (download_files={download_files})")
                
                # Process the nulaw response and create a project
                project_result = process_nulaw_response_and_create_project(response_data, False, download_files)
                
                if project_result and project_result.get('status') == 'success':
                    # Add project creation result to the response
                    response_data['project_creation'] = project_result
                    return jsonify(response_data), 200
                else:
                    # Add project creation error to the response
                    response_data['project_creation'] = {
                        'status': 'error',
                        'message': 'Failed to create project from matter context'
                    }
                    return jsonify(response_data), 200
            except Exception as proj_error:
                logger.error(f"[app] Error creating project from matter context: {str(proj_error)}")
                # Still return the original result even if project creation fails
                return result
        
        # Return the original result if there was an error in processing matter context
        return result
            
    except Exception as e:
        logger.error(f"[app] Error processing Salesforce Matter context request: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

# Helper function for processing matter context to avoid code duplication
def _process_matter_context(matter_id, sf_path=None, download_files=True):
    """
    Process a matter context request for the given matter_id
    
    Args:
        matter_id (str): The Salesforce Matter ID to retrieve context for
        sf_path (str, optional): Path to the Salesforce CLI executable
        download_files (bool, optional): Whether to download files
        
    Returns:
        flask.Response: JSON response with matter context or error
    """
    # Apply rate limiting
    if not sf_rate_limiter.is_allowed():
        logger.warning(f"[app] Rate limit exceeded for Matter ID: {matter_id}")
        return jsonify({
            'error': 'Rate limit exceeded. Please try again later.',
            'status': 'rate_limited'
        }), 429
    
    logger.info(f"[app] Fetching context for Matter ID: {matter_id}")
    
    # Set SF_CLI_PATH environment variable if provided
    if sf_path:
        logger.info(f"[app] Using provided Salesforce CLI path: {sf_path}")
        os.environ['SF_CLI_PATH'] = sf_path
    else:
        # Try to find Salesforce CLI path automatically
        auto_sf_path = find_salesforce_cli()
        if auto_sf_path:
            logger.info(f"[app] Automatically found Salesforce CLI at: {auto_sf_path}")
            os.environ['SF_CLI_PATH'] = auto_sf_path
    
    # Initialize Salesforce connection
    if not initialize_salesforce():
        logger.error("[app] Failed to initialize Salesforce connection")
        return jsonify({
            'error': 'Failed to connect to Salesforce. Please check your credentials or provide the correct sf_path.'
        }), 500
    
    # Collect and organize context for the Matter
    context = organize_matter_context(matter_id)
    
    # Return the context as JSON response
    return jsonify({
        'status': 'success',
        'matter_id': matter_id,
        'download_files': download_files,
        'context': context
    }), 200

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