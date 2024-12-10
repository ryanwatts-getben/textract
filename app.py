import json
import logging
import os
import subprocess
from flask import Flask, request, jsonify
from rag import create_index  # Ensure these are correctly imported
from flask_cors import CORS
import boto3
import tempfile
import pickle
from botocore.exceptions import ClientError
from llama_index.core import Document  # Import Document from llama_index
from dotenv import load_dotenv
import re
from pypdf import PdfReader
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.anthropic import Anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import pdfplumber  # type: ignore
from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
import io
import time

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://dev.everycase.ai",
    "http://localhost:5001"
]

# Update CORS configuration
CORS(app, resources={
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
})

AWS_UPLOAD_BUCKET_NAME = "generate-input-f5bef08a-9228-4f8c-a550-56d842b94088"

s3_client = boto3.client('s3')  # Initialize S3 client

# Supported file types
SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.json'}

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
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )

        # Create query engine with detailed configurations
        query_engine = index.as_query_engine(
            llm=llm,
            
            # Retrieval Configuration
            similarity_top_k=5,  # Number of chunks to retrieve
            node_postprocessors=[
                # Add any custom node processors here
                # Example: SimilarityPostprocessor(similarity_cutoff=0.7)
            ],
            
            # Response Generation
            response_mode="compact",  # How to synthesize the response
            response_kwargs={
                "max_tokens": 1024,  # Max response length
                "temperature": 0.7,   # Response creativity (0-1)
                "top_p": 0.9,        # Nucleus sampling parameter
                "top_k": 50,         # Top-k sampling parameter
            },
            
            # Prompt Configuration
            text_qa_template=None,  # Custom prompt template if needed
            system_prompt="""You are a helpful assistant analyzing medical documents. 
            Your role is to:
            1. Provide clear, accurate responses based on the context provided
            2. Maintain medical terminology accuracy
            3. Indicate uncertainty when information is unclear
            4. Focus on relevant medical details
            5. Avoid speculation beyond the provided context""",
            
            # Filtering and Refinement
            structured_answer_filtering=True,
            filters=[
                # Add any custom filters here
                # Example: MetadataFilter(key="document_type", value="medical_record")
            ],
            
            # Debug Options
            verbose=True,
            streaming=False,
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

def process_single_document(args):
    """Process a single document for parallel execution."""
    key, local_path, temp_dir = args
    try:
        file_name = os.path.basename(key)
        local_file_path = os.path.join(temp_dir, sanitize_filename(file_name))
        
        # Download file from S3
        s3_client.download_file(
            Bucket=AWS_UPLOAD_BUCKET_NAME,
            Key=key,
            Filename=local_file_path
        )
        
        # Read and validate content
        text = read_file_content(local_file_path)
        if text:
            return Document(
                text=text,
                metadata={
                    "source": key,
                    "file_name": file_name,
                }
            )
        return None
    except Exception as e:
        logger.error(f'[app] Error processing file {key}: {e}')
        return None

def create_or_update_index(user_id: str, project_id: str, force_refresh: bool = False, max_workers=10) -> tuple[VectorStoreIndex, str]:
    """
    Create or update index for the given user and project.
    Returns tuple of (index, status_message)
    """
    timings = {
        'start': time.time(),
        'operations': []
    }
    
    try:
        s3_prefix = f"{user_id}/{project_id}/input/"
        index_cache_key = f"{user_id}/{project_id}/index.pkl"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_cache_path = os.path.join(temp_dir, 'index.pkl')
            
            # Try loading cached index
            if not force_refresh:
                cache_start_time = time.time()
                try:
                    logger.info('[app] Attempting to load cached index from S3')
                    s3_client.download_file(AWS_UPLOAD_BUCKET_NAME, index_cache_key, index_cache_path)
                    
                    with open(index_cache_path, 'rb') as f:
                        index = pickle.load(f)
                    
                    if len(index.docstore.docs) > 0:
                        cache_duration = round(time.time() - cache_start_time, 2)
                        logger.info(f'[app] Successfully loaded cached index in {cache_duration} seconds')
                        return index, "Loaded existing index"
                    
                    logger.warning('[app] Cached index has no nodes, will create new index')
                except (ClientError, Exception) as e:
                    logger.info(f'[app] Cache miss or error, creating new index: {str(e)}')

            # Create new index
            logger.info('[app] Creating new index')
            documents = []
            
            # Start document collection timing
            doc_collection_start = time.time()
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=AWS_UPLOAD_BUCKET_NAME, Prefix=s3_prefix)
            
            # Collect all valid files first
            files_to_process = []
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page.get('Contents', []):
                    key = obj.get('Key')
                    if not key:
                        continue
                        
                    file_extension = os.path.splitext(key)[1].lower()
                    if file_extension in SUPPORTED_EXTENSIONS:
                        files_to_process.append((key, obj, temp_dir))
            
            timings['operations'].append({
                'operation': 'file_collection',
                'duration_seconds': round(time.time() - doc_collection_start, 2)
            })
            
            if not files_to_process:
                raise ValueError('No valid documents found to index')
            
            # Process files in parallel
            doc_processing_start = time.time()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(process_single_document, args): args[0]
                    for args in files_to_process
                }
                
                for future in as_completed(future_to_file):
                    key = future_to_file[future]
                    try:
                        doc = future.result()
                        if doc:
                            documents.append(doc)
                            logger.info(f'[app] Added document from {os.path.basename(key)}')
                    except Exception as e:
                        logger.error(f'[app] Error processing file {key}: {e}')

            timings['operations'].append({
                'operation': 'document_processing',
                'duration_seconds': round(time.time() - doc_processing_start, 2)
            })

            if not documents:
                raise ValueError('No valid documents were successfully processed')

            # Create index
            index_start_time = time.time()
            logger.info(f'[app] Creating new index with {len(documents)} documents')
            index = create_index(documents)
            
            timings['operations'].append({
                'operation': 'index_creation',
                'duration_seconds': round(time.time() - index_start_time, 2)
            })
            
            # Cache the index
            cache_start_time = time.time()
            with open(index_cache_path, 'wb') as f:
                pickle.dump(index, f)
            
            s3_client.upload_file(index_cache_path, AWS_UPLOAD_BUCKET_NAME, index_cache_key)
            
            timings['operations'].append({
                'operation': 'cache_saving',
                'duration_seconds': round(time.time() - cache_start_time, 2)
            })

        # Log timing summary
        total_duration = round(time.time() - timings['start'], 2)
        logger.info(f'[app] Index operation completed in {total_duration} seconds')
        for op in timings['operations']:
            logger.info(f"[app] {op['operation']}: {op['duration_seconds']} seconds")
        
        return index, f"Created new index with {len(documents)} documents in {total_duration} seconds"

    except Exception as e:
        total_duration = round(time.time() - timings['start'], 2)
        logger.error(f'[app] Error after {total_duration} seconds: {str(e)}')
        raise

@app.route('/index', methods=['POST'])
def index():
    """
    Create or update index for the specified user and project.
    Expected payload: {"userId": "user_id", "projectId": "project_id", "force_refresh": boolean}
    """
    try:
        data = request.json
        
        # Handle both naming conventions
        project_id = data.get('project_id') or data.get('projectId', '')
        user_id = data.get('user_id') or data.get('userId', '')
        force_refresh = data.get('force_refresh', True)  # Default to True for /index route
        
        if not all([project_id, user_id]):
            missing = []
            if not project_id: missing.append('project ID')
            if not user_id: missing.append('user ID')
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing)}'
            }), 400

        # Create or update the index
        try:
            _, status_message = create_or_update_index(user_id, project_id, force_refresh)
            return jsonify({
                'status': 'success',
                'message': status_message
            }), 200
            
        except ValueError as ve:
            return jsonify({
                'status': 'error',
                'message': str(ve)
            }), 404
            
    except Exception as e:
        logger.exception(f'[app] Error processing index request: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        force_refresh = data.get('force_refresh', False)  # Default to False for query route
        
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
                'message': f'Missing required fields: {", ".join(missing)}'
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

    # Method 2: pdfplumber
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text.strip() + '\n'
                        logger.info(f'[app] pdfplumber: Extracted page {i+1} with {len(page_text)} characters')
                except Exception as e:
                    logger.warning(f'[app] pdfplumber failed on page {i+1}: {str(e)}')
                    
        if text.strip():
            text_results.append(('pdfplumber', text))
    except Exception as e:
        logger.error(f'[app] pdfplumber extraction failed: {str(e)}')

    # Method 3: pdfminer
    try:
        text = pdfminer_extract_text(file_path)
        if text.strip():
            text_results.append(('pdfminer', text))
            logger.info(f'[app] pdfminer: Extracted {len(text)} characters')
    except Exception as e:
        logger.error(f'[app] pdfminer extraction failed: {str(e)}')

    # Choose the best result
    if text_results:
        # Sort by text length (assuming longer text is better)
        text_results.sort(key=lambda x: len(x[1]), reverse=True)
        method, text = text_results[0]
        logger.info(f'[app] Using {method} result with {len(text)} characters')
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
    lines = [line for line in text.split('\n') if len(line.strip()) > 3]
    text = '\n'.join(lines)
    
    return text if text.strip() else None

def read_file_content(file_path):
    """Reads and validates file content."""
    try:
        file_lower = file_path.lower()
        
        if file_lower.endswith('.pdf'):
            logger.info(f'[app] Reading PDF file: {file_path}')
            text = read_pdf_with_fallbacks(file_path)
            
        elif file_lower.endswith('.json'):
            logger.info(f'[app] Reading JSON file: {file_path}')
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, dict):
                    text = '\n'.join(f"{k}: {v}" for k, v in data.items() if v)
                elif isinstance(data, list):
                    text = '\n'.join(str(item) for item in data if item)
                else:
                    text = str(data)
                    
        else:  # Text files
            logger.info(f'[app] Reading text file: {file_path}')
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

        # Validate and clean the text
        text = clean_text(text)
        
        if not text:
            logger.warning(f'[app] No valid content extracted from {file_path}')
            return None
            
        logger.info(f'[app] Successfully extracted {len(text)} characters from {file_path}')
        return text
        
    except Exception as e:
        logger.error(f"[app] Error reading file {file_path}: {e}")
        return None

def sanitize_filename(filename):
    """
    Sanitizes the filename to remove or replace characters not allowed in file systems.
    """
    # Replace invalid characters with an underscore
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

if __name__ == "__main__":
    # Run the Flask app on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)