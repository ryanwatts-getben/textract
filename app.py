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
        # Initialize Claude
        llm = Anthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )

        # Create query engine with Claude
        query_engine = index.as_query_engine(
            llm=llm,
        )
        logger.debug(f'[rag] Query Text: {query_text}')
        response = query_engine.query(query_text)
        logger.debug(f'[rag] Response Text: {response}')
        return response.response
    except Exception as e:
        logger.error(f"[app] Error querying index: {str(e)}")
        raise

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        force_refresh = data.get('force_refresh', False)  # Default to False now
        
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

        # Define S3 paths
        s3_prefix = f"{user_id}/{project_id}/input/"
        index_cache_key = f"{user_id}/{project_id}/index.pkl"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_cache_path = os.path.join(temp_dir, 'index.pkl')
            
            # Check if index cache exists in S3
            try:
                if not force_refresh:
                    logger.info('[app] Attempting to load cached index from S3')
                    s3_client.download_file(
                        Bucket=AWS_UPLOAD_BUCKET_NAME,
                        Key=index_cache_key,
                        Filename=index_cache_path
                    )
                    
                    with open(index_cache_path, 'rb') as f:
                        index = pickle.load(f)
                    
                    # Verify the index has nodes
                    if len(index.docstore.docs) > 0:
                        logger.info(f'[app] Successfully loaded cached index with {len(index.docstore.docs)} nodes')
                        response_text = query_index(index, query_text)
                        return jsonify({'status': 'success', 'response': response_text}), 200
                    else:
                        logger.warning('[app] Cached index has no nodes, will create new index')
                        force_refresh = True
                
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.info('[app] No cached index found, will create new index')
                else:
                    logger.error(f'[app] Error accessing S3: {str(e)}')
                force_refresh = True
            except Exception as e:
                logger.error(f'[app] Error loading cached index: {str(e)}')
                force_refresh = True

            # If we need to create a new index
            if force_refresh:
                logger.info('[app] Creating new index')
                # Process documents and create index
                documents = []
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=AWS_UPLOAD_BUCKET_NAME, Prefix=s3_prefix)
                
                for page in pages:
                    if 'Contents' not in page:
                        continue
                        
                    for obj in page.get('Contents', []):
                        key = obj.get('Key')
                        if not key:
                            continue
                            
                        file_extension = os.path.splitext(key)[1].lower()
                        if file_extension not in SUPPORTED_EXTENSIONS:
                            continue
                            
                        try:
                            file_name = os.path.basename(key)
                            local_path = os.path.join(temp_dir, sanitize_filename(file_name))
                            
                            # Download file from S3
                            s3_client.download_file(
                                Bucket=AWS_UPLOAD_BUCKET_NAME,
                                Key=key,
                                Filename=local_path
                            )
                            
                            # Read and validate content
                            text = read_file_content(local_path)
                            if text:  # Only add if we got valid content
                                doc = Document(
                                    text=text,
                                    metadata={
                                        "source": key,
                                        "file_name": file_name,
                                    }
                                )
                                documents.append(doc)
                                logger.info(f'[app] Added document from {file_name} with {len(text)} characters')
                            
                        except Exception as e:
                            logger.error(f'[app] Error processing file {key}: {e}')
                            continue

                if not documents:
                    return jsonify({
                        'status': 'error',
                        'message': 'No valid documents found to index. Please check your uploaded files.'
                    }), 404

                # Create new index
                logger.info(f'[app] Creating new index with {len(documents)} documents')
                index = create_index(documents)
                
                # Save index to cache
                with open(index_cache_path, 'wb') as f:
                    pickle.dump(index, f)
                
                # Upload to S3
                s3_client.upload_file(
                    Filename=index_cache_path,
                    Bucket=AWS_UPLOAD_BUCKET_NAME,
                    Key=index_cache_key
                )

            # Query the index
            response_text = query_index(index, query_text)
            if not response_text or response_text == "Empty Response":
                return jsonify({
                    'status': 'error',
                    'message': 'No relevant information found for your query.'
                }), 404

            return jsonify({'status': 'success', 'response': response_text}), 200

    except Exception as e:
        logger.exception(f'[app] Error processing query: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

def read_file_content(file_path):
    """Reads and validates file content."""
    try:
        text = ""
        file_lower = file_path.lower()
        
        if file_lower.endswith('.pdf'):
            logger.info(f'[app] Reading PDF file: {file_path}')
            try:
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
                            else:
                                logger.warning(f'[app] Empty text extracted from page {i+1}/{num_pages}')
                        except Exception as e:
                            logger.error(f'[app] Error extracting text from page {i+1}: {str(e)}')
                    
                    # If no text was extracted, try reading raw bytes
                    if not text.strip():
                        logger.info('[app] Attempting alternative PDF extraction method')
                        for i, page in enumerate(pdf_reader.pages):
                            try:
                                # Get raw bytes and decode
                                content = page.get_contents()
                                if content:
                                    decoded = content.get_data().decode('utf-8', errors='ignore')
                                    text += decoded.strip() + '\n'
                            except Exception as e:
                                logger.error(f'[app] Error with alternative extraction on page {i+1}: {str(e)}')
                                
            except Exception as e:
                logger.error(f'[app] Error reading PDF {file_path}: {str(e)}')
                return None
                
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

        # Validate content
        text = text.strip()
        if not text:
            logger.warning(f'[app] No valid content extracted from {file_path}')
            return None
            
        # Clean up the text
        text = ' '.join(text.split())  # Remove extra whitespace
        text = text.replace('\x00', '')  # Remove null bytes
        
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