import json
import logging
import os
import subprocess
from flask import Flask, request, jsonify
from rag import create_index, query_index  # Ensure these are correctly imported
from flask_cors import CORS
import boto3
import tempfile
import pickle
from botocore.exceptions import ClientError
from llama_index.core import Document  # Import Document from llama_index
from dotenv import load_dotenv
import re
from pypdf import PdfReader

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

@app.route('/query', methods=['POST'])
def query():
    """
    Handle queries to the RAG system.
    Expects a JSON payload with 'query', 'userId', and 'projectId'.
    """
    logger.info('[app] Received /query POST request')
    try:
        data = request.json
        logger.debug(f'[app] Request JSON data: {data}')
        
        # Handle both naming conventions
        query_text = data.get('query_text') or data.get('query', '')
        project_id = data.get('project_id') or data.get('projectId', '')
        user_id = data.get('user_id') or data.get('userId', '')
        
        logger.debug(f'[app] Extracted query_text: "{query_text}", project_id: "{project_id}", user_id: "{user_id}"')
    
        if not query_text:
            logger.error('[app] Query text is missing in the request')
            return jsonify({'status': 'error', 'message': 'Query text is required'}), 400
        logger.info('[app] Query text received and validated')
    
        if not user_id or not project_id:
            logger.error('[app] User ID or Project ID is missing in the request')
            return jsonify({'status': 'error', 'message': 'User ID and Project ID are required'}), 400
        logger.info('[app] User ID and Project ID received and validated')
    
        # Define S3 paths
        s3_prefix = f"{user_id}/{project_id}/input/"
        index_cache_key = f"{user_id}/{project_id}/index.pkl"
        logger.info(f'[app] Looking for documents in S3 prefix: {s3_prefix}')
    
        # List files from S3
        documents = []
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=AWS_UPLOAD_BUCKET_NAME, Prefix=s3_prefix)
        
        # Create a temporary directory to store files
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f'[app] Created temporary directory at {temp_dir}')
            for page in pages:
                if 'Contents' not in page:
                    logger.warning(f'[app] No contents found in page for prefix: {s3_prefix}')
                    continue
                
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    file_extension = os.path.splitext(key)[1].lower()
                    logger.info(f'[app] Found file: {key} with extension: {file_extension}')
                    
                    if file_extension in SUPPORTED_EXTENSIONS:
                        logger.info(f'[app] Processing supported file: {key}')
                        file_name = os.path.basename(key)
                        local_path = os.path.join(temp_dir, sanitize_filename(file_name))

                        # Download file from S3
                        s3_client.download_file(Bucket=AWS_UPLOAD_BUCKET_NAME, Key=key, Filename=local_path)
                        logger.info(f'[app] Downloaded file to {local_path}')

                        # Read file content
                        try:
                            text = read_file_content(local_path)
                            if text:
                                documents.append(Document(text=text))
                                logger.info(f'[app] Added document from file: {key}')
                            else:
                                logger.warning(f'[app] Empty content from file: {key}')
                        except Exception as e:
                            logger.error(f'[app] Failed to process file {key}: {e}')
                    else:
                        logger.info(f'[app] Skipping unsupported file: {key}')

            if not documents:
                logger.error(f'[app] No documents found in S3 for the given project')
                return jsonify({
                    'status': 'error', 
                    'message': 'No documents found to index. Please upload documents first.'
                }), 404

            # Check if index cache exists in S3
            index_exists = False
            try:
                s3_client.head_object(Bucket=AWS_UPLOAD_BUCKET_NAME, Key=index_cache_key)
                index_exists = True
                logger.info('[app] Index cache found in S3')
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logger.info('[app] Index cache not found in S3')
                else:
                    logger.error(f'[app] Error accessing index cache in S3: {str(e)}')
                    return jsonify({'status': 'error', 'message': 'Error accessing index cache in S3'}), 500

            # Download or create index
            index = None
            index_cache_path = os.path.join(temp_dir, 'index.pkl')

            if index_exists:
                # Download index cache
                logger.info(f'[app] Downloading index cache from S3 to {index_cache_path}')
                s3_client.download_file(Bucket=AWS_UPLOAD_BUCKET_NAME, Key=index_cache_key, Filename=index_cache_path)
                logger.info('[app] Index cache downloaded successfully')

                # Load the index from the cache
                logger.info(f'[app] Loading index from {index_cache_path}')
                with open(index_cache_path, 'rb') as f:
                    index = pickle.load(f)
                logger.info('[app] Index loaded from cache successfully')
            else:
                # Create index
                logger.info('[app] Creating new index from documents')
                index = create_index(documents)
                logger.info('[app] Index created successfully')

                # Save index to cache and upload to S3
                with open(index_cache_path, 'wb') as f:
                    pickle.dump(index, f)
                logger.info('[app] Index saved locally to cache')

                # Upload index cache to S3
                logger.info(f'[app] Uploading index cache to S3 at {index_cache_key}')
                s3_client.upload_file(Filename=index_cache_path, Bucket=AWS_UPLOAD_BUCKET_NAME, Key=index_cache_key)
                logger.info('[app] Index cache uploaded to S3 successfully')

            # Query the index
            logger.info('[app] Querying the index')
            response_text = query_index(index, query_text)
            logger.info('[app] Query processed successfully')

            # Return the response
            return jsonify({'status': 'success', 'response': response_text}), 200

    except Exception as e:
        logger.exception(f'[app] Unexpected error occurred while processing query: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

def read_file_content(file_path):
    """
    Reads file content based on file type.
    Returns the text content of the file.
    """
    file_lower = file_path.lower()
    try:
        if file_lower.endswith('.pdf'):
            # Handle PDF files
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text() + '\n'
                return text
        elif file_lower.endswith('.json'):
            # Handle JSON files
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Convert JSON to string representation
                if isinstance(data, dict):
                    return ' '.join(str(v) for v in data.values() if v)
                elif isinstance(data, list):
                    return ' '.join(str(item) for item in data if item)
                else:
                    return str(data)
        else:
            # Handle text files
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    except UnicodeDecodeError:
        # Try different encoding if utf-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            logger.error(f"[app] Failed to read file with latin-1 encoding: {e}")
            raise
    except Exception as e:
        logger.error(f"[app] Failed to read file: {e}")
        raise

def sanitize_filename(filename):
    """
    Sanitizes the filename to remove or replace characters not allowed in file systems.
    """
    # Replace invalid characters with an underscore
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

if __name__ == "__main__":
    # Run the Flask app on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)