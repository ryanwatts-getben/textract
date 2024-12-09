import json
import logging
import os
import subprocess
from flask import Flask, request, jsonify
from rag import create_index, query_index, preprocess_document  # Ensure these are correctly imported
from flask_cors import CORS
import boto3
import tempfile
import pickle
from botocore.exceptions import ClientError
from llama_index.core import Document  # Import Document from llama_index
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import io

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
    logger.info(f"Processing: {message_body}")
    # Construct the command to run the appropriate split script
    command = [
        "python", script_name,
        json.dumps(message_body)
    ]
    try:
        subprocess.run(command, check=True)
        logger.info(f"Successfully processed {file_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {file_path}: {e}")
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
        logger.info(f'Total records: {total_records}, Total bills: {total_bills}')
        
        for accident_report in accident_reports:
            process_file(bucket_name, accident_report['file_path'], 'split_reports.py')
        for record in records:
            process_file(bucket_name, record['file_path'], 'split_records.py')
        for bill in bills:
            process_file(bucket_name, bill['file_path'], 'split_bills.py')
       
        
        logger.info(f"Successfully processed {total_records} records and {total_bills} bills")
        return jsonify({"status": "success", "message": f"Successfully processed {total_records} records and {total_bills} bills"}), 200
    except Exception as e:
        logger.error(f"Error occurred while processing records and bills: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# New route for handling queries
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
    
        # Define both root and input paths
        root_prefix = f"{user_id}/{project_id}/"
        input_prefix = f"{user_id}/{project_id}/input/"
        index_cache_key = f"{user_id}/{project_id}/index.pkl"
        
        logger.info(f'[app] Looking for documents in root prefix: {root_prefix} and input prefix: {input_prefix}')
        
        # List files from both locations
        documents = []
        paginator = s3_client.get_paginator('list_objects_v2')
        
        # Check both paths
        for prefix in [root_prefix, input_prefix]:
            pages = paginator.paginate(Bucket=AWS_UPLOAD_BUCKET_NAME, Prefix=prefix)
            
            file_count = 0
            for page in pages:
                if 'Contents' not in page:
                    logger.warning(f'[app] No contents found in page for prefix: {prefix}')
                    continue
                
                for obj in page.get('Contents', []):
                    file_count += 1
                    key = obj['Key']
                    file_extension = os.path.splitext(key)[1].lower()
                    logger.info(f'[app] Found file: {key} with extension: {file_extension}')
                    
                    if file_extension in SUPPORTED_EXTENSIONS:
                        logger.info(f'[app] Processing supported file: {key}')
                        # ... rest of file processing code ...
                    else:
                        logger.info(f'[app] Skipping unsupported file: {key}')

            logger.info(f'[app] Found {file_count} files in {prefix}')

        if not documents:
            logger.error(f'[app] No documents found in either location')
            return jsonify({
                'status': 'error', 
                'message': 'No documents found to index. Please upload documents first.'
            }), 404

        # Define S3 paths
        s3_prefix = f"{user_id}/{project_id}/"
        index_cache_key = f"{s3_prefix}index.pkl"
        logger.debug(f'[app] Defined S3 prefix: "{s3_prefix}" and index_cache_key: "{index_cache_key}"')
    
        # Create a temporary directory to store files
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f'[app] Created temporary directory at {temp_dir}')
            index_cache_path = os.path.join(temp_dir, 'index.pkl')
            logger.info(f'[app] Created index cache path at {index_cache_path}')

            # Check if local index cache exists and matches the current userId/projectId
            local_cache_path = 'index.pkl'  # Define a path for the local cache
            if os.path.exists(local_cache_path):
                with open(local_cache_path, 'rb') as f:
                    try:
                        local_index = pickle.load(f)
                        if local_index['user_id'] == user_id and local_index['project_id'] == project_id:
                            logger.info('[app] Using local index cache')
                            index = local_index['index']
                        else:
                            logger.info('[app] Local index cache does not match current userId/projectId')
                            index = None
                    except Exception as e:
                        logger.error(f'[app] Error loading local index cache: {str(e)}')
                        index = None
            else:
                index = None

            if index is None:
                # Check if index cache exists in S3
                try:
                    logger.debug(f'[app] Checking existence of index cache in S3 bucket "{AWS_UPLOAD_BUCKET_NAME}" with key "{index_cache_key}"')
                    s3_client.head_object(Bucket=AWS_UPLOAD_BUCKET_NAME, Key=index_cache_key)
                    logger.info('[app] Index cache found in S3')
    
                    # Download index cache
                    index_cache_path = os.path.join(temp_dir, 'index.pkl')
                    logger.debug(f'[app] Downloading index cache from S3 to "{index_cache_path}"')
                    s3_client.download_file(Bucket=AWS_UPLOAD_BUCKET_NAME, Key=index_cache_key, Filename=index_cache_path)
                    logger.info('[app] Index cache downloaded successfully')
    
                    # Load the index from the cache
                    logger.debug(f'[app] Loading index from "{index_cache_path}"')
                    with open(index_cache_path, 'rb') as f:
                        index = pickle.load(f)
                    logger.info('[app] Index loaded from cache successfully')
                
                except ClientError as e:
                    logger.error(f'[app] ClientError encountered when accessing bucket "{AWS_UPLOAD_BUCKET_NAME}": {str(e)}')
                    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.doc', '.docx', '.json'}

                    if e.response['Error']['Code'] == "404":
                        logger.info('[app] Index cache not found in S3, initiating index creation process')
                        # List and download all text files under the user's project directory
                        documents = []
                        paginator = s3_client.get_paginator('list_objects_v2')
                        pages = paginator.paginate(Bucket=AWS_UPLOAD_BUCKET_NAME, Prefix=s3_prefix)
                        logger.info(f'[app] Searching for files in bucket: {AWS_UPLOAD_BUCKET_NAME} with prefix: {s3_prefix}')

                        # Add detailed logging for S3 listing
                        file_count = 0
                        for page in pages:
                            if 'Contents' not in page:
                                logger.warning(f'[app] No contents found in page for prefix: {s3_prefix}')
                                continue
                            
                            for obj in page.get('Contents', []):
                                file_count += 1
                                key = obj['Key']
                                file_extension = os.path.splitext(key)[1].lower()
                                logger.info(f'[app] Found file: {key} with extension: {file_extension}')
                                
                                if file_extension in SUPPORTED_EXTENSIONS:
                                    logger.info(f'[app] Processing supported file: {key}')
                                else:
                                    logger.info(f'[app] Skipping unsupported file: {key}')

                        if file_count == 0:
                            logger.error(f'[app] No files found in bucket {AWS_UPLOAD_BUCKET_NAME} with prefix {s3_prefix}')
                            return jsonify({'status': 'error', 'message': f'No files found in bucket {AWS_UPLOAD_BUCKET_NAME} with prefix {s3_prefix}'}), 404
                        logger.info(f'[app] Retrieved {file_count} files from S3')
    
                        # Define cache paths
                        cache_path = os.path.join(temp_dir, 'index.pkl')
    
                        # Create index
                        logger.debug(f'[app] Creating index from documents {documents} client: {s3_client} bucket: {AWS_UPLOAD_BUCKET_NAME} key: {index_cache_key} temp_dir: {temp_dir}')
                        index = create_index(
                            documents=documents,
                            s3_client=s3_client,
                            bucket_name=AWS_UPLOAD_BUCKET_NAME,
                            index_cache_key=index_cache_key,
                            temp_dir=temp_dir,
                            cache_path=cache_path
                        )
                        logger.info('[app] Index created successfully')
    
                        # Save index to cache and upload to S3
                        logger.debug(f'[app] Saving index to cache at "{index_cache_path}"')
                        with open(index_cache_path, 'wb') as f:
                            pickle.dump(index, f)
                        logger.info('[app] Index serialized and saved to cache successfully')
    
                        logger.debug(f'[app] Uploading index cache to S3 bucket "{AWS_UPLOAD_BUCKET_NAME}" with key "{index_cache_key}"')
                        s3_client.upload_file(Filename=index_cache_path, Bucket=AWS_UPLOAD_BUCKET_NAME, Key=index_cache_key)
                        logger.info('[app] Index cache uploaded to S3 successfully')
    
                    else:
                        logger.error(f'[app] Unexpected error accessing index cache in S3: {str(e)}')
                        return jsonify({'status': 'error', 'message': 'Error accessing index cache in S3'}), 500
    
                # After creating the index, save it to the local cache
                with open(local_cache_path, 'wb') as f:
                    pickle.dump({'user_id': user_id, 'project_id': project_id, 'index': index}, f)
                logger.info('[app] Local index cache updated')

            # Query the index
            logger.debug('[app] Performing query on the index')
            response_text = query_index(index, query_text)
            logger.info('[app] Query processed successfully')
    
            # Return the response
            logger.debug('[app] Preparing JSON response with query results')
            return jsonify({'status': 'success', 'response': response_text}), 200
    
    except Exception as e:
        logger.exception(f'[app] Unexpected error occurred while processing query: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

def sanitize_filename(filename):
    # Replace invalid characters with an underscore
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

if __name__ == "__main__":
    # Run the Flask app on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)