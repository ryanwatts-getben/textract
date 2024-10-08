import requests
import json
import os
import boto3
from dotenv import load_dotenv
import concurrent.futures
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import tempfile
import fitz  # PyMuPDF
import logging
from botocore.config import Config
import threading

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure boto3 with a larger connection pool
boto3_config = Config(
    retries={'max_attempts': 10, 'mode': 'adaptive'},
    max_pool_connections=50
)

# Initialize AWS Textract client
textract_client = boto3.client(
    'textract',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION'),
    config=boto3_config
)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION'),
    config=boto3_config
)

# Create a semaphore to limit concurrent Textract calls
textract_semaphore = threading.Semaphore(20)  # Adjust this value based on your Textract limits

def create_session_with_retries():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1,
                    status_forcelist=[500, 502, 503, 504],
                    allowed_methods=["GET", "POST"])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def authenticate():
    """Authenticate with the SmartAdvocate API and return the token."""
    url = "https://app.smartadvocate.com/CaseSyncAPI/Users/authenticate"
    payload = json.dumps({
        "Username": os.getenv('SA_USERNAME'),
        "Password": os.getenv('SA_PASSWORD')
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    response.raise_for_status()
    return response.json()['token']

def filteredDocuments():
    """Fetch and filter documents from SmartAdvocate API."""
    url = "https://app.smartadvocate.com/CaseSyncAPI/case/documents/byDatePaged"
    token = authenticate()
    payload = json.dumps({
        "modifiedFromDateTime": "2024-01-01T00:00:01Z",
        "modifiedToDateTime": "2024-10-04T14:15:22Z",
        "caseNumbers": [],
        "pageRequest": {
            "currentPage": 0,
            "pageSize": 5000,
            "filterExpression": {},
            "sorts": [
                {
                    "propertyPath": "string",
                    "sortDirection": "Ascending"
                }
            ]
        }
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': token  # Note: Ensure the token is included correctly
    }

    response = requests.post(url, headers=headers, data=payload)
    response.raise_for_status()
    response_data = response.json()
    records = response_data if isinstance(response_data, list) else response_data.get('records', [])
    
    # Check if records is a list
    if not isinstance(records, list):
        logger.error("Received unexpected data structure from API. Expected a list of records.")
        return []

    # Filter records with documentFolder = "Medical Records"
    medical_records = [
        record for record in records
        if record.get('documentFolder') == "Medical Records"
    ]
    
    return medical_records

def download_pdf_content(document_id, token):
    """Download PDF content from the SmartAdvocate API."""
    url = f"https://app.smartadvocate.com/CaseSyncAPI/case/document/{document_id}/content"
    headers = {
        'Authorization': token
    }
    session = create_session_with_retries()
    response = session.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.content

def process_document(record, token):
    """Process a single document: download, extract text, and save to S3."""
    case_id = record.get('caseID')
    document_id = record.get('documentID')
    if not case_id or not document_id:
        logger.warning("Record missing caseID or documentID.")
        return

    s3_bucket_name = os.getenv('AWS_LOVELY_BUCKET')
    combined_text_key = f"{case_id}/{document_id}_full.txt"

    # Check if the combined output file already exists in S3
    try:
        s3_client.head_object(Bucket=s3_bucket_name, Key=combined_text_key)
        logger.info(f"Combined output file {combined_text_key} already exists in S3. Skipping processing.")
        return
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] != '404':
            logger.error(f"Error checking if file exists in S3: {str(e)}")
            return

    try:
        # Download PDF content
        pdf_content = download_pdf_content(document_id, token)

        # Create a temporary file to store the PDF content
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(pdf_content)

        # Open the PDF with fitz
        with fitz.open(temp_filename) as pdf_document:
            combined_text = ''
            for page_number in range(len(pdf_document)):
                page = pdf_document.load_page(page_number)
                pix = page.get_pixmap()
                image_data = pix.tobytes("png")  # Get image data in bytes

                # Use semaphore to limit concurrent Textract calls
                with textract_semaphore:
                    # Send the image to AWS Textract with exponential backoff
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            response = textract_client.detect_document_text(
                                Document={'Bytes': image_data}
                            )
                            break
                        except textract_client.exceptions.ThrottlingException:
                            if attempt == max_retries - 1:
                                raise
                            time.sleep(2 ** attempt)

                # Extract text from the response
                page_text = ''
                for item in response['Blocks']:
                    if item['BlockType'] == 'LINE':
                        page_text += item['Text'] + '\n'

                # Save the extracted text for this page
                page_text_key = f"{case_id}/{document_id}_page_{page_number}.txt"
                s3_client.put_object(Bucket=s3_bucket_name, Key=page_text_key, Body=page_text.encode('utf-8'))
                logger.info(f"Extracted text for page {page_number} saved to S3 at {page_text_key}")

                # Append to combined text
                combined_text += page_text

        # Save the combined text to S3
        s3_client.put_object(Bucket=s3_bucket_name, Key=combined_text_key, Body=combined_text.encode('utf-8'))
        logger.info(f"Combined extracted text saved to S3 at {combined_text_key}")

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)  # Clean up the temporary file

def main():
    """Main function to process documents."""
    try:
        medical_records = filteredDocuments()
        if not medical_records:
            logger.error("No medical records to process.")
            return
        max_documents_to_process = int(os.getenv('MAX_DOCUMENTS', 1000))
        max_workers = int(os.getenv('MAX_WORKERS', 50))  # Reduced from 100 to 50
        token = authenticate()

        # Limit the number of documents to process
        medical_records = medical_records[:max_documents_to_process]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for record in medical_records:
                futures.append(executor.submit(process_document, record, token))

            # Wait for all futures to complete and check for exceptions
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in processing: {str(e)}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()