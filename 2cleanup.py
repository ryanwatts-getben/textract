import json
import os
import re
import logging
import boto3
import fitz
from collections import defaultdict
from botocore.exceptions import ClientError
import sys
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize AWS clients
s3_client = boto3.client('s3')

def extract_first_date_from_content(content):
    """
    Extract the first date from the content using regex.
    Standardizes date to YYYY-MM-DD format.
    """
    date_patterns = [
        r'"Date"\s*:\s*"(\d{2}-\d{2}-\d{4})"',  # Matches "dd-mm-yyyy"
        r'"Date"\s*:\s*"(\d{4}-\d{2}-\d{2})"',  # Matches "yyyy-mm-dd"
        r'"Date"\s*:\s*"(\d{2}/\d{2}/\d{4})"',  # Matches "dd/mm/yyyy"
        r'"Date"\s*:\s*"(\d{2}/\d{2}/\d{2})"',  # Matches "mm/dd/yy"
        r'"Date"\s*:\s*"(\d{2}-\d{2}-\d{2})"',  # Matches "mm-dd-yy"
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, content)
        if match:
            date_str = match.group(1)
            
            # Standardize various formats to YYYY-MM-DD
            if '/' in date_str:
                # Convert slashes to dashes
                date_str = date_str.replace('/', '-')
            
            date_parts = date_str.split('-')
            
            if len(date_parts[0]) == 4:
                # Date is already in yyyy-mm-dd format
                standardized_date = date_str
            elif len(date_parts[2]) == 2:
                # Handle dd-mm-yy, convert yy to yyyy (assuming 20yy)
                date_parts[2] = '20' + date_parts[2]
                standardized_date = f"{date_parts[2]}-{date_parts[0]}-{date_parts[1]}"
            else:
                # Convert dd-mm-yyyy to yyyy-mm-dd
                standardized_date = f"{date_parts[2]}-{date_parts[0]}-{date_parts[1]}"
                
            return standardized_date
    
    return None  # No date found

def clean_and_extract_json(content):
    """
    Remove non-JSON text from the content and attempt to extract valid JSON.
    """
    # Attempt to find the first '{' and last '}' to extract JSON content
    start_index = content.find('{')
    end_index = content.rfind('}')
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None  # Cannot find valid JSON delimiters
    json_str = content[start_index:end_index+1]

    # Attempt to parse the extracted string as JSON
    try:
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        # Attempt common fixes
        # Remove control characters and retry
        json_str_cleaned = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        try:
            parsed_json = json.loads(json_str_cleaned)
            return parsed_json
        except json.JSONDecodeError:
            return None  # Parsing failed after attempts

def process_files(bucket_name, input_prefix):
    contents_by_date = defaultdict(list)
    all_files = []
    continuation_token = None

    # Collect all file keys
    while True:
        try:
            if continuation_token:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name, 
                    Prefix=input_prefix,
                    ContinuationToken=continuation_token
                )
            else:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name, 
                    Prefix=input_prefix
                )
        except ClientError as e:
            logger.error(f"Error listing objects in bucket {bucket_name}: {str(e)}")
            return None

        if 'Contents' in response:
            all_files.extend([obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.txt')])

        if 'NextContinuationToken' in response:
            continuation_token = response['NextContinuationToken']
        else:
            break

    if not all_files:
        logger.warning(f"No '.txt' files found in {input_prefix}")
        return None

    logger.info(f"Total '.txt' files found: {len(all_files)}")

    # Process collected files
    for file_path in all_files:
        logger.info(f"Processing file: {file_path}")
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
            content = response['Body'].read().decode('utf-8')
            date = extract_first_date_from_content(content)
            if not date:
                logger.warning(f"No date found in {file_path}")
                continue
            # Clean content and extract JSON
            parsed_json = clean_and_extract_json(content)
            if parsed_json is not None:
                contents_by_date[date].append(parsed_json)
            else:
                logger.error(f"Failed to parse JSON from file: {file_path}")
        except ClientError as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")

    return contents_by_date

def get_pdf_page_count(bucket_name, pdf_key):
    response = s3_client.get_object(Bucket=bucket_name, Key=pdf_key)
    pdf_bytes = response['Body'].read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(pdf_document)
    pdf_document.close()
    return page_count

def count_files_in_s3_prefix(bucket_name, prefix):
    file_count = 0
    continuation_token = None
    
    while True:
        list_params = {
            'Bucket': bucket_name,
            'Prefix': prefix,
            'MaxKeys': 1000
        }
        
        if continuation_token:
            list_params['ContinuationToken'] = continuation_token
        
        response = s3_client.list_objects_v2(**list_params)
        
        file_count += len(response.get('Contents', []))
        
        if not response.get('IsTruncated'):
            break
        
        continuation_token = response.get('NextContinuationToken')
    
    return file_count

def check_output_exists(bucket_name, output_key):
    try:
        s3_client.head_object(Bucket=bucket_name, Key=output_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise

def main(input_json):
    try:
        data = json.loads(input_json)
        bucket_name = data['bucket']
        input_path = data['file_path']
        user_id = data['user_id']
        case_id = data['case_id']
        file_name = data['file_name']

        input_prefix = f"{user_id}/{case_id}/records/{file_name}/clean/"
        output_prefix = f"{user_id}/{case_id}/records/{file_name}/combine/"

        # Check if all input files have been processed
        pdf_key = f"{user_id}/{case_id}/records/{file_name}.pdf"
        total_pages = get_pdf_page_count(bucket_name, pdf_key)
        input_file_count = count_files_in_s3_prefix(bucket_name, input_prefix)
        
        if input_file_count < total_pages:
            logger.warning(f"Not all pages have been processed. Expected {total_pages}, found {input_file_count}.")
            return

        logger.info(f"Processing files in: {input_prefix}")

        contents_by_date = process_files(bucket_name, input_prefix)

        if not contents_by_date:
            logger.warning("No contents to process after parsing files.")
            return

        # Output a JSON file for each date
        for date, records in contents_by_date.items():
            output_key = f"{output_prefix}records_{date}.json"
            
            # Check if output file already exists
            if check_output_exists(bucket_name, output_key):
                logger.info(f"Output file {output_key} already exists. Skipping.")
                continue
            
            try:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=output_key,
                    Body=json.dumps(records, indent=2),
                    ContentType='application/json'
                )
                logger.info(f"Saved combined records to {output_key}")
            except ClientError as e:
                logger.error(f"Error writing combined records to {output_key}: {str(e)}")

        # Prepare input for the next step (e.g., 3details.py)
        next_step_input = json.dumps({
            'file_name': file_name,
            'file_path': output_prefix,
            'bucket': bucket_name,
            'case_id': case_id,
            'user_id': user_id,
        })
        
        # Trigger the next step
        logger.info("Triggering 3details.py")
        subprocess.run(['python', '3details.py', next_step_input], check=True)

        logger.info("Processing completed successfully")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON input: {str(e)}")
    except Exception as e:
        logger.error(f"An error occurred while processing the records: {str(e)}")
if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python 2cleanup.py '<json_input>'")
        sys.exit(1)
    
    json_input = sys.argv[1]
    main(json_input)