import json
import os
import re
import logging
from datetime import datetime
from collections import defaultdict
import sys
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

# Initialize S3 client
s3_client = boto3.client('s3')

MAX_WORKERS = 10

def extract_first_date_from_content(content):
    date_patterns = [
        r'"Date"\s*:\s*"(\d{2}-\d{2}-\d{4})"',
        r'"Date"\s*:\s*"(\d{4}-\d{2}-\d{2})"',
        r'"Date"\s*:\s*"(\d{2}/\d{2}/\d{4})"',
        r'"Date"\s*:\s*"(\d{2}/\d{2}/\d{2})"',
        r'"Date"\s*:\s*"(\d{2}-\d{2}-\d{2})"',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, content)
        if match:
            date_str = match.group(1)
            
            if '/' in date_str:
                date_str = date_str.replace('/', '-')
            
            date_parts = date_str.split('-')
            
            if len(date_parts[0]) == 4:
                standardized_date = date_str
            elif len(date_parts[2]) == 2:
                date_parts[2] = '20' + date_parts[2]
                standardized_date = f"{date_parts[2]}-{date_parts[0]}-{date_parts[1]}"
            else:
                standardized_date = f"{date_parts[2]}-{date_parts[0]}-{date_parts[1]}"
                
            return standardized_date
    
    return None

def clean_and_extract_json(content):
    start_index = content.find('{')
    end_index = content.rfind('}')
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None
    json_str = content[start_index:end_index+1]

    try:
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        json_str_cleaned = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        try:
            parsed_json = json.loads(json_str_cleaned)
            return parsed_json
        except json.JSONDecodeError:
            return None

def process_s3_file(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        date = extract_first_date_from_content(content)
        if not date:
            logger.warning(f"No date found in {key}")
            return None, None
        parsed_json = clean_and_extract_json(content)
        if parsed_json is None:
            logger.error(f"Failed to parse JSON from file: {key}")
            return None, None
        return date, parsed_json
    except Exception as e:
        logger.error(f"Error processing file {key}: {str(e)}")
        return None, None

def process_files(bucket, after_clean_prefix):
    contents_by_date = defaultdict(list)
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=after_clean_prefix)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_key = {
            executor.submit(process_s3_file, bucket, obj['Key']): obj['Key']
            for page in pages
            for obj in page.get('Contents', [])
            if obj['Key'].endswith('.txt')
        }
        
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                date, parsed_json = future.result()
                if date and parsed_json:
                    contents_by_date[date].append(parsed_json)
            except Exception as e:
                logger.error(f"Error processing file {key}: {str(e)}")

    return contents_by_date

def main(input_json):
    try:
        data = json.loads(input_json)
        bucket = data['bucket']
        user_id = data['user_id']
        case_id = data['case_id']
        file_name = data['file_name']
        after_clean_prefix = data['file_path']

        logger.info(f"Processing files in: s3://{bucket}/{after_clean_prefix}")

        contents_by_date = process_files(bucket, after_clean_prefix)

        if not contents_by_date:
            logger.warning("No contents to process after parsing files.")
            return

        # Determine if it's a bill or record based on the file path
        document_type = 'bills' if '/bills/' in after_clean_prefix else 'records'
        combine_prefix = f"{user_id}/{case_id}/{document_type}/{file_name}/combine/"

        for date, records in contents_by_date.items():
            output_key = f"{combine_prefix}{date}.json"
            try:
                s3_client.put_object(
                    Bucket=bucket,
                    Key=output_key,
                    Body=json.dumps(records, indent=2)
                )
                logger.info(f"Saved combined {document_type} to s3://{bucket}/{output_key}")
            except Exception as e:
                logger.error(f"Error writing combined {document_type} to s3://{bucket}/{output_key}: {str(e)}")

        # Prepare input for the next step (after_combine.py)
        next_step_input = json.dumps({
            'bucket': bucket,
            'user_id': user_id,
            'case_id': case_id,
            'file_name': file_name,
            'file_path': combine_prefix
        })
        logger.info(f"Next step input: {next_step_input}")

        # Trigger the next step
        logger.info("Triggering complete.py")
        try:
            subprocess.run(['python', 'complete.py', next_step_input], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing after_combine.py: {e}")
            sys.exit(1)

        logger.info("Processing completed successfully")

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON input: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python combine.py '<json_input>'")
        sys.exit(1)
    
    json_input = sys.argv[1]
    main(json_input)