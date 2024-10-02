import json
import os
import logging
import boto3
import re
from datetime import datetime
from botocore.exceptions import ClientError
from copy import deepcopy
import sys

"""
This script should be run after the combine step. It DOES merge the records into a single file per date, with the child properties persisting as a list
"""

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize AWS S3 client
s3_client = boto3.client('s3')

def is_valid_json_value(value):
    """
    Check if a value can be serialized to JSON.
    """
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

def merge_json_complex(data_list):
    """
    Merge a list of JSON objects, handling nested structures and duplicates.
    """
    def merge_two(data1, data2):
        # Merge two JSON objects
        if isinstance(data1, dict) and isinstance(data2, dict):
            result = deepcopy(data1)
            for key, value in data2.items():
                if key in result:
                    # Handle merging of simple types
                    if isinstance(result[key], (str, int, float)) and isinstance(value, (str, int, float)):
                        if result[key] != value and key != "name":
                            result[key] = [str(result[key]), str(value)]
                    # Handle merging of lists
                    elif isinstance(result[key], list) and isinstance(value, list):
                        merged_list = []
                        seen = set()
                        for item in result[key] + value:
                            if isinstance(item, dict):
                                # Serialize dict to a JSON string for uniqueness
                                item_serialized = json.dumps(item, sort_keys=True)
                                if item_serialized not in seen:
                                    seen.add(item_serialized)
                                    merged_list.append(item)
                            else:
                                if item not in seen:
                                    seen.add(item)
                                    merged_list.append(item)
                        result[key] = merged_list
                    else:
                        result[key] = merge_two(result[key], value)
                else:
                    result[key] = deepcopy(value)
            return result
        elif isinstance(data1, list) and isinstance(data2, list):
            # Merge lists with unique items
            merged_list = []
            seen = set()
            for item in data1 + data2:
                if isinstance(item, dict):
                    item_serialized = json.dumps(item, sort_keys=True)
                    if item_serialized not in seen:
                        seen.add(item_serialized)
                        merged_list.append(item)
                else:
                    if item not in seen:
                        seen.add(item)
                        merged_list.append(item)
            return merged_list
        else:
            return data2 if is_valid_json_value(data2) else data1

    if not data_list:
        return {}
    
    result = data_list[0]
    for data in data_list[1:]:
        result = merge_two(result, data)
    
    return result

def get_pdf_files(bucket_name, prefix):
    """
    Retrieve a list of PDF files from an S3 bucket with a given prefix.
    """
    pdf_files = []
    continuation_token = None

    while True:
        try:
            if continuation_token:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name, 
                    Prefix=prefix,
                    ContinuationToken=continuation_token
                )
            else:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name, 
                    Prefix=prefix
                )
        except ClientError as e:
            logger.error(f"Error listing objects in bucket {bucket_name}: {str(e)}")
            return pdf_files

        if 'Contents' in response:
            pdf_files.extend([obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.pdf')])

        if 'NextContinuationToken' in response:
            continuation_token = response['NextContinuationToken']
        else:
            break

    return pdf_files

def get_files_by_date(bucket_name, prefix):
    """
    Retrieve JSON files from an S3 bucket and organize them by date extracted from filenames.
    """
    files_by_date = {}
    continuation_token = None

    while True:
        try:
            if continuation_token:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name, 
                    Prefix=prefix,
                    ContinuationToken=continuation_token
                )
            else:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name, 
                    Prefix=prefix
                )
        except ClientError as e:
            logger.error(f"Error listing objects in bucket {bucket_name}: {str(e)}")
            return files_by_date

        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.json'):
                    date = extract_date_from_filename(obj['Key'])
                    if date:
                        if date not in files_by_date:
                            files_by_date[date] = []
                        files_by_date[date].append(obj['Key'])

        if 'NextContinuationToken' in response:
            continuation_token = response['NextContinuationToken']
        else:
            break

    return files_by_date

def extract_date_from_filename(filename):
    """
    Extract a date from a filename using regex patterns.
    """
    match = re.search(r'records_(\d{4}-\d{2}-\d{2})\.json', filename)
    if not match:
        match = re.search(r'bills_(\d{4}-\d{2}-\d{2})\.json', filename)
    if not match:
        match = re.search(r'(\d{4}-\d{2}-\d{2})\.json', filename)
    if match:
        return match.group(1)
    else:
        logger.warning(f"Could not extract date from filename: {filename}")
        return None
    
def read_json_from_s3(bucket_name, key):
    """
    Read and parse a JSON file from an S3 bucket.
    """
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        logger.error(f"Error reading JSON from S3: {str(e)}")
        return []

def create_status_file(bucket_name, user_id, case_id):
    """
    Create a status.json file in an S3 bucket to indicate processing completion.
    """
    status_key = f"{user_id}/{case_id}/status.json"
    status_content = json.dumps({"status": "COMPLETED"})
    try:
        s3_client.put_object(Bucket=bucket_name, Key=status_key, Body=status_content)
        logger.info(f"Created status.json file: {status_key}")
    except ClientError as e:
        logger.error(f"Error creating status.json file: {str(e)}")

def process_and_merge_files(input_directory, output_directory):
    """
    Process and merge JSON files from the input directory, saving results to the output directory.
    """
    all_files_by_date = {}
    
    # Walk through the input directory to find .txt files
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                date = extract_date_from_filename(file)
                if date:
                    if date not in all_files_by_date:
                        all_files_by_date[date] = []
                    all_files_by_date[date].append(file_path)

    # Process the JSON files and merge them by date
    for date, files in all_files_by_date.items():
        all_data = []
        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)

        merged_data = merge_json_complex(all_data)

        if merged_data:
            output_file = os.path.join(output_directory, f"{date}.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(merged_data, f, indent=2)
            logger.info(f"Merged and saved file for date {date}: {output_file}")

    return len(all_files_by_date) > 0

def main(input_directory):
    """
    Main function to process files from the input directory and save merged results.
    """
    output_directory = os.path.join(os.path.dirname(input_directory), 'complete')

    logger.info(f"Processing files from directory: {input_directory}")

    files_processed = process_and_merge_files(input_directory, output_directory)

    if files_processed:
        logger.info("Processing completed successfully")
    else:
        logger.warning(f"No files were processed from the input directory: {input_directory}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python full-json-test.py <input_directory>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    main(input_directory)