import json
import os
import logging
import boto3
import re
from datetime import datetime
from botocore.exceptions import ClientError
from copy import deepcopy
import sys
import subprocess




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

# Initialize AWS S3 client
s3_client = boto3.client('s3')
s3_paginator = s3_client.get_paginator('list_objects_v2')

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
    Retrieve a list of PDF files from an S3 bucket with a given prefix,
    excluding files in subdirectories.
    """
    pdf_files = []
    paginator = s3_paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    for page in paginator:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.pdf'):
                    # Check if the file is directly in the prefix directory
                    relative_path = obj['Key'][len(prefix):]
                    if '/' not in relative_path:
                        pdf_files.append(obj['Key'])

    return pdf_files

def get_files_by_date(bucket_name, prefix):
    """
    Retrieve JSON files from an S3 bucket and organize them by date extracted from filenames.
    """
    files_by_date = {}
    paginator = s3_paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    for page in paginator:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.json'):
                    date = extract_date_from_filename(obj['Key'])
                    if date:
                        if date not in files_by_date:
                            files_by_date[date] = []
                        files_by_date[date].append(obj['Key'])

    return files_by_date

def extract_date_from_filename(filename):
    """
    Extract a date from a filename using regex patterns.
    """
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

def count_files_by_extension(bucket_name, prefix, extension):
    count = 0
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for item in page['Contents']:
                # Check if the file is directly in the prefix directory
                if '/' not in item['Key'][len(prefix):] and item['Key'].lower().endswith(extension.lower()):
                    count += 1
    
    return count

def create_status_file(bucket_name, user_id, case_id):
    """
    Create a status.json file in an S3 bucket to indicate processing completion.
    If reports directory exists, verifies that all PDFs have corresponding JSON files.
    """
    try:
        reports_prefix = f"{user_id}/{case_id}/reports/"
        
        try:
            # Check if reports directory exists
            pdf_count = count_files_by_extension(bucket_name, reports_prefix, '.pdf')
            if pdf_count > 0:
                # If we have PDFs, check JSON count
                json_count = count_files_by_extension(bucket_name, reports_prefix, '.json')
                logger.info(f"Found {pdf_count} PDFs and {json_count} JSONs in reports directory")
                
                if pdf_count != json_count:
                    raise Exception(
                        f"Mismatch in file counts. PDFs: {pdf_count}, JSONs: {json_count}. "
                        f"All PDFs must have corresponding JSON files before completing."
                    )
            else:
                logger.info(f"No PDFs found in reports directory: {reports_prefix}. Proceeding with status creation.")
                
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                raise
            logger.info(f"Reports directory not found or empty: {reports_prefix}. Proceeding with status creation.")
        
        # Create status file
        status_key = f"{user_id}/{case_id}/status.json"
        status_content = json.dumps({"status": "COMPLETED"})
        s3_client.put_object(Bucket=bucket_name, Key=status_key, Body=status_content)
        logger.info(f"Created status.json file: {status_key}")
        
    except ClientError as e:
        logger.error(f"S3 error while checking files or creating status.json: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in create_status_file: {str(e)}")
        raise

    
def check_combine_folder(bucket_name, prefix):
    """
    Check if the /combine/ folder exists and contains files.
    """
    paginator = s3_paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    for page in paginator:
        if 'Contents' in page and len(page['Contents']) > 0:
            return True
    
    return False

def process_and_merge_files(bucket_name, user_id, case_id, output_prefix):
    """
    Process and merge JSON files from S3, saving results to the output prefix.
    """
    records_pdf_files = get_pdf_files(bucket_name, f"{user_id}/{case_id}/records/")
    bills_pdf_files = get_pdf_files(bucket_name, f"{user_id}/{case_id}/bills/")
    
    all_pdf_files = records_pdf_files + bills_pdf_files

    # Pre-check for /combine/ folders for all PDF files
    for pdf_file in all_pdf_files:
        file_name = '.'.join(os.path.basename(pdf_file).split('.')[:-1])
        
        folder_type = 'records' if '/records/' in pdf_file else 'bills'
        combine_prefix = f"{user_id}/{case_id}/{folder_type}/{file_name}/combine/"
        if not check_combine_folder(bucket_name, combine_prefix):
            logger.error(f"Missing /combine/ folder or no files for {pdf_file}. Aborting the process.")
            return False

    all_files_by_date = {}


    
    for pdf_file in all_pdf_files:
        file_name = '.'.join(os.path.basename(pdf_file).split('.')[:-1])
        folder_type = 'records' if '/records/' in pdf_file else 'bills'
        combine_prefix = f"{user_id}/{case_id}/{folder_type}/{file_name}/combine/"


        files_by_date = get_files_by_date(bucket_name, combine_prefix)
        

        for date, files in files_by_date.items():
            if date not in all_files_by_date:
                all_files_by_date[date] = []
            all_files_by_date[date].extend(files)
    for date, files in all_files_by_date.items():
        all_data = []
        for file in files:
            data = read_json_from_s3(bucket_name, file)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

        merged_data = merge_json_complex(all_data)

        if merged_data:
            output_key = f"{output_prefix}{date}.json"

            try:
                s3_client.put_object(Bucket=bucket_name, Key=output_key, Body=json.dumps(merged_data, indent=2))
                logger.info(f"Merged and saved file for date {date}: {output_key}")
            except ClientError as e:
                logger.error(f"Error saving merged file for date {date}: {str(e)}")

    return len(all_files_by_date) > 0

def main(input_json):
    try:
        data = json.loads(input_json)
        bucket_name = data['bucket']
        user_id = data['user_id']
        case_id = data['case_id']

        output_prefix = f"{user_id}/{case_id}/complete/"

        logger.info(f"Processing files for user_id: {user_id}, case_id: {case_id}")

        files_processed = process_and_merge_files(bucket_name, user_id, case_id, output_prefix)

        if files_processed:
            create_status_file(bucket_name, user_id, case_id)
            logger.info("Processing completed successfully")

                        # Replace the commented HTML generation section with:
            transform_input = json.dumps({
                'bucket': bucket_name,
                'user_id': user_id,
                'case_id': case_id,
                'input_prefix': output_prefix,
                'output_prefix': f"{user_id}/{case_id}/transformed/"
            })

            try:
                result = subprocess.run(
                    ['python', 'transform.py', transform_input],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"transform_json.py completed successfully. Output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running transform_json.py: {e}")
                logger.error(f"transform_json.py error output: {e.stderr}")
        else:
            logger.warning(f"No files were processed for user_id: {user_id}, case_id: {case_id}")

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON input: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred while processing the records: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python complete.py '<json_input>'")
        sys.exit(1)
    
    json_input = sys.argv[1]
    main(json_input)