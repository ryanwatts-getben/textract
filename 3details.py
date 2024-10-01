import json
import os
import logging
import boto3
import re
from datetime import datetime
from botocore.exceptions import ClientError
from copy import deepcopy
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize AWS client
s3_client = boto3.client('s3')

def is_valid_json_value(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

def merge_json_complex(data_list):
    def merge_two(data1, data2):
        if isinstance(data1, dict) and isinstance(data2, dict):
            result = deepcopy(data1)
            for key, value in data2.items():
                if key in result:
                    if isinstance(result[key], (str, int, float)) and isinstance(value, (str, int, float)):
                        if result[key] != value and key != "name":
                            result[key] = [str(result[key]), str(value)]
                    elif isinstance(result[key], list) and isinstance(value, list):
                        # Merge lists with possible dicts without using set
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
    match = re.search(r'records_(\d{4}-\d{2}-\d{2})\.json', filename)
    if not match:
        match = re.search(r'(\d{4}-\d{2}-\d{2})\.json', filename)
    if match:
        return match.group(1)
    else:
        logger.warning(f"Could not extract date from filename: {filename}")
        return None
    
def read_json_from_s3(bucket_name, key):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        logger.error(f"Error reading JSON from S3: {str(e)}")
        return []

def create_status_file(bucket_name, user_id, case_id):
    status_key = f"{user_id}/{case_id}/status.json"
    status_content = json.dumps({"status": "COMPLETED"})
    try:
        s3_client.put_object(Bucket=bucket_name, Key=status_key, Body=status_content)
        logger.info(f"Created status.json file: {status_key}")
    except ClientError as e:
        logger.error(f"Error creating status.json file: {str(e)}")

def process_and_merge_files(bucket_name, user_id, case_id, output_prefix):
    records_pdf_files = get_pdf_files(bucket_name, f"{user_id}/{case_id}/records/")
    bills_pdf_files = get_pdf_files(bucket_name, f"{user_id}/{case_id}/bills/")
    
    # Combine all PDF files from records and bills
    all_pdf_files = records_pdf_files + bills_pdf_files
    
    # Pre-check for /combine/ folders for all PDF files
    for pdf_file in all_pdf_files:
        file_name = os.path.basename(pdf_file).split('.')[0]  # Extract base name from PDF file
        folder_type = 'records' if '/records/' in pdf_file else 'bills'
        combine_prefix = f"{user_id}/{case_id}/{folder_type}/{file_name}/combine/"
        
        # Check if the combine folder exists by trying to list its contents
        try:
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=combine_prefix)
            if 'Contents' not in response or len(response['Contents']) == 0:
                # No files found in the combine folder
                logger.error(f"Missing /combine/ folder or no files for {pdf_file}. Aborting the process.")
                return False  # Stop processing as a combine folder is missing or empty
        except ClientError as e:
            logger.error(f"Error checking combine folder for {pdf_file}: {str(e)}")
            return False  # Stop processing if an error occurs

    # If we pass the pre-check, proceed with the file processing
    all_files_by_date = {}
    
    for pdf_file in all_pdf_files:
        file_name = os.path.basename(pdf_file).split('.')[0]
        folder_type = 'records' if '/records/' in pdf_file else 'bills'
        combine_prefix = f"{user_id}/{case_id}/{folder_type}/{file_name}/combine/"
        files_by_date = get_files_by_date(bucket_name, combine_prefix)
        
        # Collect all files by date
        for date, files in files_by_date.items():
            if date not in all_files_by_date:
                all_files_by_date[date] = []
            all_files_by_date[date].extend(files)

    # Process the JSON files and merge them
    for date, files in all_files_by_date.items():
        all_data = []
        for file in files:
            data = read_json_from_s3(bucket_name, file)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

        merged_data = merge_json_complex(all_data)

        # Update PageNumber in References for ICD10CM, Rx, CPT, and other elements
        if 'Codes' in merged_data:
            for code_type in ['ICD10CM', 'Rx', 'CPTCodes']:
                if code_type in merged_data['Codes']:
                    for item in merged_data['Codes'][code_type]:
                        if isinstance(item, dict):
                            for code, details in item.items():
                                if isinstance(details, dict) and 'References' in details:
                                    if isinstance(details['References'], dict) and 'ThisIsPageNumberOfPDF' in details['References']:
                                        details['References']['PageNumber'] = details['References'].pop('ThisIsPageNumberOfPDF')

        # Update PageNumber in References for ProceduresOrFindings
        if 'ProceduresOrFindings' in merged_data:
            for item in merged_data['ProceduresOrFindings']:
                if isinstance(item, dict) and 'KeyWordsOrFindings' in item:
                    for finding in item['KeyWordsOrFindings']:
                        if isinstance(finding, dict):
                            for _, details in finding.items():
                                if isinstance(details, dict) and 'References' in details:
                                    if isinstance(details['References'], dict) and 'ThisIsPageNumberOfPDF' in details['References']:
                                        details['References']['PageNumber'] = details['References'].pop('ThisIsPageNumberOfPDF')

        # Update PageNumber in References for OtherInformation
        if 'OtherInformation' in merged_data:
            for item in merged_data['OtherInformation']:
                if isinstance(item, dict) and 'References' in item:
                    if isinstance(item['References'], dict) and 'ThisIsPageNumberOfPDF' in item['References']:
                        item['References']['PageNumber'] = item['References'].pop('ThisIsPageNumberOfPDF')

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

        output_prefix = f"{user_id}/{case_id}/final/"

        logger.info(f"Processing files for user_id: {user_id}, case_id: {case_id}")

        files_processed = process_and_merge_files(bucket_name, user_id, case_id, output_prefix)

        if files_processed:
            create_status_file(bucket_name, user_id, case_id)
            logger.info("Processing completed successfully")
        else:
            logger.warning(f"No files were processed for user_id: {user_id}, case_id: {case_id}")

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON input: {str(e)}")
    except Exception as e:
        logger.error(f"An error occurred while processing the records: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python 3details.py '<json_input>'")
        sys.exit(1)
    
    json_input = sys.argv[1]
    main(json_input)