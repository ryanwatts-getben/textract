import json
import os
import re
import logging
import boto3
import base64
import fitz
from collections import defaultdict
from botocore.exceptions import ClientError
from anthropic import Anthropic
import sys
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize AWS clients
s3_client = boto3.client('s3')
anthropic_client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

PROMPT_BUCKET = os.environ['PROMPT_BUCKET']

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

def encode_files_for_payload():
    encoded_images = {}
    encoded_files = {}
    
    # Encode images
    for image_name in ['training_image1.png', 'training_image2.png', 'training_image3.png']:
        response = s3_client.get_object(Bucket=PROMPT_BUCKET, Key=f"bills/step2/{image_name}")
        encoded_images[image_name] = base64.b64encode(response['Body'].read()).decode('utf-8')
    
    # Read XML and JSON files
    files_to_read = [
        'training_prompt1.xml', 'training_prompt2.xml', 'training_prompt3.xml', 'system_prompt.xml',
        'training_example_template.json', 'training_response1.json',
        'training_response2.json', 'training_response3.json'
    ]
    for file_name in files_to_read:
        response = s3_client.get_object(Bucket=PROMPT_BUCKET, Key=f"bills/step2/{file_name}")
        encoded_files[file_name] = response['Body'].read().decode('utf-8')

    return encoded_images, encoded_files

def extract_first_date_from_content_using_tsv(content, tsv_path, bucket_name, image_path):
    try:
        # Read TSV content
        response = s3_client.get_object(Bucket=bucket_name, Key=tsv_path)
        tsv_content = response['Body'].read().decode('utf-8')
        
        # Read image content
        image_response = s3_client.get_object(Bucket=bucket_name, Key=image_path)
        image_content = image_response['Body'].read()
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        
        # Prepare payload for Claude
        encoded_images, encoded_files = encode_files_for_payload()
        
        parameters = {
            'text_data': f"Content:\n{content}\n\nTSV Content:\n{tsv_content}",
            'image_data': image_base64
        }
        
        payload = construct_payload(parameters, encoded_images, encoded_files)
        
        # Process with Claude
        response = anthropic_client.messages.create(**payload)
        
        raw_content = response.content[0].text.strip()
        parsed_json = clean_and_extract_json(raw_content)
        
        if parsed_json and 'Date' in parsed_json:
            return parsed_json, parsed_json['Date']
        else:
            logger.error(f"Failed to extract date using Claude for TSV: {tsv_path}")
            return None, None
    except Exception as e:
        logger.error(f"Error in extract_first_date_from_content_using_tsv: {str(e)}")
        return None, None

def construct_payload(parameters, encoded_images, encoded_files):
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 4096,
        "temperature": 0.0,
        "system": encoded_files['system_prompt.xml'],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded_images['training_image1.png']
                        }
                    },
                    {
                        "type": "text",
                        "text": encoded_files['training_prompt1.xml']
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": encoded_files['training_response1.json']
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded_images['training_image2.png']
                        }
                    },
                    {
                        "type": "text",
                        "text": encoded_files['training_prompt2.xml']
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": encoded_files['training_response2.json']
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": parameters['image_data']
                        }
                    },
                    {
                        "type": "text",
                        "text": parameters['text_data']
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "{"
                    }
                ]
            }
        ]
    }

    return payload


def process_files(bucket_name, input_prefix, tsv_path, image_prefix):
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
            parsed_json = None
            
            # Extract page number from file path
            match = re.search(r'page_(\d+)\.txt', file_path)
            if match:
                page_num = match.group(1)
                image_path = f"{image_prefix}/page_{page_num}.png"
            else:
                logger.warning(f"Could not extract page number from file path: {file_path}")
                image_path = None
            
            if not date:
                logger.warning(f"No date found in {file_path}")
                # parsed_json, date = extract_first_date_from_content_using_tsv(content, tsv_path, bucket_name, image_path)
                # if not date:
                #     logger.warning(f"No date found in {file_path} using TSV method")
                #     continue
            else:
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

        input_prefix = f"{user_id}/{case_id}/bills/{file_name}/clean/"
        output_prefix = f"{user_id}/{case_id}/bills/{file_name}/combine/"
        tsv_path = f"{user_id}/{case_id}/bills/{file_name}/{file_name}.tsv"
        image_prefix = f"{user_id}/{case_id}/bills/{file_name}/images"

        # Check if all input files have been processed
        pdf_key = f"{user_id}/{case_id}/bills/{file_name}.pdf"
        total_pages = get_pdf_page_count(bucket_name, pdf_key)
        input_file_count = count_files_in_s3_prefix(bucket_name, input_prefix)
        
        if input_file_count < total_pages:
            logger.warning(f"Not all pages have been processed. Expected {total_pages}, found {input_file_count}.")
            return

        # Check if TSV file exists
        if not check_output_exists(bucket_name, tsv_path):
            logger.warning(f"TSV file not found: {tsv_path}")
            return

        logger.info(f"Processing files in: {input_prefix}")

        contents_by_date = process_files(bucket_name, input_prefix, tsv_path, image_prefix)

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
        logger.error(f"An error occurred while processing the bills: {str(e)}")




if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python 2cleanup-bill.py '<json_input>'")
        sys.exit(1)
    
    json_input = sys.argv[1]
    main(json_input)