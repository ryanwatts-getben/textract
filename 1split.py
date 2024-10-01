import json
import os
import logging
import fitz
import boto3
import base64
import sys
import subprocess
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic, AnthropicError, RateLimitError, APIError
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')
textract_client = boto3.client('textract')
anthropic_client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

MAX_WORKERS = 10
IMAGE_MEDIA_TYPE = "image/png"
PROMPT_BUCKET = os.environ['PROMPT_BUCKET']

def encode_files_for_payload():
    encoded_images = {}
    encoded_files = {}
    
    # Encode images
    for image_name in ['training_record_image1.png', 'training_record_image2.png']:
        response = s3_client.get_object(Bucket=PROMPT_BUCKET, Key=f"records/step1/{image_name}")
        encoded_images[image_name] = base64.b64encode(response['Body'].read()).decode('utf-8')
    
    # Read XML and JSON files
    files_to_read = [
        'training_prompt1.xml', 'training_prompt2.xml', 'system_prompt.xml',
        'training_record_response1.json', 'training_record_response2.json'
    ]
    for file_name in files_to_read:
        response = s3_client.get_object(Bucket=PROMPT_BUCKET, Key=f"records/step1/{file_name}")
        encoded_files[file_name] = response['Body'].read().decode('utf-8')

    return encoded_images, encoded_files

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
                            "data": encoded_images['training_record_image1.png']
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
                        "text": encoded_files['training_record_response1.json']
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
                            "data": encoded_images['training_record_image2.png']
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
                        "text": encoded_files['training_record_response2.json']
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
                "content": "{"
            }
        ]
    }

    return payload

def extract_json_content(text):
    start_index = text.find('{')
    end_index = text.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index:end_index+1]
    return None

def process_with_claude(filename, total_pages, extracted_text, image_media_type, img_bytes):
    try:
        encoded_images, encoded_files = encode_files_for_payload()
        
        parameters = {
            'image_data': base64.b64encode(img_bytes).decode('utf-8'),
            'text_data': f"{filename} of {total_pages} pages\n{extracted_text}"
        }
        
        payload = construct_payload(parameters, encoded_images, encoded_files)
        
        response = anthropic_client.messages.create(**payload)
        
        raw_content = '{' + response.content[0].text.strip()
        
        if raw_content:
            return raw_content
        else:
            logger.error(f"Failed to extract valid JSON from Claude's response: {raw_content[:100]}...")
            return None
    except APIError as e:
        error_message = str(e)
        if "rate_limit_error" in error_message:
            logger.error(f"Rate limit error in process_with_claude: {error_message}")
            logger.info("Applying 60-second delay due to rate limit")
            time.sleep(60)
        else:
            logger.error(f"API error in process_with_claude: {error_message}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in process_with_claude: {str(e)}", exc_info=True)
        return None

def check_output_exists(bucket_name, output_key):
    try:
        s3_client.head_object(Bucket=bucket_name, Key=output_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise

def process_page(page_num, pdf_document, bucket_name, output_prefix, filename, total_pages):
    logger.info(f"Processing page {page_num} of {filename}")
    
    textract_output_key = f"{output_prefix}/split/page_{page_num}.txt"
    claude_output_key = f"{output_prefix}/clean/page_{page_num}.txt"
    image_key = f"{output_prefix}/images/page_{page_num}.png"
    pdf_key = f"{output_prefix}/pdf/page_{page_num}.pdf"
    
    textract_exists = check_output_exists(bucket_name, textract_output_key)
    claude_exists = check_output_exists(bucket_name, claude_output_key)
    pdf_exists = check_output_exists(bucket_name, pdf_key)
    
    if textract_exists and claude_exists and pdf_exists:
        logger.info(f"All outputs exist for page {page_num}. Skipping processing.")
        return
    
    page = pdf_document.load_page(page_num - 1)  # PyMuPDF uses 0-based indexing
    
    # Save page as PDF
    if not pdf_exists:
        output_pdf = fitz.open()  # Create a new PDF document
        output_pdf.insert_pdf(pdf_document, from_page=page_num-1, to_page=page_num-1)  # Insert the specific page
        pdf_bytes = output_pdf.tobytes()  # Get the PDF as bytes
        s3_client.put_object(Bucket=bucket_name, Key=pdf_key, Body=pdf_bytes)
        logger.info(f"Saved PDF for page {page_num} to S3: {pdf_key}")
        output_pdf.close()  # Close the temporary document
    else:
        logger.info(f"PDF already exists for page {page_num}: {pdf_key}")
    
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    
    # Save image to S3 (always do this as it's needed for Claude)
    if not check_output_exists(bucket_name, image_key):
        s3_client.put_object(Bucket=bucket_name, Key=image_key, Body=img_bytes)
        logger.info(f"Saved image for page {page_num} to S3: {image_key}")
    else:
        logger.info(f"Image for page {page_num} already exists in S3.")
    
    if not textract_exists:
        # Extract text using Textract
        response = textract_client.detect_document_text(Document={'Bytes': img_bytes})
        extracted_text = '\n'.join(item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE')
        file_info = f"This is File Name: {filename} and Page Number: {page_num} to use when referencing this file.\n\n"
        full_text = file_info + extracted_text
        s3_client.put_object(Bucket=bucket_name, Key=textract_output_key, Body=full_text)
        logger.info(f"Saved Textract output for page {page_num} to S3: {textract_output_key}")
    else:
        logger.info(f"Textract output already exists for page {page_num}. Skipping Textract processing.")
        response = s3_client.get_object(Bucket=bucket_name, Key=textract_output_key)
        full_text = response['Body'].read().decode('utf-8')
    
    if not claude_exists:
        # Process with Claude (with retry)
        claude_response = process_with_claude(filename, total_pages, full_text, IMAGE_MEDIA_TYPE, img_bytes)
        if claude_response:
            s3_client.put_object(Bucket=bucket_name, Key=claude_output_key, Body=claude_response)
            logger.info(f"Saved Claude response for page {page_num} to S3: {claude_output_key}")
        else:
            logger.warning(f"Failed to process page {page_num} with Claude or extract valid JSON")
    else:
        logger.info(f"Claude output already exists for page {page_num}. Skipping Claude processing.")


def process_pdf(bucket_name, pdf_key, output_prefix):
    logger.info(f"Starting to process PDF: {pdf_key}")
    response = s3_client.get_object(Bucket=bucket_name, Key=pdf_key)
    pdf_bytes = response['Body'].read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_document)
    logger.info(f"PDF has {total_pages} pages")

    filename = os.path.basename(pdf_key)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                process_page,
                page_num,
                pdf_document,
                bucket_name,
                output_prefix,
                filename,
                total_pages
            )
            for page_num in range(1, total_pages + 1)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in processing page: {str(e)}")

    pdf_document.close()
    logger.info(f"All pages from {pdf_key} have been processed.")


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

def get_pdf_page_count(bucket_name, pdf_key):
    response = s3_client.get_object(Bucket=bucket_name, Key=pdf_key)
    pdf_bytes = response['Body'].read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(pdf_document)
    pdf_document.close()
    return page_count

def main(input_json, retry_count=0):
    logger.info(f"PDF processing started (Attempt {retry_count + 1})")
    
    try:
        data = json.loads(input_json)
        bucket_name = data['bucket']
        pdf_key = data['file_path']
        user_id = data['user_id']
        case_id = data['case_id']
        file_name = data['file_name']

        output_prefix = f"{user_id}/{case_id}/records/{file_name}"
        logger.info(f"Output prefix: {output_prefix}")

        total_pages = get_pdf_page_count(bucket_name, pdf_key)
        logger.info(f"Total pages in PDF: {total_pages}")

        process_pdf(bucket_name, pdf_key, output_prefix)
        
        split_prefix = f"{output_prefix}/split/"
        clean_prefix = f"{output_prefix}/clean/"
        
        split_file_count = count_files_in_s3_prefix(bucket_name, split_prefix)
        clean_file_count = count_files_in_s3_prefix(bucket_name, clean_prefix)
        
        logger.info(f"Files in split directory: {split_file_count}")
        logger.info(f"Files in clean directory: {clean_file_count}")

        if split_file_count == clean_file_count == total_pages:
            logger.info("All pages processed successfully. Triggering clean.py")
            
            clean_path = f"{output_prefix}/clean"
            
            clean_input = json.dumps({
                'file_name': file_name,
                'file_path': clean_path,
                'bucket': bucket_name,
                'case_id': case_id,
                'user_id': user_id,
            })
            
            logger.info("Triggering 2cleanup.py")
            subprocess.run(['python', '2cleanup.py', clean_input], check=True)
            logger.info("clean.py completed successfully")
        else:
            logger.warning("Not all pages were processed.")
            logger.warning(f"Expected {total_pages} pages, found {split_file_count} in split and {clean_file_count} in clean")
            
            if retry_count < 3:
                logger.info(f"Retrying process (Attempt {retry_count + 2})")
                main(input_json, retry_count + 1)
            else:
                logger.error("Max retry attempts reached. Process failed.")
                raise Exception("Failed to process all pages after multiple attempts")

        logger.info("PDF processing completed")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON input: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while processing the PDF: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python 1split.py '<json_input>'")
        sys.exit(1)
    
    json_input = sys.argv[1]
    main(json_input)