import json
import os
import boto3
import fitz
import logging
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger()
logger.setLevel(logging.INFO)
image_media_type = "image/png"

s3_client = boto3.client('s3')
textract_client = boto3.client('textract')
sqs_client = boto3.client('sqs')

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 10))
RECORDS_CLEANUP_QUEUE = os.environ['RECORDS_CLEANUP_QUEUE']

def send_sqs(bucket_name, file_path, file_name):
    sqs_queue_url = f'https://sqs.us-east-1.amazonaws.com/461135439633/{RECORDS_CLEANUP_QUEUE}'
    
    user_id, case_id = file_path.split('/')[0:2]
    message_body = {
        'file_name': file_name,
        'file_path': file_path,
        'bucket': bucket_name,
        'case_id': case_id,
        'user_id': user_id,
    }
    logger.info(f"Sending SQS message to cleanup queue: {message_body}")
    try:
        response = sqs_client.send_message(
            QueueUrl=sqs_queue_url,
            MessageBody=json.dumps(message_body)
        )
        logger.info(f"SQS message sent successfully. MessageId: {response['MessageId']}")
    except Exception as e:
        logger.error(f"Failed to send SQS message: {str(e)}")
        raise

def process_with_claude(pdf_key, file_name, total_pages, extracted_text, image_media_type, img_bytes):
    """
    Send content and image data to Claude API and return the response.
    """
    response = client.beta.prompt_caching.messages.create(
        model="claude-3-haiku-20240307",
        temperature=0.3,
        max_tokens=8192,
        system="{environmentVariables.CLAUDE_PROMPT}" + pdf_key,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": img_bytes,
                        },
                    },
                    {
                        "type": "text",
                        "text": file_name + "of" + total_pages + "pages" + extracted_text
                    },
                ]
            },
            {
                "role": "assistant",
                "content": '['
            },
        ],
    )  

    # Extract the text content from the response
    if isinstance(response.content, list):
        cleaned_content = ''.join(
            item.text if hasattr(item, 'text') else item.get('text', '')
            for item in response.content
        )
    else:
        cleaned_content = response.content

    # Ensure the content starts with '['
    if not cleaned_content.startswith('['):
        cleaned_content = '[' + cleaned_content

    return cleaned_content

def process_page(page_num, pdf_document, bucket_name, output_prefix, filename):
    try:
        logger.info(f"Processing page {page_num + 1} of {filename}")
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        
        # Save image
        image_key = f"{output_prefix}/images/page_{page_num + 1}.png"
        s3_client.put_object(Bucket=bucket_name, Key=image_key, Body=img_bytes)
        logger.info(f"Saved image for page {page_num + 1} to S3: {image_key}")

        # Extract text using Textract
        response = textract_client.detect_document_text(
            Document={'Bytes': img_bytes}
        )
        extracted_text = '\n'.join(item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE')
        
        # Save extracted text
        text_key = f"{output_prefix}/split/page_{page_num + 1}.txt"
        s3_client.put_object(Bucket=bucket_name, Key=text_key, Body=extracted_text)
        logger.info(f"Saved extracted text for page {page_num + 1} to S3: {text_key}")
    except Exception as e:
        logger.error(f"Error processing page {page_num + 1} of {filename}: {str(e)}")
        raise

def process_pdf(bucket_name, pdf_key, output_prefix):
    logger.info(f"Starting to process PDF: {pdf_key}")
    response = s3_client.get_object(Bucket=bucket_name, Key=pdf_key)
    pdf_bytes = response['Body'].read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_document)
    logger.info(f"PDF has {total_pages} pages")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_page, page_num, pdf_document, bucket_name, output_prefix, pdf_key)
            for page_num in range(total_pages)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in processing page: {str(e)}")

    logger.info(f"All pages from {pdf_key} have been processed.")

def handler(event, context):
    logger.info("Lambda function started")
    
    try:
        for record in event['Records']:
            body = json.loads(record['body'])
            logger.info(f'Processing SQS message: {body}')
            
            bucket_name = body['bucket']
            pdf_key = body['file_path']
            user_id = body['user_id']
            case_id = body['case_id']
            file_name = body['file_name']

            output_prefix = f"{user_id}/{case_id}/records/{file_name}"
            logger.info(f"Output prefix: {output_prefix}")

            process_pdf(bucket_name, pdf_key, output_prefix)

            details = process_with_claude(pdf_key, file_name, total_pages, extracted_text, image_media_type, img_bytes)

            # Send message to cleanup queue for split path
            split_path = f"{user_id}/{case_id}/records/{file_name}/split/"
            logger.info(f"Sending message to cleanup queue for split path: {split_path}")
            send_sqs(bucket_name, split_path, file_name)

        logger.info("PDF processing completed successfully")
        return {
            'statusCode': 200,
            'body': json.dumps('PDF processing completed successfully')
        }
    except Exception as e:
        logger.error(f"An error occurred while processing the PDFs: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }