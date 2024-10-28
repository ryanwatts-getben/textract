import json
import os
import logging
import fitz
import boto3
import base64
import sys
import re
from datetime import datetime
from collections import defaultdict
from botocore.exceptions import ClientError
from anthropic import Anthropic, APIError
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')
textract_client = boto3.client('textract')
anthropic_client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

IMAGE_MEDIA_TYPE = "image/png"
PROMPT_BUCKET = os.environ['PROMPT_BUCKET']

def normalize_date(date_str):
    date_formats = [
        "%m/%d/%y", "%m-%d-%y",
        "%m/%d/%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%Y-%m-%d",
        "%d/%m/%y", "%d-%m-%y",
        "%d/%m/%Y", "%d-%m-%Y",
        "%B %d, %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S"
    ]
    date_str = date_str.strip().title()
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    try:
        from dateutil import parser
        parsed_date = parser.parse(date_str)
        return parsed_date.strftime("%Y-%m-%d")
    except ImportError:
        pass
    except ValueError:
        pass
    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            continue
    if re.match(r'^\d{8}$', date_str):
        try:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        except:
            pass
    if re.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$', date_str):
        parts = date_str.split('.')
        if int(parts[1]) <= 12:
            try:
                return datetime(int(parts[2]), int(parts[0]), int(parts[1])).strftime("%Y-%m-%d")
            except ValueError:
                pass
        try:
            return datetime(int(parts[2]), int(parts[1]), int(parts[0])).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None

def get_rows_columns_map(table_result, blocks_map):
    rows = defaultdict(dict)
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    rows[row_index][col_index] = get_text(cell, blocks_map)
    return rows

def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] == 'SELECTED':
                            text += 'X '
    return text.strip()

def encode_files_for_payload():
    system_file = ''
    response = s3_client.get_object(Bucket=PROMPT_BUCKET, Key="reports/step1/system_prompt.xml")
    system_file = response['Body'].read().decode('utf-8')
    return system_file


# def extract_text_from_pages(pdf_document, filename):
#     pages_data = []
#     total_pages = len(pdf_document)
    
#     for page_num in range(total_pages):
#         page = pdf_document.load_page(page_num)
#         pix = page.get_pixmap()
#         img_bytes = pix.tobytes("png")
        
#         try:
#             response = textract_client.analyze_document(
#                 Document={'Bytes': img_bytes},
#                 FeatureTypes=['TABLES']
#             )
            
#             # Process text and tables
#             text_content = []
#             blocks_map = {block['Id']: block for block in response['Blocks']}
            
#             # Process regular text
#             line_blocks = [block for block in response['Blocks'] if block['BlockType'] == 'LINE']
#             for block in sorted(line_blocks, key=lambda x: (x['Geometry']['BoundingBox']['Top'], x['Geometry']['BoundingBox']['Left'])):
#                 text_content.append(block['Text'])
            
#             # Process tables
#             table_blocks = [block for block in response['Blocks'] if block['BlockType'] == 'TABLE']
#             for table in table_blocks:
#                 text_content.append("\n=== TABLE START ===")
#                 table_data = get_rows_columns_map(table, blocks_map)
#                 for row_idx in sorted(table_data.keys()):
#                     row_content = []
#                     for col_idx in sorted(table_data[row_idx].keys()):
#                         row_content.append(table_data[row_idx][col_idx])
#                     text_content.append(" | ".join(row_content))
#                 text_content.append("=== TABLE END ===\n")
            
#             extracted_text = '\n'.join(text_content)
            
#             pages_data.append({
#                 'page_number': page_num + 1,
#                 'image_data': base64.b64encode(img_bytes).decode('utf-8'),
#                 'text': extracted_text
#             })
            
#         except Exception as e:
#             logger.error(f"Error processing page {page_num + 1}: {str(e)}")
#             continue
    
#     return pages_data, total_pages

def extract_text_from_pages(pdf_document):
    pages_data = []
    total_pages = len(pdf_document)
    for page_num in range(total_pages):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")

        
        try:
            response = textract_client.detect_document_text(
                Document={'Bytes': img_bytes}
            )
            extracted_text = '\n'.join(item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE')
            
            pages_data.append({
                'page_number': page_num + 1,
                'image_data': base64.b64encode(img_bytes).decode('utf-8'),
                'text': extracted_text
            })
            
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {str(e)}")
            continue
    
    return pages_data



def construct_payload(pages_data, filename, system_file):
    # Create formatted text with page numbers
    full_text = f"{filename} of {len(pages_data)} pages\n\n"
    for page_data in pages_data:
        full_text += f"=== PAGE {page_data['page_number']} ===\n\n{page_data['text']}\n\n\n"

    # Use the first page's image as representative
    first_page_image = pages_data[0]['image_data']

    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 4096,
        "temperature": 0.0,
        "system": system_file,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": first_page_image
                        }
                    },
                    {
                        "type": "text",
                        "text": full_text
                    }
                ]
            },
            {
                "role": "assistant",
                "content": "{"
            }
        ]
    }
    return payload, full_text

def extract_date_from_response(claude_response):
    try:
        # Parse the JSON response

        response_data = json.loads(claude_response)
        if isinstance(response_data, dict):
            # Get the date from the first item, keeping the hyphenated format
            return response_data.get('Date', '')
        logger.warning("Could not find date in Claude response, using original filename")
        return None
    except json.JSONDecodeError:
        logger.warning("Could not parse Claude response for date extraction")
        return None
    except Exception as e:
        logger.warning(f"Error extracting date from response: {str(e)}")
        return None

def count_files_by_extension(bucket_name, prefix, extension):
    count = 0
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            count += sum(1 for item in page['Contents'] if item['Key'].lower().endswith(extension.lower()))
    
    return count

def process_with_claude(pdf_document, filename):
    try:
        system_file = encode_files_for_payload()
        pages_data = extract_text_from_pages(pdf_document)
        
        payload, full_text = construct_payload(pages_data, filename, system_file)
        response = anthropic_client.messages.create(**payload)
        
        if response.content:
            return '{\n' + response.content[0].text.strip(), full_text
        else:
            logger.error("Failed to get valid response from Claude")
            return None, None
            
    except APIError as e:
        error_message = str(e)
        if "rate_limit_error" in error_message:
            logger.error(f"Rate limit error: {error_message}")
            time.sleep(60)
            return process_with_claude(pdf_document, filename)  # Retry once after delay
        else:
            logger.error(f"API error: {error_message}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return None, None

def process_pdf(bucket_name, pdf_key, output_prefix, pdf_prefix):
    logger.info(f"Processing PDF: {pdf_key}")
    
    try:
        # Strip the filename from output_prefix to get base path
        base_prefix = '/'.join(output_prefix.split('/')[:-1])
        
        # Count PDF and JSON files in reports directory
        pdf_count = count_files_by_extension(bucket_name, base_prefix, '.pdf')
        json_count = count_files_by_extension(bucket_name, base_prefix, '.json')

        print('base_prefix => ', base_prefix)
        
        logger.info(f"Found {pdf_count} PDFs and {json_count} JSONs in reports directory")
        
        # If counts match, all files are processed
        if pdf_count > 0 and pdf_count == json_count:
            logger.info("All PDFs have corresponding JSON files. Skipping processing.")
            return
        
        # Continue with processing if counts don't match
        response = s3_client.get_object(Bucket=bucket_name, Key=pdf_key)
        pdf_bytes = response['Body'].read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Save individual pages as PDFs in S3
        for page_num in range(len(pdf_document)):
            output_pdf = fitz.open()  # Create a new PDF document
            output_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)  # Insert the specific page
            pdf_bytes = output_pdf.tobytes()  # Get the PDF as bytes
            page_key = f"{pdf_prefix}/page_{page_num + 1}.pdf"
            s3_client.put_object(Bucket=bucket_name, Key=page_key, Body=pdf_bytes)
            output_pdf.close()  # Close the temporary document
            logger.info(f"Saved page {page_num + 1} as PDF to S3: {page_key}")
        filename = os.path.basename(pdf_key)
        
        # Process with Claude
        claude_response, full_text = process_with_claude(pdf_document, filename)

        if claude_response:
            # Extract date from Claude response to use as filename
            date_str = extract_date_from_response(claude_response)
            if date_str:
                output_json_key = f"{base_prefix}/{date_str}.json"
            else:
                base_filename = os.path.splitext(filename)[0]
                output_json_key = f"{base_prefix}/{base_filename}.json"
                
            s3_client.put_object(
                Bucket=bucket_name,
                Key=output_json_key,
                Body=claude_response
            )
            logger.info(f"Saved processed output to: {output_json_key}")
            
            if full_text:
                output_text_key = output_json_key.replace('.json', '.txt')
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=output_text_key,
                    Body=full_text
                )
        else:
            logger.error("Failed to process PDF with Claude")
            
        pdf_document.close()
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise

def main(input_json):
    logger.info("PDF processing started")
    
    try:
        data = json.loads(input_json)
        bucket_name = data['bucket']
        pdf_key = data['file_path']
        user_id = data['user_id']
        case_id = data['case_id']
        file_name = data['file_name']

        base_filename = os.path.splitext(file_name)[0]
        output_prefix = f"{user_id}/{case_id}/reports/{file_name}"

        pdf_prefix = f"{user_id}/{case_id}/reports/{base_filename}/pdf"
        
        process_pdf(bucket_name, pdf_key, output_prefix,pdf_prefix)
        
        logger.info("PDF processing completed")
        
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON input: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python process_reports.py '<json_input>'")
        sys.exit(1)
    
    json_input = sys.argv[1]
    main(json_input)