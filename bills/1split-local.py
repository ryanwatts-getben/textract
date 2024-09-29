import sys
import os
import boto3
from dotenv import load_dotenv
import concurrent.futures
import fitz  # PyMuPDF, install via `pip install PyMuPDF`
import base64
import anthropic
import json
import logging
import traceback

# Load environment variables from .env
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

# Initialize AWS Textract client
textract_client = boto3.client(
    'textract',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def extract_text_from_page(page_image_path):
    """
    Extract text from a single page image using AWS Textract.
    """
    with open(page_image_path, 'rb') as document:
        imageBytes = document.read()

    response = textract_client.detect_document_text(
        Document={'Bytes': imageBytes}
    )

    # Extract detected text
    extracted_text = ''
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            extracted_text += item['Text'] + '\n'

    return extracted_text

def process_pdf(pdf_path, output_dir):
    """
    Process a single PDF:
    - Split it into individual page images.
    - Extract text from each page using AWS Textract.
    - Save extracted text and images.
    """
    print(f"Processing PDF: {pdf_path}")
    
    pdf_file_name = os.path.basename(pdf_path)
    pdf_document = fitz.open(pdf_path)
    total_pages = pdf_document.page_count

    # Prepare parameters for payload
    parameters = {
        'page_num': 0,
        'pdf_file_name': pdf_file_name,
        'output_dir': output_dir,
        'total_pages': total_pages
    }
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
    
        for page_num in range(1, total_pages + 1):
            parameters['page_num'] = page_num
            
            # Save page image
            page = pdf_document.load_page(page_num - 1)  # PyMuPDF page numbers start at 0
            pix = page.get_pixmap()
            image_filename = os.path.join(output_dir, f'Page_{page_num}.png')
            pix.save(image_filename)
            print(f"Saved image: {image_filename}, page: {page_num}, {output_dir}")
            # Extract text from page image concurrently
            futures.append(executor.submit(
                extract_text_and_save, image_filename, page_num, output_dir
            ))
    
    # Wait for all text extraction tasks to complete
    concurrent.futures.wait(futures)
    
    print(f"Finished processing PDF: {pdf_path}")
    return parameters  # Return parameters for use

def extract_text_and_save(image_path, page_num, output_dir):
    """
    Extract text from a page image and save the text to a file.
    """
    # Read the image bytes
    with open(image_path, 'rb') as img_file:
        imageBytes = img_file.read()
    
    # Call Textract to detect text
    response = textract_client.detect_document_text(
        Document={'Bytes': imageBytes}
    )
    
    # Extract detected text
    extracted_text = ''
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            extracted_text += item['Text'] + '\n'
    
    # Save extracted text to file
    text_filename = os.path.join(output_dir, f'Page_{page_num}.txt')
    with open(text_filename, 'w', encoding='utf-8') as text_file:
        text_file.write(extracted_text)

def encode_files_for_payload():
    """
    Encode images and read JSON files as strings from './src/data/' for reuse without parsing.
    """
    data_dir = './src/data/'
    encoded_images = {}
    encoded_jsons = {}
    
    # Encode images
    for image_name in ['training_image1.png', 'training_image2.png', 'training_image3.png']:
        image_path = os.path.join(data_dir, image_name)
        with open(image_path, 'rb') as img_file:
            encoded_images[image_name] = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Read JSON files as strings without parsing
    for json_name in [
        'training_example_template.json', 'training_prompt1.json',
        'training_prompt2.json', 'training_prompt3.json',
        'training_response1.json', 'training_response2.json',
        'training_response3.json', 'system_prompt.json'
    ]:
        json_path = os.path.join(data_dir, json_name)
        with open(json_path, 'r', encoding='utf-8') as json_file:
            encoded_jsons[json_name] = json_file.read()  # Read content as string without parsing

    return encoded_images, encoded_jsons

def construct_payload(parameters, encoded_images, encoded_jsons):
    """
    Construct the payload dictionary to send to Anthropic Claude.
    """
    # Build the payload structure
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 4096,
        "temperature": 0.0,
        "system": encoded_jsons['system_prompt.json'],
        "messages": [
            # Mocking previous interactions
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
                        "text": encoded_jsons['training_prompt1.json']
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": encoded_jsons['training_response1.json']
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
                        "text": encoded_jsons['training_prompt2.json']
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": encoded_jsons['training_response2.json']
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
                            "data": encoded_images['training_image3.png']
                        }
                    },
                    {
                        "type": "text",
                        "text": encoded_jsons['training_prompt3.json']
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": encoded_jsons['training_response3.json']
                    }
                ]
            },
            {            # Actual task
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "<actual_task_image>"
                        }
                    },
                    {
                        "type": "text",
                        "text": "<actual_task_prompt>"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<actual_task_response>"
                    }
                ]
            }
        ]
    }

    # Replace placeholders with actual data
    actual_image_path = os.path.join(parameters['output_dir'], f"Page_{parameters['page_num']}.png")
    with open(actual_image_path, 'rb') as img_file:
        actual_image_data = base64.b64encode(img_file.read()).decode('utf-8')

    actual_text_path = os.path.join(parameters['output_dir'], f"Page_{parameters['page_num']}.txt")
    with open(actual_text_path, 'r', encoding='utf-8') as text_file:
        actual_text_data = text_file.read()

    # Update the payload with actual task data
    payload['messages'][-2]['content'][0]['source']['data'] = actual_image_data
    payload['messages'][-2]['content'][1]['text'] = actual_text_data

    # Remove the placeholder response
    del payload['messages'][-1]

    return payload

def main():
    if len(sys.argv) < 2:
        print("Usage: python 1split-local.py <path_to_pdf_directory>")
        return

    input_directory = ' '.join(sys.argv[1:])
    output_directory = './src/split/'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Encode reusable images and JSONs
    encoded_images, encoded_jsons = encode_files_for_payload()

    # Process all PDFs in the input directory
    pdf_files = [
        os.path.join(input_directory, f) for f in os.listdir(input_directory)
        if f.lower().endswith('.pdf')
    ]

    parameters_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_pdf, pdf_path, output_directory) for pdf_path in pdf_files
        ]
        for future in concurrent.futures.as_completed(futures):
            parameters = future.result()
            parameters_list.append(parameters)  # Collect parameters for payload

    # Construct payloads for each processed PDF
    for parameters in parameters_list:
        payload = construct_payload(parameters, encoded_images, encoded_jsons)
        client = anthropic.Anthropic()
        response = client.messages.create(**payload)
        # Extract the text content from the response
        if isinstance(response.content, list):
            cleaned_content = ' '.join(item.text for item in response.content if hasattr(item, 'text'))
        else:
            cleaned_content = response.content
        
        # Remove any leading or trailing whitespace
        cleaned_content = cleaned_content.strip()

        # Parse the JSON string into a Python object
        try:
            parsed_content = json.loads(cleaned_content)
            with open(f"./src/split/Page_{parameters['page_num']}.json", "w", encoding='utf-8') as f:
                json.dump(parsed_content, f, ensure_ascii=False, separators=(',', ':'))
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            logging.debug(f"Problematic JSON: {cleaned_content}")
        except IOError as e:
            logging.error(f"IO error when writing file: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            logging.debug(f"Traceback: {traceback.format_exc()}")

    print("All PDFs have been processed and payloads constructed.")

if __name__ == "__main__":
    main()