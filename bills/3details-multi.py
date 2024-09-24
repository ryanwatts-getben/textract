import os
import sys
import json
from dotenv import load_dotenv
import anthropic
import re
import concurrent.futures
from concurrent.futures import as_completed
import base64
import time
import subprocess
import logging  # {{ edit_1 }}

def get_unique_path(base_path):
    """
    Generate a unique file path by appending an incrementing counter
    if the specified path already exists.
    """
    path = base_path
    counter = 1
    while os.path.exists(path):
        name, ext = os.path.splitext(base_path)
        path = f"{name}_{counter}{ext}"
        counter += 1
        if counter > 1000:  # Prevent infinite loop  # {{ edit_2 }}
            logging.error("Unable to create a unique path after 1000 attempts.")
            raise Exception("Exceeded maximum number of attempts to create a unique path.")
    return path

def extract_page_number(filename):
    """
    Extract the page number from the filename.
    """
    match = re.search(r'Page_(\d+)', filename)
    return match.group(1) if match else "Unknown"

def get_image_data(filename):
    """
    Retrieve the base64-encoded image data corresponding to the given filename.
    """
    image_path = f"{filename}.png"
    image_media_type = "image/png"
    if os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as f:
                image_content = f.read()
            image_data = base64.b64encode(image_content).decode('utf-8')
            return image_media_type, image_data
        except Exception as e:
            logging.error(f"Failed to read image file {image_path}: {e}")
            return None, None
    else:
        logging.warning(f"Image not found: {image_path}")
        return None, None

def process_with_claude(content, filename, pagenumber, image_media_type, image_data, client):
    """
    Send content and image data to the AI assistant and return the response.
    """
    if not content.strip():
        logging.warning("Content is empty, skipping API call.")
        return ""
    
    system_message = f'''You are a medical billing and records assistant. Understand that multiple dates can be represented on 1 page. Your job is to list the different procedure dates, not payment dates. You will be given a page of a medical bill and you will extract the relevant data and use MM-DD-YYYY format for Date into this JSON format: 
    {{ 
        "Page Number":"{pagenumber}",
        "{filename}", [
        {{
            "List All Charges In Full Amount":[{{
                "Date":"",
                "Patient":"",
                "Medical Facility":"",
                "Name of Doctor":"",
                "ICD10CM":[],
                "CPT Codes":[],
                "Billing Line Items From Today Only or Charge Amount":[],
                "Amount Paid Already or Payments":[],
                "Amount Still Owed or Balance Due":[],
                "Rx":[],
                "Other Medically Relevant Information":[]
            }}
        }}
        ]
    }} It is imperative that you use the MM-DD-YYYY format for Date.'''
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            temperature=0.0,
            max_tokens=4096,
            system=system_message,
            messages=[
                {"role": "user", 
                 "content": [
                     {
                         "type": "image",
                         "source": {
                             "type": "base64",
                             "media_type": image_media_type,
                             "data": image_data,
                         },
                     },
                     {
                        "type" : "text",
                        "text" : content
                    },
                 ]
                },
                {
                    "role": "assistant",
                    "content": '[{'
                },
            ],
        )  
    except Exception as e:
        logging.error(f"Error communicating with Claude API: {e}")
        return ""

    # Extract the text content from the response
    if isinstance(response.content, list):
        cleaned_content = ''.join(
            item.text if hasattr(item, 'text') else item.get('text', '')
            for item in response.content
        )
    else:
        cleaned_content = response.content

    # Ensure the content starts and ends with curly braces
    cleaned_content = cleaned_content.strip()
    if not cleaned_content.startswith('{'):
        cleaned_content = '{' + cleaned_content
    if not cleaned_content.endswith('}'):
        cleaned_content += '}'
    
    # Replace newlines with spaces to ensure valid JSON
    cleaned_content = cleaned_content.replace('\n', ' ')

    return cleaned_content

def process_file(file_path, client):
    """
    Process a single cleaned text file:
    - Extract relevant data using the AI assistant.
    - Save the output as a JSON file.
    """
    logging.info(f"Processing file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Failed to read file {file_path}: {e}")
        return

    if not content:
        logging.warning(f"File {file_path} is empty.")
        return

    original_filename = os.path.basename(file_path)
    pagenumber = extract_page_number(original_filename)
    logging.debug(f"Original filename: {original_filename}")
    logging.debug(f"Page number: {pagenumber}")

    # Get image data
    image_base_name = file_path.replace('_clean.txt', '')
    image_media_type, image_data = get_image_data(image_base_name)

    # Check if image data is available
    if image_media_type is None or image_data is None:
        logging.warning(f"No image found for {original_filename}, skipping.")
        return

    cleaned_content = process_with_claude(content, original_filename, pagenumber, image_media_type, image_data, client)
    
    if not cleaned_content:
        logging.warning(f"No content received from Claude for file {file_path}.")
        return

    logging.debug(f"Cleaned content length: {len(cleaned_content)}")
    # Save the processed content
    output_filename = f"{original_filename.replace('_clean.txt', '_details.json')}"
    output_file = os.path.join(os.path.dirname(file_path), output_filename)
    output_file = get_unique_path(output_file)
    logging.info(f"Saving processed content to: {output_file}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        logging.info(f"Processed content saved to: {output_file}")
    except Exception as e:
        logging.error(f"Failed to write to file {output_file}: {e}")

def main():
    """
    Main function to process all cleaned text files in a directory:
    - Sends them to the AI assistant for detailed extraction.
    - Saves the extracted details as JSON files.
    """
    # Configure logging  # {{ edit_3 }}
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if len(sys.argv) < 2:
        logging.error("Usage: python 3details-multi.py <path_to_directory>")
        return

    # Handle paths with spaces
    input_directory = ' '.join(sys.argv[1:])
    
    if not os.path.isdir(input_directory):
        logging.error(f"Directory not found: {input_directory}")
        return

    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logging.error("ANTHROPIC_API_KEY not found in .env file")
        return

    # Get MAX_WORKERS from .env, default to 10 if not set
    try:
        max_workers = int(os.environ.get("MAX_WORKERS", 10))
    except ValueError:
        logging.warning("Invalid MAX_WORKERS value in .env. Using default value of 10.")
        max_workers = 10

    # Collect all files to process
    file_paths = [
        os.path.join(input_directory, filename)
        for filename in os.listdir(input_directory)
        if "_clean" in filename and filename.endswith('.txt')
    ]

    if not file_paths:
        logging.warning(f"No _clean.txt files found in the directory: {input_directory}")
        return

    # Initialize the Anthropic client once and pass to threads  # {{ edit_4 }}
    client = anthropic.Anthropic(api_key=api_key)

    # Process files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file_path, client): file_path for file_path in file_paths
        }
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                future.result()
                logging.info(f"Finished processing {file_path}")
            except Exception as exc:
                logging.error(f"{file_path} generated an exception: {exc}")
        
    # After processing all files, wait 1 second and start 4combine_1k.py
    time.sleep(1)
    try:
        subprocess.run(['python', 'bills/4combine_1k.py', input_directory], check=True)
        logging.info(f"Started processing with 4combine_1k.py for directory: {input_directory}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to start 4combine_1k.py for directory {input_directory}: {e}")

    logging.info("All files have been processed.")

if __name__ == "__main__":
    main()