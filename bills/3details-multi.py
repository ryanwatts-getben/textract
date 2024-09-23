import os
import sys
import json
from dotenv import load_dotenv
import anthropic
import re
import concurrent.futures
import base64
import httpx

def get_unique_path(base_path):
    path = base_path
    counter = 1
    while os.path.exists(path):
        name, ext = os.path.splitext(base_path)
        path = f"{name}_{counter}{ext}"
        counter += 1
    return path

def extract_page_number(filename):
    match = re.search(r'_(\d+)_', filename)
    return match.group(1) if match else "Unknown"

def get_image_data(filename):
    image_path = f"{filename}.png"
    image_media_type = "image/png"
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            image_content = f.read()
        image_data = base64.b64encode(image_content).decode('utf-8')
        return image_media_type, image_data
    else:
        print(f"Image not found: {image_path}")
        return None, None

def process_with_claude(content, filename, pagenumber, image_media_type, image_data):
    if not content.strip():
        print("Content is empty, skipping API call.")
        return ""
    
    client = anthropic.Anthropic()
    system_message = f'''You are a medical billing and records assistant. Understand that multiple dates can be represented on 1 page. Your job is to list the different procedure dates, not payment dates. You will be given a page of a medical bill and you will extract the relevant data using MM-DD-YYYY format for Date into this JSON format: 
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
        }},
        {{
            "List All Payments":[{{
                "Date":"",
                "Patient":"",
                "Medical Facility":"",
                "Name of Doctor":"",
                "ICD10CM":[],
                "CPT Codes":[],
                "Billing Line Items From Today Only":[],
                "Amount Paid Already":[],
                "Amount Still Owed":[],
                "Rx":[],
                "Other Medically Relevant Information":[]
            }},
            "List All Adjustments":[{{
                "Date":"",
                "Patient":"",
                "Medical Facility":"",
                "Name of Doctor":"",
                "ICD10CM":[],
                "CPT Codes":[],
                "Billing Line Items From Today Only":[],
                "Amount Paid Already":[],
            }}
        }},
        ...
    ]
    }}'''
    
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
                    "content": '{'
                },
            ],
        )  
    
    # Handle response.content being a list
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

    # Return the cleaned content as a string
    return cleaned_content

def process_file(file_path):
    print(f"Processing file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content:
        print(f"File {file_path} is empty.")
        return

    original_filename = os.path.basename(file_path)
    pagenumber = extract_page_number(original_filename)
    print(f"Original filename: {original_filename}")
    print(f"Page number: {pagenumber}")

    # Get image data
    image_base_name = file_path.replace('_clean.txt', '')
    image_media_type, image_data = get_image_data(image_base_name)

    # Check if image data is available
    if image_media_type is None or image_data is None:
        print(f"No image found for {original_filename}, skipping.")
        return

    cleaned_content = process_with_claude(content, original_filename, pagenumber, image_media_type, image_data)
    print(f"Cleaned content length: {len(cleaned_content)}")
    print(f"Cleaned content: {cleaned_content}")
    # The cleaned_content is already a valid JSON string, so we can write it directly
    output_filename = f"{original_filename.replace('_clean.txt', '_details.json')}"
    output_file = os.path.join(os.path.dirname(file_path), output_filename)
    output_file = get_unique_path(output_file)
    print(f"Saving processed content to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

    print(f"Processed content saved to: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python 3details-multi.py <path_to_directory>")
        return

    # Join all arguments after the script name to handle paths with spaces
    input_directory = ' '.join(sys.argv[1:])
    
    if not os.path.isdir(input_directory):
        print(f"Directory not found: {input_directory}")
        return

    # Load environment variables
    load_dotenv()
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY not found in .env file")
        return

    # Get MAX_WORKERS from .env, default to 10 if not set
    max_workers = int(os.environ.get("MAX_WORKERS", 10))

    # Collect all files to process
    file_paths = [
        os.path.join(input_directory, filename)
        for filename in os.listdir(input_directory)
        if "_clean" in filename and filename.endswith('.txt')
    ]

    if not file_paths:
        print(f"No _clean.txt files found in the directory: {input_directory}")
        return

    # Process files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file_path): file_path for file_path in file_paths
        }
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                future.result()
                print(f"Finished processing {file_path}")
            except Exception as exc:
                print(f"{file_path} generated an exception: {exc}")

if __name__ == "__main__":
    main()