import os
import sys
from dotenv import load_dotenv
import anthropic
import concurrent.futures
import base64
import re

def get_unique_path(base_path):
    path = base_path
    counter = 1
    while os.path.exists(path):
        name, ext = os.path.splitext(base_path)
        path = f"{name}_{counter}{ext}"
        counter += 1
    return path

def process_with_claude(image_path):
    client = anthropic.Anthropic()
    
    # Read the image file and encode it as base64
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    image_media_type = "image/png"
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        system='Return your response in this strict JSON format: {"Date of Service: [Date of Service]", "Extracted Text: [Extracted Text]"} Extract only the text on the page, return nothing else but the text on the page unless the page text appears to be blank, then return only this string: BLANK!',
        messages=[
            {"role": "user", "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": 'Return your response in this strict JSON format: {"Date of Service: [Date of Service]", "Extracted Text: [Extracted Text]"} Extract only the text on the page, return nothing else but the text on the page unless the page text appears to be blank, then return only this string: BLANK!'
                }
            ]}
        ]
    )

    # Extract the text content from the response
    if isinstance(response.content, list):
        cleaned_content = ' '.join(item.text for item in response.content if hasattr(item, 'text'))
    else:
        cleaned_content = response.content
    
    # Remove any leading or trailing whitespace
    cleaned_content = cleaned_content.strip()
    
    return cleaned_content

def extract_filename_and_page(input_filename):
    match = re.match(r'(.+)_(\d+)\.png', input_filename)
    if match:
        filename = match.group(1)
        page_number = match.group(2)
        return f"{filename}_{page_number}", page_number
    return input_filename, "Unknown"

def process_file(file_path):
    print(f"Processing file: {file_path}")
    
    # Extract filename without extension
    input_filename = os.path.basename(file_path)
    input_name, _ = os.path.splitext(input_filename)
    
    # Extract filename and page number
    filename, page_number = extract_filename_and_page(input_filename)
    
    # Create the output file path
    output_filename = f"{input_name}_clean.txt"
    output_file = os.path.join(os.path.dirname(file_path), output_filename)
    output_file = get_unique_path(output_file)
    
    # Process with Claude
    cleaned_content = process_with_claude(file_path)
    
    # Write the cleaned content to the output file with filename and page number
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"This is File Name: {filename}.pdf and Page Number: {page_number} to use when referencing this file: {cleaned_content}")
    
    print(f"Processed and saved: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python claude_ocr.py <path_to_directory>")
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

    # Get MAX_WORKERS from .env, default to 5 if not set
    max_workers = int(os.environ.get("MAX_WORKERS", 5))

    # Process all .png files in the directory concurrently
    png_files = [
        os.path.join(input_directory, filename)
        for filename in os.listdir(input_directory)
        if filename.lower().endswith('.png')
    ]
    
    if not png_files:
        print(f"No .png files found in the directory: {input_directory}")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file_path): file_path for file_path in png_files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{file_path} generated an exception: {exc}")

if __name__ == "__main__":
    main()