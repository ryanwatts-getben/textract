import os
import sys
import json
from dotenv import load_dotenv
import anthropic
import re

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

def process_with_claude(content, filename, pagenumber):
    if not content.strip():
        print("Content is empty, skipping API call.")
        return ""
    
    client = anthropic.Anthropic()
    system_message = f'''You are a medical assistant. Extract the information and return it in a strict JSON format {{"Date":"","File Name":"{filename}","Page Number":"{pagenumber}","Patient":"","Medical Facility":"","Referred By":"" | null,"Referred To":"" | null,"Name of Doctor":"","ICD10CM":[],"CPT Codes":[],"Rx":[],"Other Medically Relevant Information":[],"Visit Summary":""}}. Ensure that "File Name" and "Page Number" are exactly as provided.'''
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        system=system_message,
        messages=[
            {"role": "user", "content": content},
            {"role": "assistant", "content": "{"}
        ]
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

    # Attempt to parse the JSON
    try:
        parsed_json = json.loads(cleaned_content)
        final_content = json.dumps(parsed_json, ensure_ascii=False, separators=(',', ':'))
        print("Parsed JSON successfully.")
        return final_content
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        print(f"Error at position {e.pos}: {cleaned_content[max(0, e.pos-20):e.pos+20]}")
        
        # If parsing fails, return the content wrapped in a JSON object
        wrapped_content = json.dumps({"raw_content": cleaned_content})
        print("Returning wrapped content.")
        return wrapped_content

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

    cleaned_content = process_with_claude(content, original_filename, pagenumber)
    print(f"Cleaned content length: {len(cleaned_content)}")

    # The cleaned_content is already a valid JSON string, so we can write it directly
    output_filename = f"{original_filename.replace('_clean.txt', '_details.json')}"
    output_file = os.path.join(os.path.dirname(file_path), output_filename)
    output_file = get_unique_path(output_file)
    print(f"Saving processed content to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

    print(f"Processed content saved to: {output_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python details.py <path_to_directory>")
        return

    input_directory = sys.argv[1]
    if not os.path.isdir(input_directory):
        print(f"Directory not found: {input_directory}")
        return

    # Load environment variables
    load_dotenv()
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY not found in .env file")
        return

    # Process all files in the directory containing "_clean" in their filename and ending with .txt
    files_found = False
    for filename in os.listdir(input_directory):
        print(f"Found file: {filename}")  # Added print statement
        if "_clean" in filename and filename.endswith('.txt'):
            file_path = os.path.join(input_directory, filename)
            process_file(file_path)
            files_found = True

    if not files_found:
        print("No files found to process.")  # Added print statement

if __name__ == "__main__":
    main()