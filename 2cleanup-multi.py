import os
import sys
from dotenv import load_dotenv
import anthropic
import concurrent.futures  # Add this import

def get_unique_path(base_path):
    path = base_path
    counter = 1
    while os.path.exists(path):
        name, ext = os.path.splitext(base_path)
        path = f"{name}_{counter}{ext}"
        counter += 1
    return path

def process_with_claude(content):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        system="Do not attempt to add any text to these files. Correct obvious spelling errors only. Return only the spell checked text.",
        messages=[
            {"role": "user", "content": content},
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

def process_file(file_path):
    print(f"Processing file: {file_path}")
    
    # Read the input file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract filename and page number
    input_filename = os.path.basename(file_path)
    input_name, _ = os.path.splitext(input_filename)
    
    # Extract page number from filename
    parts = input_name.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        page_number = parts[-1]
        filename = '_'.join(parts[:-1])
    else:
        page_number = "Unknown"
        filename = input_name

    # Create the output file path
    output_filename = f"{input_name}_clean.txt"
    output_file = os.path.join(os.path.dirname(file_path), output_filename)
    output_file = get_unique_path(output_file)
    
    # Now call process_with_claude
    cleaned_content = process_with_claude(content)
    
    # Write the cleaned content to the output file with filename and page number
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"This is File Name: {filename}_{page_number}.pdf and Page Number: {page_number} to use when referencing this file: {cleaned_content}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python 2cleanup-multi.py <path_to_directory>")
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

    # Process all .txt files in the directory concurrently
    file_paths = [
        os.path.join(input_directory, filename)
        for filename in os.listdir(input_directory)
        if filename.endswith('.txt')
    ]
    
    if not file_paths:
        print(f"No .txt files found in the directory: {input_directory}")
        return

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