import time
import os
import sys
from dotenv import load_dotenv
import anthropic
import concurrent.futures
from concurrent.futures import as_completed  # Ensure this import is present
import subprocess
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

def get_image_data(txt_file_path):
    """
    Given the path of a .txt file, return the media type and base64-encoded image data
    for the corresponding .png file in the same directory.
    """
    # Get the directory and filename without extension
    dir_path = os.path.dirname(txt_file_path)
    base_filename = os.path.splitext(os.path.basename(txt_file_path))[0]
    
    # Remove '_clean' suffix if present
    if base_filename.endswith('_clean'):
        base_filename = base_filename[:-6]
    
    # Construct the path for the .png file
    image_path = os.path.join(dir_path, f"{base_filename}.png")
    image_media_type = "image/png"
    
    print(f"Looking for image: {image_path}")  # Debug statement
    
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            image_content = f.read()
        image_data = base64.b64encode(image_content).decode('utf-8')
        return image_media_type, image_data
    else:
        print(f"Image not found: {image_path}")
        return None, None

system_message = """I am passing you the OCR text and the PNG image of a medical bill PDF page. Accuracy is key in your response. Ensure you use the MM-DD-YYYY format for Date. Use the written text and the image to compare and make sure your response is as accurate as possible. We cannot afford to miss date or write the wrong code down. Use context clues after you have exhausted your queries on the text and image to check your work and make sure you're returning quality information for this precious patient of ours."""

def process_with_claude(content, client_instance, image_media_type, image_data):
    """
    Send content and image data to Claude API and return the response.
    """
    response = client_instance.messages.create(
        model="claude-3-5-sonnet-20240620",
        temperature=0.0,
        max_tokens=4096,
        system=system_message,
        messages=[
            {
                "role": "user",
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
                        "type": "text",
                        "text": content
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

def process_file(file_path, client):
    print(f"Processing file: {file_path}")

    # Read the input file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    # Extract filename without extension
    input_filename = os.path.basename(file_path)
    filename, _ = os.path.splitext(input_filename)

    # Extract page number and base filename
    parts = filename.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        page_number = parts[-1]
        filename_without_page = '_'.join(parts[:-1])
    else:
        page_number = "Unknown"
        filename_without_page = filename

    # Get image data
    image_media_type, image_data = get_image_data(file_path)

    # Check if image data is available
    if image_media_type is None or image_data is None:
        print(f"No image found for {filename}, skipping.")
        return

    # Create the output file path
    output_filename = f"{filename}_clean.txt"
    output_file = os.path.join(os.path.dirname(file_path), output_filename)
    output_file = get_unique_path(output_file)
    
    # Now call process_with_claude with the shared client and image data
    try:
        cleaned_content = process_with_claude(content, client, image_media_type, image_data)
    except Exception as e:
        print(f"Error processing {file_path} with Claude: {e}")
        return

    # Write the cleaned content to the output file with filename and page number
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"This is File Name: {filename_without_page}_{page_number}.pdf and Page Number: {page_number} to use when referencing this file: {cleaned_content}")
    except Exception as e:
        print(f"Failed to write to {output_file}: {e}")
        return

    print(f"Successfully processed {file_path} and saved to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python 2bill-multi.py <path_to_directory>")
        return

    # Join all arguments after the script name to handle paths with spaces
    input_directory = ' '.join(sys.argv[1:])
    
    if not os.path.isdir(input_directory):
        print(f"Directory not found: {input_directory}")
        return

    # Load environment variables from the specific .env file in the 'bills' directory
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY not found in .env file")
        return

    # Get MAX_WORKERS from .env, default to 250 if not set
    try:
        max_workers = int(os.getenv('MAX_WORKERS', 250))
    except ValueError:
        print("Invalid MAX_WORKERS value in .env. Using default value of 250.")
        max_workers = 250
    print(f"MAX_WORKERS set to: {max_workers}")  # Debug statement

    # Process all .txt files in the directory and subdirectories concurrently
    file_paths = []
    for root, dirs, files in os.walk(input_directory):
        for filename in files:
            if filename.lower().endswith('.txt'):
                full_path = os.path.join(root, filename)
                file_paths.append(full_path)

    print(f"Found {len(file_paths)} .txt files to process.")  # Debug statement

    if not file_paths:
        print(f"No .txt files found in the directory: {input_directory}")
        return

    # Initialize the Anthropic client once
    client = anthropic.Anthropic()
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file, file_path, client) for file_path in file_paths]
            for future in as_completed(futures):
                try:
                    future.result()  # Will raise exception if occurred in process_file
                except Exception as exc:
                    print(f"Generated an exception: {exc}")

        print("All files have been processed.")

        # After processing, wait 1 second and start 3details-multi.py
        time.sleep(1)
        try:
            subprocess.run(['python', 'bills/3details-multi.py', input_directory], check=True)
            print(f"Started processing with 3details-multi.py for directory: {input_directory}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to start 3details-multi.py for directory {input_directory}: {e}")
    
    except Exception as e:
        print(f"An error occurred in main execution: {e}")

if __name__ == "__main__":
    main()