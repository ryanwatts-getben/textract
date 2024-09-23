import os
import sys
from dotenv import load_dotenv
import anthropic

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
        system="Avoid acknowledging the user's request with statements like 'Here is the cleaned up text from the medical record:' or it will destroy the system we have built. Understand these are OCR'd medical records and the quality of the OCR can vary. Let's first clean this OCR up the best we can to make it look like proper English, with the consideration that the medical jargon and shorthand are important to maintain. It is imperative that you only return the cleaned text or it will break the pipeline.",
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
    
    # Extract page number using regex
    match = re.search(r'_(\d+)\.txt$', input_filename)
    page_number = match.group(1) if match else "Unknown"

    # Create the output file path
    output_filename = f"{input_name}_clean.txt"
    output_file = os.path.join(os.path.dirname(file_path), output_filename)
    output_file = get_unique_path(output_file)
    
    # Now call process_with_claude
    cleaned_content = process_with_claude(content)
    
    # Write the cleaned content to the output file with filename and page number
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"This is File Name: {input_filename} and Page Number: {page_number} to use when referencing this file: {cleaned_content}")

    print(f"Processed content saved to: {output_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python cleanup.py <path_to_directory>")
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

    # Process all .txt files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_directory, filename)
            process_file(file_path)

if __name__ == "__main__":
    main()