import sys
import os
import re
from dotenv import load_dotenv
import anthropic

def standardize_date(date_str):
    """
    Standardize date format to MM-DD-YYYY.
    Handles both MM-DD-YY and MM-DD-YYYY formats.
    """
    match = re.match(r'^(\d{2})-(\d{2})-(\d{2})$', date_str)
    if match:
        month, day, year = match.groups()
        year = '20' + year
        return f'{month}-{day}-{year}'
    match = re.match(r'^(\d{2})-(\d{2})-(\d{4})$', date_str)
    if match:
        return date_str
    raise ValueError(f"Invalid date format: {date_str}")

def extract_date_from_content(content):
    """
    Extract the 'Date' value from the content string.
    """
    match = re.search(r'"Date"\s*:\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    raise ValueError("Missing 'Date' key in content.")

def process_with_claude(content, filename):
    """
    Send content to Claude API and return the response.
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        system='Review these JSON Objects, notice some are comma separated. Combine them into a single JSON Array. Ensure the Date is present and correct in the MM-DD-YYYY format. {{"Date":"VALUE","File Name":"{filename}","Page Number":"{pagenumber}","Patient":"","Medical Facility":"","Referred By":"" | null,"Referred To":"" | null,"Name of Doctor":"","ICD10CM":[],"CPT Codes":[],"Rx":[],"Other Medically Relevant Information":[],"Visit Summary":""}}',
        messages=[
            {"role": "user", "content": content},
            {"role": "assistant", "content": "{"}
        ]
    )
    return response.content[0].text if response.content else ""

def main():
    if len(sys.argv) != 2:
        print("Usage: python 5present.py <directory_path>")
        sys.exit(1)

    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        print(f"Invalid directory path: {dir_path}")
        sys.exit(1)

    load_dotenv()
    if 'ANTHROPIC_API_KEY' not in os.environ:
        print("ANTHROPIC_API_KEY not found in environment variables.")
        sys.exit(1)

    files_found = 0
    files_processed = 0
    output_files_created = []

    try:
        for filename in os.listdir(dir_path):
            if "_date" in filename and filename.endswith('.json') and os.path.isfile(os.path.join(dir_path, filename)):
                files_found += 1
                file_path = os.path.join(dir_path, filename)

                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()

                try:
                    date_str = extract_date_from_content(file_content)
                    standardized_date = standardize_date(date_str)
                except ValueError as e:
                    print(f"Error in file '{filename}': {e}")
                    continue

                try:
                    response = process_with_claude(file_content, filename)
                except Exception as e:
                    print(f"API call to Claude failed for file '{filename}': {e}")
                    continue

                new_filename = filename.replace("_date", "_final")
                output_file_path = os.path.join(dir_path, new_filename)

                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(response)

                output_files_created.append(new_filename)
                files_processed += 1
                print(f"Processed file '{filename}' -> '{new_filename}'")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    print(f"\nTotal matching files found: {files_found}")
    print(f"Total files processed: {files_processed}")
    print("Output files created:")
    for output_file in output_files_created:
        print(f"- {output_file}")

if __name__ == "__main__":
    main()