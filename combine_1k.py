import sys
import os
import re
import json
from dotenv import load_dotenv
import anthropic  # Anthropic Claude API client

def standardize_date(date_str):
    """
    Standardize date format to MM-DD-YYYY.
    Handles both MM-DD-YY and MM-DD-YYYY formats.
    """
    # Match MM-DD-YY format
    match = re.match(r'^(\d{2})-(\d{2})-(\d{2})$', date_str)
    if match:
        month, day, year = match.groups()
        year = '20' + year  # Assume years are in 2000s
        return f'{month}-{day}-{year}'
    # Match MM-DD-YYYY format
    match = re.match(r'^(\d{2})-(\d{2})-(\d{4})$', date_str)
    if match:
        return date_str
    else:
        # Handle other formats if necessary
        raise ValueError(f"Invalid date format: {date_str}")

def correct_json(json_str):
    """
    Attempt to correct common JSON issues like missing brackets or commas.
    """
    corrected = json_str.strip()
    # Ensure content starts with { and ends with }
    if not corrected.startswith('{'):
        corrected = '{' + corrected
    if not corrected.endswith('}'):
        corrected = corrected + '}'
    # Replace single quotes with double quotes
    corrected = corrected.replace("'", '"')
    # Attempt to parse the corrected JSON
    try:
        return json.loads(corrected)
    except json.JSONDecodeError:
        return None  # Return None if parsing fails

def extract_date_from_json(json_content):
    """
    Extract the 'Date' value from JSON content.
    """
    if 'Date' in json_content:
        return json_content['Date']
    else:
        raise ValueError("Missing 'Date' key in JSON.")

def send_to_claude(content):
    """
    Send content to Claude API and return the response.
    """
    client = anthropic.Client(api_key=os.environ['ANTHROPIC_API_KEY'])
    # Define the prompt or messages as per the API requirements
    # For this example, we'll send the content as-is
    response = client.completions.create(
        model="claude-v1",
        max_tokens_to_sample=1000,
        prompt=content
    )
    return response.completion

def main():
    if len(sys.argv) != 2:
        print("Usage: python combine_1k.py <directory_path>")
        sys.exit(1)

    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        print(f"Invalid directory path: {dir_path}")
        sys.exit(1)

    # Load environment variables
    load_dotenv()
    if 'ANTHROPIC_API_KEY' not in os.environ:
        print("ANTHROPIC_API_KEY not found in environment variables.")
        sys.exit(1)

    files_found = 0
    files_processed = 0
    output_files_created = []

    try:
        # List all files in the top level of the directory
        for filename in os.listdir(dir_path):
            if (
                "_date" in filename and
                filename.endswith('.json') and
                os.path.isfile(os.path.join(dir_path, filename))
            ):
                files_found += 1
                file_path = os.path.join(dir_path, filename)

                # Read file contents
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()

                # Attempt to parse the input JSON
                try:
                    json_content = json.loads(file_content)
                except json.JSONDecodeError:
                    # Try to correct JSON
                    json_content = correct_json(file_content)
                    if json_content is None:
                        print(f"Failed to parse JSON in file '{filename}'. Skipping.")
                        continue

                # Extract and standardize the Date
                try:
                    date_str = extract_date_from_json(json_content)
                    standardized_date = standardize_date(date_str)
                except ValueError as e:
                    print(f"Error in file '{filename}': {e}")
                    continue

                # Send content to Claude
                try:
                    response = send_to_claude(file_content)
                except Exception as e:
                    print(f"API call to Claude failed for file '{filename}': {e}")
                    continue

                # Attempt to parse Claude's response
                try:
                    response_json = json.loads(response)
                except json.JSONDecodeError:
                    # Try to correct JSON
                    response_json = correct_json(response)
                    if response_json is None:
                        print(f"Failed to parse JSON response for file '{filename}'. Using original response.")
                        response_json = response  # Use the raw response

                # Create new filename by replacing '_date' with '_final'
                new_filename = filename.replace("_date", "_final")
                output_file_path = os.path.join(dir_path, new_filename)

                # Write the response to the new file
                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    if isinstance(response_json, dict):
                        json.dump(response_json, outfile, ensure_ascii=False, indent=2)
                    else:
                        outfile.write(response)

                output_files_created.append(new_filename)
                files_processed += 1
                print(f"Processed file '{filename}' -> '{new_filename}'")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    # Informative output
    print(f"\nTotal matching files found: {files_found}")
    print(f"Total files processed: {files_processed}")
    print("Output files created:")
    for output_file in output_files_created:
        print(f"- {output_file}")

if __name__ == "__main__":
    main()