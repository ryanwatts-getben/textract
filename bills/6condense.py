import os
import sys
import json
import re
import logging
from dotenv import load_dotenv
import anthropic
import traceback

def extract_json_from_response(response_text):
    """
    Extract the JSON data from the assistant's response, handling any extraneous text.
    """
    # Remove code fences if present
    response_text = response_text.strip()
    response_text = re.sub(r'^```(?:json)?', '', response_text, flags=re.IGNORECASE | re.MULTILINE)
    response_text = re.sub(r'```$', '', response_text, flags=re.MULTILINE)
    response_text = response_text.strip()
    
    # Find the first occurrence of '{' or '['
    first_brace = response_text.find('{')
    first_bracket = response_text.find('[')
    if first_brace == -1 and first_bracket == -1:
        raise ValueError("No JSON object found in the response.")
    if first_brace == -1 or (first_bracket != -1 and first_bracket < first_brace):
        start = first_bracket
        open_char, close_char = '[', ']'
    else:
        start = first_brace
        open_char, close_char = '{', '}'

    # Use a stack to find the matching closing brace/bracket
    stack = []
    end = None
    for index in range(start, len(response_text)):
        char = response_text[index]
        if char == open_char:
            stack.append(char)
        elif char == close_char:
            if not stack:
                raise ValueError("No matching opening brace/bracket for closing bracket.")
            stack.pop()
            if not stack:
                end = index + 1
                break
    if end is None:
        raise ValueError("No matching closing brace/bracket found.")

    json_str = response_text[start:end]
    return json_str

def reformat_date(date_str):
    """
    Convert date from MM-DD-YYYY to YYYY-MM-DD format.
    """
    match = re.match(r'^(\d{2})-(\d{2})-(\d{4})$', date_str)
    if match:
        mm, dd, yyyy = match.groups()
        return f"{yyyy}-{mm}-{dd}"
    else:
        # If date format is unexpected, return as-is or handle accordingly
        return date_str  # Alternatively, raise an error

def merge_json_objects(json_objects):
    """
    Merge multiple JSON objects into one, handling conflicts appropriately.
    """
    # Assuming all JSON objects are dictionaries
    merged_result = {}
    for obj in json_objects:
        # Update the merged_result with the contents of obj
        # This simple merge may overwrite keys; customize as needed
        for key, value in obj.items():
            if key in merged_result:
                # Handle conflicts if necessary
                if isinstance(merged_result[key], list) and isinstance(value, list):
                    merged_result[key].extend(value)
                elif isinstance(merged_result[key], dict) and isinstance(value, dict):
                    merged_result[key] = merge_json_objects([merged_result[key], value])
                else:
                    # If values are not mergeable, you can decide how to handle it
                    merged_result[key] = value  # Overwrite for simplicity
            else:
                merged_result[key] = value
    return merged_result

def merge_with_assistant(json_objects, date_str, client):
    """
    Send the JSON objects to the assistant to merge them.
    """
    system_message = """You are a helpful assistant that merges multiple JSON records into a single JSON object. Please merge the following JSON records, ensuring that the merged JSON maintains all necessary details without data loss."""

    user_content = "Please merge the following JSON objects:\n"
    for idx, obj in enumerate(json_objects, 1):
        user_content += f"Object {idx}:\n```json\n{json.dumps(obj, indent=2)}\n```\n"

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            temperature=0.0,
            max_tokens=4096,
            system=system_message,
            messages=[
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": '{'  # Indicate that the assistant should start with '{'
                },
            ]
        )

        # Extract the assistant's reply
        if response and response.content:
            assistant_response = ''
            if isinstance(response.content, list):
                assistant_response = ''.join(
                    item.text if hasattr(item, 'text') else item.get('text', '')
                    for item in response.content
                )
            else:
                assistant_response = response.content

            assistant_response = '{' + assistant_response.strip()
            logging.debug(f"Assistant response: {assistant_response}")

            # Trim the assistant's response to extract valid JSON
            merged_json_str = extract_json_from_response(assistant_response)
            merged_json = json.loads(merged_json_str)
            return merged_json
        else:
            logging.error("No completion received from the assistant.")
            return None

    except Exception as e:
        logging.error(f"Error communicating with the assistant: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None

def process_files(input_directory, client):
    logging.info(f"Processing files in directory: {input_directory}")

    # Collect all .json files (excluding VALUE.json, VALUE!.json, VALUE2.json)
    json_file_paths = [
        os.path.join(input_directory, filename)
        for filename in os.listdir(input_directory)
        if filename.endswith('.json') and
           not filename.startswith('VALUE') and
           os.path.isfile(os.path.join(input_directory, filename))
    ]

    if not json_file_paths:
        logging.warning(f"No JSON files found in {input_directory}")
        return

    # Group files by date (assuming filenames are in 'MM-DD-YYYY.json' format)
    date_pattern = re.compile(r'(\d{2}-\d{2}-\d{4})\.json')
    date_to_files = {}
    for file_path in json_file_paths:
        filename = os.path.basename(file_path)
        match = date_pattern.search(filename)
        if match:
            date_str = match.group(1)
            date_to_files.setdefault(date_str, []).append(file_path)
        else:
            logging.warning(f"File {filename} does not match date pattern. Skipping.")

    for date_str, files in date_to_files.items():
        logging.info(f"Processing date: {date_str} with files: {files}")
        json_objects = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Attempt to parse JSON
                json_obj = json.loads(content)
                json_objects.append(json_obj)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from file {file_path}: {e}")
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")

        if len(json_objects) >= 2:
            # Try to merge JSON objects locally
            try:
                merged_json = merge_json_objects(json_objects)
                # Minify and save the merged JSON
                output_date_str = reformat_date(date_str)
                output_filename = f"{output_date_str}.json"
                output_file_path = os.path.join(input_directory, output_filename)
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_json, f, separators=(',', ':'), ensure_ascii=False)
                logging.info(f"Merged JSON saved to {output_file_path}")
                continue  # Process next date
            except Exception as e:
                logging.error(f"Error merging JSON objects for date {date_str}: {e}")
                # Proceed to assistant merging
        else:
            logging.warning(f"Not enough JSON files for date {date_str} to merge locally.")

        # If local merge failed or not possible, send to assistant
        merged_json = merge_with_assistant(json_objects, date_str, client)
        if merged_json:
            try:
                output_date_str = reformat_date(date_str)
                output_filename = f"{output_date_str}.json"
                output_file_path = os.path.join(input_directory, output_filename)
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_json, f, separators=(',', ':'), ensure_ascii=False)
                logging.info(f"Merged JSON from assistant saved to {output_file_path}")
            except Exception as e:
                logging.error(f"Failed to write merged JSON for date {date_str}: {e}")
        else:
            logging.error(f"Assistant failed to merge JSON for date {date_str}")
            # If that fails, write the files as a string in the JSON output file
            output_filename = f"{date_str}_unmerged.json"
            output_file_path = os.path.join(input_directory, output_filename)
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for idx, obj in enumerate(json_objects, 1):
                        f.write(f"Object {idx}:\n{json.dumps(obj, indent=2)}\n\n")
                logging.info(f"Saved unmerged JSON objects to {output_file_path}")
            except Exception as e:
                logging.error(f"Failed to write unmerged JSON for date {date_str}: {e}")

    logging.info("All processing tasks have been completed.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python 6condense.py <path_to_directory>")
        return

    input_directory = ' '.join(sys.argv[1:])

    if not os.path.isdir(input_directory):
        print(f"Directory not found: {input_directory}")
        return

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        logging.error("ANTHROPIC_API_KEY not found in environment variables.")
        return

    # Initialize the Anthropic client
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize Anthropic client: {e}")
        return

    process_files(input_directory, client)

if __name__ == "__main__":
    main()