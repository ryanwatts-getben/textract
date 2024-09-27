import os
import re
import logging
import sys
import json
from collections import defaultdict

# Configure logging to output to the terminal
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger()

def extract_first_date_from_content(content):
    """
    Extract the first date from the content using regex.
    Standardizes date to YYYY-MM-DD format.
    """
    date_patterns = [
        r'"Date"\s*:\s*"(\d{2}-\d{2}-\d{4})"',  # Matches "dd-mm-yyyy"
        r'"Date"\s*:\s*"(\d{4}-\d{2}-\d{2})"',  # Matches "yyyy-mm-dd"
        r'"Date"\s*:\s*"(\d{2}/\d{2}/\d{4})"',  # Matches "dd/mm/yyyy"
        r'"Date"\s*:\s*"(\d{2}/\d{2}/\d{2})"',  # Matches "dd/mm/yy"
        r'"Date"\s*:\s*"(\d{2}-\d{2}-\d{2})"',  # Matches "dd-mm-yy"
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, content)
        if match:
            date_str = match.group(1)
            
            # Standardize various formats to YYYY-MM-DD
            if '/' in date_str:
                # Convert slashes to dashes
                date_str = date_str.replace('/', '-')
            
            date_parts = date_str.split('-')
            
            if len(date_parts[0]) == 4:
                # Date is already in yyyy-mm-dd format
                standardized_date = date_str
            elif len(date_parts[2]) == 2:
                # Handle dd-mm-yy, convert yy to yyyy (assuming 20yy)
                date_parts[2] = '20' + date_parts[2]
                standardized_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
            else:
                # Convert dd-mm-yyyy to yyyy-mm-dd
                standardized_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
                
            return standardized_date
    
    return None  # No date found

def clean_and_extract_json(content):
    """
    Remove non-JSON text from the content and attempt to extract valid JSON.
    """
    # Attempt to find the first '{' and last '}' to extract JSON content
    start_index = content.find('{')
    end_index = content.rfind('}')
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None  # Cannot find valid JSON delimiters
    json_str = content[start_index:end_index+1]

    # Attempt to parse the extracted string as JSON
    try:
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        # Attempt common fixes
        # Remove control characters and retry
        json_str_cleaned = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        try:
            parsed_json = json.loads(json_str_cleaned)
            return parsed_json
        except json.JSONDecodeError:
            return None  # Parsing failed after attempts

def main():
    if len(sys.argv) != 2:
        print("Usage: python 2combine-local.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.isdir(directory_path):
        print(f"The provided path '{directory_path}' is not a directory.")
        sys.exit(1)

    logger.info(f"Processing files in directory: {directory_path}")

    contents_by_date = defaultdict(list)

    # Iterate over all .txt files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            logger.info(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    date = extract_first_date_from_content(content)
                    if not date:
                        logger.warning(f"No date found in {filename}")
                        continue

                    # Clean content and extract JSON
                    parsed_json = clean_and_extract_json(content)
                    if parsed_json is not None:
                        contents_by_date[date].append(parsed_json)
                    else:
                        logger.error(f"Failed to parse JSON from file: {filename}")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")

    if not contents_by_date:
        logger.warning("No contents to process after parsing files.")
        sys.exit(0)

    # Output a JSON file for each date
    for date, records in contents_by_date.items():
        output_filename = f"records_{date}.json"
        output_path = os.path.join(directory_path, output_filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(records, outfile, indent=2)
            logger.info(f"Saved combined records to {output_path}")
        except Exception as e:
            logger.error(f"Error writing combined records to file {output_filename}: {str(e)}")

if __name__ == "__main__":
    main()