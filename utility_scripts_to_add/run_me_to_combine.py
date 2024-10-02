import json
import os
import re
import logging
from datetime import datetime
from collections import defaultdict
import sys
import subprocess

# Set up logging configuration to display messages with timestamps and severity levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_first_date_from_content(content):
    """
    Extract the first date from the content using regex patterns.
    Standardizes the date to the format YYYY-MM-DD.
    """
    # Define regex patterns for different date formats
    date_patterns = [
        r'"Date"\s*:\s*"(\d{2}-\d{2}-\d{4})"',  # Matches "dd-mm-yyyy"
        r'"Date"\s*:\s*"(\d{4}-\d{2}-\d{2})"',  # Matches "yyyy-mm-dd"
        r'"Date"\s*:\s*"(\d{2}/\d{2}/\d{4})"',  # Matches "dd/mm/yyyy"
        r'"Date"\s*:\s*"(\d{2}/\d{2}/\d{2})"',  # Matches "mm/dd/yy"
        r'"Date"\s*:\s*"(\d{2}-\d{2}-\d{2})"',  # Matches "mm-dd-yy"
    ]
    
    # Search for the first matching date pattern in the content
    for pattern in date_patterns:
        match = re.search(pattern, content)
        if match:
            date_str = match.group(1)
            
            # Standardize the date format to YYYY-MM-DD
            if '/' in date_str:
                date_str = date_str.replace('/', '-')
            
            date_parts = date_str.split('-')
            
            if len(date_parts[0]) == 4:
                # Date is already in yyyy-mm-dd format
                standardized_date = date_str
            elif len(date_parts[2]) == 2:
                # Convert yy to yyyy (assuming 20yy)
                date_parts[2] = '20' + date_parts[2]
                standardized_date = f"{date_parts[2]}-{date_parts[0]}-{date_parts[1]}"
            else:
                # Convert dd-mm-yyyy to yyyy-mm-dd
                standardized_date = f"{date_parts[2]}-{date_parts[0]}-{date_parts[1]}"
                
            return standardized_date
    
    return None  # Return None if no date is found

def clean_and_extract_json(content):
    """
    Remove non-JSON text from the content and attempt to extract valid JSON.
    """
    # Find the first '{' and last '}' to extract JSON content
    start_index = content.find('{')
    end_index = content.rfind('}')
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None  # Return None if valid JSON delimiters are not found
    json_str = content[start_index:end_index+1]

    # Attempt to parse the extracted string as JSON
    try:
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        # Attempt to clean the string by removing control characters and retry parsing
        json_str_cleaned = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        try:
            parsed_json = json.loads(json_str_cleaned)
            return parsed_json
        except json.JSONDecodeError:
            return None  # Return None if parsing fails after attempts

def process_files(input_directory):
    """
    Process files from the input directory and organize contents by date.
    """
    contents_by_date = defaultdict(list)
    all_files = []

    # Collect all .txt files from the input directory
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.txt'):
                all_files.append(os.path.join(root, file))

    if not all_files:
        logger.warning(f"No '.txt' files found in {input_directory}")
        return None

    logger.info(f"Total '.txt' files found: {len(all_files)}")

    # Process each collected file
    for file_path in all_files:
        logger.info(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            date = extract_first_date_from_content(content)
            if not date:
                logger.warning(f"No date found in {file_path}")
                continue
            # Clean content and extract JSON
            parsed_json = clean_and_extract_json(content)
            if parsed_json is not None:
                contents_by_date[date].append(parsed_json)
            else:
                logger.error(f"Failed to parse JSON from file: {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")

    return contents_by_date

def main(input_directory):
    """
    Main function to process files from the input directory and save merged results.
    """
    # Define the output directory for combined files
    output_directory = os.path.join(input_directory, 'combine')
    os.makedirs(output_directory, exist_ok=True)

    logger.info(f"Processing files in: {input_directory}")

    # Process files and organize contents by date
    contents_by_date = process_files(input_directory)

    if not contents_by_date:
        logger.warning("No contents to process after parsing files.")
        return

    # Output a JSON file for each date
    for date, records in contents_by_date.items():
        output_file = os.path.join(output_directory, f"records_{date}.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2)
            logger.info(f"Saved combined records to {output_file}")
        except Exception as e:
            logger.error(f"Error writing combined records to {output_file}: {str(e)}")

    # Prepare input for the next step (e.g., 3details.py)
    next_step_input = json.dumps({
        'file_path': output_directory,
    })
    
    # Trigger the next step
    logger.info("Triggering 3details.py")
    try:
        subprocess.run(['python', '3details.py', next_step_input], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing 3details.py: {e}")
        sys.exit(1)

    logger.info("Processing completed successfully")

if __name__ == "__main__":
    # Ensure the script is run with the correct number of arguments
    if len(sys.argv) != 2:
        logger.error("Usage: python run_me_to_combine.py <input_directory>")
        sys.exit(1)
    
    # Get the input directory from command-line arguments
    input_directory = sys.argv[1]
    main(input_directory)