import sys
import os
import re
import time
import subprocess
import logging
import base64
import anthropic  # Ensure the anthropic package is installed
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def standardize_date(date_str):
    """Ensure the date is in MM-DD-YYYY format."""
    # Replace slashes with dashes for consistency
    normalized_date = date_str.replace('/', '-')
    # Enforce MM-DD-YYYY format and ensure it starts with '20'
    match = re.match(r'^(\d{2})-(\d{2})-(20\d{2})$', normalized_date)
    if match:
        return normalized_date
    else:
        raise ValueError(f"Date '{date_str}' is not in MM-DD-YYYY format.")

def collect_surrounding_pages(page_number, dir_path, total_pages):
    """Collect up to 5 pages before and after the given page number."""
    page_number = int(page_number)
    start_page = max(1, page_number - 5)
    end_page = min(total_pages, page_number + 5)

    pages = []
    for n in range(start_page, end_page + 1):
        image_file = os.path.join(dir_path, f"Page_{n}.png")
        text_file = os.path.join(dir_path, f"Page_{n}.txt")

        files_exist = True
        if not os.path.isfile(image_file):
            logging.warning(f"Warning: {image_file} does not exist.")
            files_exist = False
        if not os.path.isfile(text_file):
            logging.warning(f"Warning: {text_file} does not exist.")
            files_exist = False

        if files_exist:
            # Read and encode image data
            with open(image_file, "rb") as img_f:
                image_data = base64.b64encode(img_f.read()).decode('utf-8')
            # Read text data
            with open(text_file, "r", encoding='utf-8') as txt_f:
                text_data = txt_f.read()

            pages.append({
                "page_number": n,
                "image_data": image_data,
                "text_data": text_data
            })
    return pages

def process_with_claude(all_missing_pages):
    """Send all the collected pages to the AI assistant to extract missing dates at once."""
    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logging.error("ANTHROPIC_API_KEY not found in environment variables.")
        return {}

    client = anthropic.Anthropic(api_key=api_key)

    # Prepare messages for Claude
    messages = []
    system_message = """You are a helpful assistant that extracts missing 'Date' information from provided pages. 
For each page, return the page number and the extracted date in MM-DD-YYYY format."""

    messages.append({"role": "system", "content": system_message})

    # For each missing page, include the surrounding pages
    for page_info in all_missing_pages:
        page_number = page_info['page_number']
        pages = page_info['pages']

        content = f"Extract the missing 'Date' for Page {page_number} based on the following pages:\n"
        for p in pages:
            content += f"\n--- Page {p['page_number']} ---\n"
            content += f"Image (base64-encoded): {p['image_data']}\n"
            content += f"Text:\n{p['text_data']}\n"

        messages.append({"role": "user", "content": content})

    # Concatenate all user messages
    full_prompt = anthropic.HUMAN_PROMPT
    for message in messages:
        if message['role'] == 'system':
            full_prompt += message['content'] + anthropic.HUMAN_PROMPT
        elif message['role'] == 'user':
            full_prompt += message['content'] + anthropic.HUMAN_PROMPT

    # Send the messages to Claude
    try:
        response = client.completions.create(
            model="claude-3-5-sonnet-20240620",
            temperature=0.0,
            max_tokens=4096,
            prompt=full_prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT]
        )
        # Parse the response to extract dates for each page
        completion = response.completion.strip()
        # Assuming the assistant returns data in a structured format like JSON or a list
        # Parse and map the extracted dates to page numbers
        extracted_dates = {}
        # For example, if response is in format: "Page X: MM-DD-YYYY\nPage Y: MM-DD-YYYY"
        for line in completion.splitlines():
            match = re.match(r'Page\s+(\d+):\s*([\d\-]+)', line)
            if match:
                pg_num = match.group(1)
                date_str = match.group(2)
                extracted_dates[pg_num] = date_str
        return extracted_dates
    except Exception as e:
        logging.error(f"Error communicating with Claude: {e}")
        return {}

def main():
    if len(sys.argv) != 2:
        logging.error("Usage: python 4combine_1k.py <directory_path>")
        sys.exit(1)

    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        logging.error(f"Invalid directory path: {dir_path}")
        sys.exit(1)

    files_processed = 0
    output_files = {}  # Tracks which output files have been initiated
    missing_date_pages = []  # Collect all pages with missing dates
    total_pages = 0

    try:
        # Get total number of pages
        for filename in os.listdir(dir_path):
            if filename.startswith("Page_") and filename.endswith(".txt"):
                match = re.match(r'Page_(\d+)\.txt', filename)
                if match:
                    page_num = int(match.group(1))
                    total_pages = max(total_pages, page_num)

        # Iterate over all JSON files with "_details.json" in their filename
        for filename in os.listdir(dir_path):
            if (
                filename.endswith('_details.json') and
                os.path.isfile(os.path.join(dir_path, filename))
            ):
                file_path = os.path.join(dir_path, filename)
                files_processed += 1

                logging.info(f"Processing file: {filename}")

                # Read the entire content of the JSON file as a string
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                except Exception as e:
                    logging.error(f"Failed to read file '{filename}': {e}")
                    continue

                # Use regex to extract all JSON objects within the string
                json_objects = re.findall(r'\{[^{}]*\}', content, re.DOTALL)

                for json_obj in json_objects:
                    # Extract the 'Date' value using regex without parsing entire JSON
                    date_match = re.search(r'"Date"\s*:\s*"([^"]+)"', json_obj)
                    if date_match:
                        date_str = date_match.group(1).strip()
                        try:
                            standardized_date = standardize_date(date_str)
                        except ValueError as ve:
                            logging.warning(f"Skipping JSON object in file '{filename}': {ve}")
                            continue

                        # Create the output filename by removing '_details' and adding '_date.json'
                        input_base_name = filename.replace('_details.json', '')
                        output_filename = f"{input_base_name}_date.json"
                        output_file_path = os.path.join(dir_path, output_filename)

                        # Initialize the file with '[' if not already done
                        if output_filename not in output_files:
                            try:
                                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                                    outfile.write('[' + json_obj)
                                output_files[output_filename] = True
                                logging.info(f"Created new output file: {output_filename}")
                            except Exception as e:
                                logging.error(f"Failed to write to file '{output_filename}': {e}")
                                continue
                        else:
                            # Append a comma and the JSON object
                            try:
                                with open(output_file_path, 'a', encoding='utf-8') as outfile:
                                    outfile.write(',' + json_obj)
                                logging.info(f"Appended to output file: {output_filename}")
                            except Exception as e:
                                logging.error(f"Failed to append to file '{output_filename}': {e}")
                                continue
                    else:
                        # When 'Date' key is missing
                        logging.warning(f"No 'Date' key found in JSON object in file '{filename}'. Will attempt to extract date using surrounding pages later.")
                        # Extract page number from filename
                        match = re.search(r'Page_(\d+)_details\.json', filename)
                        if match:
                            page_number = match.group(1)
                            # Collect surrounding pages
                            pages = collect_surrounding_pages(page_number, dir_path, total_pages)
                            if not pages:
                                logging.warning(f"No surrounding pages found for page {page_number}. Skipping.")
                                continue
                            missing_date_pages.append({
                                'page_number': page_number,
                                'pages': pages,
                                'filename': filename,
                                'json_obj': json_obj
                            })
                        else:
                            logging.warning(f"Could not determine page number from filename '{filename}'. Skipping.")
                            continue

        # Process all missing date pages at once
        if missing_date_pages:
            logging.info("Processing pages with missing dates...")
            extracted_dates = process_with_claude(missing_date_pages)
            for page_info in missing_date_pages:
                page_number = page_info['page_number']
                json_obj = page_info['json_obj']
                filename = page_info['filename']

                if page_number in extracted_dates:
                    date_str = extracted_dates[page_number]
                    try:
                        standardized_date = standardize_date(date_str)
                    except ValueError as ve:
                        logging.warning(f"Extracted date is invalid for page {page_number}: {ve}")
                        continue

                    # Insert the extracted date into the json_obj
                    json_obj_with_date = re.sub(r'\{', f'{{"Date":"{standardized_date}", ', json_obj, 1)

                    # Create the output filename
                    input_base_name = filename.replace('_details.json', '')
                    output_filename = f"{input_base_name}_date.json"
                    output_file_path = os.path.join(dir_path, output_filename)

                    # Initialize or append to the output file
                    if output_filename not in output_files:
                        try:
                            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                                outfile.write('[' + json_obj_with_date)
                            output_files[output_filename] = True
                            logging.info(f"Created new output file: {output_filename}")
                        except Exception as e:
                            logging.error(f"Failed to write to file '{output_filename}': {e}")
                            continue
                    else:
                        try:
                            with open(output_file_path, 'a', encoding='utf-8') as outfile:
                                outfile.write(',' + json_obj_with_date)
                            logging.info(f"Appended to output file: {output_filename}")
                        except Exception as e:
                            logging.error(f"Failed to append to file '{output_filename}': {e}")
                            continue
                else:
                    logging.warning(f"Date could not be extracted for page {page_number}. Skipping.")
        else:
            logging.info("No pages with missing dates to process.")

        # After processing all files, close the JSON arrays by adding ']'
        for output_filename in output_files:
            output_file_path = os.path.join(dir_path, output_filename)
            try:
                with open(output_file_path, 'a', encoding='utf-8') as outfile:
                    outfile.write(']')
                logging.info(f"Closed JSON array in output file: {output_filename}")
            except Exception as e:
                logging.error(f"Failed to close JSON array in file '{output_filename}': {e}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

    # Print summary
    logging.info(f"\nTotal files processed: {files_processed}")
    if output_files:
        logging.info("Output files:")
        for output_filename in output_files:
            logging.info(f"- {output_filename}")
    else:
        logging.info("No matching files found.")

    # After processing all files, wait 1 second and start 5present-multi.py
    time.sleep(1)
    try:
        subprocess.run(['python', 'bills/5present-multi.py', dir_path], check=True)
        logging.info(f"Started processing with 5present-multi.py for directory: {dir_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to start 5present-multi.py for directory '{dir_path}': {e}")

if __name__ == "__main__":
    main()
