import sys
import os
import re
import time
import subprocess

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

def main():
    if len(sys.argv) != 2:
        print("Usage: python 4combine_1k.py <directory_path>")
        sys.exit(1)

    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        print(f"Invalid directory path: {dir_path}")
        sys.exit(1)

    files_processed = 0
    output_files = {}  # Tracks which output files have been initiated

    try:
        # Iterate over all JSON files with "_details" in their name
        for filename in os.listdir(dir_path):
            if (
                "_details" in filename and
                filename.endswith('.json') and
                os.path.isfile(os.path.join(dir_path, filename))
            ):
                file_path = os.path.join(dir_path, filename)
                files_processed += 1

                # Read the entire content of the JSON file as a string
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Use regex to extract all JSON objects within the string
                # This pattern assumes that JSON objects do not contain nested braces
                json_objects = re.findall(r'\{[^{}]*\}', content, re.DOTALL)

                for json_obj in json_objects:
                    # Extract the 'Date' value using regex without parsing entire JSON
                    date_match = re.search(r'"Date"\s*:\s*"([^"]+)"', json_obj)
                    if date_match:
                        date_str = date_match.group(1).strip()
                        try:
                            standardized_date = standardize_date(date_str)
                        except ValueError as ve:
                            print(f"Skipping JSON object in file '{filename}': {ve}")
                            continue

                        # Define the output filename
                        output_filename = f"{standardized_date}_final.json"
                        output_file_path = os.path.join(dir_path, output_filename)

                        # Initialize the file with '[' if not already done
                        if output_filename not in output_files:
                            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                                outfile.write('[' + json_obj)
                            output_files[output_filename] = True
                        else:
                            # Append a comma and the JSON object
                            with open(output_file_path, 'a', encoding='utf-8') as outfile:
                                outfile.write(',' + json_obj)
                    else:
                        print(f"No 'Date' key found in JSON object in file '{filename}'. Skipping this object.")
        
        # After processing all files, close the JSON arrays by adding ']'
        for output_filename in output_files:
            output_file_path = os.path.join(dir_path, output_filename)
            with open(output_file_path, 'a', encoding='utf-8') as outfile:
                outfile.write(']')
    
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    # Print summary
    print(f"\nTotal files processed: {files_processed}")
    print("Output files:")
    for output_filename in output_files:
        print(f"- {output_filename}")

    # After processing all files, wait 1 second and start 5present-multi.py
    time.sleep(1)
    try:
        subprocess.run(['python', 'bills/5present-multi.py', dir_path], check=True)
        print(f"Started processing with 5present-multi.py for directory: {dir_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start 5present-multi.py for directory {dir_path}: {e}")

if __name__ == "__main__":
    main()
