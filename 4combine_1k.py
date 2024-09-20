import sys
import os
import re

def standardize_date(date_str):
    """Standardize date format to MM-DD-YYYY."""
    # Check if date is in MM-DD-YY format and convert to MM-DD-YYYY
    match = re.match(r'^(\d{2})-(\d{2})-(\d{2})$', date_str)
    if match:
        month, day, year = match.groups()
        year = '20' + year  # Assumes years are in 2000s
        return f'{month}-{day}-{year}'
    else:
        return date_str  # Return as-is if already in desired format
def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('VALUE_date.json'):
            old_path = os.path.join(directory, filename)
            new_filename = filename.replace('VALUE_date.json', 'VALUE!.json')       
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

def extract_date_from_json_string(json_str):
    """Extract the value of the 'Date' key from a JSON string without parsing the entire JSON."""
    match = re.search(r'"Date"\s*:\s*"([^"]+)"', json_str)
    if match:
        return match.group(1)
    else:
        raise ValueError("Missing 'Date' key in JSON string.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python combine_1k.py <directory_path>")
        sys.exit(1)

    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        print(f"Invalid directory path: {dir_path}")
        sys.exit(1)

    files_processed = 0
    output_files = set()

    try:
        # List all files in the top level of the directory
        for filename in os.listdir(dir_path):
            if (
                "_details" in filename and
                filename.endswith('.json') and
                os.path.isfile(os.path.join(dir_path, filename))
            ):
                file_path = os.path.join(dir_path, filename)
                files_processed += 1

                # Read file contents as a string
                with open(file_path, 'r') as file:
                    json_str = file.read()

                try:
                    # Extract and standardize the Date
                    date_str = extract_date_from_json_string(json_str)
                    standardized_date = standardize_date(date_str)
                except ValueError as e:
                    print(f"Error in file '{filename}': {e}")
                    continue

                # Prepare the output file path
                output_filename = f"{standardized_date}_date.json"
                output_file_path = os.path.join(dir_path, output_filename)

                # Check for existing files with both YY and YYYY formats
                if os.path.exists(output_file_path):
                    # Append to existing file
                    with open(output_file_path, 'a') as outfile:
                        if os.path.getsize(output_file_path) > 0:
                            outfile.write(',')
                        outfile.write(json_str)
                    action = "Appended to"
                else:
                    # Create a new file
                    with open(output_file_path, 'w') as outfile:
                        outfile.write(json_str)
                    action = "Created"
                
                output_files.add(output_filename)
                print(f"{action} file '{output_filename}'")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    # Basic output
    print(f"\nTotal files processed: {files_processed}")
    print("Output files:")
    for output_file in output_files:
        print(f"- {output_file}")

    # Rename files after processing
    rename_files(dir_path)

if __name__ == "__main__":
    main()
