import os
import sys
import re

def process_files(input_directory):
    """
    Process all '_date.json' files in the input directory as plain text,
    extract the first 'Date' value and all 'Page Number' values,
    create VALUE.json,
    compare with VALUE!.json to create VALUE2.json.
    """
    # Dictionary to hold dates and their associated page numbers
    date_pages = {}

    # Collect all '_date.json' file paths
    file_paths = [
        os.path.join(input_directory, filename)
        for filename in os.listdir(input_directory)
        if filename.endswith('_date.json')
    ]

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the first instance of "Date"
        date_match = re.search(r'"Date"\s*:\s*"([^"]+)"', content)
        if date_match:
            date_value = date_match.group(1).strip()
        else:
            date_value = 'Unknown'

        # Find all instances of "Page Number"
        page_numbers = re.findall(r'"Page Number"\s*:\s*"(\d+)"', content)
        page_numbers = list(map(int, page_numbers))  # Convert to integers

        # Update the dictionary
        if date_value in date_pages:
            date_pages[date_value].extend(page_numbers)
        else:
            date_pages[date_value] = page_numbers

    # Remove duplicates and sort the page numbers for each date
    for date in date_pages:
        date_pages[date] = sorted(set(date_pages[date]))

    # Write to VALUE.json
    value_json_path = os.path.join(input_directory, 'VALUE.json')
    with open(value_json_path, 'w', encoding='utf-8') as f:
        f.write('{\n    "Date": [\n')
        for i, (date_str, pages_list) in enumerate(date_pages.items()):
            pages_str = ', '.join(str(p) for p in pages_list)
            f.write(f'        {{\n            "{date_str}": [{pages_str}]\n        }}')
            if i < len(date_pages) - 1:
                f.write(',\n')
            else:
                f.write('\n')
        f.write('    ]\n}')

    print(f"VALUE.json has been written to {value_json_path}")

    # Now read VALUE!.json to get missing pages
    value_excl_json_path = os.path.join(input_directory, 'VALUE!.json')
    if not os.path.exists(value_excl_json_path):
        print("VALUE!.json not found in the directory.")
        return

    with open(value_excl_json_path, 'r', encoding='utf-8') as f:
        value_excl_content = f.read()

    # Extract "Page Number" from each JSON object in VALUE!.json
    missing_pages = re.findall(r'"Page Number"\s*:\s*"(\d+)"', value_excl_content)
    missing_pages = list(map(int, missing_pages))

    # Get a list of all pages (pages in date_pages plus missing_pages)
    all_pages = set()
    for pages in date_pages.values():
        all_pages.update(pages)
    all_pages.update(missing_pages)
    all_pages = sorted(all_pages)

    # Assign missing pages based on sequential inference
    for missing_page in missing_pages:
        assigned_date = 'Unknown'
        prev_page = missing_page - 1
        next_page = missing_page + 1

        # Try to find dates for adjacent pages
        prev_date = None
        next_date = None

        for date, pages in date_pages.items():
            if prev_page in pages:
                prev_date = date
            if next_page in pages:
                next_date = date

        if prev_date == next_date and prev_date is not None:
            assigned_date = prev_date
        elif prev_date and not next_date:
            assigned_date = prev_date
        elif next_date and not prev_date:
            assigned_date = next_date
        else:
            # If neither prev_page nor next_page have dates, try further
            # For simplicity, we'll keep 'Unknown' for this implementation
            pass

        if assigned_date != 'Unknown':
            date_pages.setdefault(assigned_date, []).append(missing_page)
        else:
            print(f"Unable to assign a date to page {missing_page}")

    # Remove duplicates and sort the page numbers for each date
    for date in date_pages:
        date_pages[date] = sorted(set(date_pages[date]))

    # Write to VALUE2.json
    value2_json_path = os.path.join(input_directory, 'VALUE2.json')
    with open(value2_json_path, 'w', encoding='utf-8') as f:
        f.write('{\n    "Date": [\n')
        for i, (date_str, pages_list) in enumerate(date_pages.items()):
            pages_str = ', '.join(str(p) for p in pages_list)
            f.write(f'        {{\n            "{date_str}": [{pages_str}]\n        }}')
            if i < len(date_pages) - 1:
                f.write(',\n')
            else:
                f.write('\n')
        f.write('    ]\n}')

    print(f"VALUE2.json has been written to {value2_json_path}")
    print("Processing complete.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python 6condense.py <path_to_directory>")
        return

    input_directory = ' '.join(sys.argv[1:])

    if not os.path.isdir(input_directory):
        print(f"Directory not found: {input_directory}")
        return

    process_files(input_directory)

if __name__ == "__main__":
    main()