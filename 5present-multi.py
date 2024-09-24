import sys
import os
import re
from dotenv import load_dotenv
import anthropic
import concurrent.futures  # Import for concurrency

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

def extract_page_number(filename):
    """
    Extract page number from filename.
    """
    match = re.search(r'Page_(\d+)', filename)
    if match:
        return match.group(1)
    else:
        return "Unknown"
prompt = """<instructions>
Claude, please combine the medical records from the provided JSON files using the following guidelines:

1. Update the "Page_$ref" values with the actual page numbers where each piece of information is found.

2. In the "General" array:
   <general_array>
   - Combine all unique values for Referred By, Referred To, Name of Doctor, ICD10CM, CPT Codes, Rx, and Other Medically Relevant Information from all pages.
   - Include a list of repeated values across multiple pages that are not being specifically tracked.
   - Ensure all values are unique - do not repeat any information.
   - For the "Daily Summary" key, write a precise, accurate, and comprehensive summary of the day's events based on the notes. Present this summary as a single paragraph, interpreting any ICD-10-CM and CPT codes when mentioned.
   </general_array>

3. In the "Page Numbers That Contain The Following Information" array:
   <page_numbers_array>
   - For each key (Referred By, Referred To, Name of Doctor, ICD10CM, CPT Codes, Rx, Other Medically Relevant Information), list all page numbers where that information appears.
   - Format each entry as follows:
     {"$key": [
       {"$value": "$page_numbers"},
       {"$value": "$page_numbers"},
       ...
     ]}
   - Where $key is the category (e.g., "ICD10CM"), $value is the specific item (e.g., "G43.309"), and $page_numbers is a comma-separated list of page numbers where that item appears (e.g., "5, 9, 13, 27").
   </page_numbers_array>

4. Maintain the overall structure of the provided example output, filling in all relevant information from the input JSON files.

5. Ensure that all variable information (indicated by $ in the example) is replaced with actual data from the input files.

6. If any information is missing or not applicable, use null or an empty array [] as appropriate.

7. Pay close attention to detail, ensuring all information is accurately transferred and combined from the input files to the output JSON structure.

8. When writing the "Daily Summary", be sure to provide a comprehensive overview of the day's medical events, treatments, diagnoses, and any other relevant information, interpreting medical codes for clarity.

9. Discard other Key Value pairs from your previous JSON and only use the ones in the instructions. Failure to do so will result in catastrophic failure of our entire pipeline.

Here is the golden output that will thrill my heart if you can produce it:

```json
  {
    "Date": "08-25-2023",
    "General": [
      {
        "Medically Relevant Information That Is Present On All Pages": [
          {
            "Patient": "Jim Stephenson",
            "Medical Facility": "America's PT Palace",
            "Referred By": null,
            "Referred To": null,
            "Name of Doctor": ["Tbone Placeholder, PT"],
            "ICD10CM": [
              {
                "V65.01": "1",
                "V65.02": "2",
                "R51.89": "3"
              }
            ],
            "CPT Codes": [
              {
                "97802": "4"
              }
            ],
            "Rx": [
              {
                "Naproxen": "5",
                "Floricet": "6"
              }
            ],
            "Procedures Performed or Findings Made": [
              {
                "Brief Summary": "Physical Therapy, occipital HA, neck pain f/u MVA, possible whiplash"
              }
            ],
            "Daily Summary": ""
          }
        ]
      }
    ],
    "Page Numbers That Contain The Following Information": [
      {
        "Name of Doctor": [
          {
            "Tbone Placeholder, PT": "1"
          }
        ]
      },
      {
        "ICD10CM": [
          {
            "V65.01": "2"
          },
          {
            "V65.02": "3"
          }
        ]
      },
      {
        "CPT Codes": [
          {
            "97802": "4"
          }
        ]
      },
      {
        "Rx": [
          {
            "Floricet": "6"
          },
          {
            "Naproxen": "5"
          }
        ]
      },
      {
        "Procedures Performed or Findings Made": [
          {
            "Physical Therapy, occipital HA, neck pain f/u MVA, possible whiplash": "2"
          }
        ]
      }
    ],
    "PDF": "/pdf/385-390cmc.pdf"
  }
```

Please process the provided JSON input files and generate a combined output following these instructions and the structure of the example output provided.
</instructions>"""

def process_with_claude(content, filename):
    """
    Send content to Claude API and return the response.
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        system=prompt,
        messages=[
            {"role": "user", "content": content},
            {"role": "assistant", "content": "{"}
        ]
    )
    return '{' + response.content[0].text if response.content else ""

def process_file(file_path):
    print(f"Processing file: {file_path}")
    filename = os.path.basename(file_path)

    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    try:
        date_str = extract_date_from_content(file_content)
        standardized_date = standardize_date(date_str)
    except ValueError as e:
        print(f"Error in file '{filename}': {e}")
        return

    try:
        response = process_with_claude(file_content, filename)
    except Exception as e:
        print(f"API call to Claude failed for file '{filename}': {e}")
        return

    new_filename = filename.replace("_date", "_final")
    output_file_path = os.path.join(os.path.dirname(file_path), new_filename)

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        outfile.write(response)

    print(f"Processed file '{filename}' -> '{new_filename}'")

def main():
    if len(sys.argv) != 2:
        print("Usage: python 5present-multi.py <directory_path>")
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

    # Get MAX_WORKERS from .env, default to 10 if not set
    max_workers = int(os.environ.get("MAX_WORKERS", 10))

    # Collect all files to process
    file_paths = [
        os.path.join(dir_path, filename)
        for filename in os.listdir(dir_path)
        if "_date" in filename and filename.endswith('.json') and os.path.isfile(os.path.join(dir_path, filename))
    ]

    if not file_paths:
        print("No matching files found.")
        return

    total_files = len(file_paths)
    files_processed = 0

    # Process files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file_path): file_path for file_path in file_paths
        }
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                future.result()
                files_processed += 1
            except Exception as exc:
                print(f"{file_path} generated an exception: {exc}")

    print(f"\nTotal files found: {total_files}")
    print(f"Total files processed: {files_processed}")

if __name__ == "__main__":
    main()