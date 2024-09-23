import os
import sys
import json
from dotenv import load_dotenv
import anthropic
import re

def get_unique_path(base_path):
    path = base_path
    counter = 1
    while os.path.exists(path):
        name, ext = os.path.splitext(base_path)
        # Remove any existing counter to prevent infinite loops
        name = re.sub(r'_\d+$', '', name)
        path = f"{name}_{counter}{ext}"
        counter += 1
    return path

def process_with_claude(content1, content2):
    content1 = content1.replace('"Same Date?": "True"', '')
    content2 = content2.replace('"Same Date?": "True"', '')
    content1 = content1.replace('Same Date? True', '')
    content2 = content2.replace('Same Date? True', '')
    client = anthropic.Anthropic()

    system_prompt = (
        'Are these records from the same date?'
        'If "yes" they are from the same date, combine and condense the shared values in a key named "General" and '
        'separate any unique information in arrays with the Key equal to the {"Page Number": ""} value in that record, I will write VALUE in the example as a placeholder to demonstrate where the Page Number VALUE would go'
        'Example Return: {"Date": "", "General": [{...}], VALUE: [{...}], VALUE: [{...}]}, ... '
        'If "no" they are not from the same date, respond exactly and only with {"Same Date?": "False"}. '
        'If the key "General" exists, continue to add to it anything general about the pages. Create a new array for the new page you are adding. '
        "If a diagnosis or procedure occurs on one page and not the other, it is unique and needs to be wrapped in that page's array. "
        'Avoid returning {"Same?": "True"}; it will destroy the system we have built. Just return the combined JSON in the shape I provided. '
        'It is important that you stick to that shape. Only return valid JSON; otherwise, it will disrupt the system we have built.'
    )

    # Log content1 and content2
    print("\n--- process_with_claude ---")
    print("Content1:")
    print(content1)
    print("Content2:")
    print(content2)

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        system=system_prompt,
        messages=[
            {"role": "user", "content": f"\n{system_prompt}\n\nFirst Medical Record:\n{content1}\n\nSecond Medical Record:\n{content2}"}
        ]
    )

    # Extract the content from the response
    if isinstance(response.content, list):
        response_content = ''.join(item.text for item in response.content if hasattr(item, 'text'))
    else:
        response_content = response.content

    # Minify response_content before printing it
    try:
        response_json = json.loads(response_content)
        minified_response = json.dumps(response_json, separators=(',', ':'))
    except json.JSONDecodeError:
        minified_response = response_content.strip()

    # Log Claude's response
    print("Claude's Response:")
    print(minified_response)

    # Ensure the content starts and ends with curly braces
    cleaned_content = minified_response.strip()
    if not cleaned_content.startswith('{'):
        cleaned_content = '{' + cleaned_content
    if not cleaned_content.endswith('}'):
        cleaned_content += '}'

    # Replace newlines with spaces
    cleaned_content = cleaned_content.replace('\n', ' ')

    # Handle JSON parsing
    try:
        parsed_json = json.loads(cleaned_content)
        final_content = json.dumps(parsed_json, ensure_ascii=False, separators=(',', ':'))
        print("Parsed JSON successfully.")
        return final_content
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        error_context = cleaned_content[max(0, e.pos - 20):e.pos + 20]
        print(f"Error at position {e.pos}: {error_context}")
        # Save the raw content for inspection
        print("Returning raw content.")
        return cleaned_content
def get_file_number(filename):
    # Extract the number X from filenames matching '_X_details.json'
    match = re.search(r'_(\d+)_details\.json$', filename)
    return int(match.group(1)) if match else float('inf')

def main():
    if len(sys.argv) != 2:
        print("Usage: python combine.py <path_to_directory>")
        return

    input_directory = sys.argv[1]
    if not os.path.isdir(input_directory):
        print(f"Directory not found: {input_directory}")
        return

    # Load environment variables
    load_dotenv()
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY not found in environment variables")
        return

    # Initialize Anthropics client
    anthropic.api_key = os.environ["ANTHROPIC_API_KEY"]

    # Get list of _details.json files
    files = [f for f in os.listdir(input_directory) if f.endswith('_details.json')]
    # Sort files based on the number in the filename
    files.sort(key=get_file_number)

    i = 0
    while i < len(files):
        current_file = files[i]
        current_file_path = os.path.join(input_directory, current_file)

        with open(current_file_path, 'r', encoding='utf-8') as f:
            content1 = f.read()
        combined_files = [current_file]
        combined_numbers = [get_file_number(current_file)]

        print(f"\nProcessing file: {current_file}")

        j = i + 1
        while j < len(files):
            next_file = files[j]
            next_file_path = os.path.join(input_directory, next_file)

            with open(next_file_path, 'r', encoding='utf-8') as f:
                content2 = f.read()

            print(f"\nProcessing files: {current_file} and {next_file}")
            # Use Claude to check if records are from the same date
            claude_response = process_with_claude(content1, content2)

            # Check Claude's response
            try:
                response_json = json.loads(claude_response)
                if response_json == {"Same Date?": "False"}:
                    print("Same Date? False")
                    # Keep content1 as is
                    break  # Exit the inner loop
                else:
                    print("Same Date? True")
                    # Update content1 to be Claude's response
                    content1 = claude_response  # Use combined content for the next iteration
                    combined_files.append(next_file)
                    combined_numbers.append(get_file_number(next_file))
                    j += 1
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed: {e}")
                print("Assuming records are not from the same date.")
                break  # Exit the inner loop if parsing fails

        # Save the combined content
        start_num = min(combined_numbers)
        end_num = max(combined_numbers)
        if start_num == end_num:
            output_filename = f"GSMCRecords_{start_num}_final_{start_num}.json"
        else:
            output_filename = f"GSMCRecords_{start_num}_final_{start_num}-{end_num}.json"
        output_file = os.path.join(input_directory, output_filename)
        output_file = get_unique_path(output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content1)
        print(f"\nSaving File: {output_file}")

        # Update the main loop index
        i = j

if __name__ == "__main__":
    main()