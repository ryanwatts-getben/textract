import sys
import os
import re
import logging
from dotenv import load_dotenv
import anthropic
import concurrent.futures
import time
import subprocess
import traceback
import base64

# Load environment variables from .env file
load_dotenv()

# Configure logging with different levels and handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to stdout
        logging.FileHandler("bills/5present-multi.log")  # Also log to a file
    ]
)

def standardize_date(date_str):
    """
    Ensure the date is in MM-DD-YYYY format.
    """
    # Replace slashes with dashes for consistency
    normalized_date = date_str.replace('/', '-')
    # Enforce MM-DD-YYYY format
    match = re.match(r'^(\d{2})-(\d{2})-(\d{4})$', normalized_date)
    if match:
        return normalized_date
    else:
        raise ValueError(f"Date '{date_str}' is not in MM-DD-YYYY format.")

def extract_date_from_content(content, file_path, client):
    """
    Extract the 'Date' value from the content string using regular expressions.
    If not found, use Claude to identify the date from the page content and image.
    """
    date_match = re.search(r'"Date"\s*:\s*"([^"]+)"', content)
    if date_match:
        date_str = date_match.group(1).strip()
        try:
            return standardize_date(date_str)
        except ValueError as ve:
            logging.error(f"Invalid date format extracted: {ve}")
            raise ve
    else:
        logging.warning(f"Missing 'Date' key in content for file: {file_path}")
        return identify_date_with_claude(file_path, client)

def identify_date_with_claude(file_path, client):
    """
    Send the page content and image to Claude to identify the date.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        image_media_type, image_data = get_image_data(file_path)
        if image_media_type is None or image_data is None:
            logging.error(f"Failed to get image data for {file_path}")
            return None

        system_message = """You are a helpful assistant that identifies the date on a medical bill or record. 
        The date should be in MM-DD-YYYY format. If multiple dates are present, identify the most prominent or relevant date, 
        which is likely the date of service or the bill date. If no exact date is found, provide your best estimate based on 
        the available information, and explain your reasoning."""

        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            temperature=0.0,
            system=system_message,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Please identify the date on this medical document. Here's the text content:\n\n{text_content}"
                        },
                    ]
                },
            ],
        )

        # Extract the date from Claude's response
        if response and response.content:
            date_match = re.search(r'\b(\d{2}-\d{2}-\d{4})\b', response.content[0].text)
            if date_match:
                return standardize_date(date_match.group(1))
            else:
                logging.error(f"Claude couldn't identify a date for {file_path}")
                return None
        else:
            logging.error(f"No response received from Claude for {file_path}")
            return None

    except Exception as e:
        logging.error(f"Error identifying date with Claude for {file_path}: {e}")
        return None

def get_image_data(file_path):
    """
    Given the path of a .txt file, return the media type and base64-encoded image data
    for the corresponding .png file in the same directory.
    """
    dir_path = os.path.dirname(file_path)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Construct the path for the .png file
    image_path = os.path.join(dir_path, f"{base_filename}.png")
    image_media_type = "image/png"
    
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            image_content = f.read()
        image_data = base64.b64encode(image_content).decode('utf-8')
        return image_media_type, image_data
    else:
        logging.warning(f"Image not found: {image_path}")
        return None, None

def find_json_object(content):
    """
    Find the first JSON object in the content string.
    """
    start_brace = content.find('{')
    start_bracket = content.find('[')
    if start_brace == -1 and start_bracket == -1:
        raise ValueError("No valid JSON object start found.")
    if start_brace == -1:
        start_index = start_bracket
    elif start_bracket == -1:
        start_index = start_brace
    else:
        start_index = min(start_brace, start_bracket)

    # Stack to keep track of braces and brackets
    stack = []
    end_index = start_index
    for i, char in enumerate(content[start_index:], start=start_index):
        if char == '{' or char == '[':
            stack.append(char)
        elif char == '}' or char == ']':
            if not stack:
                raise ValueError("Unbalanced JSON braces.")
            opening = stack.pop()
            if (opening == '{' and char != '}') or (opening == '[' and char != ']'):
                raise ValueError("Mismatched JSON braces.")
            if not stack:
                end_index = i
                break

    if stack:
        raise ValueError("Unbalanced JSON braces at the end of content.")

    json_obj = content[start_index:end_index + 1]
    return json_obj

def get_unique_path(base_path):
    """
    Generate a unique file path by appending an incrementing counter
    if the specified path already exists.
    """
    path = base_path
    counter = 1
    while os.path.exists(path):
        name, ext = os.path.splitext(base_path)
        path = f"{name}_{counter}{ext}"
        counter += 1
        if counter > 1000:  # Prevent infinite loop
            logging.error("Unable to create a unique path after 1000 attempts.")
            raise Exception("Exceeded maximum number of attempts to create a unique path.")
    return path

def process_with_claude(combined_content, client):
    """
    Send combined content to the AI assistant (Claude) and return the response.
    """
    system_message = """<rewritten_instructions>
please combine the medical bills from the provided JSON files into a single JSON object, following these guidelines:

1. Combine all the entries into one JSON structure.
2. Ensure that all data corresponding to the same date is merged appropriately.
3. Do not duplicate entries; make sure all values are unique.
4. Keep the "Date" field and ensure it is in MM-DD-YYYY format.
5. Preserve the structure of the JSON objects.

2. In the "PatientInformation" array:
   <PatientInformation>
      - Save the "Date" value as the filename of the current file and force MM-DD-YYYY format.
      - Combine all unique values for Patient, MedicalFacility, NameOfDoctor, ReferredTo, and ReferredBy
      - Cleanup the names of "Patient", "MedicalFacility", and "NameOfDoctor" to be consistent. Patient and Doctor names should read FirstName LastName.
      - Ensure all values are unique - do not repeat any information.
   </PatientInformation>

3. In the "Codes" array:
   <Codes>
   - For each key (Referred By, Referred To, Name of Doctor, ICD10CM, CPT Codes, Rx, Billing Line Items From Today Only, Amount Paid Already, Amount Still Owed, Other Medically Relevant Information), list all page numbers where that information appears.
   - Format each entry as follows:
     {"$key": [
       {"$value": "$page_numbers"},
       {"$value": "$page_numbers"},
       ...
     ]}
   - Where $key is the category (e.g., "ICD10CM"), $value is the specific item (e.g., "G43.309"), and $page_numbers is a comma-separated list of page numbers where that item appears (e.g., "5, 9, 13, 27").
   </Codes>
{{"Date":"08-25-2023","PatientInformation":{{"Patient":"Jim Stephenson","MedicalFacility":"America's PT Palace","NameOfDoctor":["Tbone Placeholder, PT"],"ReferredTo":[{{"NameOfDoctor":["Dr. Jones"],"MedicalFacility":["Emergency Room of America"]}}],"ReferredBy":[{{"NameOfDoctor":["Dr. Jones"],"MedicalFacility":["Emergency Room of America"]}}]}},"Codes":[{{"Type":"CPT","Code":"97802","Description":"Medical Procedure","Documents":[{{"PDF":"/pdf/385-390cmc.pdf","PageNumbers":["4","5","6"]}},{{"PDF":"/pdf/385-390-billing.pdf","PageNumbers":["1","2","3"]}}],"BillingDetails":{{"Charges":"35.52","InsurancePayment":"0.96","ContractualAdjustment":"34.56","PatientPayment":"7.69","Adjustment":"0.00","ResponsibleParty":"0.00"}}}},{{"Type":"ICD10CM","Code":"V65.01","Description":"Person consulting for explanation of examination or test findings","Documents":[{{"PDF":"/pdf/385-390cmc.pdf","PageNumbers":["2","3"]}}]}},{{"Type":"ICD10CM","Code":"V65.02","Description":"Person consulting for immunization","Documents":[{{"PDF":"/pdf/385-390cmc.pdf","PageNumbers":["3"]}}]}},{{"Type":"ICD10CM","Code":"R51.89","Description":"Other specified headache","Documents":[{{"PDF":"/pdf/385-390cmc.pdf","PageNumbers":["4"]}}]}},{{"Type":"ICD10CM","Code":"S06.5X0A","Description":"Traumatic subdural hemorrhage without loss of consciousness, initial encounter","Documents":[{{"PDF":"/pdf/385-390-billing.pdf","PageNumbers":["4","5"]}}]}},{{"Type":"ICD10CM","Code":"S32.000A","Description":"Fracture of unspecified part of lumbar vertebra, initial encounter for closed fracture","Documents":[{{"PDF":"/pdf/385-390-billing.pdf","PageNumbers":["6"]}}]}},{{"Type":"Rx","Code":"Naproxen","Description":"Nonsteroidal anti-inflammatory drug (NSAID) used to treat pain and inflammation","Documents":[{{"PDF":"/pdf/385-390cmc.pdf","PageNumbers":["5","6"]}},{{"PDF":"/pdf/385-390-billing.pdf","PageNumbers":["2"]}}]}},{{"Type":"Rx","Code":"Floricet","Description":"Combination medication used to treat tension headaches","Documents":[{{"PDF":"/pdf/385-390cmc.pdf","PageNumbers":["4"]}}]}},{{"Type":"HCPCS","Code":"J1100","Description":"INJ Dexamethasone Sodium Phosphate 1 MG","Documents":[{{"PDF":"/pdf/385-390-billing.pdf","PageNumbers":["6"]}}],"BillingDetails":{{"Charges":"35.52","InsurancePayment":"0.96","ContractualAdjustment":"34.56","PatientPayment":"7.69","Adjustment":"0.00","ResponsibleParty":"0.00"}}}}],"ProceduresOrFindings":[{{"KeyWordsOrFindings":"Physical Therapy","Documents":[{{"PDF":"/pdf/385-390cmc.pdf","PageNumbers":["4"]}},{{"PDF":"/pdf/385-390-billing.pdf","PageNumbers":["6"]}}]}},{{"KeyWordsOrFindings":"Occipital Headache","Documents":[{{"PDF":"/pdf/385-390cmc.pdf","PageNumbers":["5"]}}]}},{{"KeyWordsOrFindings":"Neck Pain Follow-up MVA","Documents":[{{"PDF":"/pdf/385-390cmc.pdf","PageNumbers":["4"]}}]}},{{"KeyWordsOrFindings":"Possible Whiplash","Documents":[{{"PDF":"/pdf/385-390cmc.pdf","PageNumbers":["4","5"]}}]}}],"FinancialSummary":{{"TotalCharges":"71.04","AmountPaid":"7.69","AmountOwed":"63.35","Documents":[{{"PDF":"/pdf/385-390-billing.pdf","PageNumbers":["7"]}}]}}}}
```
The result will be 1 file output for each date. Combine all the JSON objects into a single JSON file for each date. Please process the provided JSON input files and generate a combined output following these instructions and the structure of the example output provided.
</instructions>"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            temperature=0.0,
            system=system_message,
            messages=[
                {"role": "user", "content": combined_content},
                {"role": "assistant", "content": "{"}  # Indicate that the assistant should start the response with '{'
            ],
        )
        
        # Extract the assistant's reply
        if response and response.content:
            cleaned_content = ''.join(
                item.text if hasattr(item, 'text') else item.get('text', '')
                for item in response.content
            )
            return cleaned_content.strip()
        else:
            logging.error("No completion received from Claude.")
            return ""
    except Exception as e:
        logging.error(f"Error communicating with Claude: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return ""

def main():
    """
    Main function to process all '_details.json' files in a directory.
    """
    if len(sys.argv) != 2:
        logging.error("Usage: python 5present-multi.py <directory_path>")
        sys.exit(1)

    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        logging.error(f"Invalid directory path: {dir_path}")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logging.error("ANTHROPIC_API_KEY not found in environment variables.")
        sys.exit(1)

    # Initialize the Anthropic client
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize Anthropic client: {e}")
        sys.exit(1)

    # Collect all '_details.json' files in the directory
    file_paths = [
        os.path.join(dir_path, filename)
        for filename in os.listdir(dir_path)
        if filename.endswith('_details.json') and os.path.isfile(os.path.join(dir_path, filename))
    ]

    if not file_paths:
        logging.warning(f"No '_details.json' files found in the directory: {dir_path}")
        return

    # Dictionary to group file contents by date
    date_to_contents = {}

    # Extract dates and group file contents
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Find the first JSON object in the content
            json_obj = find_json_object(content)
            # Extract the 'Date' value using regex or Claude if necessary
            standardized_date = extract_date_from_content(json_obj, file_path, client)
            if standardized_date:
                # Group contents by date
                date_to_contents.setdefault(standardized_date, []).append(json_obj)
                logging.info(f"File '{file_path}' added to date group '{standardized_date}'.")
            else:
                logging.warning(f"Couldn't determine date for file '{file_path}'. Skipping.")
        except Exception as e:
            logging.error(f"Error processing file '{file_path}': {e}")

    # Process each date group
    for date, contents in date_to_contents.items():
        combined_content = '\n'.join(contents)
        # Send to Claude for combining into single JSON object
        logging.info(f"Sending combined content for date '{date}' to Claude.")
        result = process_with_claude(combined_content, client)

        if result:
            # Save the result to a single output file per date
            try:
                output_filename = f"{date}.json"
                output_file_path = os.path.join(dir_path, output_filename)
                output_file_path = get_unique_path(output_file_path)
                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(result)
                logging.info(f"Combined output saved to '{output_file_path}'.")
            except Exception as e:
                logging.error(f"Failed to write combined output for date '{date}': {e}")
        else:
            logging.error(f"Failed to get combined content from Claude for date '{date}'.")

    logging.info("All files have been processed.")

    # After processing all files, wait 1 second and start 6condense.py
    time.sleep(1)
    try:
        subprocess.run(['python', 'bills/6condense.py', dir_path], check=True)
        logging.info(f"Started processing with 6condense.py for directory: {dir_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to start 6condense.py for directory '{dir_path}': {e}")

    logging.info("All processing tasks have been completed.")

if __name__ == "__main__":
    main()