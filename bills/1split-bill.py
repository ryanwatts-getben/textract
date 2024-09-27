import PyPDF2
import anthropic
import sys
import os
from dotenv import load_dotenv
import boto3
import fitz  # PyMuPDF
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import base64

# Load environment variables from .env file
load_dotenv()

client = anthropic.Anthropic()


# Configure logging  # {{ edit_2 }}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_unique_path(base_path):
    """
    Generate a unique file or directory path by appending an incrementing counter
    if the specified path already exists.
    """
    path = base_path
    counter = 1
    # Loop until a unique path is found or a reasonable limit is reached
    while os.path.exists(path):
        name, ext = os.path.splitext(base_path)
        path = f"{name}_{counter}{ext}"
        counter += 1
        if counter > 1000:  # Prevent infinite loop  # {{ edit_3 }}
            logging.error("Unable to create a unique path after 1000 attempts.")
            raise Exception("Exceeded maximum number of attempts to create a unique path.")
    return path

def process_page(page_num, pdf_path, unique_pdf_output_folder, textract_client, filename, total_pages):
    """
    Process a single page by:
    - Rendering it to an image.
    - Extracting text from the image using AWS Textract.
    - Saving the extracted text to a .txt file.
    """
    try:
        # Open the PDF using PyMuPDF (ensure thread safety by opening per thread)  # {{ edit_4 }}
        with fitz.open(pdf_path) as pdf_document:
            # Load the specific page from the PDF
            page = pdf_document.load_page(page_num)
            # Render the page to a pixmap (image)
            pix = page.get_pixmap()
            # Define the output image file path
            output_image_file = os.path.join(unique_pdf_output_folder, f"Page_{page_num + 1}.png")
            # Save the image to the output path
            pix.save(output_image_file)
    
        # Read image bytes for Textract processing
        with open(output_image_file, 'rb') as img_file:
            image_bytes = img_file.read()
    
        # Use AWS Textract to detect text in the image
        response = textract_client.detect_document_text(
            Document={'Bytes': image_bytes}
        )
    
        # Initialize a string to hold the extracted text
        extracted_text = ''
        # Iterate over the detected text blocks
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                # Append each line of text to the extracted text
                extracted_text += item['Text'] + '\n'
    
        # Define the output text file path
        output_text_file = os.path.join(unique_pdf_output_folder, f"Page_{page_num + 1}.txt")
        # Write the extracted text to the file
        with open(output_text_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)

        # Now, process with Claude
        claude_response = process_with_claude(filename, str(total_pages), extracted_text, image_media_type, image_bytes, client)
        if claude_response:
            # Save the Claude response to a file named Page_(n)_clean.txt
            output_clean_text_file = os.path.join(unique_pdf_output_folder, f"Page_{page_num + 1}_clean.txt")
            with open(output_clean_text_file, 'w', encoding='utf-8') as f:
                f.write(claude_response)
            logging.info(f"Page {page_num + 1} of {filename} processed and Claude response saved.")
        else:
            logging.warning(f"Failed to process {filename} with Claude")

        logging.info(f"Page {page_num + 1} of {filename} processed and saved.")
    
    except Exception as e:
        # Handle any exceptions that occur during processing
        logging.error(f"Error processing page {page_num + 1} of {filename}: {e}")
    return page_num, image_bytes, extracted_text

system = f"""Read this medical bill and respond only using this tool to leverage a uniform JSON format in the "output_schema" to extract the data, Respond using the output_schema only: tools = [{{"name": "parse_medical_record"}},{{"description": "Parses an OCR'd medical record and returns structured data in JSON format."}}, {{"input_schema": {{"type": "object"}},"properties": {{"ocr_text": {{"type": "string"}},{{"description": "The OCR'd text from a medical record."}},}}}},"required": ["ocr_text"]}},"output_schema": {{"type": "object","properties": {{"IsPageDated":{{"type":"boolean"}},"IsPageBlank":{{"type":"boolean"}},"IsPageContentContinuedFromPreviousPage":{{"type":"boolean"}},"Date":{{"type":"string"}},"PathToRecordPDFfilename":{{"type":["string","null"]}},"PathToBillingPDFfilename":{{"type":["string","null"]}},"PatientInformation":{{"type":"object","properties":{{"PatientFirstNameLastName":{{"type":"string"}},"MedicalFacility":{{"type":"string"}},"DoctorFirstNameLastName":{{"type":"array","items":{{"type":"string"}}}},"ReferredTo":{{"type":"array","items":{{"type":"object","properties":{{"DoctorOrFacilityName":{{"type":["string","null"]}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}},"ReferredBy":{{"type":"array","items":{{"type":"object","properties":{{"DoctorOrFacilityName":{{"type":["string","null"]}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}}}}}},"Codes":{{"type":"object","properties":{{"ICD10CM":{{"type":"array","items":{{"type":"object","additionalProperties":{{"type":"object","properties":{{"Description":{{"type":["string","null"]}},"BillingLineItem":{{"type":["string","null"]}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}}}},"Rx":{{"type":"array","items":{{"type":"object","additionalProperties":{{"type":"object","properties":{{"Dosage":{{"type":["string","null"]}},"Frequency":{{"type":["string","null"]}},"Duration":{{"type":["string","null"]}},"BillingLineItem":{{"type":["string","null"]}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}}}},"CPTCodes":{{"type":"array","items":{{"type":"object","additionalProperties":{{"type":"object","properties":{{"Description":{{"type":["string","null"]}},"BillingLineItem":{{"type":["string","null"]}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}}}}}}}},"ProceduresOrFindings":{{"type":"array","items":{{"type":"object","properties":{{"KeyWordsOrFindings":{{"type":"array","items":{{"type":"object","additionalProperties":{{"type":["string","null"]}},"properties":{{"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}}}}}}}},"DailyFinancialSummary":{{"type":"array","items":{{"type":"object","properties":{{"TotalChargesToday":{{"type":"string"}},"AmountPaidByPatient":{{"type":"string"}},"AmountAdjusted":{{"type":"string"}},"AmountPaidByInsurance":{{"type":"string"}},"AmountOwed":{{"type":"string"}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}},"OtherInformation":{{"type":"array","items":{{"type":"object","properties":{{"Date of Birth":{{"type":["string","null"]}},"Insurance":{{"type":["string","null"]}},"Insured's ID":{{"type":["string","null"]}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}},"required":["IsPageDated","IsPageBlank","IsPageContentContinuedFromPreviousPage"]}}"""

image_media_type="image/png"

def process_with_claude(filename, total_pages, extracted_text, image_media_type, img_bytes, client):
    """
    Send content and image data to Claude API and return the response.
    """
    try:
        # Encode the image bytes to base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            temperature=0.3,
            max_tokens=8192,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type,
                                "data": img_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"{filename} of {total_pages} pages\n{extracted_text}"
                        },
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{'
                },
            ],
        )    

        # Extract the text content from the response
        if isinstance(response.content, list):
            cleaned_content = ''.join(
                item.text if hasattr(item, 'text') else item.get('text', '')
                for item in response.content
            )
        else:
            cleaned_content = response.content

        # Ensure the content starts with '['
        if not cleaned_content.startswith('{'):
            cleaned_content = '{' + cleaned_content

        return cleaned_content
    except Exception as e:
        logging.error(f"Error in process_with_claude: {str(e)}", exc_info=True)
        return None
def main():
    """
    Main function to process all PDF files in a directory:
    - Split PDFs into individual pages.
    - Extract text from each page using AWS Textract.
    - Save the extracted text.
    """
    # Check for the PDF directory argument
    if len(sys.argv) == 2:
        pdf_directory = sys.argv[1]
    else:
        logging.error("No PDF directory provided. Please provide a PDF directory path.")
        return

    # Retrieve AWS credentials and region from environment variables
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')

    if not aws_region:
        logging.error("AWS_REGION is not specified in the environment variables.")
        return

    # Get MAX_WORKERS from .env, default to 10 if not set or invalid
    try:
        max_workers = int(os.getenv('MAX_WORKERS', 10))
    except ValueError:
        logging.warning("Invalid MAX_WORKERS value in .env. Using default value of 10.")
        max_workers = 10

    # Initialize AWS Textract client with region and credentials
    try:
        textract_client = boto3.client(
            'textract',
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
    except Exception as e:
        logging.error(f"Failed to initialize AWS Textract client: {e}")
        return

    try:
        all_pages = []  # List to hold processing tasks for all pages
        unique_pdf_folders = []  # List to store output folders for each PDF
        # Iterate over all files in the provided directory
        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                pdf_filename = os.path.basename(pdf_path)
                folder_name = os.path.splitext(pdf_filename)[0]

                # Create a unique output folder for the split pages
                pdf_output_folder = os.path.join(os.path.dirname(pdf_path), folder_name, 'split')
                unique_pdf_output_folder = get_unique_path(pdf_output_folder)
                os.makedirs(unique_pdf_output_folder)
                unique_pdf_folders.append(unique_pdf_output_folder)

                # Open the PDF file using PyPDF2 to get the total number of pages
                try:
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        total_pages = len(pdf_reader.pages)
                        
                except Exception as e:
                    logging.error(f"Failed to read PDF file {pdf_path}: {e}")
                    continue

                # Add each page to the processing list
                all_pages.extend([
                    (page_num, pdf_path, unique_pdf_output_folder, textract_client, filename, total_pages)
                    for page_num in range(total_pages)
                ])

        if not all_pages:
            logging.warning("No PDF files found to process.")
            return
    
        # Process all pages concurrently using a thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_page, *args) for args in all_pages]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error during page processing: {str(e)}", exc_info=True)

        logging.info("All pages from all PDFs have been processed.")

        # Short delay before starting the next script
        time.sleep(1)

        # # Start 2bill-multi.py for each unique PDF output folder
        # for folder in unique_pdf_folders:
        #     # Execute the next script with the output folder as an argument
        #     try:
        #         subprocess.run(['python', 'bills/2bill-multi.py', folder], check=True)
        #         logging.info(f"Started processing with 2bill-multi.py for folder: {folder}")
        #     except subprocess.CalledProcessError as e:
        #         logging.error(f"Failed to start 2bill-multi.py for folder {folder}: {e}")

    except Exception as e:
        logging.error(f"Error While processing the PDFs: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    main()