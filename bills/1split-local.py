import os
import sys
import logging
import fitz  # PyMuPDF for PDF operations
import boto3  # For AWS Textract
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger()

# Retrieve AWS credentials and region from environment variables
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
aws_region = os.environ.get('AWS_REGION')

# Initialize AWS Textract client with specified credentials and region
textract_client = boto3.client(
    'textract',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Initialize Anthropic client for Claude
anthropic_client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

# Constants
MAX_WORKERS = 10
IMAGE_MEDIA_TYPE = "image/png"
system = f"""Read this medical bill and respond only using this tool to leverage a uniform JSON format in the "output_schema" to organize the raw text data into a uniform output_schema: tools = [{{"name": "parse_medical_record"}},{{"description": "Parses an OCR'd medical record and returns structured data in JSON format."}}, {{"input_schema": {{"type": "object"}},"properties": {{"ocr_text": {{"type": "string"}},{{"description": "The OCR'd text from a medical record."}},}}}},"required": ["ocr_text"]}},"output_schema": {{"type": "object","properties": {{"IsPageDated":true,"IsPageBlank":false,"IsPageContentContinuedFromPreviousPage":false,"Date":"2024-09-27","PathToRecordPDFfilename":null,"PathToBillingPDFfilename":"/documents/billing/billing.pdf","ThisIsPageNumberOfBillingPDF":"1","TotalPagesInBillingPDF":"1","PatientInformation":{{"PatientFirstNameLastName":"John Doe","MedicalFacility":"Sunrise Medical Center","DoctorFirstNameLastName":["Dr. Emily Smith","Dr. Michael Brown"],"ReferredTo":[{{"DoctorOrFacilityName":"Dr. Sarah Johnson","References":{{"PageReferences":"1","BillingOrRecordDocument":"Record"}}}},{{"DoctorOrFacilityName":"Wellness Clinic","References":{{"PageReferences":"1","BillingOrRecordDocument":"Billing"}}}}],"ReferredBy":[{{"DoctorOrFacilityName":"Dr. Andrew Lee","References":{{"PageReferences":"1","BillingOrRecordDocument":"Record"}}}}]}},"Codes":{{"ICD10CM":[{{"A01.1":{{"Description":"Typhoid meningitis","CostForTreatment":"$1,200","References":{{"PageReferences":"1","BillingOrRecordDocument":"Record"}}}}}},{{"B02.2":{{"Description":"Zoster meningitis","CostForTreatment":"$900","References":{{"PageReferences":"1","BillingOrRecordDocument":"Billing"}}}}}}],"Rx":[{{"MedicationA":{{"Dosage":"500mg","Frequency":"Twice a day","Duration":"10 days","CostForTreatment":"$50","References":{{"PageReferences":"1","BillingOrRecordDocument":"Record"}}}}}},{{"MedicationB":{{"Dosage":"250mg","Frequency":"Once a day","Duration":"5 days","CostForTreatment":"$30","References":{{"PageReferences":"1","BillingOrRecordDocument":"Billing"}}}}}}],"CPTCodes":[{{"99213":{{"Description":"Office visit, established patient","CostForTreatment":"$150","References":{{"PageReferences":"1","BillingOrRecordDocument":"Record"}}}}}},{{"93000":{{"Description":"Electrocardiogram","CostForTreatment":"$200","References":{{"PageReferences":"1","BillingOrRecordDocument":"Billing"}}}}}}]}},"ProceduresOrFindings":[{{"KeyWordsOrFindings":[{{"Finding1":"Hypertension","Finding2":"Elevated cholesterol"}}],"References":{{"PageReferences":"1","BillingOrRecordDocument":"Record"}}}},{{"KeyWordsOrFindings":[{{"Finding1":"Minor abrasion","Finding2":"No signs of infection"}}],"References":{{"PageReferences":"1","BillingOrRecordDocument":"Billing"}}}}],"DailyFinancialSummary":[{{"TotalChargesToday":"$2,000","AmountPaidByPatient":"$500","AmountAdjusted":"$100","AmountPaidByInsurance":"$1,300","AmountOwed":"$100","References":{{"PageReferences":"1","BillingOrRecordDocument":"Billing"}}}}],"OtherInformation":[{{"Date of Birth":"1985-06-15","Insurance":"HealthPlus Insurance","Insured's ID":"HP123456789","References":{{"PageReferences":"1","BillingOrRecordDocument":"Record"}}}}]}}"""

def process_with_claude(filename, total_pages, extracted_text, image_media_type, img_bytes, client):
    """
    Send content and image data to Claude API and return the response.
    """
    try:
        # Encode the image bytes to base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=8192,
            temperature=0.3,
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
                            "text": "It is imperative you only return parsed data into your JSON output or it will break our workflow. "f"{filename} of {total_pages} pages\n{extracted_text}"
                        },
                    ]
                },
                {
                    "role": "assistant",
                    "content": '{'
                },
            ],
        )

        cleaned_content = response.content[0].text.strip()

        # Ensure the content starts with '{'
        if not cleaned_content.startswith('{'):
            cleaned_content = '{' + cleaned_content

        return cleaned_content
    except Exception as e:
        logger.error(f"Error in process_with_claude: {str(e)}", exc_info=True)
        return None

def process_page(page_num, pdf_document, output_dir, filename, total_pages):
    try:
        logger.info(f"Processing page {page_num + 1} of {filename}")
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")

        # Save image locally
        image_filename = f"page_{page_num + 1}.png"
        image_path = os.path.join(output_dir, image_filename)
        with open(image_path, 'wb') as img_file:
            img_file.write(img_bytes)
        logger.info(f"Saved image for page {page_num + 1} locally: {image_path}")

        # Extract text using Textract
        response = textract_client.detect_document_text(
            Document={'Bytes': img_bytes}
        )
        extracted_text = '\n'.join(item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE')

        # Add file name and page number as the first line
        file_info = f"This is File Name: {filename} and Page Number: {page_num + 1} to use when referencing this file.\n\n"
        full_text = file_info + extracted_text

        # Save extracted text locally
        text_filename = f"page_{page_num + 1}.txt"
        text_path = os.path.join(output_dir, text_filename)
        with open(text_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(full_text)
        logger.info(f"Saved extracted text for page {page_num + 1} locally: {text_path}")

        # Process with Claude
        claude_response = process_with_claude(filename, total_pages, extracted_text, IMAGE_MEDIA_TYPE, img_bytes, anthropic_client)
        if claude_response:
            # Save the Claude response locally
            clean_text_filename = f"page_{page_num + 1}_clean.txt"
            clean_text_path = os.path.join(output_dir, clean_text_filename)
            with open(clean_text_path, 'w', encoding='utf-8') as clean_txt_file:
                clean_txt_file.write(claude_response)
            logger.info(f"Saved Claude response for page {page_num + 1} locally: {clean_text_path}")
        else:
            logger.warning(f"Failed to process page {page_num + 1} with Claude")

    except Exception as e:
        logger.error(f"Error processing page {page_num + 1} of {filename}: {str(e)}")
        raise

def process_pdf(pdf_path, output_dir):
    logger.info(f"Starting to process PDF: {pdf_path}")
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    logger.info(f"PDF has {total_pages} pages")

    filename = os.path.basename(pdf_path)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                process_page,
                page_num,
                pdf_document,
                output_dir,
                filename,
                total_pages
            )
            for page_num in range(total_pages)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in processing page: {str(e)}")

    logger.info(f"All pages from {pdf_path} have been processed.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python 1split-local.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.isdir(directory_path):
        print(f"The provided path '{directory_path}' is not a directory.")
        sys.exit(1)

    logger.info(f"Processing PDFs in directory: {directory_path}")

    # Process each PDF file in the directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            output_dir = directory_path  # Output to the same input directory
            try:
                process_pdf(pdf_path, output_dir)
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {str(e)}")

if __name__ == "__main__":
    main()