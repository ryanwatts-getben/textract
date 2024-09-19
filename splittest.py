import PyPDF2
import sys
import os
import boto3
import time
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def extract_text_from_pdf_page(pdf_reader, page_num):
    try:
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        if text.strip():
            return text
        else:
            return None
    except Exception as e:
        print(f"PyPDF2 extraction failed: {str(e)}")
        return None

def extract_text_using_aws_textract(pdf_path, page_num):
    try:
        s3 = boto3.client('s3',
                          aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                          region_name=os.environ.get('AWS_REGION'))
        textract_client = boto3.client('textract',
                                       aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                                       aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                                       region_name=os.environ.get('AWS_REGION'))

        textract_bucket = os.environ.get('AWS_TEXTRACT_BUCKET_NAME')
        folder_name = '019207c4-e513-7d59-8697-6411084a2633'
        filename = os.path.basename(pdf_path)
        s3_key = f'{folder_name}/{filename}'

        # Upload PDF to S3
        s3.upload_file(pdf_path, textract_bucket, s3_key)

        # Start Textract job
        response = textract_client.start_document_text_detection(
            DocumentLocation={
                'S3Object': {
                    'Bucket': textract_bucket,
                    'Name': s3_key
                }
            }
        )

        job_id = response['JobId']

        # Wait for the job to complete
        while True:
            time.sleep(5)
            job_status = textract_client.get_document_text_detection(JobId=job_id)
            status = job_status['JobStatus']
            if status in ['SUCCEEDED', 'FAILED']:
                break

        if status == 'SUCCEEDED':
            # Retrieve the results
            response = textract_client.get_document_text_detection(JobId=job_id)
            
            # Extract text from the response
            extracted_text = ""
            for item in response['Blocks']:
                if item['BlockType'] == 'LINE':
                    extracted_text += item['Text'] + "\n"
            
            return extracted_text.strip()
        else:
            print(f"Textract job failed for page {page_num + 1}")
            return None

    except Exception as e:
        print(f"AWS Textract extraction failed: {str(e)}")
        return None

def get_unique_path(base_path):
    path = base_path
    counter = 1
    while os.path.exists(path):
        name, ext = os.path.splitext(base_path)
        path = f"{name}_{counter}{ext}"
        counter += 1
    return path

def process_pdf(pdf_path, input_directory):
    pdf_filename = os.path.basename(pdf_path)
    folder_name = os.path.splitext(pdf_filename)[0]

    # Create the output folder
    output_folder = os.path.join(input_directory, folder_name, 'split')
    output_folder = get_unique_path(output_folder)
    os.makedirs(output_folder)

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)

            for page_num in range(total_pages):
                # Extract text from the current page
                extracted_text = extract_text_from_pdf_page(reader, page_num)
                
                # If PyPDF2 fails, use AWS Textract
                if extracted_text is None:
                    extracted_text = extract_text_using_aws_textract(pdf_path, page_num)
                
                if extracted_text is None:
                    extracted_text = f"Unable to extract text from page {page_num + 1}"

                # Create the output file path for this page
                output_file = os.path.join(output_folder, f"Page_{page_num + 1}.txt")
                output_file = get_unique_path(output_file)

                # Write the extracted text to the output file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)

                print(f"Page {page_num + 1} extracted and saved to: {output_file}")

        print(f"All pages from '{pdf_filename}' extracted and saved in folder: {output_folder}")

    except Exception as e:
        print(f"An error occurred while processing '{pdf_filename}': {str(e)}")

def main():
    if len(sys.argv) == 2:
        input_directory = sys.argv[1]
    else:
        print("Usage: python splittest.py <path_to_directory>")
        return

    if not os.path.isdir(input_directory):
        print(f"Directory not found: {input_directory}")
        return

    # Find all PDF files in the input directory
    pdf_files = [
        os.path.join(input_directory, filename)
        for filename in os.listdir(input_directory)
        if filename.lower().endswith('.pdf') and os.path.isfile(os.path.join(input_directory, filename))
    ]

    if not pdf_files:
        print("No PDF files found in the directory.")
        return

    for pdf_path in pdf_files:
        process_pdf(pdf_path, input_directory)

if __name__ == "__main__":
    main()