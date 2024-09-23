import PyPDF2
import sys
import os
import boto3
import time
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import concurrent.futures

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

def extract_text_using_aws_textract(file_path, page_num):
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
        filename = os.path.basename(file_path)
        s3_key = f'{folder_name}/{filename}'

        # Upload PDF to S3
        s3.upload_file(file_path, textract_bucket, s3_key)

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

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else None
    except Exception as e:
        print(f"Image text extraction failed: {str(e)}")
        return None

def get_unique_path(base_path):
    path = base_path
    counter = 1
    while os.path.exists(path):
        name, ext = os.path.splitext(base_path)
        path = f"{name}_{counter}{ext}"
        counter += 1
    return path

def process_file(file_path):
    file_filename = os.path.basename(file_path)

    try:
        if file_path.lower().endswith('.pdf'):
            process_pdf(file_path)
        else:
            print(f"Unsupported file type: {file_filename}")
            return

        print(f"File '{file_filename}' processed.")

    except Exception as e:
        print(f"An error occurred while processing '{file_filename}': {str(e)}")

def process_pdf(pdf_path):
    import os

    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    directory = os.path.dirname(pdf_path)
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    # Create a subdirectory named after the PDF file
    output_dir = os.path.join(directory, base_filename)
    os.makedirs(output_dir, exist_ok=True)

    for page_num in range(total_pages):
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        # Save images inside the subdirectory with names like page_1.png
        output_file = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(output_file)
        print(f"Page {page_num + 1} converted and saved to: {output_file}")

    pdf_document.close()
def main():
    if len(sys.argv) == 2:
        input_directory = sys.argv[1]
    else:
        print("Usage: python splittest.py <path_to_directory>")
        return

    if not os.path.isdir(input_directory):
        print(f"Directory not found: {input_directory}")
        return

    # Load environment variables
    load_dotenv()
    max_workers = int(os.environ.get("MAX_WORKERS", 10))

    # Find all .pdf files in the input directory
    files = [
        os.path.join(input_directory, filename)
        for filename in os.listdir(input_directory)
        if filename.lower().endswith('.pdf') and os.path.isfile(os.path.join(input_directory, filename))
    ]

    if not files:
        print("No .pdf files found in the directory.")
        return

    # Process files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file_path): file_path for file_path in files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                future.result()
                print(f"Finished processing {file_path}")
            except Exception as exc:
                print(f"{file_path} generated an exception: {exc}")

if __name__ == "__main__":
    main()