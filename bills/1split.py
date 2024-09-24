import PyPDF2
import sys
import os
from dotenv import load_dotenv
import boto3
import fitz  # PyMuPDF
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

def get_unique_path(base_path):
    path = base_path
    counter = 1
    while os.path.exists(path):
        name, ext = os.path.splitext(base_path)
        path = f"{name}_{counter}{ext}"
        counter += 1
    return path

def process_page(page_num, pdf_document, unique_pdf_output_folder, textract_client, filename):
    """
    Process a single page: save as image, extract text using Textract, and save to a text file.
    """
    try:
        # Save the page as an image using PyMuPDF
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        output_image_file = os.path.join(unique_pdf_output_folder, f"Page_{page_num + 1}.png")
        pix.save(output_image_file)

        # Read image bytes for Textract
        with open(output_image_file, 'rb') as img_file:
            image_bytes = img_file.read()

        # Extract text using AWS Textract
        response = textract_client.detect_document_text(
            Document={'Bytes': image_bytes}
        )

        # Process Textract response
        extracted_text = ''
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                extracted_text += item['Text'] + '\n'

        # Save the extracted text to Page_X.txt
        output_text_file = os.path.join(unique_pdf_output_folder, f"Page_{page_num + 1}.txt")
        with open(output_text_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)

        print(f"Page {page_num + 1} of {filename} processed and saved.")

    except Exception as e:
        print(f"Error processing page {page_num + 1} of {filename}: {e}")

def main():
    if len(sys.argv) == 2:
        pdf_directory = sys.argv[1]
    else:
        print("No PDF directory provided. Please provide a PDF directory path.")
        return

    # Retrieve AWS credentials and region from environment variables
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')

    if not aws_region:
        print("AWS_REGION is not specified in the environment variables.")
        return

    # Get MAX_WORKERS from .env, default to 10 if not set
    try:
        max_workers = int(os.getenv('MAX_WORKERS', 10))
    except ValueError:
        print("Invalid MAX_WORKERS value in .env. Using default value of 10.")
        max_workers = 10

    # Initialize AWS Textract client with region and credentials
    textract_client = boto3.client(
        'textract',
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    try:
        all_pages = []
        unique_pdf_folders = []  # {{ edit_1 }}
        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                pdf_filename = os.path.basename(pdf_path)
                folder_name = os.path.splitext(pdf_filename)[0]

                # Create the output folder in the original PDF's path
                pdf_output_folder = os.path.join(os.path.dirname(pdf_path), folder_name, 'split')
                unique_pdf_output_folder = get_unique_path(pdf_output_folder)
                os.makedirs(unique_pdf_output_folder)
                unique_pdf_folders.append(unique_pdf_output_folder)  # {{ edit_2 }}

                # Open the PDF file using PyPDF2
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    total_pages = len(pdf_reader.pages)

                # Open the PDF using PyMuPDF (fitz)
                pdf_document = fitz.open(pdf_path)

                # Collect all pages from all PDFs
                all_pages.extend([
                    (page_num, pdf_document, unique_pdf_output_folder, textract_client, filename)
                    for page_num in range(total_pages)
                ])

        # Process all pages concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_page, *args) for args in all_pages]
            for future in as_completed(futures):
                future.result()  # This will raise exceptions if any occurred

        print("All pages from all PDFs have been processed.")  # {{ edit_3 }}

        # Wait for 1 second before starting 2bill-multi.py
        time.sleep(1)

        # Start 2bill-multi.py for each unique PDF output folder  # {{ edit_4 }}
        for folder in unique_pdf_folders:  # {{ edit_5 }}
            subprocess.run(['python', 'bills/2bill-multi.py', folder], check=True)  # {{ edit_6 }}
            print(f"Started processing with 2bill-multi.py for folder: {folder}")

    except Exception as e:
        print(f"An error occurred while processing the PDFs: {str(e)}")

if __name__ == "__main__":
    main()