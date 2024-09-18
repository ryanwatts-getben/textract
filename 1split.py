import PyPDF2
import sys
import os

def extract_text_from_pdf_page(pdf_reader, page_num):
    try:
        page = pdf_reader.pages[page_num]
        return page.extract_text()
    except Exception as e:
        return f"An error occurred on page {page_num + 1}: {str(e)}"

def get_unique_path(base_path):
    path = base_path
    counter = 1
    while os.path.exists(path):
        name, ext = os.path.splitext(base_path)
        path = f"{name}_{counter}{ext}"
        counter += 1
    return path

def main():
    if len(sys.argv) == 2:
        pdf_path = sys.argv[1]
    else:
        print("No PDF path provided. Please provide a PDF path.")
        return

    # Extract the filename without extension
    pdf_filename = os.path.basename(pdf_path)
    folder_name = os.path.splitext(pdf_filename)[0]

    # Create the output folder
    base_output_folder = os.path.join('textract', folder_name, 'split')
    output_folder = get_unique_path(base_output_folder)
    os.makedirs(output_folder)

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)

            for page_num in range(total_pages):
                # Extract text from the current page
                extracted_text = extract_text_from_pdf_page(reader, page_num)
                
                # Create the output file path for this page
                output_file = os.path.join(output_folder, f"Page_{page_num + 1}.txt")
                output_file = get_unique_path(output_file)

                # Write the extracted text to the output file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)

                print(f"Page {page_num + 1} extracted and saved to: {output_file}")

        print(f"All pages extracted and saved in folder: {output_folder}")

    except Exception as e:
        print(f"An error occurred while processing the PDF: {str(e)}")

if __name__ == "__main__":
    main()