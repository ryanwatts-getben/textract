import sys
import os
import fitz  # PyMuPDF

def split_pdf(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Get the directory and base name of the input PDF file
    input_dir = os.path.dirname(pdf_path)
    filename = os.path.splitext(os.path.basename(pdf_path))[0]

    # Iterate over each page in the PDF
    for page_num in range(len(pdf_document)):
        # Create a new PDF for the current page
        new_pdf = fitz.open()  # Empty PDF
        new_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)

        # Define the output path for the single-page PDF
        output_path = os.path.join(input_dir, f"{filename}_page_{page_num + 1}.pdf")

        # Save the single-page PDF to the output directory
        new_pdf.save(output_path)

    # Close the original PDF document
    pdf_document.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python onlysplit.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    split_pdf(pdf_path)
