import os
import sys
import fitz  # PyMuPDF

def convert_pdf_to_images(pdf_path, output_folder):
    print(f"Converting PDF: {pdf_path}")
    print(f"Output folder: {output_folder}")
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder created/confirmed: {output_folder}")

    # Get the base name of the PDF file (without extension)
    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"PDF base name: {pdf_base_name}")

    # Open the PDF file
    print("Starting PDF to image conversion...")
    pdf_document = fitz.open(pdf_path)
    print(f"PDF has {len(pdf_document)} pages")

    # Convert each page to an image
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        output_file = os.path.join(output_folder, f"{pdf_base_name}_Page{page_num+1}.png")
        pix.save(output_file)
        print(f"Saved: {output_file}")

    pdf_document.close()

def main():
    print("Script started")
    if len(sys.argv) != 2:
        print("Usage: python script.py <pdf_folder>")
        sys.exit(1)

    pdf_folder = sys.argv[1]
    print(f"PDF folder: {pdf_folder}")
    output_folder = os.path.join(pdf_folder, "proof")
    print(f"Output folder set to: {output_folder}")

    # Process all PDF files in the input folder
    print("Scanning for PDF files...")
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files")

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"Processing: {pdf_path}")
        convert_pdf_to_images(pdf_path, output_folder)

    print("Script completed")

if __name__ == "__main__":
    main()