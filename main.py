import PyPDF2
import sys
import os

#pdf_path = r"C:\Users\custo\OneDrive\Desktop\medscan\!lovely\Lepera\GSMC Records.pdf.pdf"
#pdf_path = ""
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        return f"An error occurred: {str(e)}"

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
    output_folder = os.path.join('textract', folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Create the output file path
    output_file = os.path.join(output_folder, f"{folder_name}.txt")

    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Write the extracted text to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_text)

    print(f"Text extracted and saved to: {output_file}")

if __name__ == "__main__":
    main()