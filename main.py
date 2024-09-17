import PyPDF2
import sys

# pdf_path = "/Users/ryanwatts/med-scan/textract/textract/scenarios.pdf"
pdf_path = ""
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
        print("No PDF path provided. Using default pdf_path.")
        # pdf_path is already defined at the top
    extracted_text = extract_text_from_pdf(pdf_path)
    
    print(extracted_text)

if __name__ == "__main__":
    main()