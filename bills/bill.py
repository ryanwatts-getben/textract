import pandas as pd
import camelot
from PyPDF2 import PdfReader, PdfWriter
import os
import re  # Imported for regular expressions

input_pdf = """C:/Users/custo/OneDrive/Desktop/medscan/textract/textract/!bill/Mcleod Carolina Forest ER Bill 7-10-2023 and PT Bills 10-4-23 through 2-28-24.pdf"""

def split_pdf(input_pdf, output_dir):
    try:
        print("Splitting PDF into individual pages...")
        reader = PdfReader(input_pdf)
        for page_num, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            output_filename = os.path.join(output_dir, f'page_{page_num + 1}.pdf')
            with open(output_filename, 'wb') as out_file:
                writer.write(out_file)
            print(f"Created {output_filename}")
        print("PDF has been split into individual pages.")
    except Exception as e:
        print(f"Error splitting PDF: {e}")

def extract_text_from_pages(total_pages, output_dir):
    try:
        print("Extracting tables from pages using camelot...")
        for page_num in range(1, total_pages + 1):
            page_file = os.path.join(output_dir, f'page_{page_num}.pdf')
            if os.path.exists(page_file):
                try:
                    tables = camelot.read_pdf(page_file, pages='1')
                    if tables:
                        output_txt = os.path.join(output_dir, f'Page_{page_num}.txt')
                        with open(output_txt, 'w') as f:
                            for table in tables:
                                f.write(table.df.to_string(index=False))
                                f.write('\n\n')
                        print(f"Extracted tables from {page_file} and saved to {output_txt}")
                    else:
                        print(f"No tables found in {page_file}")
                except Exception as e:
                    print(f"Error extracting tables from {page_file}: {e}")
            else:
                print(f"{page_file} does not exist.")
    except Exception as e:
        print(f"Error during table extraction: {e}")

def identify_billing_pages(page_texts):
    # Compile regex patterns for billing keywords with case-insensitive flags
    billing_patterns = [
        re.compile(r'\binvoice\b', re.IGNORECASE),
        re.compile(r'\breceipt\b', re.IGNORECASE),
        re.compile(r'\bamount\s+due\b', re.IGNORECASE),
        re.compile(r'\bbilling\s+statement\b', re.IGNORECASE),
        re.compile(r'\btotal\s+due\b', re.IGNORECASE),
        re.compile(r'\bbalance\b', re.IGNORECASE),
        re.compile(r'\bpayment\b', re.IGNORECASE),
        # Add more regex patterns below to match your specific use case
        re.compile(r'\bservice\s+charge\b', re.IGNORECASE),
        re.compile(r'\bdue\s+date\b', re.IGNORECASE),
        re.compile(r'\bpatient\s+account\b', re.IGNORECASE),
        re.compile(r'\btransaction\s+summary\b', re.IGNORECASE),
        re.compile(r'\bsubtotal\b', re.IGNORECASE),
        re.compile(r'\btax\b', re.IGNORECASE),
        re.compile(r'\bfee\b', re.IGNORECASE),
        re.compile(r'\bfees\b', re.IGNORECASE),
        re.compile(r'\bcredit\b', re.IGNORECASE),
        re.compile(r'\bdebit\b', re.IGNORECASE),
        # Example of adding phrases
        re.compile(r'\bpayment\s+received\b', re.IGNORECASE),
        re.compile(r'\binvoice\s+number\b', re.IGNORECASE),
        re.compile(r'\baccount\s+statement\b', re.IGNORECASE),
    ]
    billing_pages = []
    try:
        for page_num, text in enumerate(page_texts, start=1):
            if any(pattern.search(text) for pattern in billing_patterns):
                billing_pages.append(page_num)
                print(f"Page {page_num} identified as billing page.")
    except Exception as e:
        print(f"Error identifying billing pages: {e}")
    return billing_pages

def create_billing_pdf(input_pdf, billing_pages, output_dir):
    try:
        reader = PdfReader(input_pdf)
        writer = PdfWriter()
        for page_num in billing_pages:
            writer.add_page(reader.pages[page_num - 1])  # Zero-based index
        billing_pdf_path = os.path.join(output_dir, 'billing_pages.pdf')
        with open(billing_pdf_path, 'wb') as out_file:
            writer.write(out_file)
        print(f"Billing pages have been extracted into '{billing_pdf_path}'.")
    except Exception as e:
        print(f"Error creating billing PDF: {e}")

def extract_tables_from_billing_pages(billing_pdf_path):
    try:
        tables = camelot.read_pdf(billing_pdf_path, pages='all')
        print(f"Extracted {len(tables)} tables from billing pages.")
        return tables
    except Exception as e:
        print(f"Error extracting tables from billing pages: {e}")
        return []

def analyze_extracted_tables(tables):
    try:
        if isinstance(tables, camelot.core.TableList):
            data_frames = [table.df for table in tables]
        else:
            data_frames = tables  # In case you're using pdfplumber and get a list of lists
        combined_df = pd.concat(data_frames, ignore_index=True)
        # Perform data cleaning and analysis as needed
        return combined_df
    except Exception as e:
        print(f"Error analyzing extracted tables: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def process_pdf(input_pdf):
    try:
        # Determine output directory based on input file name
        input_dir = os.path.dirname(input_pdf)
        base_name = os.path.splitext(os.path.basename(input_pdf))[0]
        output_dir = os.path.join(input_dir, base_name)
        
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        else:
            print(f"Output directory already exists: {output_dir}")
    
        # Step 1: Split the PDF
        split_pdf(input_pdf, output_dir)
        
        # Step 2: Extract text from pages
        total_pages = len(PdfReader(input_pdf).pages)
        extract_text_from_pages(total_pages, output_dir)
        
        # Step 3: Identify billing pages
        page_texts = [os.path.join(output_dir, f'Page_{page_num}.txt') for page_num in range(1, total_pages + 1)]
        billing_pages = identify_billing_pages(page_texts)
        
        if not billing_pages:
            print("No billing pages found.")
            return
        
        # Step 4: Create a PDF of billing pages
        create_billing_pdf(input_pdf, billing_pages, output_dir)
        
        # Step 5: Extract tables from billing pages
        billing_pdf_path = os.path.join(output_dir, 'billing_pages.pdf')
        tables = extract_tables_from_billing_pages(billing_pdf_path)
        
        if not tables:
            print("No tables found in billing pages.")
            return
        
        # Step 6: Analyze the data
        combined_df = analyze_extracted_tables(tables)
        
        if not combined_df.empty:
            # Now you can work with combined_df for your project needs
            print(combined_df.head())
        else:
            print("No data available after analysis.")
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")

if __name__ == '__main__':
    print("Starting PDF processing...")
    process_pdf(input_pdf)
    print("PDF processing completed.")

# Example usage:
# process_pdf('your_input_pdf_file.pdf')
