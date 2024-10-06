import json
import os
import boto3
from jinja2 import Environment, BaseLoader
from datetime import datetime
import sys
from dotenv import load_dotenv
import logging
import traceback

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize S3 client
s3 = boto3.client('s3')

# Get bucket and template file name from environment variables
PROMPT_BUCKET = os.getenv('PROMPT_BUCKET')
TEMPLATE_FILE = os.getenv('TEMPLATE_FILE')

def sanitize_filename(filename):
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def parse_date(date_string):
    if not date_string:
        return 'Unknown_Date'
    date_formats = ['%Y-%m-%d', '%m/%d/%y', '%m/%d/%Y', '%Y/%m/%d']
    for fmt in date_formats:
        try:
            return datetime.strptime(date_string, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return 'Unknown_Date'

def process_page_numbers(references):
    if isinstance(references, dict) and 'ThisIsPageNumberOfPDF' in references:
        page_numbers = references['ThisIsPageNumberOfPDF']
        if isinstance(page_numbers, str):
            page_numbers = [page_numbers]
        elif isinstance(page_numbers, int):
            page_numbers = [str(page_numbers)]
        else:
            page_numbers = [str(num) for num in page_numbers]
        references['ThisIsPageNumberOfPDF'] = page_numbers
    return references

def process_codes(codes):
    if not isinstance(codes, dict):
        return {}
    for category in codes.values():
        if isinstance(category, list):
            for item in category:
                if isinstance(item, dict):
                    for code_info in item.values():
                        if isinstance(code_info, dict) and 'References' in code_info:
                            code_info['References'] = process_page_numbers(code_info['References'])
    return codes

def process_lab_results(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ['Chemistry', 'LabResults']:
                processed_results = {}
                for test, result in value.items():
                    if isinstance(result, str):
                        parts = result.split()
                        if len(parts) > 1:
                            processed_results[test] = {
                                'Result': parts[0],
                                'Range': ' '.join(parts[1:])
                            }
                        else:
                            processed_results[test] = {'Result': result, 'Range': ''}
                    elif isinstance(result, dict):
                        processed_results[test] = result
                data[key] = processed_results
            elif isinstance(value, (dict, list)):
                process_lab_results(value)
    elif isinstance(data, list):
        for item in data:
            process_lab_results(item)

def get_template_from_s3():
    try:
        response = s3.get_object(Bucket=PROMPT_BUCKET, Key=TEMPLATE_FILE)
        template_content = response['Body'].read().decode('utf-8')
        return template_content
    except Exception as e:
        logger.error(f"Error retrieving template from S3: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def count_s3_objects(bucket, prefix):
    try:
        paginator = s3.get_paginator('list_objects_v2')
        count = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                count += len(page['Contents'])
        return count
    except Exception as e:
        logger.error(f"Error counting S3 objects: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main(input_json, retry_count=0):
    try:
        # Parse input JSON
        data = json.loads(input_json)
        bucket_name = data['bucket']
        user_id = data['user_id']
        case_id = data['case_id']

        logger.info(f"HTML generation started (Attempt {retry_count + 1})")

        complete_prefix = f"{user_id}/{case_id}/complete/"
        html_prefix = f"{user_id}/{case_id}/html/"

        # Count files in complete and html directories
        complete_file_count = count_s3_objects(bucket_name, complete_prefix)
        html_file_count = count_s3_objects(bucket_name, html_prefix)

        logger.info(f"Files in complete directory: {complete_file_count}")
        logger.info(f"Files in html directory: {html_file_count}")

        if complete_file_count == html_file_count and complete_file_count > 0:
            logger.info("All HTML files already generated. Skipping HTML generation.")
            return

        # Check if environment variables are set
        if not PROMPT_BUCKET or not TEMPLATE_FILE:
            raise ValueError("PROMPT_BUCKET and TEMPLATE_FILE must be set in the .env file")

        # Get the template from S3
        template_content = get_template_from_s3()
        env = Environment(loader=BaseLoader())
        template = env.from_string(template_content)

        # List all JSON files in the input prefix
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=complete_prefix)

        for page in pages:
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.json'):
                    try:
                        # Read the JSON file from S3
                        response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                        file_content = response['Body'].read().decode('utf-8')
                        json_data = json.loads(file_content)

                        # Extract necessary data from JSON
                        date = parse_date(json_data.get('Date'))
                        patient_info = json_data.get('PatientInformation', {})
                        codes = process_codes(json_data.get('Codes', {}))
                        procedures_or_findings = json_data.get('ProceduresOrFindings', [])
                        daily_financial_summary = json_data.get('DailyFinancialSummary', [])
                        other_information = json_data.get('OtherInformation', [])

                        # Process lab results
                        process_lab_results(procedures_or_findings)

                        # Render the HTML using the template
                        rendered_html = template.render(
                            date=date,
                            patient_information=patient_info,
                            codes=codes,
                            procedures_or_findings=procedures_or_findings,
                            daily_financial_summary=daily_financial_summary,
                            other_information=other_information
                        )

                        # Define output HTML file path in S3
                        safe_filename = sanitize_filename(f"{date}.html")
                        output_key = f"{html_prefix}{safe_filename}"

                        # Write the rendered HTML to S3
                        s3.put_object(Bucket=bucket_name, Key=output_key, Body=rendered_html, ContentType='text/html')

                        logger.info(f"Generated HTML for {date}: s3://{bucket_name}/{output_key}")
                    except Exception as e:
                        logger.error(f"Error processing file {obj['Key']}: {str(e)}")
                        logger.error(traceback.format_exc())

        # Recount files after processing
        html_file_count = count_s3_objects(bucket_name, html_prefix)
        logger.info(f"Files in html directory after processing: {html_file_count}")

        if complete_file_count != html_file_count:
            logger.warning("Not all HTML files were generated.")
            logger.warning(f"Expected {complete_file_count} files, found {html_file_count} in html")
            
            if retry_count < 3:
                logger.info(f"Retrying process (Attempt {retry_count + 2})")
                return main(input_json, retry_count + 1)
            else:
                logger.error("Max retry attempts reached. Process failed.")
                raise Exception("Failed to generate all HTML files after multiple attempts")

        logger.info("HTML generation completed successfully")

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding input JSON: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during HTML generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python generate_html.py '<json_input>'")
        sys.exit(1)

    json_input = sys.argv[1]
    try:
        main(json_input)
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)