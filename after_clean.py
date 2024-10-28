import json
import os
import logging
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from datetime import datetime
from fuzzywuzzy import fuzz
import sys
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')

MAX_WORKERS = 1000

def smart_capitalize(name):
    def capitalize_part(part):
        if part.lower().startswith("o'"):
            return "O'" + part[2:].capitalize()
        sub_parts = part.split("'")
        sub_parts = [sp.capitalize() for sp in sub_parts]
        part = "'".join(sub_parts)
        sub_parts = part.split("-")
        sub_parts = [sp.capitalize() for sp in sub_parts]
        part = "-".join(sub_parts)
        return part

    parts = name.split()
    capitalized_parts = [capitalize_part(part) for part in parts]
    return ' '.join(capitalized_parts)

def clean_name(name):
    titles_degrees_salutations = [
        "MD", "DO", "Dr.", "DR", "Dr", "M.D.", "D.O.", "PhD", "Ph.D.",
        "DDS", "D.D.S.", "DVM", "D.V.M.", "Mr.", "Mrs.", "Miss", "Ms.",
        "Mr", "Mrs", "Miss", "Ms", "APRN", "R1", "MT)", "MD)", "MT)",
        "Md", "md"
    ]
    
    for title in titles_degrees_salutations:
        name = re.sub(
            r'\b' + re.escape(title) + r'\b|\B' + re.escape(title) + r'\b|^' + re.escape(title) + r'\s|\s' + re.escape(title) + r'$|\s' + re.escape(title) + r'\s',
            ' ', name, flags=re.IGNORECASE
        )
    
    if ',' in name:
        parts = name.split(',')
        if len(parts) >= 2:
            last_name = parts[0].strip()
            first_names = ' '.join(parts[1:]).strip()
            name = f"{first_names} {last_name}"
    
    name = re.sub(r'[^\w\s\'\-]', '', name)
    name = re.sub(r'\d+', '', name)
    name = re.sub(r'\b[a-zA-Z]\b(?!\')', '', name)
    name = re.sub(r'\s+', ' ', name)
    
    name = smart_capitalize(name)
    name = name.strip()
    
    parts = name.split()
    if len(parts) == 2:
        first_name, last_name = parts
        name = f"{first_name} {last_name}"
    elif len(parts) > 2:
        first_name = parts[0]
        last_name = parts[-1]
        middle_names = ' '.join(parts[1:-1])
        name = f"{first_name} {middle_names} {last_name}"
    
    return name

def remove_duplicates(names):
    unique_names = []
    for name in names:
        if not any(fuzz.ratio(name, unique_name) > 80 for unique_name in unique_names):
            unique_names.append(name)
    return unique_names

def normalize_date(date_str):
    date_formats = [
        "%m/%d/%y", "%m-%d-%y",
        "%m/%d/%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%Y-%m-%d",
        "%d/%m/%y", "%d-%m-%y",
        "%d/%m/%Y", "%d-%m-%Y",
        "%B %d, %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S"
    ]

    date_str = date_str.strip().title()

    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    try:
        from dateutil import parser
        parsed_date = parser.parse(date_str)
        return parsed_date.strftime("%Y-%m-%d")
    except ImportError:
        pass
    except ValueError:
        pass

    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            continue

    if re.match(r'^\d{8}$', date_str):
        try:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        except:
            pass

    if re.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$', date_str):
        parts = date_str.split('.')
        if int(parts[1]) <= 12:
            try:
                return datetime(int(parts[2]), int(parts[0]), int(parts[1])).strftime("%Y-%m-%d")
            except ValueError:
                pass
        try:
            return datetime(int(parts[2]), int(parts[1]), int(parts[0])).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None

def update_references(data, page_number_str, path_to_filename_pdf, total_pages, document_type):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "References" and isinstance(value, dict):
                value['ThisIsPageNumberOfPDF'] = page_number_str
                value['PathToFilenamePDF'] = path_to_filename_pdf
                value['DocumentType'] = document_type
                value['TotalPagesInPDF'] = total_pages
            else:
                update_references(value, page_number_str, path_to_filename_pdf, total_pages, document_type)
    elif isinstance(data, list):
        for item in data:
            update_references(item, page_number_str, path_to_filename_pdf, total_pages, document_type)

def reconfigure_json_structure(data, page_number, path_to_filename_pdf, document_type):
    page_number_str = str(page_number)

    data.pop("IsPageDated", None)
    data.pop("IsPageBlank", None)
    data.pop("IsPageContentContinuedFromPreviousPage", None)
    data.pop("PathToRecordPDFfilename", None)
    data.pop("PathToBillingPDFfilename", None)

    total_pages = data.pop("TotalPagesInPDF", None)
    if total_pages is None:
        total_pages = data.pop("TotalPagesInBillingPDF", None)

    update_references(data, page_number_str, path_to_filename_pdf, total_pages, document_type)
    
    data.pop("ThisIsPageNumberOfPDF", None)
    
    data["Reference"] = page_number_str
    
    return data

def process_json_file(bucket, input_key, output_key, page_number, path_to_filename_pdf, document_type):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=input_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        
        if "PatientInformation" in data:
            patient_info = data["PatientInformation"]
            
            if "PatientFirstNameLastName" in patient_info:
                original_patient_name = patient_info["PatientFirstNameLastName"]
                cleaned_patient_name = clean_name(original_patient_name)
                patient_info["PatientFirstNameLastName"] = cleaned_patient_name
                logger.info(f"Cleaned patient name in {output_key}: {cleaned_patient_name}")
            
            if "DoctorFirstNameLastName" in patient_info and isinstance(patient_info["DoctorFirstNameLastName"], list):
                original_names = patient_info["DoctorFirstNameLastName"]
                cleaned_names = [clean_name(name) for name in original_names]
                cleaned_names = remove_duplicates(cleaned_names)
                patient_info["DoctorFirstNameLastName"] = cleaned_names
                logger.info(f"Cleaned doctor names in {output_key}: {cleaned_names}")
            
            if "ReferredTo" in patient_info and isinstance(patient_info["ReferredTo"], list):
                cleaned_names = [clean_name(name) for name in patient_info["ReferredTo"]]
                cleaned_names = remove_duplicates(cleaned_names)
                patient_info["ReferredTo"] = cleaned_names
            
            if "ReferredBy" in patient_info and isinstance(patient_info["ReferredBy"], list):
                cleaned_names = [clean_name(name) for name in patient_info["ReferredBy"]]
                cleaned_names = remove_duplicates(cleaned_names)
                patient_info["ReferredBy"] = cleaned_names
        
        if "Date" in data:
            original_date = data["Date"]
            if original_date is not None:
                normalized_date = normalize_date(original_date)
                if normalized_date:
                    data["Date"] = normalized_date
                    logger.info(f"Normalized date in {output_key}: {normalized_date}")
            else:
                logger.info(f"Date is None in {output_key}")
        
        data = reconfigure_json_structure(data, page_number, path_to_filename_pdf, document_type)
        
        s3_client.put_object(Bucket=bucket, Key=output_key, Body=json.dumps(data, indent=2))
        logger.info(f"Processed and uploaded: {output_key}")
    except Exception as e:
        logger.error(f"Error processing file {input_key}: {e}")


def extract_page_info(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        first_line = response['Body'].readline().decode('utf-8').strip()
        match = re.match(r'This is File Name: (.*) and Page Number: (\d+)', first_line)
        if match:
            path_to_filename_pdf = match.group(1)
            page_number = int(match.group(2))
            return path_to_filename_pdf, page_number
    except Exception as e:
        logger.error(f"Error reading page info from {key}: {e}")
    return None, None

def check_output_exists(bucket_name, output_key):
    try:
        s3_client.head_object(Bucket=bucket_name, Key=output_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise

def process_files(bucket, clean_prefix, after_clean_prefix, document_type):
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=clean_prefix)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for page in pages:
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.txt'):
                    input_key = obj['Key']
                    output_key = input_key.replace(clean_prefix, after_clean_prefix)
                    split_key = input_key.replace('/clean/', '/split/')
                    
                    # Check if both input and output files exist
                    input_exists = check_output_exists(bucket, input_key)
                    output_exists = check_output_exists(bucket, output_key)
                    
                    if input_exists and output_exists:
                        logger.info(f"Both input and output files exist for {input_key}. Skipping processing.")
                        continue
                    
                    path_to_filename_pdf, page_number = extract_page_info(bucket, split_key)
                    if page_number is not None and path_to_filename_pdf is not None:
                        future = executor.submit(
                            process_json_file,
                            bucket,
                            input_key,
                            output_key,
                            page_number,
                            path_to_filename_pdf,
                            document_type
                        )
                        futures.append(future)
                    else:
                        logger.warning(f"Could not extract page info for {input_key}")

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in processing file: {str(e)}")


def count_s3_objects(bucket, prefix):
    """
    Count the number of objects in an S3 bucket with a given prefix.
    This function handles pagination to count more than 1000 objects.
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            count += len(page['Contents'])
    return count

def main(input_json, retry_count=0):
    try:
        data = json.loads(input_json)
        bucket = data['bucket']
        user_id = data['user_id']
        case_id = data['case_id']
        file_name = data['file_name']
        file_path = data['file_path']

        logger.info(f"Processing started (Attempt {retry_count + 1})")

        # Determine if it's a bill or record based on the file path
        document_type = 'Billing' if '/bills/' in file_path else 'Record'
        folder_type = 'bills' if document_type == 'Billing' else 'records'

        clean_prefix = f"{user_id}/{case_id}/{folder_type}/{file_name}/clean/"
        after_clean_prefix = f"{user_id}/{case_id}/{folder_type}/{file_name}/after-clean/"

        # Count files in clean and after-clean directories
        clean_file_count = count_s3_objects(bucket, clean_prefix)
        after_clean_file_count = count_s3_objects(bucket, after_clean_prefix)

        logger.info(f"Files in clean directory: {clean_file_count}")
        logger.info(f"Files in after-clean directory: {after_clean_file_count}")

        if clean_file_count == after_clean_file_count and clean_file_count > 0:
            logger.info("All files already processed. Skipping cleaning step.")
        else:
            logger.info("Processing files")
            process_files(bucket, clean_prefix, after_clean_prefix, document_type)

            # Recount files after processing
            after_clean_file_count = count_s3_objects(bucket, after_clean_prefix)
            logger.info(f"Files in after-clean directory after processing: {after_clean_file_count}")

            if clean_file_count != after_clean_file_count:
                logger.warning("Not all files were processed.")
                logger.warning(f"Expected {clean_file_count} files, found {after_clean_file_count} in after-clean")
                
                if retry_count < 1:
                    logger.info(f"Retrying process (Attempt {retry_count + 1})")
                    return main(input_json, retry_count + 1)
                else:
                    logger.error("Max retry attempts reached. Process failed.")

        logger.info("All files processed successfully")

        # Update file_path to after_clean_prefix
        after_clean_path = f"{user_id}/{case_id}/{folder_type}/{file_name}/after-clean"

        # Prepare input for combine.py
        combine_input = json.dumps({
            'bucket': bucket,
            'user_id': user_id,
            'case_id': case_id,
            'file_name': file_name,
            'file_path': after_clean_path
        })

        # Call combine.py as a subprocess
        logger.info("Calling combine.py")
        try:
            result = subprocess.run(['python', 'combine.py', combine_input], 
                                    check=True, 
                                    capture_output=True, 
                                    text=True)
            logger.info(f"combine.py completed successfully. Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running combine.py: {e}")
            logger.error(f"combine.py error output: {e.stderr}")
            raise

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON input: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while processing the files: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python clean.py '<json_input>'")
        sys.exit(1)
    
    json_input = sys.argv[1]
    main(json_input)