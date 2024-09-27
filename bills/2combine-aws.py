import json
import os
import boto3
import anthropic
import base64
import logging
import tiktoken
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')
anthropic_client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 250))
RECORDS_DETAILS_QUEUE = os.environ['RECORDS_DETAILS_QUEUE']
SYSTEM_PROMPT= """Read this medical bill and respond only using this tool to leverage a uniform JSON format in the "output_schema" to extract the data, Do not return this explanation text in your response. Respond using the output_schema only: tools = [{{"name": "parse_medical_record"}},{{"description": "Parses an OCR'd medical record and returns structured data in JSON format."}}, {{"input_schema": {{"type": "object"}},"properties": {{"ocr_text": {{"type": "string"}},{{"description": "The OCR'd text from a medical record."}},}}}},"required": ["ocr_text"]}},"output_schema": {{"type": "object","properties": {{"IsPageDated":{{"type":"boolean"}},"IsPageBlank":{{"type":"boolean"}},"IsPageContentContinuedFromPreviousPage":{{"type":"boolean"}},"Date":{{"type":"string"}},"PathToRecordPDFfilename":{{"type":["string","null"]}},"PathToBillingPDFfilename":{{"type":["string","null"]}},"PatientInformation":{{"type":"object","properties":{{"PatientFirstNameLastName":{{"type":"string"}},"MedicalFacility":{{"type":"string"}},"DoctorFirstNameLastName":{{"type":"array","items":{{"type":"string"}}}},"ReferredTo":{{"type":"array","items":{{"type":"object","properties":{{"DoctorOrFacilityName":{{"type":["string","null"]}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}},"ReferredBy":{{"type":"array","items":{{"type":"object","properties":{{"DoctorOrFacilityName":{{"type":["string","null"]}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}}}},"Codes":{{"type":"object","properties":{{"ICD10CM":{{"type":"array","items":{{"type":"object","additionalProperties":{{"type":"object","properties":{{"Description":{{"type":["string","null"]}},"BillingLineItem":{{"type":["string","null"]}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}},"Rx":{{"type":"array","items":{{"type":"object","additionalProperties":{{"type":"object","properties":{{"Dosage":{{"type":["string","null"]}},"Frequency":{{"type":["string","null"]}},"Duration":{{"type":["string","null"]}},"BillingLineItem":{{"type":["string","null"]}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}},"CPTCodes":{{"type":"array","items":{{"type":"object","additionalProperties":{{"type":"object","properties":{{"Description":{{"type":["string","null"]}},"BillingLineItem":{{"type":["string","null"]}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}}}}}}}},"ProceduresOrFindings":{{"type":"array","items":{{"type":"object","properties":{{"KeyWordsOrFindings":{{"type":"array","items":{{"type":"object","additionalProperties":{{"type":["string","null"]}},"properties":{{"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}}}}}},"DailyFinancialSummary":{{"type":"array","items":{{"type":"object","properties":{{"TotalChargesToday":{{"type":"string"}},"AmountPaidByPatient":{{"type":"string"}},"AmountAdjusted":{{"type":"string"}},"AmountPaidByInsurance":{{"type":"string"}},"AmountOwed":{{"type":"string"}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}},"OtherInformation":{{"type":"array","items":{{"type":"object","properties":{{"Date of Birth":{{"type":["string","null"]}},"Insurance":{{"type":["string","null"]}},"Insured's ID":{{"type":["string","null"]}},"References":{{"type":"object","properties":{{"PageReferences":{{"type":["string","null"]}},"BillingOrRecordDocument":{{"type":["string","null"]}}}}}}}}}}}}}},"required":["IsPageDated","IsPageBlank","IsPageContentContinuedFromPreviousPage"]}}"""
MAX_TOKENS = 180000

def send_sqs(bucket_name, file_path, file_name):
    sqs_queue_url = f'https://sqs.us-east-1.amazonaws.com/461135439633/{RECORDS_DETAILS_QUEUE}'
    user_id, case_id = file_path.split('/')[0:2]
    message_body = {
        'file_name': file_name,
        'file_path': file_path,
        'bucket': bucket_name,
        'case_id': case_id,
        'user_id': user_id,
    }
    logger.info(f"Sending SQS message to details queue: {message_body}")
    try:
        response = sqs_client.send_message(
            QueueUrl=sqs_queue_url,
            MessageBody=json.dumps(message_body)
        )
        logger.info(f"SQS message sent successfully. MessageId: {response['MessageId']}")
    except Exception as e:
        logger.error(f"Failed to send SQS message: {str(e)}")
        raise

def extract_dates_from_content(content):
    """
    Extract dates from the content using regex.
    Standardizes dates to dd-mm-yyyy format.
    """
    date_patterns = [
        r'"Date"\s*:\s*"(\d{2}-\d{2}-\d{4})"',  # Matches "dd-mm-yyyy"
        r'"Date"\s*:\s*"(\d{2}-\d{2}-\d{2})"',  # Matches "dd-mm-yy"
    ]
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, content)
        for date_str in matches:
            # Standardize date to dd-mm-yyyy
            date_parts = date_str.split('-')
            if len(date_parts[2]) == 2:
                # Convert yy to yyyy (assuming 20yy)
                date_parts[2] = '20' + date_parts[2]
            standardized_date = '-'.join(date_parts)
            dates.append(standardized_date)
    return dates

def merge_records(accumulator, new_record):
    """
    Merge two records according to the schema, combining their properties.
    """
    for key, value in new_record.items():
        if key == "Date":
            continue
        elif key not in accumulator:
            accumulator[key] = value
        else:
            if isinstance(value, dict):
                if isinstance(accumulator[key], dict):
                    accumulator[key] = merge_records(accumulator[key], value)
                else:
                    accumulator[key] = value  # Overwrite with new value
            elif isinstance(value, list):
                if isinstance(accumulator[key], list):
                    accumulator[key] = combine_lists(accumulator[key], value)
                else:
                    accumulator[key] = value  # Overwrite with new list
            else:
                accumulator[key] = value  # Overwrite with new value
    return accumulator

def combine_lists(list1, list2):
    """
    Combine two lists, removing duplicates based on their content.
    """
    combined = list1.copy()
    for item2 in list2:
        if item2 not in combined:
            combined.append(item2)
    return combined

def group_records_by_date(records):
    """
    Consolidate records based on Date, preserving all properties as per the schema.
    """
    consolidated = defaultdict(dict)

    for record in records:
        date = record.get("Date")
        if date:
            if date not in consolidated:
                consolidated[date] = record
            else:
                consolidated[date] = merge_records(consolidated[date], record)

    combined_records = list(consolidated.values())
    return combined_records

def handler(event, context):
    logger.info("Event triggered")
    
    try:
        for record in event['Records']:
            body = json.loads(record['body'])
            logger.info(f'Processing body: {body}')
            
            bucket_name = body['bucket']
            file_path = body['file_path']
            user_id = body['user_id']
            case_id = body['case_id']
            file_name = body['file_name']
            
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=file_path)
            file_keys = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.txt')]
            logger.info(f"{len(file_keys)} .txt files found for processing")
            
            all_records = []
            for file_key in file_keys:
                logger.info(f"Processing file: {file_key}")
                try:
                    file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
                    content = file_obj['Body'].read().decode('utf-8')
                    dates = extract_dates_from_content(content)
                    if not dates:
                        logger.warning(f"No dates found in {file_key}")
                        continue

                    # Parse JSON content
                    try:
                        json_content = json.loads(content)
                        if isinstance(json_content, list):
                            all_records.extend(json_content)
                        else:
                            all_records.append(json_content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON in {file_key}: {str(e)}")
                        continue

                except Exception as e:
                    logger.error(f"Error processing file {file_key}: {str(e)}")

            if not all_records:
                logger.warning("No records to process after parsing files.")
                continue

            # Consolidate records by Date
            consolidated_records = group_records_by_date(all_records)

            # Save consolidated records to S3 or process further as needed
            output_key = f"{user_id}/{case_id}/records/{file_name}/combined/combined_records.json"
            s3_client.put_object(Bucket=bucket_name, Key=output_key, Body=json.dumps(consolidated_records))
            logger.info(f"Saved consolidated records to {output_key}")

            # Optionally, send consolidated_records to Claude or further processing
            # For example:
            # for record in consolidated_records:
            #     date = record["Date"]
            #     claude_response = send_to_claude(date, json.dumps(record))
            #     # Handle Claude's response as needed

        logger.info("Processing completed successfully")
        return {
            'statusCode': 200,
            'body': json.dumps('Processing completed successfully')
        }
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }