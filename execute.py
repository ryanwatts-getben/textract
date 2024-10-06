import json
import logging
import os
import subprocess
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def process_file(bucket_name, file_path, script_name):
    user_id, case_id = file_path.split('/')[:2]
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    message_body = {
        'file_name': file_name_without_extension,
        'file_path': file_path,
        'bucket': bucket_name,
        'case_id': case_id,
        'user_id': user_id,
    }
    logger.info(f"Processing: {message_body}")
    
    # Construct the command to run the appropriate split script
    command = [
        sys.executable,  # Use the current Python interpreter
        script_name,
        json.dumps(message_body)
    ]
    
    logger.info(f"Executing command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True)
        logger.info(f"Successfully processed {file_path} by calling {script_name}")
        logger.info(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {file_path}: {e}")
        logger.error(f"Error output: {e.stderr}")

def main(json_input):
    try:
        data = json.loads(json_input)
        bucket_name = data['bucket']
        records = data.get('records', [])
        bills = data.get('bills', [])
        total_records = len(records)
        total_bills = len(bills)
        logger.info(f'Total records: {total_records}, Total bills: {total_bills}')
        
        for record in records:
            process_file(bucket_name, record['file_path'], 'split_records.py')
        
        for bill in bills:
            process_file(bucket_name, bill['file_path'], 'split_bills.py')
        
        logger.info(f"Successfully processed {total_records} records and {total_bills} bills")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON input: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred while processing records and bills: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python execute.py '<json_input>'")
        sys.exit(1)
    
    json_input = sys.argv[1]
    main(json_input)