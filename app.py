import json
import logging
import os
import subprocess
from flask import Flask, request, jsonify
from rag import load_documents, create_index, query_index
from flask_cors import CORS  # Optional: for cross-origin requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Optional: Enable CORS

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
        "python", script_name,
        json.dumps(message_body)
    ]
    try:
        subprocess.run(command, check=True)
        logger.info(f"Successfully processed {file_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        bucket_name = data['bucket']
        records = data.get('records', [])
        bills = data.get('bills', [])
        accident_reports = data.get('reports', [])
        total_records = len(records)
        total_bills = len(bills)
        logger.info(f'Total records: {total_records}, Total bills: {total_bills}')
        
        for accident_report in accident_reports:
            process_file(bucket_name, accident_report['file_path'], 'split_reports.py')
        for record in records:
            process_file(bucket_name, record['file_path'], 'split_records.py')
        for bill in bills:
            process_file(bucket_name, bill['file_path'], 'split_bills.py')
       
        
        logger.info(f"Successfully processed {total_records} records and {total_bills} bills")
        return jsonify({"status": "success", "message": f"Successfully processed {total_records} records and {total_bills} bills"}), 200
    except Exception as e:
        logger.error(f"Error occurred while processing records and bills: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# New route for handling queries
@app.route('/query', methods=['POST'])
def query():
    """
    Handle queries to the RAG system.
    Expects a JSON payload with a 'query' field containing the user's question.
    """
    try:
        data = request.json
        query_text = data.get('query', '')

        if not query_text:
            logger.error('[app] Query text is missing in the request')
            return jsonify({'status': 'error', 'message': 'Query text is required'}), 400

        # Load documents and create index
        documents = load_documents(directory="./data")
        index = create_index(documents, data_directory="./data")
        logger.info('[app] Index loaded and created successfully')

        # Query the index
        response_text = query_index(index, query_text)
        logger.info('[app] Query processed successfully')

        # Return the response
        return jsonify({'status': 'success', 'response': response_text}), 200

    except Exception as e:
        logger.error(f'[app] Error occurred while processing query: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)