import boto3
import json
import os
from flask.json import JSONEncoder
from dotenv import load_dotenv
from app import app

load_dotenv()

def upload_swagger_docs():
    with app.test_request_context():
        swagger_json = app.test_client().get('/swagger.json').json

    # Format JSON with indentation
    formatted_json = json.dumps(swagger_json, indent=2)

    # Upload to S3
    s3 = boto3.client('s3')
    bucket_name = os.getenv('S3_SWAGGER_DOCS', 'swagger-medchron-api')
    
    try:
        s3.put_object(
            Bucket=bucket_name,
            Key='swagger.json',
            Body=formatted_json,
            ContentType='application/json'
        )
        print(f"Successfully uploaded swagger.json to s3://{bucket_name}/swagger.json")
    except Exception as e:
        print(f"Error uploading swagger docs: {str(e)}")

if __name__ == "__main__":
    upload_swagger_docs() 