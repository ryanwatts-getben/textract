import requests
import json
import os
import sys
import argparse
import datetime
import time
from dotenv import load_dotenv
from salesforce_refresh_token import get_salesforce_headers, refresh_salesforce_token, salesforce_request, verify_credentials

# Load environment variables from .env file
load_dotenv()

# Initialize headers and instance URL
headers = None
SALESFORCE_INSTANCE_URL = None

def initialize_salesforce():
    """Initialize Salesforce connection and verify credentials.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global headers, SALESFORCE_INSTANCE_URL
    
    print("[get_documents] Initializing Salesforce connection...")
    
    # First verify Salesforce CLI credentials
    cli_available = verify_credentials()
    if not cli_available:
        print("[get_documents] ⚠️ Salesforce CLI credentials are not properly configured.")
        print("\n[get_documents] To fix this issue:")
        print("  Option 1: Run the script with the --sf-path argument:")
        print("    python salesforce_get_all_documents_by_matter_id.py --sf-path \"C:\\path\\to\\sf.cmd\"")
        print("\n  Option 2: Set the SF_CLI_PATH environment variable:")
        print("    In PowerShell: $env:SF_CLI_PATH = \"C:\\path\\to\\sf.cmd\"")
        print("    In CMD: set SF_CLI_PATH=C:\\path\\to\\sf.cmd")
        print("\n  Option 3: Find where sf is installed:")
        print("    In PowerShell: where.exe sf")
        print("    Then use that path with option 1 or 2")
        return False
    
    # Get Salesforce credentials and headers with token validation/refresh
    headers, SALESFORCE_INSTANCE_URL = get_salesforce_headers()
    
    if not headers or not SALESFORCE_INSTANCE_URL:
        print("[get_documents] ❌ Failed to obtain Salesforce authentication.")
        print("[get_documents] Please ensure you are logged in with the Salesforce CLI:")
        print("    sf org login web --instance-url https://louisfirm.my.salesforce.com --alias louisfirm")
        return False
    
    print(f"[get_documents] ✅ Successfully connected to Salesforce: {SALESFORCE_INSTANCE_URL}")
    return True

# Helper function to handle 401 errors with automatic token refresh
def execute_salesforce_request(request_func, *args, **kwargs):
    """
    Execute a Salesforce API request with automatic token refresh on 401 errors.
    
    Args:
        request_func: The requests function to execute (e.g., requests.get, requests.post)
        *args: Positional arguments to pass to the request function
        **kwargs: Keyword arguments to pass to the request function
        
    Returns:
        requests.Response: The response from the Salesforce API
    """
    global headers, SALESFORCE_INSTANCE_URL
    max_retries = 1  # Only retry once to avoid infinite loops
    
    # Make sure we have headers and instance URL
    if not headers or not SALESFORCE_INSTANCE_URL:
        if not initialize_salesforce():
            raise Exception("Failed to initialize Salesforce connection. Please check CLI configuration.")
    
    for attempt in range(max_retries + 1):
        try:
            # Make the request
            response = request_func(*args, **kwargs)
            
            # Check if we got a 401 Unauthorized error
            if response.status_code == 401 and attempt < max_retries:
                print("[get_documents] Received 401 error, refreshing token...")
                
                # Refresh the token with force_refresh=True to bypass cache
                new_token, new_instance_url = refresh_salesforce_token()
                
                if new_token:
                    # Update our global headers
                    headers["Authorization"] = f"Bearer {new_token}"
                    if new_instance_url:
                        SALESFORCE_INSTANCE_URL = new_instance_url
                    
                    # Update the headers in the request kwargs
                    if 'headers' in kwargs:
                        kwargs['headers']["Authorization"] = f"Bearer {new_token}"
                    
                    # Continue to the next attempt
                    continue
            
            # Raise an exception for bad responses
            response.raise_for_status()
            return response
        
        except requests.exceptions.RequestException as e:
            # On the last attempt, raise the exception
            if attempt == max_retries:
                print(f"[get_documents] Request error after {max_retries + 1} attempts: {e}")
                raise
            
            # For connection errors, wait a moment before retrying
            if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                print(f"[get_documents] Connection error: {e}. Retrying...")
                time.sleep(1)  # Wait 1 second before retry
            else:
                # For other errors, just raise
                raise

def escape_soql_string(value):
    """
    Escape a string value for safe use in a SOQL query.
    
    Args:
        value (str): The string value to escape
        
    Returns:
        str: The escaped string
    """
    if value is None:
        return "null"
    
    # Escape single quotes by doubling them
    return str(value).replace("'", "\\'")

def get_documents_by_matter_id(matter_id):
    """
    Get documents associated with a specific Matter ID using Salesforce ContentDocumentLink.
    
    Args:
        matter_id (str): The Salesforce Matter ID
        
    Returns:
        list: List of document details
    """
    print(f"[get_documents] Retrieving documents for Matter ID: {matter_id}")
    
    # Input validation
    if not matter_id:
        print("[get_documents] Error: matter_id cannot be empty")
        return {"error": "matter_id cannot be empty"}
    
    # Escape the matter_id to prevent SOQL injection
    escaped_matter_id = escape_soql_string(matter_id)
    
    # Initialize Salesforce connection if needed
    if not headers or not SALESFORCE_INSTANCE_URL:
        if not initialize_salesforce():
            return {"error": "Failed to initialize Salesforce connection"}
    
    # Step 1: Query ContentDocumentLink to find documents linked to the matter
    query = (
        f"SELECT ContentDocumentId, LinkedEntityId, ContentDocument.Title, "
        f"ContentDocument.FileType, ContentDocument.FileExtension, ContentDocument.CreatedDate "
        f"FROM ContentDocumentLink "
        f"WHERE LinkedEntityId = '{escaped_matter_id}'"
    )
    
    query_url = f"{SALESFORCE_INSTANCE_URL}/services/data/v63.0/query?q={query}"
    
    try:
        response = execute_salesforce_request(requests.get, query_url, headers=headers)
        
        document_links = response.json()
        
        if document_links["totalSize"] == 0:
            print(f"[get_documents] No documents found for Matter ID: {matter_id}")
            return []
        
        print(f"[get_documents] Found {document_links['totalSize']} documents")
        
        # Step 2: Get ContentVersion details for each document
        documents = []
        for record in document_links["records"]:
            content_document_id = record["ContentDocumentId"]
            
            # Escape the content_document_id to prevent SOQL injection
            escaped_content_document_id = escape_soql_string(content_document_id)
            
            # Query ContentVersion to get the latest version of the document
            version_query = (
                f"SELECT Id, Title, PathOnClient, FileType, FileExtension, VersionNumber, "
                f"ContentSize, ContentDocumentId, CreatedDate, LastModifiedDate "
                f"FROM ContentVersion "
                f"WHERE ContentDocumentId = '{escaped_content_document_id}' "
                f"AND IsLatest = true"
            )
            
            version_url = f"{SALESFORCE_INSTANCE_URL}/services/data/v63.0/query?q={version_query}"
            version_response = execute_salesforce_request(requests.get, version_url, headers=headers)
            
            version_data = version_response.json()
            
            if version_data["totalSize"] > 0:
                version = version_data["records"][0]
                
                # Add document details to the list
                document = {
                    "ContentDocumentId": content_document_id,
                    "Title": version["Title"],
                    "FileName": version["PathOnClient"],
                    "FileType": version["FileType"],
                    "FileExtension": version["FileExtension"],
                    "VersionNumber": version["VersionNumber"],
                    "ContentSize": version["ContentSize"],
                    "ContentVersionId": version["Id"],
                    "CreatedDate": version["CreatedDate"],
                    "LastModifiedDate": version["LastModifiedDate"],
                    "DownloadUrl": f"{SALESFORCE_INSTANCE_URL}/services/data/v63.0/sobjects/ContentVersion/{version['Id']}/VersionData"
                }
                
                documents.append(document)
        
        return documents
    
    except requests.exceptions.RequestException as e:
        print(f"[get_documents] Error retrieving documents: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[get_documents] Response status: {e.response.status_code}")
            print(f"[get_documents] Response body: {e.response.text}")
        return []

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Get documents for a Salesforce Matter')
    parser.add_argument('--matter-id', required=True, help='Salesforce Matter ID')
    parser.add_argument('--output', default="matter_documents.json", help='Output filename (default: matter_documents.json)')
    parser.add_argument('--sf-path', help='Path to the Salesforce CLI executable (sf)')
    args = parser.parse_args()
    
    # If SF_CLI_PATH is provided as an argument, set it in the environment
    if args.sf_path:
        os.environ['SF_CLI_PATH'] = args.sf_path
        print(f"[get_documents] Using Salesforce CLI path: {args.sf_path}")
    
    # Initialize Salesforce connection
    if not initialize_salesforce():
        print("[get_documents] ❌ Unable to connect to Salesforce. Exiting.")
        sys.exit(1)
    
    # Use the provided Matter ID
    matter_id = args.matter_id
    
    print(f"[get_documents] Using Matter ID: {matter_id}")
    
    # Get documents for the Matter
    try:
        documents = get_documents_by_matter_id(matter_id)
        
        if not documents:
            print("[get_documents] No documents found.")
            sys.exit(0)
            
        print(f"[get_documents] Retrieved {len(documents)} documents")
        
        # Save the documents to a file
        with open(args.output, "w") as f:
            json.dump(documents, f, indent=2)
        
        print(f"[get_documents] Documents saved to: {args.output}")
        
        print("\n[get_documents] Done!")
    except Exception as e:
        print(f"[get_documents] Error: {e}")
        print("[get_documents] Unable to retrieve documents. Please check your Salesforce connection and try again.")
        sys.exit(1)
