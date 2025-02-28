#!/usr/bin/env python
import json
import requests
import logging
import sys
import datetime
import os
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger('salesforce_create_new_client')

def post_to_nulaw(matter_id: str, sf_path: Optional[str] = None, download_files: bool = True) -> Optional[Dict[str, Any]]:
    """
    Make a POST request to the /nulaw endpoint to get matter context
    
    Args:
        matter_id (str): The Salesforce Matter ID
        sf_path (str, optional): Path to the Salesforce CLI executable
        download_files (bool): Whether to download files from SharePoint (defaults to True)
        
    Returns:
        dict or None: Response data or None if request failed
    """
    logger.info(f"[salesforce_create_new_client] Fetching matter context for ID: {matter_id}")
    
    url = "http://localhost:5000/nulaw"
    headers = {
        "Content-Type": "application/json"
    }
    
    # Handle the case where download_files is passed as a string
    if isinstance(download_files, str):
        download_files = download_files.lower() != 'false'
    
    logger.info(f"[salesforce_create_new_client] Setting download_files={download_files}")
    
    payload = {
        "matter_id": matter_id,
        "download_files": download_files
    }
    
    if sf_path:
        payload["sf_path"] = sf_path
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"[salesforce_create_new_client] Error making POST request to /nulaw: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"[salesforce_create_new_client] Response status: {e.response.status_code}")
            logger.error(f"[salesforce_create_new_client] Response body: {e.response.text}")
        return None

def format_project_payload(nulaw_response: Dict[str, Any], download_files: bool = True) -> Optional[Dict[str, Any]]:
    """
    Format the response from /nulaw endpoint into the project creation payload
    
    Args:
        nulaw_response (dict): Response from the /nulaw endpoint
        download_files (bool): Whether to download files from SharePoint (defaults to True)
        
    Returns:
        dict or None: Formatted payload for project creation or None if formatting failed
    """
    try:
        # Handle the case where download_files is passed as a string
        if isinstance(download_files, str):
            download_files = download_files.lower() != 'false'
        
        logger.info(f"[salesforce_create_new_client] Formatting payload with download_files={download_files}")
        
        if nulaw_response.get('status') != 'success' or 'context' not in nulaw_response:
            logger.error(f"[salesforce_create_new_client] Invalid response from /nulaw: {nulaw_response}")
            return None
        
        context = nulaw_response['context']
        matter = context.get('matter', {})
        
        # Extract required fields
        client_name = matter.get('Client_Full_Name__c', '')
        
        # Process incident date (trim time off)
        incident_date = matter.get('Accident_Date_Time__c', '')
        if incident_date:
            # Try to extract just the date part (YYYY-MM-DD)
            try:
                # Check if the date has a time component
                if 'T' in incident_date:
                    incident_date = incident_date.split('T')[0]
                # Check if it has spaces (indicating a different format)
                elif ' ' in incident_date:
                    incident_date = incident_date.split(' ')[0]
            except Exception as e:
                logger.warning(f"[salesforce_create_new_client] Error processing incident date: {e}")
        
        # Get treatments total billed amount
        meds_total = matter.get('Treatments_Total_Billed_Amount__c', '')
        # Convert meds_total to string to satisfy API requirements
        meds_total_str = str(meds_total) if meds_total != '' else ''
        
        # Get case type and sub case type for injury_type_other
        case_type = matter.get('nu_law__Case_Type__c', '')
        sub_case_type = matter.get('nu_law__Sub_Case_Type__c', '')
        injury_type_other = f"{case_type} - {sub_case_type}" if case_type and sub_case_type else ""
        
        # Collect additional context
        additional_context = {
            "incident_details": context.get('matter', {}).get('categorized_fields', {}).get('incident_details', {}),
            "basic": context.get('matter', {}).get('categorized_fields', {}).get('basic', {}),
            "medical_treatment": context.get('matter', {}).get('categorized_fields', {}).get('medical_treatment', {}),
            "treatment_decisions": context.get('matter', {}).get('categorized_fields', {}).get('treatment_decisions', {}),
            "insurance_financial": context.get('matter', {}).get('categorized_fields', {}).get('insurance_financial', {}),
            "pre_existing_conditions": context.get('matter', {}).get('categorized_fields', {}).get('pre_existing_conditions', {}),
            "treatment_records": context.get('treatment_records', {}),
            "treatments": context.get('treatment_records', {}).get('treatment_amounts', {}),
            "insurance_information": {"records": context.get('insurance_information', [])}
        }
        
        # Format the project creation payload
        payload = {
            "clientName": client_name,
            "projectEmailAddress": "skinner@everycase.ai",  # Always this
            "incidentDate": incident_date,
            "description": "Injury Type: MVA",  # Always this
            "source": "NULAW",  # Always this
            "download_files": download_files,  # Add the download_files flag
            "ProjectAdditionalInfo": {
                "MedsTotal": meds_total_str,  # Use string version here
                "FormFields": {
                    "client_name": client_name,
                    "email": "skinner@everycase.ai",  # Always this
                    "incident_date": incident_date,
                    "medical_bills_total": meds_total_str,  # Also use string version here
                    "injury_type": "MVA",  # Always this
                    "source": "NULAW",  # Always this
                    "injury_type_other": injury_type_other,
                    "download_files": download_files,  # Also add to form fields
                    "additional_context": json.dumps(additional_context)
                }
            }
        }
        
        return payload
    except Exception as e:
        logger.error(f"[salesforce_create_new_client] Error formatting project payload: {e}")
        return None

def create_project(project_payload: Dict[str, Any], matter_id: str = 'unknown', timestamp: str = None, logs_dir: str = '') -> Optional[Dict[str, Any]]:
    """
    Send a POST request to create a new project
    
    Args:
        project_payload (dict): Formatted payload for project creation
        matter_id (str): Salesforce Matter ID for logging purposes
        timestamp (str): Timestamp for log filename
        logs_dir (str): Directory for logging files
        
    Returns:
        dict or None: Response data or None if request failed
    """
    logger.info("[salesforce_create_new_client] Creating new project")
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Log the request payload to a JSON file for debugging
    try:
        post_filename = os.path.join(logs_dir, f'salesforce_create_new_client_post_{matter_id}_{timestamp}.json')
        with open(post_filename, 'w') as f:
            json.dump(project_payload, f, indent=2)
        logger.info(f"[salesforce_create_new_client] Request payload logged to {post_filename}")
    except Exception as log_error:
        logger.error(f"[salesforce_create_new_client] Failed to log request payload: {str(log_error)}")
    
    url = "http://localhost:5000/api/project/create"
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json=project_payload)
        response.raise_for_status()
        
        # Log the response payload
        try:
            response_data = response.json()
            response_filename = os.path.join(logs_dir, f'salesforce_create_new_client_response_{matter_id}_{timestamp}.json')
            with open(response_filename, 'w') as f:
                json.dump(response_data, f, indent=2)
            logger.info(f"[salesforce_create_new_client] Response payload logged to {response_filename}")
        except Exception as log_error:
            logger.error(f"[salesforce_create_new_client] Failed to log response: {str(log_error)}")
            
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"[salesforce_create_new_client] Error creating project: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"[salesforce_create_new_client] Response status: {e.response.status_code}")
            logger.error(f"[salesforce_create_new_client] Response body: {e.response.text}")
            
            # Log error response if available
            try:
                error_filename = os.path.join(logs_dir, f'salesforce_create_new_client_error_{matter_id}_{timestamp}.txt')
                with open(error_filename, 'w') as f:
                    f.write(f"Status: {e.response.status_code}\n\n")
                    f.write(f"Body: {e.response.text}")
                logger.info(f"[salesforce_create_new_client] Error response logged to {error_filename}")
            except Exception as log_error:
                logger.error(f"[salesforce_create_new_client] Failed to log error response: {str(log_error)}")
                
        return None

def process_nulaw_response_and_create_project(nulaw_response: Dict[str, Any], dry_run: bool = False, download_files: bool = True) -> Optional[Dict[str, Any]]:
    """
    Process a response from the /nulaw endpoint and create a new project
    
    This function is designed to be called from the Flask app after the /nulaw endpoint
    processes a request.
    
    Args:
        nulaw_response (dict): Response from the /nulaw endpoint
        dry_run (bool): If True, only format the payload without creating the project
        download_files (bool): Whether to download files from SharePoint (defaults to True)
        
    Returns:
        dict or None: Response from project creation or formatted payload (if dry_run)
    """
    # Handle the case where download_files is passed as a string
    if isinstance(download_files, str):
        download_files = download_files.lower() != 'false'
    
    logger.info(f"[salesforce_create_new_client] Processing with download_files={download_files}")
    
    # Get matter ID for logging
    matter_id = nulaw_response.get('matter_id', 'unknown')
    
    # Generate timestamp for log filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    logs_dir = 'logs'
    try:
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            logger.info(f"[salesforce_create_new_client] Created logs directory: {logs_dir}")
    except Exception as e:
        logger.error(f"[salesforce_create_new_client] Error creating logs directory: {str(e)}")
        logs_dir = ''  # Fall back to current directory
    
    # Log the input nulaw response for debugging
    try:
        response_filename = os.path.join(logs_dir, f'salesforce_nulaw_response_{matter_id}_{timestamp}.json')
        with open(response_filename, 'w') as f:
            json.dump(nulaw_response, f, indent=2)
        logger.info(f"[salesforce_create_new_client] Input nulaw response logged to {response_filename}")
    except Exception as log_error:
        logger.error(f"[salesforce_create_new_client] Failed to log nulaw response: {str(log_error)}")
    
    # Format the payload for project creation
    project_payload = format_project_payload(nulaw_response, download_files)
    if not project_payload:
        logger.error("[salesforce_create_new_client] Failed to format project payload")
        return None
    
    # Log the formatted project payload for debugging
    try:
        payload_filename = os.path.join(logs_dir, f'salesforce_formatted_payload_{matter_id}_{timestamp}.json')
        with open(payload_filename, 'w') as f:
            json.dump(project_payload, f, indent=2)
        logger.info(f"[salesforce_create_new_client] Formatted project payload logged to {payload_filename}")
    except Exception as log_error:
        logger.error(f"[salesforce_create_new_client] Failed to log formatted payload: {str(log_error)}")
    
    # If it's a dry run, just return the formatted payload
    if dry_run:
        logger.info("[salesforce_create_new_client] Dry run - returning formatted payload")
        return {
            "status": "success",
            "message": "Dry run - project would be created with this payload",
            "payload": project_payload
        }
    
    # Create the project
    response = create_project(project_payload, matter_id, timestamp, logs_dir)
    if response:
        logger.info(f"[salesforce_create_new_client] Successfully created project")
        return {
            "status": "success",
            "message": "Project created successfully",
            "project": response,
            "original_matter_id": matter_id
        }
    else:
        logger.error("[salesforce_create_new_client] Failed to create project")
        return {
            "status": "error",
            "message": "Failed to create project"
        }

def main():
    """Main function to handle CLI arguments and execute the process"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a new client project from Salesforce Matter data')
    parser.add_argument('--matter-id', required=True, help='Salesforce Matter ID')
    parser.add_argument('--sf-path', help='Path to the Salesforce CLI executable (sf)')
    parser.add_argument('--dry-run', action='store_true', help='Only display the payload without creating the project')
    parser.add_argument('--download-files', dest='download_files', action='store_true', default=True, 
                        help='Download files from SharePoint (default: True)')
    parser.add_argument('--no-download-files', dest='download_files', action='store_false',
                        help='Do not download files from SharePoint')
    
    args = parser.parse_args()
    
    # Convert string values if they came in as strings
    if isinstance(args.download_files, str):
        args.download_files = args.download_files.lower() != 'false'
    
    logger.info(f"[salesforce_create_new_client] Starting with download_files={args.download_files}")
    
    # Step 1: Fetch data from /nulaw endpoint
    nulaw_response = post_to_nulaw(args.matter_id, args.sf_path, args.download_files)
    if not nulaw_response:
        logger.error("[salesforce_create_new_client] Failed to get matter context")
        sys.exit(1)
    
    # Step 2 & 3: Format payload and create project
    result = process_nulaw_response_and_create_project(nulaw_response, args.dry_run, args.download_files)
    
    if result:
        # Print the result for inspection
        print(json.dumps(result, indent=2))
        if args.dry_run:
            logger.info("[salesforce_create_new_client] Dry run completed successfully")
        else:
            logger.info(f"[salesforce_create_new_client] Project created successfully (download_files={args.download_files})")
    else:
        logger.error("[salesforce_create_new_client] Failed to process nulaw response and create project")
        sys.exit(1)
    
if __name__ == "__main__":
    main()
