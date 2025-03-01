#!/usr/bin/env python3

import os
import requests
import json
from dotenv import load_dotenv, dotenv_values
import datetime
import argparse
import sys
from salesforce_refresh_token import get_salesforce_headers, refresh_salesforce_token, salesforce_request, verify_credentials, get_cli_credentials
import time
from share_point_get_documents import get_sharepoint_documents_for_matter

# Set the Matter ID
MATTER_ID = "a0OUR000004DwOr2AK"

# Load environment variables from .env file
load_dotenv()

# Initialize headers and instance URL
headers = None
SALESFORCE_INSTANCE_URL = None

# Note: Removed _insurance_metadata_cache global variable as it's no longer needed

def set_salesforce_auth_globals(auth_headers, auth_instance_url):
    """
    Set global authentication headers and instance URL directly.
    This allows bypassing CLI authentication if we already have valid credentials.
    
    Args:
        auth_headers (dict): Pre-authenticated Salesforce API headers with valid token
        auth_instance_url (str): Salesforce instance URL
        
    Returns:
        bool: True if successful, False otherwise
    """
    global headers, SALESFORCE_INSTANCE_URL
    
    if not auth_headers or not auth_instance_url:
        print("[get_all_context] Invalid auth headers or instance URL provided")
        return False
    
    # Set global variables
    headers = auth_headers
    SALESFORCE_INSTANCE_URL = auth_instance_url
    
    # Verify the provided credentials by making a simple API call
    test_url = f"{SALESFORCE_INSTANCE_URL}/services/data/v63.0/sobjects/"
    try:
        response = requests.get(test_url, headers=headers)
        if response.status_code == 200:
            print(f"[get_all_context] ✅ Successfully connected to Salesforce with provided credentials: {SALESFORCE_INSTANCE_URL}")
            return True
        else:
            print(f"[get_all_context] ❌ Provided credentials failed verification. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"[get_all_context] ❌ Error verifying provided credentials: {str(e)}")
        return False

def initialize_salesforce():
    """
    Initialize the Salesforce connection using stored credentials.
    
    Returns:
        bool: True if initialized successfully, False otherwise
    """
    global headers, SALESFORCE_INSTANCE_URL
    
    try:
        # If we already have headers and URL, we're already initialized
        if headers and SALESFORCE_INSTANCE_URL:
            print("[get_all_context] Already initialized with Salesforce credentials")
            return True
        
        # Get access token from environment or .env file
        load_dotenv()
        
        access_token = os.environ.get("SALESFORCE_ACCESS_TOKEN")
        instance_url = os.environ.get("SALESFORCE_INSTANCE_URL")
        
        # If not found in environment, try to load from .env file directly
        if not access_token or not instance_url:
            print("[get_all_context] Salesforce credentials not found in environment, checking .env file directly")
            try:
                env_vars = dotenv_values()
                if "SALESFORCE_ACCESS_TOKEN" in env_vars and "SALESFORCE_INSTANCE_URL" in env_vars:
                    access_token = env_vars["SALESFORCE_ACCESS_TOKEN"]
                    instance_url = env_vars["SALESFORCE_INSTANCE_URL"]
                    print("[get_all_context] Found Salesforce credentials in .env file")
            except Exception as e:
                print(f"[get_all_context] Error reading .env file: {e}")
        
        # If we still don't have credentials, try to get them from the CLI
        if not access_token or not instance_url:
            print("[get_all_context] Salesforce credentials not found, attempting to get from CLI")
            try:
                # Use the refresh script to get credentials from CLI
                token, url = get_cli_credentials()
                
                if token and url:
                    access_token = token
                    instance_url = url
                    print(f"[get_all_context] Successfully retrieved token from CLI: {token[:10]}...")
            except Exception as e:
                print(f"[get_all_context] Error getting credentials from CLI: {e}")
        
        # If we still don't have credentials, try to refresh the token
        if not access_token or not instance_url:
            print("[get_all_context] No Salesforce credentials found, attempting to refresh token")
            try:
                # Use the refresh script to refresh the token
                token, url = refresh_salesforce_token()
                
                if token and url:
                    access_token = token
                    instance_url = url
                    print(f"[get_all_context] Successfully refreshed token: {token[:10]}...")
                    
                    # Update environment for future calls
                    os.environ["SALESFORCE_ACCESS_TOKEN"] = token
                    os.environ["SALESFORCE_INSTANCE_URL"] = url
                    
                    # Update .env file
                    from salesforce_refresh_token import update_env_file
                    update_env_file("SALESFORCE_ACCESS_TOKEN", token)
                    update_env_file("SALESFORCE_INSTANCE_URL", url)
            except Exception as e:
                print(f"[get_all_context] Error refreshing token: {e}")
        
        # Validate that we have the required credentials
        if not access_token:
            print("[get_all_context] No Salesforce access token found")
            return False
        
        if not instance_url:
            print("[get_all_context] No Salesforce instance URL found")
            return False
        
        # Initialize global variables
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        SALESFORCE_INSTANCE_URL = instance_url.rstrip("/")  # Ensure no trailing slash
        
        # Test a simple API call to verify the token
        try:
            versions_url = f"{SALESFORCE_INSTANCE_URL}/services/data/"
            response = requests.get(versions_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                print("[get_all_context] Successfully verified Salesforce connection")
                
                # Store credentials for other functions
                set_salesforce_auth_globals(headers, SALESFORCE_INSTANCE_URL)
                
                return True
            elif response.status_code == 401:
                print("[get_all_context] Invalid Salesforce token (401 Unauthorized)")
                
                # Clear invalid token from environment
                if 'SALESFORCE_ACCESS_TOKEN' in os.environ:
                    os.environ.pop('SALESFORCE_ACCESS_TOKEN')
                    print("[get_all_context] Cleared invalid token from environment")
                
                # Try refreshing the token once more
                try:
                    from salesforce_refresh_token import refresh_salesforce_token
                    token, url = refresh_salesforce_token()
                    
                    if token and url:
                        # Update headers with new token
                        headers = {
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json"
                        }
                        SALESFORCE_INSTANCE_URL = url.rstrip("/")
                        
                        # Test the new token
                        versions_url = f"{SALESFORCE_INSTANCE_URL}/services/data/"
                        response = requests.get(versions_url, headers=headers, timeout=30)
                        
                        if response.status_code == 200:
                            print("[get_all_context] Successfully verified Salesforce connection after token refresh")
                            
                            # Store credentials for other functions
                            set_salesforce_auth_globals(headers, SALESFORCE_INSTANCE_URL)
                            
                            # Update environment
                            os.environ["SALESFORCE_ACCESS_TOKEN"] = token
                            os.environ["SALESFORCE_INSTANCE_URL"] = url
                            
                            # Update .env file
                            from salesforce_refresh_token import update_env_file
                            update_env_file("SALESFORCE_ACCESS_TOKEN", token)
                            update_env_file("SALESFORCE_INSTANCE_URL", url)
                            
                            return True
                        else:
                            print(f"[get_all_context] Verification failed after token refresh with status: {response.status_code}")
                            return False
                    else:
                        print("[get_all_context] Failed to refresh token")
                        return False
                except Exception as e:
                    print(f"[get_all_context] Error during token refresh: {e}")
                    return False
            else:
                print(f"[get_all_context] Salesforce verification failed with status: {response.status_code}")
                return False
        except Exception as e:
            print(f"[get_all_context] Error verifying Salesforce connection: {e}")
            return False
    except Exception as e:
        print(f"[get_all_context] Error initializing Salesforce: {e}")
        return False

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
                print("[get_all_context] Received 401 error, refreshing token...")
                
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
                print(f"[get_all_context] Request error after {max_retries + 1} attempts: {e}")
                raise
            
            # For connection errors, wait a moment before retrying
            if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                print(f"[get_all_context] Connection error: {e}. Retrying...")
                time.sleep(1)  # Wait 1 second before retry
            else:
                # For other errors, just raise
                raise

def sf_query(query):
    """
    Execute a SOQL query against Salesforce with automatic authentication handling.
    
    Args:
        query (str): The SOQL query to execute
        
    Returns:
        dict: The JSON response from Salesforce
    """
    global headers, SALESFORCE_INSTANCE_URL
    
    # Make sure we're initialized
    if not headers or not SALESFORCE_INSTANCE_URL:
        if not initialize_salesforce():
            raise Exception("Failed to initialize Salesforce connection")
    
    try:
        # Batch multiple queries together if they're similar to reduce API calls
        result = salesforce_request(
            'get', 
            '/services/data/v63.0/query', 
            params={'q': query}
        )
        
        if result:
            return result
        else:
            raise Exception("Failed to execute query")
    except Exception as e:
        print(f"[get_all_context] Query error: {e}")
        raise

# Add batch query functionality to reduce API calls for similar queries
def sf_batch_query(queries):
    """
    Execute multiple SOQL queries in a single composite request.
    
    Args:
        queries (list): List of SOQL queries to execute
        
    Returns:
        list: List of JSON responses from Salesforce
    """
    if not queries:
        return []
    
    # If there's only one query, use the regular sf_query function
    if len(queries) == 1:
        return [sf_query(queries[0])]
    
    print(f"[get_all_context] Executing batch query with {len(queries)} queries")
    
    global headers, SALESFORCE_INSTANCE_URL
    
    # Make sure we're initialized
    if not headers or not SALESFORCE_INSTANCE_URL:
        if not initialize_salesforce():
            raise Exception("Failed to initialize Salesforce connection")
    
    # Limit batch size to 25 queries at a time (Salesforce composite API limit)
    MAX_BATCH_SIZE = 25
    all_results = []
    
    # Process queries in batches of MAX_BATCH_SIZE
    for i in range(0, len(queries), MAX_BATCH_SIZE):
        batch_queries = queries[i:i+MAX_BATCH_SIZE]
        print(f"[get_all_context] Processing batch {i//MAX_BATCH_SIZE + 1} of {(len(queries) + MAX_BATCH_SIZE - 1)//MAX_BATCH_SIZE} ({len(batch_queries)} queries)")
        
        try:
            # Create a composite request with all queries in this batch
            composite_request = {
                "compositeRequest": [
                    {
                        "method": "GET",
                        "url": f"/services/data/v63.0/query?q={requests.utils.quote(query)}",
                        "referenceId": f"query{j}"
                    }
                    for j, query in enumerate(batch_queries)
                ]
            }
            
            # Execute the composite request
            result = salesforce_request(
                'post', 
                '/services/data/v63.0/composite', 
                data=composite_request
            )
            
            if result and "compositeResponse" in result:
                # Extract the individual query results
                batch_results = []
                
                for response in result["compositeResponse"]:
                    if response.get("httpStatusCode") == 200:
                        batch_results.append(response.get("body"))
                    else:
                        print(f"[get_all_context] Error in batch query: {response.get('body', {}).get('message', 'Unknown error')}")
                        # Add None for failed queries to maintain order
                        batch_results.append(None)
                
                all_results.extend(batch_results)
            else:
                raise Exception("Failed to execute batch query")
                
        except Exception as e:
            print(f"[get_all_context] Batch query error: {e}")
            # Fall back to individual queries for this batch
            print(f"[get_all_context] Falling back to individual queries for batch {i//MAX_BATCH_SIZE + 1}")
            for query in batch_queries:
                try:
                    all_results.append(sf_query(query))
                except Exception as inner_e:
                    print(f"[get_all_context] Individual query error: {inner_e}")
                    all_results.append(None)
    
    return all_results

def get_matter_details(matter_id):
    """
    Get detailed information about a Matter record.
    
    Args:
        matter_id (str): The Salesforce Matter ID
        
    Returns:
        dict: Matter record details
    """
    print(f"[get_all_context] Retrieving Matter details for ID: {matter_id}")
    
    # Define field categories for better organization
    field_categories = {
        "basic": [
            "Id", 
            "Name", 
            "nu_law__Email_Identification_Number__c",
            "Client_Full_Name__c",
            "Birthdate_Copy__c",
            "Client_Age_Copy__c", 
            "Language_Preferred__c"
        ],
        "case_details": [ 
            "nu_law__Open_Date__c",
            "nu_law__Status__c", 
            "Case_Status__c",
            "Case_Phase__c",
            "Case_Age__c",
            "SOL__c",
            "Presuit_vs_Litigation__c",
            "Phase_Treatment__c"
        ],
        "incident_details": [
            "Accident_Date_Time__c",
            "Accident_Description__c",
            "City__c",
            "County__c",
            "State__c",
            "Driver_Passenger_or_Pedestrian__c",
            "Client_Placement_in_Vehicle__c",
            "Police_responded__c",
            "Client_Facts__c"
        ],
        "medical_treatment": [
            "nu_law__Case_Type__c",
            "nu_law__Sub_Case_Type__c",
            "Treatment_Level__c",
            "MRI_Findings__c",
            "TBI_Initial_Test__c",
            "TBI_Final_Test__c",
            "Ortho_Eval_Date_First_of__c",
            "MRI_Date_First_of__c",
            "ER__c",
            "Ambulance__c",
            "TBI_Status__c",
            "Latest_Special_Treatment_Note__c",
            "Treatments_Total_Billed_Amount__c"
        ],
        "treatment_decisions": [
            "Ortho_Evall_Referral_Decision__c",
            "Ortho_Eval_Referral_Decision_Date__c",
            "RFA_Decision__c",
            "RFA_Decision_Date__c",
            "RFA_Decline_Reason__c",
            "Injection_Decision__c",
            "Injection_Decision_Date__c",
            "Injection_Decline_Reason__c",
            "Surgery_Decision__c",
            "Surgery_Decision_Date__c",
            "Surgery_Decline_Reason__c",
            "TBI_Initial_Test_Referral_Decision__c",
            "TBI_Initial_Test_Referral_Decision_Date__c"
        ],
        "insurance_financial": [
            "Limits_BI1__c", 
            "Limits_BI2__c", 
            "Limits_Total__c",
            "Email_BI1_Insurance_Company__c",
            "Email_BI2_Insurance_Company__c",
            "Commercial_Policy__c",
            "Health_Insurance_Medicaid_Medicare__c",
            "Treatment_Limits__c",
            "Dec_Page_BI1__c"
        ],
        "kpis": [
            "Case_Value__c", 
            "Case_Valuation_Low__c",
            "Case_Valuation_High__c",
            "Projected_Value_Actual__c",
            "Total_Injury_Costs__c"
        ],
        "client_information": [
            "Client_Update_Schedule__c",
            "Welcome_Client_Call_Date__c",
            "Orthopedic_Update_Call_Date__c",
            "Last_Client_Contact_Date__c",
            "Last_Client_Contact_Outcome__c",
            "Client_Last_Connected__c",
            "Review_No_Social_Media_Policy__c"
        ],
        "case_progress_kpis": [
            "Case_Summary_File_Review_KPI__c",
            "Client_Care_File_Review_KPI__c",
            "RFA_KPI__c",
            "TBI_Initial_Test_KPI__c",
            "TBI_Final_Test_KPI__c",
            "Orthopedic_Evaluation_KPI__c",
            "Crash_Report_Status_KPI__c"
        ],
        "pre_existing_conditions": [
            "Prior_Injuries__c",
            "Prior_PI_Case__c",
            "Prior_Accident__c",
            "Prior_Attorney__c",
            "Bankruptcy__c",
            "Child_Support__c",
            "Injured_during_Work__c",
            "Aggravating_Factors__c"
        ],
        "file_management": [
            "Sharepoint_Folder__c",
            "Case_Summary_Link__c"
        ]
    }
    
    # First, get basic fields to ensure the record exists
    basic_fields = ", ".join(field_categories["basic"])
    query = f"SELECT {basic_fields} FROM nu_law__Matter__c WHERE Id = '{matter_id}'"
    
    try:
        query_result = sf_query(query)
        
        if query_result["totalSize"] == 0:
            print(f"[get_all_context] Matter record not found: {matter_id}")
            return None
        
        matter_record = query_result["records"][0]
        print(f"[get_all_context] Found Matter: {matter_record['Name']}")
        
        # Create a dictionary to store fields by category
        categorized_fields = {category: {} for category in field_categories.keys()}
        categorized_fields["basic"] = {field: matter_record.get(field) for field in field_categories["basic"] if field in matter_record}
        
        # Now query each category of fields
        for category, fields in field_categories.items():
            if category == "basic":
                continue  # Already queried
                
            # Create a comma-separated list of fields
            field_list = ", ".join(fields)
            
            try:
                category_query = f"SELECT {field_list} FROM nu_law__Matter__c WHERE Id = '{matter_id}'"
                category_result = sf_query(category_query)
                
                if category_result["totalSize"] > 0:
                    category_record = category_result["records"][0]
                    
                    # Add fields to the matter record and categorized fields
                    for field in fields:
                        if field in category_record:
                            matter_record[field] = category_record[field]
                            categorized_fields[category][field] = category_record[field]
                    
                    print(f"[get_all_context] Successfully retrieved {category} fields")
            except Exception as e:
                print(f"[get_all_context] Error retrieving {category} fields: {e}")
                
                # Try individual fields if the category query fails
                for field in fields:
                    try:
                        field_query = f"SELECT {field} FROM nu_law__Matter__c WHERE Id = '{matter_id}'"
                        field_result = sf_query(field_query)
                        
                        if field_result["totalSize"] > 0:
                            field_value = field_result["records"][0][field]
                            matter_record[field] = field_value
                            categorized_fields[category][field] = field_value
                            print(f"[get_all_context] Successfully retrieved field: {field}")
                    except Exception as field_error:
                        print(f"[get_all_context] Error retrieving field {field}: {field_error}")
        
        # Add the categorized fields to the matter record
        matter_record["categorized_fields"] = categorized_fields
        
        return matter_record
        
    except Exception as e:
        print(f"[get_all_context] Error retrieving Matter details: {e}")
        return {"error": str(e)}

def get_client_details(client_id):
    """
    Get details of a Client (Account) record.
    
    Args:
        client_id (str): The Salesforce Account ID
        
    Returns:
        dict: Account record details
    """
    if not client_id:
        return None
        
    print(f"[get_all_context] Retrieving Client details for ID: {client_id}")
    
    # Query the Account record with key fields
    query = f"SELECT Id, Name, PersonEmail, PersonMobilePhone, nu_law__Folder_Path__c, nu_law__Folder_Id__c, nu_law__Folder_Name__c FROM Account WHERE Id = '{client_id}'"
    
    try:
        query_result = sf_query(query)
        
        if query_result["totalSize"] == 0:
            print(f"[get_all_context] Client record not found: {client_id}")
            return None
        
        client_record = query_result["records"][0]
        print(f"[get_all_context] Found Client: {client_record['Name']}")
        
        return client_record
    
    except Exception as e:
        print(f"[get_all_context] Error retrieving Client details: {e}")
        return {"error": str(e)}

def get_related_records(matter_id):
    """
    Retrieve related records (Tasks, Events, Notes) for a Matter.
    
    Args:
        matter_id (str): The Salesforce Matter ID
        
    Returns:
        dict: Dictionary of related records
    """
    print(f"[get_all_context] Retrieving related records for Matter ID: {matter_id}")
    
    related_records = {
        "tasks": [],
        "events": [],
        "notes": []
    }
    
    # Query Tasks related to the Matter
    task_query = f"SELECT Id, Subject, Status, ActivityDate, Description, CreatedDate FROM Task WHERE WhatId = '{matter_id}' ORDER BY CreatedDate DESC LIMIT 50"
    
    try:
        task_data = sf_query(task_query)
        
        if task_data["totalSize"] > 0:
            related_records["tasks"] = task_data["records"]
            print(f"[get_all_context] Found {task_data['totalSize']} related Tasks")
    
    except Exception as e:
        print(f"[get_all_context] Error retrieving related Tasks: {e}")
    
    # Query Events related to the Matter
    event_query = f"SELECT Id, Subject, StartDateTime, EndDateTime, Description, CreatedDate FROM Event WHERE WhatId = '{matter_id}' ORDER BY CreatedDate DESC LIMIT 50"
    
    try:
        event_data = sf_query(event_query)
        
        if event_data["totalSize"] > 0:
            related_records["events"] = event_data["records"]
            print(f"[get_all_context] Found {event_data['totalSize']} related Events")
    
    except Exception as e:
        print(f"[get_all_context] Error retrieving related Events: {e}")
    
    # Query Notes related to the Matter
    note_query = f"SELECT Id, Title, Body, CreatedDate FROM Note WHERE ParentId = '{matter_id}' ORDER BY CreatedDate DESC LIMIT 50"
    
    try:
        note_data = sf_query(note_query)
        
        if note_data["totalSize"] > 0:
            related_records["notes"] = note_data["records"]
            print(f"[get_all_context] Found {note_data['totalSize']} related Notes")
    
    except Exception as e:
        print(f"[get_all_context] Error retrieving related Notes: {e}")
    
    return related_records

def get_treatment_details(url):
    """
    Fetch additional treatment details from a Salesforce URL.
    
    Args:
        url (str): The Salesforce API URL to fetch details from
        
    Returns:
        dict: The treatment details or None if request failed
    """
    global headers, SALESFORCE_INSTANCE_URL
    
    print(f"[get_all_context] Fetching treatment details from URL: {url}")
    
    # Make sure we're initialized
    if not headers or not SALESFORCE_INSTANCE_URL:
        if not initialize_salesforce():
            print("[get_all_context] Failed to initialize Salesforce connection")
            return None
    
    # Extract the relative URL if it's a full URL
    if url.startswith('http'):
        # Extract the path part of the URL
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        url = parsed_url.path
    
    # Remove leading slash if present for consistency with salesforce_request
    if url.startswith('/'):
        url = url[1:]
    
    try:
        # Use salesforce_request to fetch the details
        details = salesforce_request('get', f'/{url}')
        
        if details:
            print(f"[get_all_context] Successfully retrieved treatment details")
            return details
        else:
            print(f"[get_all_context] Failed to retrieve treatment details from {url}")
            return None
    
    except Exception as e:
        print(f"[get_all_context] Error fetching treatment details: {e}")
        return None

def get_treatment_records(matter_id):
    """
    Get comprehensive treatment records and related information for a Matter.
    
    Args:
        matter_id (str): The Salesforce Matter ID
        
    Returns:
        dict: Dictionary containing all treatment-related information
    """
    print(f"[get_all_context] Retrieving comprehensive treatment records for Matter ID: {matter_id}")
    
    treatment_data = {
        "matter_treatment_info": {},
        "treatments": [],
        "treatment_objects": {},
        "providers": set(),  # Using a set to avoid duplicate providers
        "treatment_amounts": {}
    }
    
    # 1. Get Matter-level treatment fields individually to handle errors
    matter_treatment_fields = [
        # Overall treatment status and notes
        "Treatment_Level__c",
        "Latest_Special_Treatment_Note__c",
        "Treatment_Limits__c",
        "Treatment_Done_Interval__c",
        "Treatment_Plan_Records_Audit__c",
        "Treatments_Total_Billed_Amount__c",
        "Latest_Treatment_Note_Date__c",
        "Client_Care_Treatment_Level__c",
        
        # TBI treatment related fields
        "TBI_Treatment_Plan_Complete__c",
        "TBI_Status__c",
        "TBI_Initial_Test_Referral_Decision__c",
        "TBI_Initial_Test_Decline_Reason__c",
        "TBI_Initial_Test__c",
        "TBI_Final_Test__c", 
        "TBI_Final_Test_Decline_Reason__c",
        "Do_you_have_any_TBI_symptoms__c",
        
        # Surgery related fields
        "Surgery__c",
        "Surgery_Date__c",
        "Surgery_Decision__c", 
        "Surgery_Decline_Reason__c",
        "Surgery_Date_First_of__c",
        "Surgery_Scheduled_Date_Time__c",
        
        # Injection related fields
        "Injections__c",
        "Injection_Decision__c",
        "Injection_Decline_Reason__c",
        "Injection_Decision_Date__c",
        "Injection_Date_First_of__c",
        "Injection_Scheduled_Date_Tim__c",
        
        # RFA (Radio Frequency Ablation) related fields
        "RFA__c",
        "RFA_Decision__c",
        "RFA_Decision_Date__c",
        "RFA_Decline_Reason__c",
        "RFA_Scheduled_Date_Time__c",
        
        # Diagnostic testing and evaluation fields
        "MRI_Findings__c",
        "MRI_Date_First_of__c",
        "MRI_Scheduled_Date_Time__c",
        "Orthopedic_Evaluation_Date__c",
        "Ortho_Eval_Date_First_of__c",
        "Orthopedic_Scheduled_Date_Tim__c",
        "Ortho_Evall_Referral_Decision__c",
        "Ortho_Eval_Referral_Decision_Date__c",
        "Ortho_Eval_Referral_Decline_Reason__c",
        
        # Initial medical care fields
        "ER__c",
        "Ambulance__c",
        "Chiropractic_Date__c",
        
        # Medical management and insurance fields
        "Medical_Management_Status__c",
        "Medical_Management_Notes__c",
        "Health_Insurance_Medicaid_Medicare__c"
    ] 
    print("[get_all_context] Retrieving Matter-level treatment fields...")
    for field in matter_treatment_fields:
        try:
            query = f"SELECT {field} FROM nu_law__Matter__c WHERE Id = '{matter_id}'"
            result = sf_query(query)
            
            if result["totalSize"] > 0 and field in result["records"][0]:
                treatment_data["matter_treatment_info"][field] = result["records"][0][field]
        except Exception as e:
            print(f"[get_all_context] Field {field} not available: {e}")
    
    # 2. Discover treatment-related objects that exist in the Salesforce instance
    treatment_objects = discover_treatment_objects()
    
    # 3. For each discovered object, get fields and query records
    for obj_name, obj_info in treatment_objects.items():
        print(f"[get_all_context] Processing object: {obj_name}")
        
        # Get the available fields for this object
        fields = get_object_fields(obj_name)
        
        if not fields:
            continue
        
        # Ensure we have Id and Name fields at a minimum
        if "Id" not in fields or "Name" not in fields:
            print(f"[get_all_context] Object {obj_name} missing Id or Name fields")
            continue
        
        # Start with basic fields
        query_fields = ["Id", "Name"]
        
        # Look for Matter reference field - try common patterns
        matter_field = None
        potential_matter_fields = [
            "nu_law__Matter__c", 
            "Matter__c",
            "nu_law__Matter_Link__c"
        ]
        
        # Add any fields with "matter" in their name
        matter_fields = [field for field in fields if "matter" in field.lower()]
        potential_matter_fields.extend(matter_fields)
        
        for field in potential_matter_fields:
            if field in fields:
                matter_field = field
                break
        
        if not matter_field:
            print(f"[get_all_context] Cannot find Matter reference field for {obj_name}")
            continue
        
        # Add treatment-related fields if they exist
        treatment_related_field_keywords = [
            "treatment", "provider", "date", "type", "notes", "cost", "status", 
            "completed", "visit", "manager"
        ]
        
        for field in fields:
            if (any(keyword in field.lower() for keyword in treatment_related_field_keywords)
                and field not in query_fields):
                query_fields.append(field)
        
        # Prepare and execute the query
        fields_str = ", ".join(query_fields)
        query = f"SELECT {fields_str} FROM {obj_name} WHERE {matter_field} = '{matter_id}'"
        
        try:
            print(f"[get_all_context] Executing query for {obj_name}...")
            result = sf_query(query)
            
            if result["totalSize"] > 0:
                print(f"[get_all_context] Found {result['totalSize']} {obj_name} records")
                treatment_data["treatment_objects"][obj_name] = result["records"]
                
                # Add to treatments list for compatibility
                if "treatment" in obj_name.lower():
                    # Iterate through each treatment record
                    for treatment_record in result["records"]:
                        # Add the record to the treatments list
                        treatment_data["treatments"].append(treatment_record)
                        
                        # Check if the record has attributes with a URL
                        if "attributes" in treatment_record and "url" in treatment_record["attributes"]:
                            # Fetch additional details from the URL
                            url = treatment_record["attributes"]["url"]
                            details = get_treatment_details(url)
                            
                            # If details were fetched successfully, add them to the treatment record
                            if details:
                                treatment_record["detailed_info"] = details
                
                # Look for provider information
                for record in result["records"]:
                    for field, value in record.items():
                        if field.lower().endswith("provider__c") and value:
                            provider_name_field = field[:-3] + "r.Name"
                            if provider_name_field in record:
                                treatment_data["providers"].add((
                                    value,
                                    record[provider_name_field]["Name"]
                                ))
            else:
                print(f"[get_all_context] No {obj_name} records found")
        
        except Exception as e:
            print(f"[get_all_context] Error querying {obj_name}: {e}")
            # Log the query for debugging purposes
            print(f"[get_all_context] Failed query: {query}")
            # Continue with the next object instead of failing completely
            continue
    
    # 4. Get treatment amounts individually to handle errors
    amount_fields = [
        "Nu_OM__All_Treatments_Amounts__c",
        "Nu_OM__Treatments_Total_Amount__c",
        "Treatments_Total_Billed_Amount__c"
    ]
    
    print("[get_all_context] Retrieving treatment amounts...")
    for field in amount_fields:
        try:
            query = f"SELECT {field} FROM nu_law__Matter__c WHERE Id = '{matter_id}'"
            result = sf_query(query)
            
            if result["totalSize"] > 0 and field in result["records"][0]:
                treatment_data["treatment_amounts"][field] = result["records"][0][field]
        except Exception as e:
            print(f"[get_all_context] Treatment amount field {field} not available: {e}")
    
    # Convert providers set to list for JSON serialization
    treatment_data["providers"] = [
        {"id": provider[0], "name": provider[1]}
        for provider in treatment_data["providers"]
    ]
    
    return treatment_data

def discover_treatment_objects():
    """
    Identify treatment-related objects in the Salesforce instance.
    
    Returns:
        dict: Dictionary of existing object names
    """
    print("[get_all_context] Discovering treatment-related objects...")
    
    # We'll check these potential treatment-related objects
    potential_objects = [
        "nu_law__Treatment_Record_Visit__c",
        "Nu_OM__Treatment__c",
        "Nu_OM__Treatment_Visit__c",
        "Injury_Treatment__c",
        "Treatment__c",
        "Medical_Treatment__c",
        "Treatment_Plan__c",
        "Treatment_Record__c",
        "TreatmentItem__c"
    ]
    
    # Objects to exclude - these special objects cannot be queried with standard SOQL
    excluded_suffixes = ["__ChangeEvent", "__History", "__Feed", "__Share"]
    
    discovered_objects = {}
    
    # Check global objects describe
    try:
        url = f"{SALESFORCE_INSTANCE_URL}/services/data/v63.0/sobjects/"
        response = execute_salesforce_request(requests.get, url, headers=headers)
        
        # Get all object names from the response
        all_object_names = [obj["name"] for obj in response.json()["sobjects"]]
        
        # Filter for treatment-related objects, excluding special objects
        for obj_name in all_object_names:
            # Skip objects with excluded suffixes
            if any(obj_name.endswith(suffix) for suffix in excluded_suffixes):
                continue
                
            if "treatment" in obj_name.lower() or any(pot_obj.lower() == obj_name.lower() for pot_obj in potential_objects):
                print(f"[get_all_context] Found treatment-related object: {obj_name}")
                discovered_objects[obj_name] = {"name": obj_name}
    
    except requests.exceptions.RequestException as e:
        print(f"[get_all_context] Error retrieving Salesforce object list: {e}")
        
        # Fallback to individual object checks
        for obj_name in potential_objects:
            try:
                # Skip objects with excluded suffixes
                if any(obj_name.endswith(suffix) for suffix in excluded_suffixes):
                    continue
                    
                print(f"[get_all_context] Checking if {obj_name} exists...")
                obj_url = f"{SALESFORCE_INSTANCE_URL}/services/data/v63.0/sobjects/{obj_name}/describe"
                obj_response = execute_salesforce_request(requests.get, obj_url, headers=headers)
                
                # If we got here, the object exists
                print(f"[get_all_context] Found treatment object: {obj_name}")
                discovered_objects[obj_name] = {"name": obj_name}
            except requests.exceptions.RequestException:
                # Object doesn't exist or can't be queried
                print(f"[get_all_context] Object {obj_name} not found")
                pass
    
    return discovered_objects

def get_object_fields(object_name):
    """
    Get available fields for a Salesforce object.
    
    Args:
        object_name (str): The API name of the Salesforce object
        
    Returns:
        list: List of field names
    """
    print(f"[get_all_context] Getting fields for {object_name}...")
    
    # Skip special objects that cannot be queried with standard SOQL
    excluded_suffixes = ["__ChangeEvent", "__History", "__Feed", "__Share"]
    if any(object_name.endswith(suffix) for suffix in excluded_suffixes):
        print(f"[get_all_context] Skipping special object type: {object_name}")
        return []
        
    try:
        url = f"{SALESFORCE_INSTANCE_URL}/services/data/v63.0/sobjects/{object_name}/describe"
        response = execute_salesforce_request(requests.get, url, headers=headers)
        
        # Extract field names from the response
        fields = [field["name"] for field in response.json()["fields"]]
        print(f"[get_all_context] Found {len(fields)} fields for {object_name}")
        return fields
    
    except requests.exceptions.RequestException as e:
        print(f"[get_all_context] Error retrieving fields for {object_name}: {e}")
        return []

def get_insurance_information(matter_id):
    """
    Get insurance information related to a Matter.
    
    Args:
        matter_id (str): The Salesforce Matter ID
        
    Returns:
        dict: Dictionary containing insurance records
    """
    print(f"[get_all_context] Retrieving insurance information for Matter ID: {matter_id}")
    
    insurance_data = {
        "records": []
    }
    
    # Query insurance records using nu_law__Matter_Link__c field
    try:
        print(f"[get_all_context] Querying insurance records using nu_law__Matter_Link__c...")
        
        # Use a more focused field list
        insurance_fields = [
            "Id", "Name", "CreatedDate", "LastModifiedDate", 
            "nu_law__Insurance_Carrier__c", "nu_law__Insurance_External_ID__c", 
            "nu_law__Limits_Maximum__c", "nu_law__Limits_Minimum__c", 
            "nu_law__Matter_Link__c", "nu_law__Monthly_Premium__c",
            "nu_law__Policy_Begin_Date__c", "nu_law__Policy_End_Date__c", 
            "nu_law__Policy_Holder__c", "nu_law__Policy_Number__c",
            "nu_law__Type_of_Insurance__c", "Adjuster_Notes__c", 
            "Claim_Number__c", "Insured_Name__c", "Policy_Limit__c",
            "MedPay__c", "Adjuster_Name__c", "Adjuster_Phone__c", "Adjuster_Email__c"
        ]
        
        # Split into chunks of 20 fields to avoid exceeding SOQL query length limits
        field_chunks = [insurance_fields[i:i+20] for i in range(0, len(insurance_fields), 20)]
        
        # Build queries for each chunk
        chunk_queries = []
        for chunk in field_chunks:
            query = f"SELECT {', '.join(chunk)} FROM nu_law__Insurance__c WHERE nu_law__Matter_Link__c = '{matter_id}'"
            chunk_queries.append(query)
            
        print(f"[get_all_context] Executing {len(chunk_queries)} chunked queries for insurance")
        
        # Execute all queries in a batch if possible
        results = sf_batch_query(chunk_queries)
        
        # Process results and merge fields
        insurance_records = []
        
        for i, result in enumerate(results):
            if result and result.get("records"):
                # For the first chunk with results, store all records
                if not insurance_records:
                    insurance_records = result.get("records", [])
                    print(f"[get_all_context] Found {len(insurance_records)} insurance records in chunk {i+1}")
                # For subsequent chunks, merge the fields into existing records
                else:
                    print(f"[get_all_context] Merging fields from chunk {i+1}")
                    for j, record in enumerate(result.get("records", [])):
                        if j < len(insurance_records):
                            # Remove attributes to avoid duplicates
                            if "attributes" in record:
                                del record["attributes"]
                            # Update the existing record with new fields
                            insurance_records[j].update(record)
        
        if insurance_records:
            print(f"[get_all_context] Successfully retrieved {len(insurance_records)} complete insurance records")
            insurance_data["records"] = insurance_records
        else:
            print(f"[get_all_context] No insurance records found for matter ID: {matter_id}")
    
    except Exception as e:
        print(f"[get_all_context] Error retrieving insurance records: {e}")
        # Try a simplified approach with fewer fields as fallback
        try:
            print("[get_all_context] Attempting fallback query with fewer fields...")
            minimal_fields = ["Id", "Name", "nu_law__Insurance_Carrier__c", "nu_law__Policy_Number__c", "nu_law__Type_of_Insurance__c"]
            query = f"SELECT {', '.join(minimal_fields)} FROM nu_law__Insurance__c WHERE nu_law__Matter_Link__c = '{matter_id}'"
            result = sf_query(query)
            
            if result and result.get("records"):
                insurance_data["records"] = result.get("records")
                print(f"[get_all_context] Fallback successful - found {len(result.get('records'))} insurance records")
            else:
                print("[get_all_context] Fallback query returned no results")
        except Exception as fallback_e:
            print(f"[get_all_context] Fallback query failed: {fallback_e}")
    
    return insurance_data

def minify_context(data):
    """
    Remove null values from a nested dictionary or list.
    
    This function removes:
    - Python None values
    - String "null" values
    
    Args:
        data: Dictionary or list to minify
        
    Returns:
        Minified data structure with null values removed
    """
    if isinstance(data, dict):
        return {
            key: minify_context(value) 
            for key, value in data.items() 
            if value is not None and value != "null"
        }
    elif isinstance(data, list):
        return [minify_context(item) for item in data if item is not None and item != "null"]
    else:
        return data

def organize_matter_context(matter_id, download_files=False):
    """
    Collect and organize all context information for a Matter.
    
    Args:
        matter_id (str): The Salesforce Matter ID
        download_files (bool, optional): Whether to download files from SharePoint (defaults to False)
        
    Returns:
        dict: Organized Matter context
    """
    print(f"[get_all_context] Collecting context for Matter ID: {matter_id}")
    
    # Get Matter details
    matter = get_matter_details(matter_id)
    if not matter:
        return {"error": f"Matter not found: {matter_id}"}
    
    # Get Client details if available
    client = None
    if matter.get("nu_law__Client__c"):
        client = get_client_details(matter["nu_law__Client__c"])
    
    # Documents are now retrieved separately through the /nulawdocs/ endpoint
    documents = []
    
    # Get related records
    related_records = get_related_records(matter_id)
    
    # Get treatment records
    treatment_records = get_treatment_records(matter_id)
    
    # Get insurance information
    insurance_information = get_insurance_information(matter_id)
    
    # Organize SharePoint information
    sharepoint_info = {}
    if "categorized_fields" in matter and "sharepoint" in matter["categorized_fields"]:
        sharepoint_info = matter["categorized_fields"]["sharepoint"]
    
    # If we didn't find any SharePoint fields, add a note
    if not sharepoint_info:
        print("[get_all_context] No SharePoint folder information found for this Matter")
        sharepoint_info = {"note": "No SharePoint folder information found for this Matter"}
    
    # Create organized context
    context = {
        "matter": matter,
        "client": client,
        "sharepoint": sharepoint_info,
        "documents": documents,  # Empty array - documents now fetched separately
        "related_records": related_records,
        "treatment_records": treatment_records,
        "insurance_information": insurance_information,
        "collected_at": datetime.datetime.now().isoformat()
    }
    
    # Add SharePoint documents to the context
    try:
        print("[get_all_context] Retrieving SharePoint documents...")
        from share_point_get_documents import get_sharepoint_documents_for_matter, enable_debug_logging
        
        # Enable debug logging for detailed information
        enable_debug_logging()
        
        # Create a "sharepoint_files" directory in the same location as the output file
        sharepoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sharepoint_files")
        
        print(f"[get_all_context] Download files set to: {download_files}")
        
        # Try to get SharePoint documents with recursive folder traversal and download only if requested
        sharepoint_documents = get_sharepoint_documents_for_matter(
            context, 
            download_docs=download_files,  # Only download if requested
            target_dir=sharepoint_dir,  # Save to the sharepoint_files directory
            recursive=True  # Get documents from all subfolders
        )
        
        if sharepoint_documents:
            print(f"[get_all_context] Retrieved {len(sharepoint_documents)} SharePoint documents")
            if download_files:
                print(f"[get_all_context] Files saved to: {sharepoint_dir}")
            context["sharepoint_documents"] = sharepoint_documents
            if download_files:
                context["sharepoint_files_dir"] = sharepoint_dir
        else:
            print("[get_all_context] No SharePoint documents found or there was an error retrieving them")
            context["sharepoint_documents"] = []
    except Exception as e:
        print(f"[get_all_context] Error retrieving SharePoint documents: {str(e)}")
        context["sharepoint_documents"] = []
        context["sharepoint_error"] = str(e)
    
    return context

def print_context_summary(context):
    """
    Print a summary of the gathered context.
    
    Args:
        context (dict): The context dictionary
    """
    print("\n" + "="*80)
    print("MATTER CONTEXT SUMMARY")
    print("="*80)
    
    # Check if we have a valid matter
    if "matter" in context and "error" not in context["matter"]:
        matter = context["matter"]
        print(f"Matter: {matter.get('Name')} (ID: {matter.get('Id')})")
        
        # Handle potential None value for Matter Number
        print(f"Matter Number: {matter.get('API_ID__c') or matter.get('nu_law__Matter_Number__c') or 'Not available'}")
        
        if "client" in context and context["client"] is not None and "error" not in context.get("client", {}):
            client = context["client"]
            print(f"Client: {client.get('Name')} (ID: {client.get('Id')})")
        
        # For collections, check if they exist and are not None before getting length
        if "documents" in context and context["documents"] is not None:
            print(f"Documents: {len(context['documents'])}")
        else:
            print("Documents: None")
        
        if "related_records" in context and context["related_records"] is not None:
            related_records_count = 0
            for rec_type, records in context["related_records"].items():
                if records is not None:
                    related_records_count += len(records)
            print(f"Related Records: {related_records_count}")
        else:
            print("Related Records: None")
        
        # Use treatment_records instead of treatments
        if "treatment_records" in context and context["treatment_records"] is not None:
            if "treatments" in context["treatment_records"] and context["treatment_records"]["treatments"] is not None:
                print(f"Treatment Records: {len(context['treatment_records']['treatments'])}")
            else:
                print("Treatment Records: 0")
        else:
            print("Treatment Records: None")
        
        # Use insurance_information instead of insurance
        if "insurance_information" in context and context["insurance_information"] is not None:
            if "records" in context["insurance_information"] and context["insurance_information"]["records"] is not None:
                print(f"Insurance Records: {len(context['insurance_information']['records'])}")
            else:
                print("Insurance Records: 0")
        else:
            print("Insurance Records: None")
    else:
        error_msg = context.get("matter", {}).get("error", "Matter not found")
        print(f"Error: {error_msg}")
    
    print("="*80)

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Collect and organize context information for a Matter')
    parser.add_argument('--matter-id', default=MATTER_ID, help='Salesforce Matter ID (default: a0OUR000004DwOr2AK)')
    parser.add_argument('--output', default="matter_context.json", help='Output filename (default: matter_context.json)')
    parser.add_argument('--sf-path', help='Path to the Salesforce CLI executable (sf)')
    args = parser.parse_args()
    
    # If SF_CLI_PATH is provided as an argument, set it in the environment
    if args.sf_path:
        os.environ['SF_CLI_PATH'] = args.sf_path
        print(f"[get_all_context] Using Salesforce CLI path: {args.sf_path}")
    
    # Initialize Salesforce connection
    if not initialize_salesforce():
        print("[get_all_context] ❌ Unable to connect to Salesforce. Exiting.")
        sys.exit(1)
    
    # Use the provided Matter ID or the default
    matter_id = args.matter_id
    
    print(f"[get_all_context] Using Matter ID: {matter_id}")
    
    # Collect and organize context for the Matter
    try:
        context = organize_matter_context(matter_id)
        
        # Print a summary of the context
        print_context_summary(context)
        
        # Save the context to a file using json.dump
        print(f"[get_all_context] Saving context to file: {args.output}")
        
        # Save the full context
        with open(args.output, "w") as f:
            json.dump(context, f, indent=2)
        
        print(f"[get_all_context] Context saved to: {args.output}")
        
        # Save a minified version
        minified_filename = args.output.replace(".json", "_minified.json")
        minified_context = minify_context(context)
        
        with open(minified_filename, "w") as f:
            json.dump(minified_context, f, indent=2)
        
        print(f"[get_all_context] Minified context saved to: {minified_filename}")
        
        print("\n[get_all_context] Done!")
    except Exception as e:
        print(f"[get_all_context] Error: {e}")
        print("[get_all_context] Unable to complete the context gathering. Please check your Salesforce connection and try again.")
        sys.exit(1)
