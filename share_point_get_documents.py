#!/usr/bin/env python3
"""
SharePoint Document Retrieval Module
Handles retrieving documents from SharePoint folders using Microsoft Graph API
"""
import os
import requests
import json
import logging
import urllib.parse
import base64
from typing import Dict, List, Optional, Any, Tuple
from share_point_refresh_token import get_bearer_token, get_auth_headers

# Set up logging with proper namespace
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[share_point_get_documents] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

# Constants
SHAREPOINT_DOMAIN = "https://netorg430999.sharepoint.com/sites/MatterDocs/"
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"

def enable_debug_logging():
    """Enable debug level logging for this module"""
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled for SharePoint document retrieval")

def print_item_details(item, prefix=""):
    """
    Print detailed information about a SharePoint item for debugging
    
    Args:
        item (dict): The SharePoint item
        prefix (str): Prefix for the log message
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
        
    try:
        item_type = "Folder" if "folder" in item else "File" if "file" in item else "Unknown"
        item_name = item.get("name", "Unknown")
        item_id = item.get("id", "No ID")
        
        # Create a pretty-printed version of the item with limited depth
        item_json = json.dumps({k: v for k, v in item.items() 
                               if k not in ['@odata.context']}, indent=2)
        
        logger.debug(f"{prefix}Details for {item_type} '{item_name}' (ID: {item_id}):\n{item_json}")
    except Exception as e:
        logger.debug(f"Error printing item details: {e}")

def get_sharepoint_site_id() -> Optional[str]:
    """
    Retrieves the SharePoint site ID for the MatterDocs site
    
    Returns:
        str: The site ID if successful, None otherwise
    """
    headers = get_auth_headers()
    if not headers.get("Authorization"):
        logger.error("Failed to get authorization headers")
        return None
    
    # Extract domain and site path
    parsed_url = urllib.parse.urlparse(SHAREPOINT_DOMAIN)
    hostname = parsed_url.netloc
    site_path = parsed_url.path
    
    # Create the API URL to get the site ID
    url = f"{GRAPH_API_ENDPOINT}/sites/{hostname}:{site_path}"
    
    try:
        logger.info(f"Requesting SharePoint site ID for {SHAREPOINT_DOMAIN}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        site_data = response.json()
        site_id = site_data.get("id")
        
        if site_id:
            logger.info(f"Successfully retrieved site ID: {site_id}")
            return site_id
        else:
            logger.error("No site ID found in the response")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting SharePoint site ID: {str(e)}")
        return None

def get_drive_id(site_id: str) -> Optional[str]:
    """
    Retrieves the document library drive ID for the SharePoint site
    
    Args:
        site_id (str): The SharePoint site ID
        
    Returns:
        str: The drive ID if successful, None otherwise
    """
    headers = get_auth_headers()
    if not headers.get("Authorization"):
        logger.error("Failed to get authorization headers")
        return None
    
    # Create the API URL to get the default document library drive
    url = f"{GRAPH_API_ENDPOINT}/sites/{site_id}/drive"
    
    try:
        logger.info(f"Requesting drive ID for site ID: {site_id}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        drive_data = response.json()
        drive_id = drive_data.get("id")
        
        if drive_id:
            logger.info(f"Successfully retrieved drive ID: {drive_id}")
            return drive_id
        else:
            logger.error("No drive ID found in the response")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting drive ID: {str(e)}")
        return None

def get_folder_id(drive_id: str, folder_path: str) -> Optional[str]:
    """
    Retrieves the folder ID for a specific path in the SharePoint drive
    
    Args:
        drive_id (str): The SharePoint drive ID
        folder_path (str): The folder path relative to the drive root
        
    Returns:
        str: The folder ID if successful, None otherwise
    """
    headers = get_auth_headers()
    if not headers.get("Authorization"):
        logger.error("Failed to get authorization headers")
        return None
    
    # Clean up the folder path by removing leading and trailing slashes
    folder_path = folder_path.strip('/')
    
    # Create the API URL to get the folder ID
    url = f"{GRAPH_API_ENDPOINT}/drives/{drive_id}/root:/{urllib.parse.quote(folder_path)}"
    
    try:
        logger.info(f"Requesting folder ID for path: {folder_path}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        folder_data = response.json()
        folder_id = folder_data.get("id")
        
        if folder_id:
            logger.info(f"Successfully retrieved folder ID: {folder_id}")
            return folder_id
        else:
            logger.error("No folder ID found in the response")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting folder ID: {str(e)}")
        return None

def list_folder_contents(drive_id: str, folder_id: str) -> List[Dict[str, Any]]:
    """
    Lists all items in a SharePoint folder
    
    Args:
        drive_id (str): The SharePoint drive ID
        folder_id (str): The folder ID
        
    Returns:
        List[Dict[str, Any]]: List of items in the folder
    """
    headers = get_auth_headers()
    if not headers.get("Authorization"):
        logger.error("Failed to get authorization headers")
        return []
    
    # Create the API URL to list folder contents
    url = f"{GRAPH_API_ENDPOINT}/drives/{drive_id}/items/{folder_id}/children"
    
    try:
        logger.info(f"Listing contents of folder ID: {folder_id}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        contents_data = response.json()
        items = contents_data.get("value", [])
        
        # Check if there are more items to fetch (pagination)
        next_link = contents_data.get("@odata.nextLink")
        while next_link:
            logger.info("Fetching next page of folder contents")
            next_response = requests.get(next_link, headers=headers)
            next_response.raise_for_status()
            
            next_data = next_response.json()
            items.extend(next_data.get("value", []))
            next_link = next_data.get("@odata.nextLink")
        
        # Debug log the raw response
        logger.debug(f"Raw API response for folder contents (first 1000 chars):\n{json.dumps(contents_data)[:1000]}...")
        
        # Count the types of items
        folders_count = sum(1 for item in items if "folder" in item)
        files_count = sum(1 for item in items if "file" in item)
        other_count = len(items) - folders_count - files_count
        
        logger.info(f"Found {len(items)} items in folder ({folders_count} folders, {files_count} files, {other_count} other)")
        
        # Print details for each item in debug mode
        for i, item in enumerate(items):
            print_item_details(item, prefix=f"Item {i+1}/{len(items)}: ")
            
        return items
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error listing folder contents: {str(e)}")
        return []

def get_documents_recursively(drive_id: str, folder_id: str, folder_path: str = "", download_docs: bool = False, target_dir: str = None) -> List[Dict[str, Any]]:
    """
    Recursively lists and processes all documents in a SharePoint folder and its subfolders
    
    Args:
        drive_id (str): The SharePoint drive ID
        folder_id (str): The folder ID
        folder_path (str): Current folder path for filename prefixing
        download_docs (bool): Whether to download the documents
        target_dir (str, optional): Directory to save downloaded files
        
    Returns:
        List[Dict[str, Any]]: List of all documents found
    """
    logger.info(f"Processing folder{' path: ' + folder_path if folder_path else ''} (ID: {folder_id})")
    
    # Get items in the current folder
    items = list_folder_contents(drive_id, folder_id)
    if not items:
        return []
        
    all_documents = []
    
    # Process each item
    for item in items:
        item_name = item.get("name", "Unknown")
        
        # If it's a folder, recursively process it
        if "folder" in item:
            logger.debug(f"Found subfolder: {item_name}")
            # Build the subfolder path for nested items
            subfolder_path = f"{folder_path}_{item_name}" if folder_path else item_name
            
            # Get the folder ID and recursively process it
            subfolder_id = item.get("id")
            if subfolder_id:
                # Recursively get documents from this subfolder
                subfolder_docs = get_documents_recursively(
                    drive_id, 
                    subfolder_id, 
                    subfolder_path, 
                    download_docs, 
                    target_dir
                )
                all_documents.extend(subfolder_docs)
        
        # If it's a file, process it
        elif "file" in item:
            logger.debug(f"Processing file: {item_name}")
            
            # Create document metadata
            document = {
                "id": item.get("id"),
                "name": item_name,
                "originalName": item_name,  # Keep the original name
                "folder_path": folder_path,  # Save the folder path
                "webUrl": item.get("webUrl"),
                "createdDateTime": item.get("createdDateTime"),
                "lastModifiedDateTime": item.get("lastModifiedDateTime"),
                "size": item.get("size"),
                "fileType": os.path.splitext(item_name)[1][1:] if "." in item_name else ""
            }
            
            # Prepend folder path to filename if we have a folder path
            if folder_path:
                # Keep file extension separate
                filename, file_ext = os.path.splitext(item_name)
                document["name"] = f"{folder_path}_{filename}{file_ext}"
            
            # Download document content if requested
            if download_docs:
                # Use the modified filename with folder path for saving
                download_result = download_file(
                    drive_id, 
                    item.get("id"), 
                    target_dir,
                    custom_filename=document["name"]  # Use the modified name with folder path
                )
                
                if download_result:
                    document["downloaded"] = True
                    if not target_dir:
                        # If target_dir is None, we didn't save the file, so include content as base64
                        document["contentBase64"] = base64.b64encode(download_result[1]).decode('utf-8')
                else:
                    document["downloaded"] = False
            
            all_documents.append(document)
            
    return all_documents

def download_file(drive_id: str, item_id: str, target_dir: str = None, custom_filename: str = None) -> Optional[Tuple[str, bytes]]:
    """
    Downloads a file from SharePoint
    
    Args:
        drive_id (str): The SharePoint drive ID
        item_id (str): The item ID of the file
        target_dir (str, optional): Directory to save the file, if None the file is not saved
        custom_filename (str, optional): Custom filename to use instead of the original name
        
    Returns:
        Optional[Tuple[str, bytes]]: Tuple of (filename, file_content) if successful, None otherwise
    """
    headers = get_auth_headers()
    if not headers.get("Authorization"):
        logger.error("Failed to get authorization headers")
        return None
    
    # First get item metadata to get the file name
    metadata_url = f"{GRAPH_API_ENDPOINT}/drives/{drive_id}/items/{item_id}"
    
    try:
        logger.info(f"Getting metadata for item ID: {item_id}")
        metadata_response = requests.get(metadata_url, headers=headers)
        metadata_response.raise_for_status()
        
        metadata = metadata_response.json()
        original_filename = metadata.get("name")
        
        if not original_filename:
            logger.error("No file name found in metadata")
            return None
        
        # Use custom filename if provided, otherwise use original
        file_name = custom_filename if custom_filename else original_filename
        
        # Now download the file content
        download_url = f"{GRAPH_API_ENDPOINT}/drives/{drive_id}/items/{item_id}/content"
        
        logger.info(f"Downloading file: {original_filename}" + (f" as {file_name}" if custom_filename else ""))
        content_response = requests.get(download_url, headers=headers)
        content_response.raise_for_status()
        
        file_content = content_response.content
        
        # Save the file if target directory is provided
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, file_name)
            
            with open(file_path, 'wb') as f:
                f.write(file_content)
            logger.info(f"Saved file to: {file_path}")
        
        return (file_name, file_content)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file: {str(e)}")
        return None

def get_documents_from_sharepoint_folder(folder_path: str, download_docs: bool = False, target_dir: str = None, recursive: bool = True) -> List[Dict[str, Any]]:
    """
    Main function to retrieve documents from a SharePoint folder
    
    Args:
        folder_path (str): The folder path in SharePoint
        download_docs (bool): Whether to download the documents or just get metadata
        target_dir (str, optional): Directory to save downloaded files if download_docs is True
        recursive (bool): Whether to recursively get documents from subfolders
        
    Returns:
        List[Dict[str, Any]]: List of document metadata with optional content
    """
    logger.info(f"Getting documents from SharePoint folder: {folder_path}")
    
    if not folder_path:
        logger.error("No folder path provided")
        return []
    
    # Step 1: Get the SharePoint site ID
    site_id = get_sharepoint_site_id()
    if not site_id:
        logger.error("Failed to get SharePoint site ID")
        return []
    
    # Step 2: Get the drive ID
    drive_id = get_drive_id(site_id)
    if not drive_id:
        logger.error("Failed to get drive ID")
        return []
    
    # Step 3: Get the folder ID
    folder_id = get_folder_id(drive_id, folder_path)
    if not folder_id:
        logger.error(f"Failed to get folder ID for path: {folder_path}")
        return []
    
    # Step 4: Process the folder - recursively if specified
    if recursive:
        # Get folder name for path construction
        folder_name = folder_path.split('/')[-1] if '/' in folder_path else folder_path
        documents = get_documents_recursively(drive_id, folder_id, "", download_docs, target_dir)
    else:
        # Step 4 (Legacy): List the folder contents (non-recursive)
        items = list_folder_contents(drive_id, folder_id)
        
        # Step 5 (Legacy): Process and return the documents (non-recursive)
        documents = []
        folders_count = 0
        files_count = 0
        
        for item in items:
            # Log item properties for debugging
            item_name = item.get("name", "Unknown")
            has_folder_prop = "folder" in item
            has_file_prop = "file" in item
            
            if has_folder_prop:
                logger.debug(f"Found folder: {item_name}")
                folders_count += 1
                continue  # Skip folders
            
            # If it doesn't have a file property, log and skip
            if not has_file_prop:
                logger.debug(f"Item {item_name} has neither folder nor file property - skipping")
                continue
                
            # It's a file - process it
            files_count += 1
            logger.debug(f"Processing file: {item_name}")
            
            document = {
                "id": item.get("id"),
                "name": item_name,
                "webUrl": item.get("webUrl"),
                "createdDateTime": item.get("createdDateTime"),
                "lastModifiedDateTime": item.get("lastModifiedDateTime"),
                "size": item.get("size"),
                "fileType": os.path.splitext(item_name)[1][1:] if "." in item_name else ""
            }
            
            # Download document content if requested
            if download_docs:
                download_result = download_file(drive_id, item.get("id"), target_dir)
                if download_result:
                    document["downloaded"] = True
                    if not target_dir:
                        # If target_dir is None, we didn't save the file, so include content as base64
                        document["contentBase64"] = base64.b64encode(download_result[1]).decode('utf-8')
                else:
                    document["downloaded"] = False
            
            documents.append(document)
        
        logger.info(f"Found {len(items)} total items in folder ({folders_count} folders, {files_count} files)")
    
    logger.info(f"Retrieved {len(documents)} documents from folder{' and subfolders' if recursive else ''}")
    return documents

def get_sharepoint_documents_for_matter(matter_context: Dict[str, Any], download_docs: bool = False, target_dir: str = None, recursive: bool = True) -> List[Dict[str, Any]]:
    """
    Retrieve SharePoint documents for a matter based on context from get_all_context_by_matter.py
    
    Args:
        matter_context (Dict[str, Any]): Matter context dictionary from get_all_context_by_matter.py
        download_docs (bool): Whether to download the documents or just get metadata
        target_dir (str, optional): Directory to save downloaded files if download_docs is True
        recursive (bool): Whether to recursively get documents from subfolders
        
    Returns:
        List[Dict[str, Any]]: List of document metadata with optional content
    """
    # Extract SharePoint folder path from matter context
    sharepoint_folder = None
    
    # First try to find it in the matter object
    if "matter" in matter_context and isinstance(matter_context["matter"], dict):
        sharepoint_folder = matter_context["matter"].get("Sharepoint_Folder__c")
    
    # If not found, try looking in categorized_fields if available
    if not sharepoint_folder and "matter" in matter_context and "categorized_fields" in matter_context["matter"]:
        categories = matter_context["matter"]["categorized_fields"]
        
        # Look in file_management category first
        if "file_management" in categories:
            sharepoint_folder = categories["file_management"].get("Sharepoint_Folder__c")
        
        # If still not found, check other likely categories
        if not sharepoint_folder:
            for category in ["basic", "case_details"]:
                if category in categories:
                    sharepoint_folder = categories[category].get("Sharepoint_Folder__c")
                    if sharepoint_folder:
                        break
    
    # If still not found, look in the sharepoint section directly
    if not sharepoint_folder and "sharepoint" in matter_context:
        sharepoint_folder = matter_context["sharepoint"].get("Sharepoint_Folder__c")
    
    if not sharepoint_folder:
        logger.error("No SharePoint folder found in matter context")
        return []
    
    # Clean up the folder path
    sharepoint_folder = sharepoint_folder.strip()
    
    if not sharepoint_folder:
        # Log the issue
        logger.warning("No SharePoint folder found in matter context")
        
        # For debugging: check if matter ID is available to create a fallback path
        matter_id = None
        if "matter" in matter_context and isinstance(matter_context["matter"], dict):
            matter_id = matter_context["matter"].get("Id")
        
        if matter_id:
            # Use the matter ID as a fallback folder path for testing
            logger.warning(f"Using matter ID as fallback SharePoint folder path: {matter_id}")
            sharepoint_folder = matter_id
        else:
            logger.error("Cannot create fallback SharePoint folder path - no matter ID found")
            return []
    
    # Get the documents from the SharePoint folder
    return get_documents_from_sharepoint_folder(sharepoint_folder, download_docs, target_dir, recursive)

if __name__ == "__main__":
    # Example usage
    import sys
    
    # Process command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Retrieve documents from SharePoint folder')
    parser.add_argument('folder_path', help='Path to SharePoint folder')
    parser.add_argument('--download', action='store_true', help='Download the documents')
    parser.add_argument('--target-dir', help='Directory to save downloaded files')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--no-recursive', action='store_true', help='Disable recursive folder traversal')
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        enable_debug_logging()
        logger.debug("Debug mode enabled")
    
    # Get documents from SharePoint
    recursive = not args.no_recursive
    documents = get_documents_from_sharepoint_folder(
        args.folder_path, 
        args.download, 
        args.target_dir, 
        recursive
    )
    
    # Print results
    if documents:
        # Just print the document count and first few documents for large results
        if len(documents) > 10:
            print(json.dumps(documents[:10], indent=2))
            print(f"\n... and {len(documents) - 10} more documents (showing first 10 only)")
        else:
            print(json.dumps(documents, indent=2))
        
        print(f"\nFound {len(documents)} documents in folder{' and subfolders' if recursive else ''}: {args.folder_path}")
        
        if args.download and args.target_dir:
            print(f"Files downloaded to: {os.path.abspath(args.target_dir)}")
    else:
        print(f"No documents found in folder{' or subfolders' if recursive else ''}: {args.folder_path}")
        print("Try running with --debug for more information")
