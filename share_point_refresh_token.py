#!/usr/bin/env python3
"""
SharePoint API Authentication Module
Handles retrieving OAuth bearer tokens for Microsoft Graph API access
for both Windows and Linux environments.
"""
import os
import sys
import requests
import json
import logging
from typing import Dict, Optional, Any, Tuple
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='[share_point_refresh_token] %(message)s')
logger = logging.getLogger(__name__)

# Token cache to reduce API calls
_token_cache = {
    "token": None,
    "expiry": 0  # Timestamp when token expires
}

def ensure_file_permissions():
    """
    Ensure the script has appropriate file permissions in Linux environments.
    This is important for production environments.
    """
    if not sys.platform.startswith('win'):
        try:
            # Get current script path
            script_path = os.path.abspath(__file__)
            
            # Check if file is executable
            is_executable = os.access(script_path, os.X_OK)
            if not is_executable:
                logger.info("Script is not executable. Attempting to add execute permission.")
                # Add execute permission (chmod +x)
                os.chmod(script_path, os.stat(script_path).st_mode | 0o111)
                logger.info(f"Execute permission added to {script_path}")
        except Exception as e:
            logger.error(f"Error setting file permissions: {str(e)}")

def load_credentials() -> Tuple[str, str, str, str]:
    """
    Load Microsoft Graph API credentials from environment variables or .env file.
    
    Returns:
        tuple: (tenant_id, client_id, client_secret, scope)
    """
    # Load environment variables if not already loaded
    load_dotenv()
    
    # Try to get credentials from environment
    tenant_id = os.environ.get('MS_TENANT_ID', '')
    client_id = os.environ.get('MS_CLIENT_ID', '')
    client_secret = os.environ.get('MS_CLIENT_SECRET', '')
    scope = os.environ.get('MS_API_SCOPE', 'https://graph.microsoft.com/.default')
    
    # Check if credentials are available
    if not all([tenant_id, client_id, client_secret]):
        logger.warning("One or more Microsoft Graph API credentials are missing.")
        
        # Log what credentials are available/missing for debugging
        logger.debug(f"MS_TENANT_ID available: {bool(tenant_id)}")
        logger.debug(f"MS_CLIENT_ID available: {bool(client_id)}")
        logger.debug(f"MS_CLIENT_SECRET available: {bool(client_secret)}")
        
        logger.warning("Consider setting MS_TENANT_ID, MS_CLIENT_ID, and MS_CLIENT_SECRET environment variables.")
        
        # Try to load from .env file explicitly (in case load_dotenv() didn't work)
        try:
            from dotenv import dotenv_values
            config = dotenv_values(".env")
            
            if not tenant_id and 'MS_TENANT_ID' in config:
                tenant_id = config['MS_TENANT_ID']
                logger.debug("Loaded MS_TENANT_ID from .env file")
                
            if not client_id and 'MS_CLIENT_ID' in config:
                client_id = config['MS_CLIENT_ID']
                logger.debug("Loaded MS_CLIENT_ID from .env file")
                
            if not client_secret and 'MS_CLIENT_SECRET' in config:
                client_secret = config['MS_CLIENT_SECRET']
                logger.debug("Loaded MS_CLIENT_SECRET from .env file")
        except Exception as e:
            logger.warning(f"Error trying to load from .env file: {str(e)}")
    
    return tenant_id, client_id, client_secret, scope

def get_bearer_token(force_refresh=False) -> Optional[str]:
    """
    Retrieves a bearer token from the Microsoft OAuth endpoint using client credentials flow.
    
    Args:
        force_refresh (bool): Force a new token to be retrieved even if cached token exists
    
    Returns:
        str: The bearer token if successful, None otherwise
    """
    global _token_cache
    import time
    
    # Use cached token if available and not forcing refresh
    current_time = time.time()
    if not force_refresh and _token_cache["token"] and current_time < _token_cache["expiry"]:
        logger.info("Using cached bearer token")
        return _token_cache["token"]
    
    # Load credentials
    tenant_id, client_id, client_secret, scope = load_credentials()
    
    # Verify we have the required credentials
    if not (tenant_id and client_id and client_secret):
        logger.error("Cannot get bearer token: Missing SharePoint credentials")
        logger.error("Please set MS_TENANT_ID, MS_CLIENT_ID, and MS_CLIENT_SECRET environment variables")
        return None
    
    logger.info(f"Requesting new token from Microsoft OAuth endpoint (Tenant: {tenant_id[:6]}...)")
    
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
    }
    
    try:
        logger.info("Requesting bearer token from Microsoft OAuth endpoint")
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        token_data = response.json()
        bearer_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)  # Default to 1 hour if not specified
        
        if bearer_token:
            logger.info("Successfully retrieved bearer token")
            
            # Cache the token with expiry time (subtract 5 minutes for safety)
            _token_cache["token"] = bearer_token
            _token_cache["expiry"] = current_time + (expires_in - 300)
            
            return bearer_token
        else:
            logger.error("No access token found in the response")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting bearer token: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
        return None

def get_auth_headers() -> Dict[str, str]:
    """
    Get headers for Microsoft Graph API with authentication
    
    Returns:
        Dict[str, str]: Headers with bearer token
    """
    token = get_bearer_token()
    
    if not token:
        logger.error("Failed to get bearer token for authentication headers")
        return {}
        
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

def verify_token(token: str) -> bool:
    """
    Verify if a token is valid by making a test request to Microsoft Graph API.
    
    Args:
        token (str): The bearer token to verify
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    if not token:
        return False
        
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    
    # Make a simple request to verify token
    test_url = "https://graph.microsoft.com/v1.0/me"
    try:
        response = requests.get(test_url, headers=headers, timeout=10)
        
        # If we get a 401, token is invalid
        if response.status_code == 401:
            logger.warning("SharePoint token verification failed: Unauthorized (401)")
            return False
            
        # Any non-error response means token is valid
        if response.status_code < 400:
            logger.info("SharePoint token verification successful")
            return True
            
        # For other errors, log but consider token might still be valid
        logger.warning(f"SharePoint token verification got unexpected status: {response.status_code}")
        # We'll be optimistic and assume token might still be valid
        return True
    except Exception as e:
        logger.warning(f"Error verifying SharePoint token: {str(e)}")
        # Be conservative and assume token is invalid on exceptions
        return False

def get_reliable_token(max_attempts=3) -> Optional[str]:
    """
    Get a reliable SharePoint token with verification and retries.
    
    Args:
        max_attempts (int): Maximum number of refresh attempts
        
    Returns:
        Optional[str]: A verified token or None if all attempts fail
    """
    for attempt in range(max_attempts):
        # Always force refresh on attempts after the first
        force_refresh = (attempt > 0)
        
        # Get token
        token = get_bearer_token(force_refresh=force_refresh)
        
        # No token was obtained
        if not token:
            logger.warning(f"Failed to get SharePoint token on attempt {attempt+1}/{max_attempts}")
            continue
            
        # Verify token works
        if verify_token(token):
            return token
        else:
            logger.warning(f"Token verification failed on attempt {attempt+1}/{max_attempts}")
            # Clear token cache to force a fresh token on next attempt
            global _token_cache
            _token_cache = {"token": None, "expiry": 0}
            
    # All attempts failed
    logger.error(f"Failed to get reliable SharePoint token after {max_attempts} attempts")
    return None

def sharepoint_request(method, endpoint, data=None, params=None) -> Optional[Dict]:
    """
    User-friendly function to make a Microsoft Graph API request with automatic authentication.
    
    Args:
        method (str): HTTP method (get, post, patch, delete)
        endpoint (str): API endpoint (e.g., '/sites/{site-id}/lists')
        data (dict, optional): Request payload for POST/PATCH requests
        params (dict, optional): URL parameters for GET requests
        
    Returns:
        dict or None: Response data or None if request failed
    """
    headers = get_auth_headers()
    
    if "Authorization" not in headers:
        logger.error("Unable to get valid authentication for SharePoint request")
        return None
    
    base_url = "https://graph.microsoft.com/v1.0"
    url = f"{base_url}{endpoint}"
    
    try:
        method = method.lower()
        if method == 'get':
            response = requests.get(url, headers=headers, params=params)
        elif method == 'post':
            response = requests.post(url, headers=headers, json=data)
        elif method == 'patch':
            response = requests.patch(url, headers=headers, json=data)
        elif method == 'delete':
            response = requests.delete(url, headers=headers)
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            return None
        
        response.raise_for_status()
        if response.status_code != 204:  # No content
            return response.json()
        return {"success": True}  # Return something for no-content responses
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making {method.upper()} request to {url}: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
        return None

def verify_credentials():
    """
    Verify that SharePoint credentials are valid by making a test request.
    
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    logger.info("Verifying SharePoint credentials")
    
    # Check environment variables
    tenant_id, client_id, client_secret, scope = load_credentials()
    
    if not all([tenant_id, client_id, client_secret]):
        logger.warning("Missing one or more required environment variables:")
        if not tenant_id:
            logger.warning("- MS_TENANT_ID is not set")
        if not client_id:
            logger.warning("- MS_CLIENT_ID is not set")
        if not client_secret:
            logger.warning("- MS_CLIENT_SECRET is not set")
            
        logger.info("\nTo set these variables:")
        if sys.platform.startswith('win'):
            logger.info("For PowerShell: $env:MS_TENANT_ID = 'your-tenant-id'")
            logger.info("For CMD: set MS_TENANT_ID=your-tenant-id")
        else:
            logger.info("For bash/zsh: export MS_TENANT_ID=your-tenant-id")
            
        logger.info("\nOr create a .env file with these variables")
    
    # Test token retrieval
    token = get_bearer_token(force_refresh=True)
    if not token:
        logger.error("Failed to retrieve bearer token")
        return False
    
    # Test a simple API call
    try:
        # Make a simple request to test the token
        response = sharepoint_request('get', '/me')
        if response:
            logger.info("✅ SharePoint credentials are valid")
            return True
        else:
            logger.error("❌ SharePoint test request failed")
            return False
    except Exception as e:
        logger.error(f"❌ Error testing SharePoint credentials: {str(e)}")
        return False

if __name__ == "__main__":
    # Check file permissions for Linux environments
    ensure_file_permissions()
    
    print("\n================== SharePoint Authentication Check ==================\n")
    
    # Determine platform
    is_windows = sys.platform.startswith('win')
    
    print(f"Running on {'Windows' if is_windows else 'Linux/Unix'} environment")
    
    # Load .env file if present
    load_dotenv()
    print("\nChecking environment variables:")
    
    # Check for environment variables
    env_vars = {
        "MS_TENANT_ID": os.getenv("MS_TENANT_ID"),
        "MS_CLIENT_ID": os.getenv("MS_CLIENT_ID"),
        "MS_CLIENT_SECRET": os.getenv("MS_CLIENT_SECRET"),
    }
    
    for var, value in env_vars.items():
        if value:
            masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "****"
            print(f"  ✅ {var}: {masked_value}")
        else:
            print(f"  ❌ {var}: Not set")
    
    # Verify credentials
    verification_result = verify_credentials()
    
    if verification_result:
        print("\n================== Testing SharePoint API Query ==================\n")
        
        # Example: List SharePoint sites
        sites = sharepoint_request('get', '/sites?search=*')
        if sites and 'value' in sites:
            print(f"✅ Successfully retrieved {len(sites['value'])} SharePoint sites")
            for site in sites['value'][:5]:  # Show first 5 sites
                print(f"  - {site.get('displayName', 'Unknown')} ({site.get('webUrl', 'No URL')})")
        else:
            print("❌ Failed to retrieve SharePoint sites")
    else:
        print("\n❌ SharePoint authentication failed")
        
        # Show help for setting up credentials
        print("\nTo set up SharePoint credentials:")
        print("1. Create a .env file in the same directory as this script")
        print("2. Add the following lines to the .env file:")
        print("   MS_TENANT_ID=your-tenant-id")
        print("   MS_CLIENT_ID=your-client-id")
        print("   MS_CLIENT_SECRET=your-client-secret")
        print("\nOr set these environment variables in your system")