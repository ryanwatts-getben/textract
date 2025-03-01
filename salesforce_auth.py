#!/usr/bin/env python3

import os
import requests
import json
import sys
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

def authenticate_with_username_password():
    """
    Authenticate with Salesforce using username and password (plus security token).
    This method is suitable for headless server environments.
    
    Returns:
        tuple: (access_token, instance_url) or (None, None) if authentication fails
    """
    print("[salesforce_auth] Attempting to authenticate with username and password")
    
    # Get credentials from environment
    username = os.getenv("SALESFORCE_USERNAME")
    password = os.getenv("SALESFORCE_PASSWORD")
    security_token = os.getenv("SALESFORCE_SECURITY_TOKEN", "")  # Optional
    client_id = os.getenv("SALESFORCE_CLIENT_ID", "PlatformCLI")  # Default to PlatformCLI as in SF CLI
    client_secret = os.getenv("SALESFORCE_CLIENT_SECRET", "")
    
    # Check if credentials are available
    if not username or not password:
        print("[salesforce_auth] Error: Username or password not provided in environment variables")
        print("[salesforce_auth] Please set SALESFORCE_USERNAME and SALESFORCE_PASSWORD environment variables")
        return None, None
        
    # Combine password and security token if token is provided
    password_with_token = password + security_token
        
    # Get the login endpoint
    login_url = os.getenv("SALESFORCE_LOGIN_URL", "https://login.salesforce.com/services/oauth2/token")
    
    # Prepare payload for OAuth token request
    payload = {
        "grant_type": "password",
        "client_id": client_id,
        "client_secret": client_secret,
        "username": username,
        "password": password_with_token
    }
    
    print(f"[salesforce_auth] Authenticating user: {username}")
    print(f"[salesforce_auth] Using login URL: {login_url}")
    
    try:
        # Make request to get token
        response = requests.post(login_url, data=payload)
        
        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            access_token = data.get("access_token")
            instance_url = data.get("instance_url")
            
            print(f"[salesforce_auth] Successfully authenticated with username/password method")
            print(f"[salesforce_auth] Instance URL: {instance_url}")
            
            # Save token to environment variables for current session
            os.environ["SALESFORCE_ACCESS_TOKEN"] = access_token
            os.environ["SALESFORCE_INSTANCE_URL"] = instance_url
            
            return access_token, instance_url
        else:
            print(f"[salesforce_auth] Authentication failed with status code: {response.status_code}")
            print(f"[salesforce_auth] Response: {response.text}")
                
    except Exception as e:
        print(f"[salesforce_auth] Error during authentication: {e}")
        
    return None, None

def test_salesforce_connection(access_token, instance_url):
    """
    Test the Salesforce connection by making a simple API call.
    
    Args:
        access_token (str): Salesforce access token
        instance_url (str): Salesforce instance URL
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    if not access_token or not instance_url:
        print("[salesforce_auth] No valid access token or instance URL provided")
        return False
    
    # Create headers with the access token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Test with a simple limits API call which doesn't require specific permissions
    try:
        url = f"{instance_url}/services/data/v58.0/limits"
        print(f"[salesforce_auth] Testing connection with: {url}")
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print("[salesforce_auth] Connection test successful!")
            return True
        else:
            print(f"[salesforce_auth] Connection test failed with status code: {response.status_code}")
            print(f"[salesforce_auth] Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"[salesforce_auth] Error testing connection: {e}")
        return False

def save_to_env_file(access_token, instance_url):
    """
    Save credentials to .env file.
    
    Args:
        access_token (str): Salesforce access token
        instance_url (str): Salesforce instance URL
    """
    try:
        env_path = '.env'
        env_exists = os.path.exists(env_path)
        
        # Read existing .env file if it exists
        env_vars = {}
        if env_exists:
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value

        # Update with new token
        env_vars['SALESFORCE_ACCESS_TOKEN'] = access_token
        env_vars['SALESFORCE_INSTANCE_URL'] = instance_url
        
        # Write back to .env file
        with open(env_path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
                
        print(f"[salesforce_auth] Saved credentials to {os.path.abspath(env_path)}")
        return True
        
    except Exception as e:
        print(f"[salesforce_auth] Error saving to .env file: {e}")
        return False

def main():
    """Main function to handle authentication and testing"""
    print("\n================== Salesforce Authentication ==================\n")
    
    # Attempt authentication
    access_token, instance_url = authenticate_with_username_password()
    
    if access_token and instance_url:
        print("\n✅ Authentication successful!")
        print(f"Token: {access_token[:10]}... (Token retrieved successfully)")
        print(f"Instance URL: {instance_url}")
        
        # Test connection
        print("\n================== Testing API Connection ==================\n")
        if test_salesforce_connection(access_token, instance_url):
            # Save credentials to .env file
            save_to_env_file(access_token, instance_url)
            
            # Display a sample query example
            print("\n================== Example Usage ==================\n")
            print("You can now use the token to make Salesforce API requests:")
            print("""
import requests

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

# Example: Query Accounts
query = "SELECT Id, Name FROM Account LIMIT 5"
response = requests.get(
    f"{instance_url}/services/data/v58.0/query", 
    headers=headers, 
    params={"q": query}
)

if response.status_code == 200:
    data = response.json()
    print(f"Found {data['totalSize']} accounts")
    for record in data['records']:
        print(f"Account: {record['Name']}")
""")
        else:
            print("\n❌ Connection test failed")
    else:
        print("\n❌ Authentication failed")
        print("Please check your environment variables and try again")
        print("\nRequired environment variables:")
        print("- SALESFORCE_USERNAME: Your Salesforce username")
        print("- SALESFORCE_PASSWORD: Your Salesforce password")
        print("- SALESFORCE_SECURITY_TOKEN: Your Salesforce security token (if required)")
        print("\nOptional environment variables:")
        print("- SALESFORCE_CLIENT_ID: OAuth client ID (defaults to PlatformCLI)")
        print("- SALESFORCE_CLIENT_SECRET: OAuth client secret")
        print("- SALESFORCE_LOGIN_URL: Login URL (defaults to https://login.salesforce.com/services/oauth2/token)")

if __name__ == "__main__":
    main() 