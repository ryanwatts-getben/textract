#!/usr/bin/env python3

import os
import requests
import json
import subprocess
import sys
import shutil
from dotenv import load_dotenv
import time

# Allow configuring the CLI path through env var or directly in the script
SF_CLI_PATH = os.getenv("SF_CLI_PATH", "sf")  # Default to "sf" command

# Add this at the top of the file after the imports
_token_cache = {
    "token": None,
    "instance_url": None,
    "expiry": 0  # Timestamp when token expires
}

def find_sf_cli():
    """
    Find the Salesforce CLI executable path.
    
    Returns:
        str: Path to the Salesforce CLI executable or None if not found
    """
    global SF_CLI_PATH
    
    # First check if the current SF_CLI_PATH works
    if shutil.which(SF_CLI_PATH):
        return SF_CLI_PATH
    
    # Determine platform
    is_windows = sys.platform.startswith('win')
    
    # Try to find sf in common locations
    possible_paths = ["sf"]  # Default PATH for all platforms
    
    if is_windows:
        # Windows-specific paths
        windows_paths = [
            "sf.cmd",                            # Windows cmd wrapper
            "sf.exe",                            # Windows executable
            os.path.expanduser("~/AppData/Local/sf/bin/sf.exe"),  # Windows install location
            os.path.expanduser("~/AppData/Roaming/npm/sf.cmd"),   # npm Windows location
            "C:\\Program Files\\sfdx\\bin\\sf.exe",  # Possible install location
            os.path.join(sys.prefix, "Scripts", "sf.exe"),  # Python venv location
        ]
        possible_paths.extend(windows_paths)
    else:
        # Linux/Mac-specific paths
        unix_paths = [
            os.path.expanduser("~/.npm/bin/sf"),        # npm global install location
            "/usr/local/bin/sf",                        # Common location on Unix
            "/usr/bin/sf",                              # Another common location on Unix
            os.path.expanduser("~/.local/bin/sf"),      # User-specific bin directory
            os.path.join(sys.prefix, "bin", "sf"),      # Python venv location for Unix
        ]
        possible_paths.extend(unix_paths)
    
    # Add common npm global install (works on both platforms)
    possible_paths.append(os.path.expanduser("~/.npm/sf"))
    
    # Check each possible path
    for path in possible_paths:
        sf_resolved = shutil.which(path)
        if sf_resolved:
            SF_CLI_PATH = sf_resolved
            return sf_resolved
    
    # If we can't find it automatically, try platform-specific commands
    try:
        if is_windows:
            # Windows uses 'where'
            result = subprocess.run(['where', 'sf'], capture_output=True, text=True)
        else:
            # Linux/Mac uses 'which'
            result = subprocess.run(['which', 'sf'], capture_output=True, text=True)
            
        if result.returncode == 0:
            sf_path = result.stdout.strip().split('\n')[0]  # Take first match
            if os.path.exists(sf_path):
                SF_CLI_PATH = sf_path
                return sf_path
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return None

def run_sf_command(args):
    """
    Run a Salesforce CLI command with proper path handling.
    
    Args:
        args (list): Command arguments (excluding 'sf')
        
    Returns:
        tuple: (returncode, stdout, stderr)
    """
    sf_path = find_sf_cli()
    if not sf_path:
        return 1, "", "Salesforce CLI not found in PATH. Please set SF_CLI_PATH env variable."
    
    try:
        result = subprocess.run(
            [sf_path] + args,
            capture_output=True,
            text=True
        )
        return result.returncode, result.stdout, result.stderr
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        return 1, "", f"Error running Salesforce CLI: {e}"

def get_cli_credentials(org_alias='louisfirm'):
    """
    Get Salesforce credentials from CLI if available.
    
    Args:
        org_alias (str): The alias of the Salesforce org to use
    
    Returns:
        tuple: (access_token, instance_url) or (None, None) if CLI not available
    """
    try:
        # First check if the org exists
        returncode, stdout, stderr = run_sf_command(['org', 'list', '--json'])
        
        if returncode == 0:
            org_list = json.loads(stdout)
            orgs = org_list.get('result', {}).get('nonScratchOrgs', [])
            found_org = False
            for org in orgs:
                if org.get('alias') == org_alias:
                    found_org = True
                    break
            
            if not found_org:
                print(f"[salesforce_refresh] Org alias '{org_alias}' not found in CLI")
                # List available orgs
                if orgs:
                    print("[salesforce_refresh] Available orgs:")
                    for org in orgs:
                        if org.get('alias'):
                            print(f"  - {org.get('alias')}")
                return None, None
        
        # Run sf org display command
        returncode, stdout, stderr = run_sf_command(['org', 'display', '--target-org', org_alias, '--json'])
        
        if returncode == 0:
            data = json.loads(stdout)
            return (
                data['result']['accessToken'],
                data['result']['instanceUrl']
            )
        return None, None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[salesforce_refresh] Error parsing CLI output: {e}")
        return None, None

def authenticate_with_username_password(username=None, password=None, security_token=None, instance_url=None):
    """
    Authenticate with Salesforce using username and password (plus security token).
    This method is suitable for headless server environments.
    
    Args:
        username (str): Salesforce username
        password (str): Salesforce password
        security_token (str): Salesforce security token (may be needed depending on IP restrictions)
        instance_url (str): Salesforce instance URL (e.g., https://louisfirm.my.salesforce.com)
        
    Returns:
        tuple: (access_token, instance_url) or (None, None) if authentication fails
    """
    print("[salesforce_refresh] Attempting to authenticate with username and password")
    
    # Get credentials from environment if not provided
    if not username:
        username = os.getenv("SALESFORCE_USERNAME")
    if not password:
        password = os.getenv("SALESFORCE_PASSWORD")
    if not security_token:
        security_token = os.getenv("SALESFORCE_SECURITY_TOKEN")
    if not instance_url:
        instance_url = os.getenv("SALESFORCE_INSTANCE_URL", "https://louisfirm.my.salesforce.com")
        
    # Check if credentials are available
    if not username or not password:
        print("[salesforce_refresh] Username or password not provided")
        return None, None
        
    # Combine password and security token if token is provided
    if security_token:
        password_with_token = password + security_token
    else:
        password_with_token = password
        
    # Get the login endpoint - this is the same for all Salesforce instances
    login_url = "https://login.salesforce.com/services/oauth2/token"
    
    # Prepare payload for OAuth token request
    payload = {
        "grant_type": "password",
        "client_id": os.getenv("SALESFORCE_CLIENT_ID", "PlatformCLI"),  # Default to PlatformCLI for SF CLI
        "client_secret": os.getenv("SALESFORCE_CLIENT_SECRET", ""),
        "username": username,
        "password": password_with_token
    }
    
    try:
        # Make request to get token
        response = requests.post(login_url, data=payload)
        
        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            access_token = data.get("access_token")
            instance_url = data.get("instance_url")
            
            print(f"[salesforce_refresh] Successfully authenticated with username/password method")
            
            # Save token to environment variables
            if access_token and instance_url:
                if "update_env_file" in globals():
                    update_env_file("SALESFORCE_ACCESS_TOKEN", access_token)
                    update_env_file("SALESFORCE_INSTANCE_URL", instance_url)
                
                return access_token, instance_url
        else:
            print(f"[salesforce_refresh] Authentication failed with status code: {response.status_code}")
            print(f"[salesforce_refresh] Response: {response.text}")
                
    except Exception as e:
        print(f"[salesforce_refresh] Error during username-password authentication: {e}")
        
    return None, None

def refresh_salesforce_token(update_env=True, org_alias='louisfirm'):
    """
    Refresh Salesforce access token using CLI.
    
    Args:
        update_env (bool): Whether to update the .env file with the new token
        org_alias (str): The alias of the Salesforce org to use with CLI
        
    Returns:
        tuple: (new_token, instance_url) - The new access token and instance URL
    """
    print("[salesforce_refresh] Attempting to refresh Salesforce access token using CLI...")
    
    # Try to get token from CLI
    token, instance_url = get_cli_credentials(org_alias)
    if token and instance_url:
        print(f"[salesforce_refresh] Successfully obtained token from CLI for org '{org_alias}'")
        if update_env:
            update_env_file("SALESFORCE_ACCESS_TOKEN", token)
            update_env_file("SALESFORCE_INSTANCE_URL", instance_url)
            print("[salesforce_refresh] Updated .env file with CLI token")
        return token, instance_url
    
    # If no token from CLI, try to refresh CLI auth by running login
    print("[salesforce_refresh] No token available from CLI, trying to refresh CLI authentication...")
    print("[salesforce_refresh] You may see a browser window open for authentication.")
    
    try:
        returncode, stdout, stderr = run_sf_command([
            'org', 'login', 'web', 
            '--instance-url', 'https://louisfirm.my.salesforce.com',
            '--alias', org_alias,
            '--set-default'
        ])
        
        if returncode == 0:
            print(f"[salesforce_refresh] Successfully authenticated with Salesforce CLI")
            # Try getting the token again
            return get_cli_credentials(org_alias)
        else:
            print(f"[salesforce_refresh] CLI login failed: {stderr}")
            
            # If we're in a headless environment (likely server), try username-password method
            if "PortInUseError" in stderr or "browser" in stderr.lower() or "display" in stderr.lower():
                print("[salesforce_refresh] Detected headless environment, trying username-password authentication")
                return authenticate_with_username_password()
    except Exception as e:
        print(f"[salesforce_refresh] Error during CLI login: {e}")
        
        # Try username-password method as fallback
        print("[salesforce_refresh] Falling back to username-password authentication")
        return authenticate_with_username_password()
    
    return None, None

def update_env_file(key, value):
    """
    Update a key-value pair in the .env file.
    
    Args:
        key (str): The environment variable key
        value (str): The new value to set
    """
    try:
        env_path = ".env"
        
        # Check if .env file exists, create if not
        if not os.path.exists(env_path):
            with open(env_path, 'w') as file:
                file.write(f"{key}={value}\n")
            return
        
        # Read the current .env file
        with open(env_path, 'r') as file:
            lines = file.readlines()
        
        # Find and replace the line with the key
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                updated = True
                break
        
        # If the key wasn't found, add it
        if not updated:
            lines.append(f"{key}={value}\n")
        
        # Write the updated content back to the .env file
        with open(env_path, 'w') as file:
            file.writelines(lines)
    except Exception as e:
        print(f"[salesforce_refresh] Error updating .env file: {e}")

def verify_credentials():
    """
    Verify that Salesforce CLI is properly configured.
    
    Returns:
        bool: True if CLI is available and configured
    """
    print("[salesforce_refresh] Verifying Salesforce CLI credentials...")
    
    # Check CLI availability
    sf_path = find_sf_cli()
    if sf_path:
        print(f"[salesforce_refresh] ✅ Salesforce CLI found at: {sf_path}")
        
        # Get version
        returncode, stdout, stderr = run_sf_command(['--version'])
        if returncode == 0:
            print(f"[salesforce_refresh] ✅ CLI version: {stdout.strip()}")
            
            # Check available orgs
            returncode, stdout, stderr = run_sf_command(['org', 'list', '--json'])
            if returncode == 0:
                try:
                    orgs_data = json.loads(stdout)
                    orgs = orgs_data.get('result', {}).get('nonScratchOrgs', [])
                    
                    if orgs:
                        print("[salesforce_refresh] ✅ Salesforce orgs found:")
                        for org in orgs:
                            org_status = "✅ " if org.get('isDefaultUsername') else "  "
                            if org.get('alias'):
                                print(f"  {org_status}{org.get('alias')} ({org.get('username')})")
                            else:
                                print(f"  {org_status}{org.get('username')}")
                        return True
                    else:
                        print("[salesforce_refresh] ❌ No Salesforce orgs found")
                        is_headless = not sys.platform.startswith('win') or 'DISPLAY' not in os.environ
                        
                        if is_headless:
                            print("[salesforce_refresh] ❗ Detected headless environment (no display/browser)")
                            print("[salesforce_refresh] For headless authentication options:")
                            print("  1. sfdxurl method: sf org login sfdx-url -f auth_url.txt")
                            print("  2. JWT method: sf org login jwt -u username -f server.key -i client_id")
                            print("  3. Username-password: sf org login username-password -u username -p password+token")
                            print("\nSee deployment_instructions.md for detailed steps.")
                        else:
                            print("[salesforce_refresh] Please authenticate with:")
                            print("  sf org login web --instance-url https://louisfirm.my.salesforce.com --alias louisfirm")
                except json.JSONDecodeError:
                    print("[salesforce_refresh] ❌ Error parsing CLI output")
                    print(f"[salesforce_refresh] Raw output: {stdout}")
            else:
                print("[salesforce_refresh] ❌ Unable to list Salesforce orgs")
                print(f"[salesforce_refresh] Error: {stderr}")
        else:
            print("[salesforce_refresh] ❌ Salesforce CLI test command failed")
            print(f"[salesforce_refresh] Error: {stderr}")
    else:
        print("[salesforce_refresh] ❌ Salesforce CLI not found in PATH")
        print("[salesforce_refresh] To fix this issue:")
        print("  1. Set the SF_CLI_PATH environment variable to the full path of the sf command")
        
        if sys.platform.startswith('win'):
            print("     For example: set SF_CLI_PATH=C:\\Program Files\\sfdx\\bin\\sf.exe")
        else:
            print("     For example: export SF_CLI_PATH=/usr/local/bin/sf")
            
        print("  2. Or edit this script to set SF_CLI_PATH directly at the top")
        print("  3. Or make sure the Salesforce CLI is in your system PATH")
        
        if sys.platform.startswith('win'):
            print("\n[salesforce_refresh] To find where sf is installed, run in PowerShell:")
            print("  where.exe sf")
        else:
            print("\n[salesforce_refresh] To find where sf is installed, run:")
            print("  which sf")
        return False
    
    return True

def get_salesforce_headers(org_alias='louisfirm', force_refresh=False):
    """
    Get valid headers for Salesforce API requests, refreshing token if needed.
    
    Args:
        org_alias (str): The alias of the Salesforce org to use with CLI
        force_refresh (bool): Force refresh the token even if cached
    
    Returns:
        tuple: (headers, instance_url) for Salesforce API requests with valid token
    """
    global _token_cache
    current_time = time.time()
    
    # Check if we have a cached token that's still valid (tokens typically last 2 hours)
    if not force_refresh and _token_cache["token"] and _token_cache["instance_url"] and current_time < _token_cache["expiry"]:
        print("[salesforce_refresh] Using cached token")
        headers = {
            "Authorization": f"Bearer {_token_cache['token']}",
            "Content-Type": "application/json"
        }
        return headers, _token_cache["instance_url"]
    
    # Try to get credentials from CLI 
    token, instance_url = get_cli_credentials(org_alias)
    
    # If CLI fails, try the stored token in .env as a backup
    if not token:
        try:
            load_dotenv()
            token = os.getenv("SALESFORCE_ACCESS_TOKEN")
            instance_url = os.getenv("SALESFORCE_INSTANCE_URL")
            
            # If still no token, see if we can refresh the CLI auth
            if not token or not instance_url:
                token, instance_url = refresh_salesforce_token(org_alias=org_alias)
        except Exception as e:
            print(f"[salesforce_refresh] Error loading .env credentials: {e}")
    
    # If we still don't have credentials, verify and exit
    if not token or not instance_url:
        print("[salesforce_refresh] No valid credentials found")
        verify_credentials()
        return None, None
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Only test if the token is valid if we're forcing a refresh or don't have a cached token
    if force_refresh or not _token_cache["token"]:
        # Test if the token is valid
        test_url = f"{instance_url}/services/data/v63.0/sobjects/"
        try:
            response = requests.get(test_url, headers=headers)
            
            # If token is invalid, refresh it
            if response.status_code == 401:
                print("[salesforce_refresh] Access token is invalid or expired, refreshing...")
                new_token, new_instance_url = refresh_salesforce_token(org_alias=org_alias)
                
                if new_token:
                    token = new_token
                    headers["Authorization"] = f"Bearer {token}"
                    if new_instance_url and new_instance_url != instance_url:
                        instance_url = new_instance_url
                    print("[salesforce_refresh] Using refreshed token for requests")
                else:
                    print("[salesforce_refresh] Failed to refresh token")
                    verify_credentials()
                    return None, None
            else:
                print("[salesforce_refresh] Current token is valid")
        
        except requests.exceptions.RequestException as e:
            print(f"[salesforce_refresh] Error checking token: {e}")
            # Try refreshing the token
            new_token, new_instance_url = refresh_salesforce_token(org_alias=org_alias)
            if new_token:
                token = new_token
                headers["Authorization"] = f"Bearer {token}"
                if new_instance_url and new_instance_url != instance_url:
                    instance_url = new_instance_url
            else:
                verify_credentials()
                return None, None
    else:
        print("[salesforce_refresh] Skipping token validation (using new token from CLI)")
    
    # Cache the token with a 1 hour 45 minutes expiry (conservative for 2 hour token)
    _token_cache["token"] = token
    _token_cache["instance_url"] = instance_url
    _token_cache["expiry"] = current_time + (105 * 60)  # 1 hour 45 minutes in seconds
    
    return headers, instance_url

def salesforce_request(method, endpoint, data=None, params=None, org_alias='louisfirm'):
    """
    User-friendly function to make a Salesforce API request with automatic authentication.
    
    Args:
        method (str): HTTP method (get, post, patch, delete)
        endpoint (str): API endpoint (e.g., '/services/data/v63.0/query')
        data (dict, optional): Request payload for POST/PATCH requests
        params (dict, optional): URL parameters for GET requests
        org_alias (str): The alias of the Salesforce org to use with CLI
        
    Returns:
        dict or None: Response data or None if request failed
    """
    headers, instance_url = get_salesforce_headers(org_alias)
    
    if not headers or not instance_url:
        print("[salesforce_refresh] Unable to get valid authentication for Salesforce request")
        return None
    
    url = f"{instance_url}{endpoint}"
    
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
            print(f"[salesforce_refresh] Unsupported HTTP method: {method}")
            return None
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"[salesforce_refresh] Error making {method.upper()} request to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[salesforce_refresh] Response status: {e.response.status_code}")
            print(f"[salesforce_refresh] Response body: {e.response.text}")
        return None

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
                print(f"[salesforce_refresh] Script is not executable. Attempting to add execute permission.")
                # Add execute permission (chmod +x)
                os.chmod(script_path, os.stat(script_path).st_mode | 0o111)
                print(f"[salesforce_refresh] Execute permission added to {script_path}")
        except Exception as e:
            print(f"[salesforce_refresh] Error setting file permissions: {e}")

if __name__ == "__main__":
    # Check file permissions for Linux environments
    ensure_file_permissions()
    
    print("\n================== Salesforce CLI Path Check ==================\n")
    sf_path = find_sf_cli()
    
    # Determine platform
    is_windows = sys.platform.startswith('win')
    
    if sf_path:
        print(f"✅ Found Salesforce CLI at: {sf_path}")
    else:
        print("❌ Could not find Salesforce CLI automatically")
        print("If you know where sf is installed, set the SF_CLI_PATH environment variable:")
        
        if is_windows:
            print("For PowerShell: $env:SF_CLI_PATH = 'C:\\path\\to\\sf.exe'")
            print("For CMD: set SF_CLI_PATH=C:\\path\\to\\sf.exe")
            print("\nAttempting to find sf using 'where' command...")
            cmd = 'where'
        else:
            print("For bash/zsh: export SF_CLI_PATH=/path/to/sf")
            print("\nAttempting to find sf using 'which' command...")
            cmd = 'which'
        
        try:
            result = subprocess.run([cmd, 'sf'], capture_output=True, text=True)
            if result.returncode == 0:
                paths = result.stdout.strip().split('\n')
                print(f"Found potential paths with '{cmd} sf':")
                for path in paths:
                    print(f"  - {path}")
                print("\nTo use one of these paths, set SF_CLI_PATH environment variable:")
                
                if is_windows:
                    print(f"  For PowerShell: $env:SF_CLI_PATH = '{paths[0]}'")
                    print(f"  For CMD: set SF_CLI_PATH={paths[0]}")
                else:
                    print(f"  For bash/zsh: export SF_CLI_PATH={paths[0]}")
                
                # Try to use it anyway by setting it in the current process
                if paths and os.path.exists(paths[0]):
                    print(f"\nAutomatically using found path: {paths[0]}")
                    SF_CLI_PATH = paths[0]
                    sf_path = paths[0]  # Update sf_path for the check below
            else:
                print(f"❌ '{cmd} sf' command did not find any paths")
        except Exception as e:
            print(f"❌ Failed to run '{cmd} sf' command: {e}")
            
    # If sf is in PATH but Python can't see it, display debugging info
    if not sf_path:
        print("\n================== PATH Debugging Info ==================\n")
        print("System PATH variable:")
        for path in os.environ.get('PATH', '').split(os.pathsep):
            print(f"  - {path}")
        
        if not is_windows:
            print("\nFor Linux/Unix systems, make sure:")
            print("1. Salesforce CLI is installed: npm install -g @salesforce/cli")
            print("2. The installation directory is in your PATH")
            print("3. The sf executable has execute permissions (chmod +x /path/to/sf)")
            print("4. Try creating a symlink: sudo ln -s /path/to/sf /usr/local/bin/sf")
    
    print("\n================== Salesforce CLI Authentication Check ==================\n")
    cli_available = verify_credentials()
    
    if cli_available:
        print("\n================== Attempting Token Refresh ==================\n")
        token, instance_url = refresh_salesforce_token()
        
        if token:
            print(f"\n✅ Authentication successful!")
            print(f"New token: {token[:10]}... (Token refreshed successfully)")
            print(f"Instance URL: {instance_url}")
            
            # Test sample query
            print("\n================== Testing API Query ==================\n")
            query = "SELECT Id, Name FROM Account LIMIT 5"
            result = salesforce_request('get', '/services/data/v63.0/query', params={'q': query})
            
            if result:
                print(f"✅ API query successful! Found {result.get('totalSize', 0)} records")
            else:
                print("❌ API query failed")
        else:
            print("\n❌ Authentication failed")
            print("Please ensure you are logged into Salesforce CLI with: sf org login web")
    else:
        print("\n❌ Salesforce CLI not configured properly")
        print("Please check the issues above and try again") 