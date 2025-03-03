#!/usr/bin/env python3

import os
import requests
import json
import subprocess
import sys
import shutil
from dotenv import load_dotenv, dotenv_values
import time

# Load environment variables from .env file at the start
load_dotenv()

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
        print(f"[salesforce_refresh] Verified CLI path exists: {SF_CLI_PATH}")
        return SF_CLI_PATH
    
    # Determine platform
    is_windows = sys.platform.startswith('win')
    print(f"[salesforce_refresh] Detected platform: {'Windows' if is_windows else 'Unix/Linux'}")
    
    # Check common executable names first
    cli_names = ['sf', 'sfdx']
    if is_windows:
        # On Windows, check both .cmd and .exe extensions
        cli_names = ['sf.cmd', 'sf.exe', 'sf', 'sfdx.cmd', 'sfdx.exe', 'sfdx']
    
    # First try the PATH environment for these executable names
    for name in cli_names:
        cli_path = shutil.which(name)
        if cli_path:
            print(f"[salesforce_refresh] Found CLI in PATH: {cli_path}")
            SF_CLI_PATH = cli_path
            return cli_path
    
    # If not found in PATH, search common installation locations
    possible_paths = []
    
    if is_windows:
        # Windows-specific paths
        npm_paths = [
            os.path.expanduser("~\\AppData\\Roaming\\npm\\sf.cmd"),
            os.path.expanduser("~\\AppData\\Roaming\\npm\\sfdx.cmd"),
            "C:\\Program Files\\sfdx\\bin\\sf.cmd",
            "C:\\Program Files\\sfdx\\bin\\sf.exe",
            os.path.expanduser("~\\AppData\\Local\\sfdx\\bin\\sf.cmd"),
            os.path.expanduser("~\\AppData\\Local\\sfdx\\bin\\sf.exe"),
            os.path.expanduser("~\\AppData\\Local\\sf\\bin\\sf.cmd"),
            os.path.expanduser("~\\AppData\\Local\\sf\\bin\\sf.exe"),
            # Node modules paths
            os.path.expanduser("~\\AppData\\Roaming\\npm\\node_modules\\@salesforce\\cli\\bin\\sf.js"),
            os.path.expanduser("~\\AppData\\Roaming\\npm\\node_modules\\sfdx-cli\\bin\\sfdx.js")
        ]
        possible_paths.extend(npm_paths)
    else:
        # Unix/Linux paths
        unix_paths = [
            "/usr/local/bin/sf",
            "/usr/bin/sf",
            "/usr/local/bin/sfdx",
            "/usr/bin/sfdx",
            os.path.expanduser("~/.npm-global/bin/sf"),
            os.path.expanduser("~/.npm-global/bin/sfdx"),
            # Node.js paths on Unix
            os.path.expanduser("~/.npm-global/lib/node_modules/@salesforce/cli/bin/sf.js"),
            os.path.expanduser("~/.npm-global/lib/node_modules/sfdx-cli/bin/sfdx.js")
        ]
        possible_paths.extend(unix_paths)
    
    # Check each location
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[salesforce_refresh] Found CLI at: {path}")
            SF_CLI_PATH = path
            return path
    
    # If not found through direct paths, try running commands to locate it
    try:
        if is_windows:
            print("[salesforce_refresh] Trying to locate CLI using Windows commands...")
            # Try with PowerShell first (more reliable)
            try:
                result = subprocess.run(
                    ['powershell', '-Command', "Get-Command sf.cmd -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path"],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    sf_path = result.stdout.strip()
                    print(f"[salesforce_refresh] PowerShell found CLI at: {sf_path}")
                    if os.path.exists(sf_path):
                        SF_CLI_PATH = sf_path
                        return sf_path
            except Exception as ps_error:
                print(f"[salesforce_refresh] PowerShell command failed: {ps_error}")
                
            # Fall back to CMD where command
            for cmd_name in ['sf.cmd', 'sf.exe', 'sfdx.cmd', 'sfdx.exe']:
                try:
                    result = subprocess.run(
                        ['where', cmd_name], 
                        capture_output=True, text=True, timeout=10
                    )
                    
                    if result.returncode == 0:
                        # Take the first path found
                        sf_path = result.stdout.strip().split('\n')[0]
                        print(f"[salesforce_refresh] CMD found CLI at: {sf_path}")
                        if os.path.exists(sf_path):
                            SF_CLI_PATH = sf_path
                            return sf_path
                except Exception as cmd_error:
                    print(f"[salesforce_refresh] CMD 'where {cmd_name}' failed: {cmd_error}")
                    
            # Last resort: try to find Node.js and run with node directly
            try:
                node_path = shutil.which('node')
                if node_path:
                    print(f"[salesforce_refresh] Found Node.js at: {node_path}")
                    # Common paths for salesforce CLI JavaScript files
                    js_paths = [
                        os.path.expanduser("~\\AppData\\Roaming\\npm\\node_modules\\@salesforce\\cli\\bin\\sf.js"),
                        os.path.expanduser("~\\AppData\\Roaming\\npm\\node_modules\\sfdx-cli\\bin\\sfdx.js")
                    ]
                    
                    for js_path in js_paths:
                        if os.path.exists(js_path):
                            print(f"[salesforce_refresh] Found CLI JS file at: {js_path}")
                            # Use node to run the JS file directly
                            SF_CLI_PATH = f"{node_path} {js_path}"
                            return SF_CLI_PATH
            except Exception as node_error:
                print(f"[salesforce_refresh] Error checking for Node.js: {node_error}")
                
        else:
            # Unix/Linux which command
            for cmd_name in ['sf', 'sfdx']:
                try:
                    result = subprocess.run(['which', cmd_name], capture_output=True, text=True)
                    if result.returncode == 0:
                        sf_path = result.stdout.strip()
                        print(f"[salesforce_refresh] Unix 'which' found CLI at: {sf_path}")
                        if os.path.exists(sf_path):
                            SF_CLI_PATH = sf_path
                            return sf_path
                except Exception as e:
                    print(f"[salesforce_refresh] Unix 'which {cmd_name}' failed: {e}")
    except Exception as e:
        print(f"[salesforce_refresh] Error trying to locate CLI: {e}")
    
    print("[salesforce_refresh] Could not find Salesforce CLI automatically")
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
        # On Windows, shell=True can help resolve PATH issues
        is_windows = sys.platform.startswith('win')
        use_shell = is_windows
        
        # Check if we need to use Node.js to run the CLI
        use_node = " " in sf_path and sf_path.startswith("node")
        
        if use_node:
            # For Node.js execution, don't modify the command
            cmd = sf_path + " " + " ".join(args)
            print(f"[salesforce_refresh] Running Node.js command: {cmd}")
        # For Windows with shell=True, we need to pass a single string instead of a list
        elif use_shell:
            # Properly quote the path if it contains spaces
            quoted_path = f'"{sf_path}"' if " " in sf_path else sf_path
            cmd = f'{quoted_path} {" ".join(args)}'
            print(f"[salesforce_refresh] Running Windows command: {cmd}")
        else:
            cmd = [sf_path] + args
            print(f"[salesforce_refresh] Running Unix command: {' '.join(cmd)}")
            
        # Execute the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=use_shell,  # Use shell on Windows
            timeout=60  # Add timeout to prevent hanging
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out after 60 seconds"
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        return 1, "", f"Error running Salesforce CLI: {e}"

def get_cli_credentials(org_alias='louisfirm'):
    """
    Get the Salesforce access token from the CLI.
    
    Args:
        org_alias (str): The Salesforce org alias to use
        
    Returns:
        tuple: (token, instance_url) or (None, None) if not found
    """
    try:
        print(f"[salesforce_refresh] Attempting to get credentials from Salesforce CLI for org '{org_alias}'")
        
        # First check if we have any orgs available
        returncode, stdout, stderr = run_sf_command(['org', 'list', '--json'])
        
        if returncode != 0:
            print(f"[salesforce_refresh] Failed to list orgs: {stderr}")
            return None, None
        
        try:
            orgs_data = json.loads(stdout)
            
            # Check if the specified alias exists
            available_orgs = orgs_data.get('result', {}).get('nonScratchOrgs', [])
            aliases = [org.get('alias') for org in available_orgs if org.get('alias')]
            
            if not aliases:
                print("[salesforce_refresh] No org aliases found in Salesforce CLI")
                return None, None
                
            if org_alias not in aliases:
                print(f"[salesforce_refresh] Warning: Org alias '{org_alias}' not found in CLI")
                print(f"[salesforce_refresh] Available aliases: {', '.join(aliases)}")
                
                # If alias not found but we have at least one org, use the first one with an alias
                first_alias = next((org.get('alias') for org in available_orgs if org.get('alias')), None)
                if first_alias:
                    print(f"[salesforce_refresh] Using available org alias '{first_alias}' instead")
                    org_alias = first_alias
                else:
                    print("[salesforce_refresh] No usable org aliases found")
                    return None, None
        except json.JSONDecodeError:
            print(f"[salesforce_refresh] Failed to parse org list output as JSON")
            print(f"[salesforce_refresh] Output: {stdout}")
            return None, None
        
        # Check if the token for the alias (now verified to exist) is available
        if org_alias:
            try:
                # Try the modern `sf org display` command first
                returncode, stdout, stderr = run_sf_command(['org', 'display', '--json', '--target-org', org_alias])
                
                if returncode == 0:
                    data = json.loads(stdout)
                    if 'result' in data:
                        access_token = data['result'].get('accessToken')
                        instance_url = data['result'].get('instanceUrl')
                        
                        if access_token and instance_url:
                            print(f"[salesforce_refresh] Successfully retrieved token from CLI for org '{org_alias}'")
                            
                            # Update .env file with the token for later use
                            update_env_file("SALESFORCE_ACCESS_TOKEN", access_token)
                            update_env_file("SALESFORCE_INSTANCE_URL", instance_url)
                            
                            return access_token, instance_url
                else:
                    print(f"[salesforce_refresh] Failed to get org details: {stderr}")
                    print("[salesforce_refresh] Trying alternate command format...")
                    
                    # Try the older format in case user has sfdx CLI
                    returncode, stdout, stderr = run_sf_command(['force:org:display', '--json', '--targetusername', org_alias])
                    
                    if returncode == 0:
                        data = json.loads(stdout)
                        if 'result' in data:
                            access_token = data['result'].get('accessToken')
                            instance_url = data['result'].get('instanceUrl')
                            
                            if access_token and instance_url:
                                print(f"[salesforce_refresh] Successfully retrieved token using older sfdx CLI command for org '{org_alias}'")
                                
                                # Update .env file with the token for later use
                                update_env_file("SALESFORCE_ACCESS_TOKEN", access_token)
                                update_env_file("SALESFORCE_INSTANCE_URL", instance_url)
                                
                                return access_token, instance_url
                    else:
                        print(f"[salesforce_refresh] Alternate command also failed: {stderr}")
            except Exception as e:
                print(f"[salesforce_refresh] Error getting org details from CLI: {e}")
        
        print("[salesforce_refresh] Could not retrieve valid token from Salesforce CLI")
        return None, None
    except Exception as e:
        print(f"[salesforce_refresh] Error getting CLI credentials: {e}")
        return None, None

def print_diagnostics():
    """
    Print detailed diagnostics about the environment to help debug authentication issues.
    This is useful when running in server/headless environments.
    """
    print("\n========== SALESFORCE AUTHENTICATION DIAGNOSTICS ==========")
    
    # Check Python version
    import sys
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check environment
    import os
    print("\nEnvironment variables:")
    env_vars_to_check = [
        "SALESFORCE_ACCESS_TOKEN", 
        "SALESFORCE_INSTANCE_URL",
        "SALESFORCE_USERNAME",
        "SALESFORCE_PASSWORD",
        "MS_TENANT_ID",
        "MS_CLIENT_ID",
        "MS_CLIENT_SECRET"
    ]
    
    for var in env_vars_to_check:
        if var in os.environ:
            # Mask the value for security
            val = os.environ[var]
            if len(val) > 8:
                masked_val = val[:4] + "*" * (len(val) - 8) + val[-4:]
            else:
                masked_val = "****"
            print(f"  {var}: {masked_val}")
        else:
            print(f"  {var}: Not set")
    
    # Check for .env file
    print("\n.env file:")
    if os.path.exists(".env"):
        print("  .env file exists")
        try:
            from dotenv import dotenv_values
            env_values = dotenv_values(".env")
            for key in env_vars_to_check:
                if key in env_values:
                    val = env_values[key]
                    if len(val) > 8:
                        masked_val = val[:4] + "*" * (len(val) - 8) + val[-4:]
                    else:
                        masked_val = "****"
                    print(f"  {key} in .env: {masked_val}")
        except Exception as e:
            print(f"  Error reading .env file: {str(e)}")
    else:
        print("  .env file does not exist")
    
    # Check SF CLI
    print("\nSalesforce CLI:")
    sf_path = find_sf_cli()
    if sf_path:
        print(f"  SF CLI found at: {sf_path}")
        try:
            use_shell = sys.platform.startswith('win')
            if use_shell:
                cmd = f'"{sf_path}" --version'
            else:
                cmd = [sf_path, "--version"]
                
            result = subprocess.run(cmd, capture_output=True, text=True, shell=use_shell)
            
            if result.returncode == 0:
                print(f"  SF CLI version: {result.stdout.strip()}")
            else:
                print(f"  Error getting SF CLI version: {result.stderr}")
        except Exception as e:
            print(f"  Error running SF CLI: {str(e)}")
    else:
        print("  SF CLI not found")
    
    # Check network connectivity
    print("\nNetwork connectivity:")
    try:
        import socket
        socket.create_connection(("louisfirm.my.salesforce.com", 443), timeout=5)
        print(" Can connect to Salesforce (louisfirm.my.salesforce.com:443)")
    except Exception as e:
        print(f"  Cannot connect to Salesforce: {str(e)}")
    
    try:
        socket.create_connection(("graph.microsoft.com", 443), timeout=5)
        print("  Can connect to Microsoft Graph API (graph.microsoft.com:443)")
    except Exception as e:
        print(f"  Cannot connect to Microsoft Graph API: {str(e)}")
    
    # Check for ports in use
    print("\nPorts in use:")
    oauth_ports = [1717, 1718, 1719, 1720, 1721]
    for port in oauth_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("localhost", port))
            sock.close()
            print(f"  Port {port} is available")
        except Exception:
            print(f"  Port {port} is in use or unavailable")
    
    print("\n========== END DIAGNOSTICS ==========\n")

def authenticate_with_username_password():
    """
    Authenticates with Salesforce using username-password flow.
    Returns a tuple of (token, instance_url) if successful, otherwise (None, None).
    """
    try:
        # Get credentials, checking the environment first
        client_id = os.environ.get('SALESFORCE_CLIENT_ID')
        client_secret = os.environ.get('SALESFORCE_CLIENT_SECRET')
        username = os.environ.get('SALESFORCE_USERNAME')
        password = os.environ.get('SALESFORCE_PASSWORD')
        security_token = os.environ.get('SALESFORCE_SECURITY_TOKEN', '')
        
        # If not in environment, try reading from .env file
        if not all([client_id, client_secret, username, password]):
            print("[salesforce_refresh] Some credentials not found in environment, trying .env file")
            load_dotenv()
            client_id = os.environ.get('SALESFORCE_CLIENT_ID') or dotenv_values().get('SALESFORCE_CLIENT_ID')
            client_secret = os.environ.get('SALESFORCE_CLIENT_SECRET') or dotenv_values().get('SALESFORCE_CLIENT_SECRET')
            username = os.environ.get('SALESFORCE_USERNAME') or dotenv_values().get('SALESFORCE_USERNAME')
            password = os.environ.get('SALESFORCE_PASSWORD') or dotenv_values().get('SALESFORCE_PASSWORD')
            security_token = os.environ.get('SALESFORCE_SECURITY_TOKEN') or dotenv_values().get('SALESFORCE_SECURITY_TOKEN', '')
        
        # Print debug info about which credentials were found
        print(f"[salesforce_refresh] Username: {username or 'Not found'}")
        print(f"[salesforce_refresh] Client ID: {'Found' if client_id else 'Not found'}")
        print(f"[salesforce_refresh] Client Secret: {'Found' if client_secret else 'Not found'}")
        print(f"[salesforce_refresh] Security Token: {'Found' if security_token else 'Not found (may be required)'}")
        
        if not all([client_id, client_secret, username, password]):
            missing = []
            if not client_id: missing.append("SALESFORCE_CLIENT_ID")
            if not client_secret: missing.append("SALESFORCE_CLIENT_SECRET")
            if not username: missing.append("SALESFORCE_USERNAME")
            if not password: missing.append("SALESFORCE_PASSWORD")
            
            print(f"[salesforce_refresh] Missing required credentials: {', '.join(missing)}")
            return None, None
        
        # Security token warning if not provided
        if not security_token:
            print("[salesforce_refresh] WARNING: Security token not provided! This may cause authentication to fail.")
            print("[salesforce_refresh] Security token is required unless your IP address is whitelisted in Salesforce.")
            print("[salesforce_refresh] You can reset your security token in Salesforce: Settings > My Personal Information > Reset Security Token")
        
        # Try multiple possible URLs
        urls_to_try = [
            f"os.environ.get('SALESFORCE_INSTANCE_URL')",
        ]
        
        # Remove any duplicate URLs
        urls_to_try = list(set(urls_to_try))
        
        for url in urls_to_try:
            # First try with security token
            if security_token:
                pwd_with_token = f"{password}{security_token}"
                print(f"[salesforce_refresh] Trying authentication with security token to: {url}")
                
                data = {
                    'grant_type': 'password',
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'username': username,
                    'password': pwd_with_token
                }
                
                response = requests.post(url, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    update_env_file("SALESFORCE_ACCESS_TOKEN", result.get('access_token'))
                    return result.get('access_token'), result.get('instance_url')
                    
            # Then try without security token if we didn't succeed
            print(f"[salesforce_refresh] Trying authentication without security token to: {url}")
            
            data = {
                'grant_type': 'password',
                'client_id': client_id,
                'client_secret': client_secret,
                'username': username,
                'password': password
            }
            
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('access_token'), result.get('instance_url')
            
            print(f"[salesforce_refresh] Authentication failed with status code {response.status_code} for {url}")
            try:
                error_info = response.json()
                print(f"[salesforce_refresh] Response: {error_info}")
                
                # Provide more helpful information based on error code
                if 'error' in error_info:
                    if error_info['error'] == 'invalid_grant':
                        if 'authentication failure' in error_info.get('error_description', ''):
                            print("[salesforce_refresh] Authentication failure. Possible causes:")
                            print("  - Incorrect username or password")
                            print("  - Missing or incorrect security token (if your IP is not whitelisted)")
                            print("  - Account is locked or has IP restrictions")
                    elif error_info['error'] == 'invalid_client_id':
                        print("[salesforce_refresh] Invalid client ID. Please check your Connected App configuration.")
                    elif error_info['error'] == 'inactive_user':
                        print("[salesforce_refresh] User account is inactive. Please contact your Salesforce administrator.")
                
            except Exception as json_err:
                print(f"[salesforce_refresh] Failed to parse error response: {response.text}")
        
        print("[salesforce_refresh] All authentication URLs failed")
        return None, None
    except Exception as e:
        print(f"[salesforce_refresh] Error in username-password authentication: {e}")
        return None, None

def is_headless_environment():
    """
    Detect if running in a headless environment (no display/GUI).
    This is important for server environments where browser-based auth won't work.
    
    Returns:
        bool: True if running in a headless environment
    """
    # Check for common server/headless indicators
    if 'SSH_CONNECTION' in os.environ or 'SSH_TTY' in os.environ:
        return True
        
    # For Linux/Unix, check for DISPLAY environment variable
    if not sys.platform.startswith('win') and 'DISPLAY' not in os.environ:
        return True
        
    # For all platforms, check if we're running in a known CI/CD environment
    ci_env_vars = ['CI', 'JENKINS_URL', 'TRAVIS', 'CIRCLECI', 'GITHUB_ACTIONS', 'GITLAB_CI']
    for var in ci_env_vars:
        if var in os.environ:
            return True
            
    return False

def refresh_salesforce_token(org_alias='louisfirm'):
    """
    Refreshes the Salesforce token using the Salesforce CLI.
    Returns a tuple of (token, instance_url) if successful, otherwise (None, None).
    """
    try:
        # Try to get the token via CLI first
        token, instance_url = get_cli_credentials(org_alias)
        if token and instance_url:
            return token, instance_url
        
        # If token not found in CLI, try to login again
        if not is_headless_environment():
            print("[salesforce_refresh] Token not found in CLI, attempting to refresh CLI authentication")
            try:
                # Removed the --oauth-local-port parameter which isn't supported in newer CLI versions
                login_command = [
                    'org', 'login', 'web',
                    '--instance-url', 'os.environ.get("SALESFORCE_INSTANCE_URL")',
                    '--alias', org_alias,
                    '--set-default'
                ]
                
                returncode, stdout, stderr = run_sf_command(login_command)
                
                if returncode == 0:
                    print("[salesforce_refresh] CLI login successful, attempting to get token")
                    return get_cli_credentials(org_alias)
                else:
                    if "Nonexistent flag: --oauth-local-port" in stderr:
                        print("[salesforce_refresh] CLI login failed with unsupported flag. Your CLI version doesn't support --oauth-local-port")
                    else:
                        print(f"[salesforce_refresh] CLI login failed: {stderr}")
            except Exception as e:
                print(f"[salesforce_refresh] CLI login failed with error: {e}")
        else:
            print("[salesforce_refresh] Skipping browser-based authentication in headless environment")
    
        # As a fallback, try username-password flow
        print("[salesforce_refresh] Falling back to username-password authentication")
        return authenticate_with_username_password()
    except Exception as e:
        print(f"[salesforce_refresh] Error refreshing token: {e}")
        return None, None

def update_env_file(key, value):
    """
    Update a key-value pair in the .env file
    
    Args:
        key (str): The environment variable name
        value (str): The value to set
    """
    if not os.path.exists(".env"):
        print(f"[salesforce_refresh] Creating new .env file")
        with open(".env", "w") as f:
            f.write(f"{key}={value}\n")
        return
    
    # Read the current .env file
    with open(".env", "r") as f:
        lines = f.readlines()
    
    # Check if the key already exists
    key_exists = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            key_exists = True
            break
    
    # If the key doesn't exist, add it
    if not key_exists:
        lines.append(f"{key}={value}\n")
    
    # Write the updated .env file
    with open(".env", "w") as f:
        f.writelines(lines)
    
    print(f"[salesforce_refresh] Updated {key} in .env file")

def verify_credentials():
    """
    Verify if Salesforce credentials are available via CLI or environment.
    Also validates CLI operation on the current system.
    
    Returns:
        bool: True if credentials are available and CLI works, False otherwise
    """
    # First check if CLI is available at all
    sf_path = find_sf_cli()
    if not sf_path:
        print("[salesforce_refresh] Salesforce CLI not found")
        return False
    
    # Test CLI operation with a simple command
    try:
        print(f"[salesforce_refresh] Testing CLI operation with '--version' command...")
        cmd = None
        is_windows = sys.platform.startswith('win')
        
        # Check if we're using Node.js to run CLI
        if " " in sf_path and sf_path.startswith("node"):
            # Node.js execution
            cmd = f"{sf_path} --version"
            shell = True
        elif is_windows:
            # Windows execution
            cmd = f'"{sf_path}" --version' if " " in sf_path else f'{sf_path} --version'
            shell = True
        else:
            # Unix execution
            cmd = [sf_path, "--version"]
            shell = False
        
        print(f"[salesforce_refresh] Running CLI test: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=shell,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"[salesforce_refresh] CLI test successful: {result.stdout.strip()}")
            cli_works = True
        else:
            print(f"[salesforce_refresh] CLI test failed: {result.stderr}")
            cli_works = False
    except Exception as e:
        print(f"[salesforce_refresh] CLI test failed with exception: {e}")
        cli_works = False
    
    # Now check if we already have a token via CLI
    if cli_works:
        print(f"[salesforce_refresh] CLI works, checking for existing org connection...")
        try:
            returncode, stdout, stderr = run_sf_command(['org', 'list', '--json'])
            if returncode == 0:
                try:
                    data = json.loads(stdout)
                    orgs = data.get('result', {}).get('nonScratchOrgs', [])
                    if orgs:
                        print(f"[salesforce_refresh] Found {len(orgs)} connected Salesforce orgs")
                        return True
                    else:
                        print("[salesforce_refresh] No orgs connected to CLI")
                except json.JSONDecodeError:
                    print(f"[salesforce_refresh] Could not parse CLI output as JSON")
            else:
                print(f"[salesforce_refresh] CLI org list failed: {stderr}")
        except Exception as e:
            print(f"[salesforce_refresh] Error checking CLI orgs: {e}")
    
    # Fall back to checking environment variables if CLI is not available
    username = os.getenv("SALESFORCE_USERNAME") 
    password = os.getenv("SALESFORCE_PASSWORD")
    instance_url = os.getenv("SALESFORCE_INSTANCE_URL")
    client_id = os.getenv("SALESFORCE_CLIENT_ID")
    client_secret = os.getenv("SALESFORCE_CLIENT_SECRET")
    
    # Also check .env file
    if (not username or not password) and os.path.exists(".env"):
        try:
            from dotenv import dotenv_values
            env_values = dotenv_values(".env")
            if "SALESFORCE_USERNAME" in env_values and "SALESFORCE_PASSWORD" in env_values:
                print("[salesforce_refresh] Username and password found in .env file")
                return True
            if "SALESFORCE_INSTANCE_URL" in env_values and "SALESFORCE_CLIENT_ID" in env_values and "SALESFORCE_CLIENT_SECRET" in env_values:
                print("[salesforce_refresh] Instance URL, client ID, and client secret found in .env file")
                return True
        except Exception as e:
            print(f"[salesforce_refresh] Error reading .env file: {e}")
    
    if username and password and instance_url and client_id and client_secret:
        print("[salesforce_refresh] Username and password:   environment variables")
        print(f"[salesforce_refresh] Instance URL: TRUE")
        print(f"[salesforce_refresh] Client ID: TRUE")
        print(f"[salesforce_refresh] Client Secret: TRUE")
        return True
    else:
        print(f"[salesforce_refresh] Username: {username}")
        print(f"[salesforce_refresh] Password: {password}")
        print(f"[salesforce_refresh] Instance URL: {instance_url}")
        print(f"[salesforce_refresh] Client ID: {client_id}")
        print(f"[salesforce_refresh] Client Secret: {client_secret}")
        return False

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

def salesforce_request(method, url, headers=None, params=None, data=None, json_data=None, timeout=30):
    """
    Make a request to Salesforce API with proper authentication and error handling.
    
    Args:
        method (str): HTTP method ('get', 'post', etc.)
        url (str): Full URL or path (if path, instance_url will be prepended)
        headers (dict, optional): HTTP headers. Authorization will be added if not present.
        params (dict, optional): Query parameters.
        data (dict, optional): Form data.
        json_data (dict, optional): JSON data (for POST/PATCH).
        timeout (int, optional): Request timeout in seconds.
        
    Returns:
        dict or None: JSON response data or None if request failed.
    """
    import requests
    
    # Initialize headers if not provided
    if headers is None:
        # Get cached token or refresh if needed
        token, instance_url = None, None
        
        # Try to get from environment variables first
        token = os.getenv("SALESFORCE_ACCESS_TOKEN")
        instance_url = os.getenv("SALESFORCE_INSTANCE_URL")
        
        # If not in env vars, try to get from .env file directly
        if not token or not instance_url:
            try:
                from dotenv import dotenv_values
                env_values = dotenv_values(".env")
                if not token and "SALESFORCE_ACCESS_TOKEN" in env_values:
                    token = env_values["SALESFORCE_ACCESS_TOKEN"]
                if not instance_url and "SALESFORCE_INSTANCE_URL" in env_values:
                    instance_url = env_values["SALESFORCE_INSTANCE_URL"]
            except Exception as e:
                print(f"[salesforce_refresh] Error reading .env file: {e}")
        
        # If still no token, try to refresh
        if not token or not instance_url:
            token, instance_url = refresh_salesforce_token()
            if not token or not instance_url:
                print("[salesforce_refresh] Failed to get Salesforce token")
                return None
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    # If URL is a path, prepend instance_url
    if not url.startswith('http'):
        # Find instance_url in headers or use from prior steps
        instance_url = None
        
        # Try to extract from Authorization header
        if headers and 'Authorization' in headers:
            auth_parts = headers['Authorization'].split()
            if len(auth_parts) == 2 and auth_parts[0].lower() == 'bearer':
                # This is a token, so we need an instance URL
                instance_url = os.getenv("SALESFORCE_INSTANCE_URL")
                if not instance_url:
                    try:
                        from dotenv import dotenv_values
                        env_values = dotenv_values(".env")
                        if "SALESFORCE_INSTANCE_URL" in env_values:
                            instance_url = env_values["SALESFORCE_INSTANCE_URL"]
                    except Exception:
                        pass
        
        if not instance_url:
            print("[salesforce_refresh] No instance URL found for request")
            return None
            
        url = f"{instance_url.rstrip('/')}/{url.lstrip('/')}"
    
    # Make the request with proper error handling
    try:
        method = method.lower()
        
        if method == 'get':
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
        elif method == 'post':
            response = requests.post(url, headers=headers, params=params, data=data, json=json_data, timeout=timeout)
        elif method == 'patch':
            response = requests.patch(url, headers=headers, params=params, data=data, json=json_data, timeout=timeout)
        elif method == 'delete':
            response = requests.delete(url, headers=headers, params=params, timeout=timeout)
        else:
            print(f"[salesforce_refresh] Unsupported HTTP method: {method}")
            return None
        
        # Check if request was successful
        if response.status_code >= 200 and response.status_code < 300:
            if response.content:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    print(f"[salesforce_refresh] Error parsing response as JSON: {response.text}")
                    return response.text
            return {}
        else:
            print(f"[salesforce_refresh] API request failed with status code: {response.status_code}")
            print(f"[salesforce_refresh] Response: {response.text}")
            
            # Check for specific error types
            if response.status_code == 401:
                print("[salesforce_refresh] Authentication failed. Token may have expired.")
                # Clear cached token to force a refresh next time
                if os.getenv("SALESFORCE_ACCESS_TOKEN"):
                    os.environ.pop("SALESFORCE_ACCESS_TOKEN")
            
            return None
    except requests.RequestException as e:
        print(f"[salesforce_refresh] Request error: {e}")
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
    try:
        print("\n================== Salesforce CLI Path Check ==================\n")
        
        sf_path = find_sf_cli()
        if sf_path:
            print(f"✅ Found Salesforce CLI at: {sf_path}")
            
            # On Linux, check permissions
            if not sys.platform.startswith('win'):
                try:
                    # Check if the file is executable
                    is_executable = os.access(sf_path, os.X_OK)
                    print(f"CLI executable permissions: {'Yes' if is_executable else 'No'}")
                    
                    if not is_executable and os.path.exists(sf_path):
                        print(f"Attempting to make CLI executable...")
                        try:
                            os.chmod(sf_path, os.stat(sf_path).st_mode | 0o111)  # Add execute permission
                            print(f"Added execute permission to {sf_path}")
                        except Exception as chmod_error:
                            print(f"Failed to set execute permission: {chmod_error}")
                except Exception as e:
                    print(f"Error checking CLI permissions: {e}")
        else:
            print("❌ Salesforce CLI not found in PATH")
            # Print some debugging information about where we looked
            print("\nSearched the following locations:")
            
            is_windows = sys.platform.startswith('win')
            
            if is_windows:
                common_locations = [
                    os.path.expanduser("~\\AppData\\Roaming\\npm\\sf.cmd"),
                    os.path.expanduser("~\\AppData\\Roaming\\npm\\sfdx.cmd"),
                    "C:\\Program Files\\sfdx\\bin\\sf.cmd",
                    os.path.expanduser("~\\AppData\\Local\\sfdx\\bin\\sf.cmd")
                ]
                
                for loc in common_locations:
                    print(f"  - {loc}: {'Exists' if os.path.exists(loc) else 'Not found'}")
                
                # Try to find it using system commands
                try:
                    cmd = 'where'
                    print(f"\nAttempting to find CLI using '{cmd} sf'...")
                    
                    # Try various CLI names
                    for cli_name in ['sf.cmd', 'sf.exe', 'sfdx.cmd', 'sfdx.exe']:
                        try:
                            result = subprocess.run(['where', cli_name], capture_output=True, text=True, timeout=5)
                            if result.returncode == 0:
                                paths = result.stdout.strip().split('\n')
                                print(f"Found potential paths with 'where {cli_name}':")
                                for path in paths:
                                    print(f"  - {path}")
                                
                                # Try to use it by setting it in the current process
                                if paths and os.path.exists(paths[0]):
                                    print(f"\nAttempting to use: {paths[0]}")
                                    SF_CLI_PATH = paths[0]
                                    sf_path = paths[0]  # Update sf_path for later use
                            else:
                                print(f"'where {cli_name}' did not find any paths")
                        except Exception as cli_error:
                            print(f"Error running 'where {cli_name}': {cli_error}")
                    
                    # Try Node.js directly if we can find it
                    node_path = shutil.which('node')
                    if node_path:
                        print(f"\nFound Node.js at: {node_path}")
                        js_files = [
                            os.path.expanduser("~\\AppData\\Roaming\\npm\\node_modules\\@salesforce\\cli\\bin\\sf.js"),
                            os.path.expanduser("~\\AppData\\Roaming\\npm\\node_modules\\sfdx-cli\\bin\\sfdx.js")
                        ]
                        
                        for js_file in js_files:
                            if os.path.exists(js_file):
                                print(f"Found CLI JS file: {js_file}")
                                SF_CLI_PATH = f"{node_path} {js_file}"
                                sf_path = SF_CLI_PATH  # Update sf_path for later use
                                print(f"Set SF_CLI_PATH to use Node.js directly: {SF_CLI_PATH}")
                except Exception as e:
                    print(f"Error searching for CLI: {e}")
            else:
                common_locations = [
                    "/usr/local/bin/sf", 
                    "/usr/bin/sf",
                    os.path.expanduser("~/.npm-global/bin/sf")
                ]
                
                for loc in common_locations:
                    print(f"  - {loc}: {'Exists' if os.path.exists(loc) else 'Not found'}")
                
                # Try to find using which command
                try:
                    for cmd_name in ['sf', 'sfdx']:
                        result = subprocess.run(['which', cmd_name], capture_output=True, text=True)
                        if result.returncode == 0:
                            path = result.stdout.strip()
                            print(f"Found CLI using 'which {cmd_name}': {path}")
                            SF_CLI_PATH = path
                            sf_path = path  # Update sf_path for later use
                except Exception as e:
                    print(f"Error running 'which' command: {e}")
            
            # Check if we have Node.js and npm available for installation
            node_available = shutil.which('node')
            npm_available = shutil.which('npm')
            
            print(f"\nNode.js available: {'Yes' if node_available else 'No'}")
            print(f"npm available: {'Yes' if npm_available else 'No'}")
            
            if npm_available:
                print("\nYou can install Salesforce CLI with:")
                print("  npm install -g @salesforce/cli")
                
                # Offer to install for them
                try:
                    install_sf = input("\nWould you like to install Salesforce CLI now? (y/n): ")
                    if install_sf.lower() in ['y', 'yes']:
                        print("\nInstalling Salesforce CLI...")
                        subprocess.run(['npm', 'install', '-g', '@salesforce/cli'], check=True)
                        print("\nInstallation complete. Rerunning CLI check...")
                        sf_path = find_sf_cli()  # Check again after installation
                except Exception as install_error:
                    print(f"Error during installation: {install_error}")
            else:
                print("\nTo install Salesforce CLI:")
                if is_windows:
                    print("1. Install Node.js from https://nodejs.org/")
                    print("2. Open Command Prompt as administrator")
                    print("3. Run: npm install -g @salesforce/cli")
                else:
                    print("1. Install Node.js and npm:")
                    print("   - Ubuntu/Debian: sudo apt install nodejs npm")
                    print("   - CentOS/RHEL: sudo yum install nodejs npm")
                    print("   - macOS: brew install node")
                    print("2. Run: npm install -g @salesforce/cli")
        
        print("\n================== Salesforce CLI Authentication Check ==================\n")
        cli_available = verify_credentials()
        
        # Check if there are existing orgs and set the org_alias accordingly
        existing_org_alias = None
        if cli_available:
            try:
                returncode, stdout, stderr = run_sf_command(['org', 'list', '--json'])
                if returncode == 0:
                    data = json.loads(stdout)
                    orgs = data.get('result', {}).get('nonScratchOrgs', [])
                    if orgs:
                        # Get the default org or the first one with an alias
                        for org in orgs:
                            if org.get('alias'):
                                if org.get('isDefaultUsername') or not existing_org_alias:
                                    existing_org_alias = org.get('alias')
                        if existing_org_alias:
                            print(f"[salesforce_refresh] Found existing org with alias: {existing_org_alias}")
                            print(f"[salesforce_refresh] Using this org alias instead of the default 'louisfirm'")
            except Exception as e:
                print(f"[salesforce_refresh] Error checking existing orgs: {e}")
        
        org_alias = existing_org_alias or 'louisfirm'
        
        # For headless environments, try direct authentication first
        if is_headless_environment():
            print("\n================== Detected Headless Environment ==================\n")
            print("Trying direct username-password authentication without relying on CLI...")
            token, instance_url = authenticate_with_username_password()
            
            if token:
                print(f"\n✅ Authentication successful with username-password flow!")
                print(f"New token: {token[:10]}... (Token obtained successfully)")
                print(f"Instance URL: {instance_url}")
                
                # Update .env with the token
                update_env_file("SALESFORCE_ACCESS_TOKEN", token)
                update_env_file("SALESFORCE_INSTANCE_URL", instance_url)
                
                # Test sample query
                print("\n================== Testing API Query ==================\n")
                query = "SELECT Id, Name FROM Account LIMIT 5"
                result = salesforce_request('get', '/services/data/v57.0/query', 
                                            params={'q': query}, 
                                            headers={"Authorization": f"Bearer {token}"})
                
                if result and 'records' in result:
                    print(f"✅ API query successful! Found {len(result.get('records', []))} records")
                else:
                    print("❌ API query failed")
            else:
                print("\n❌ Direct authentication failed, trying CLI-based methods...")
                # Continue with CLI-based authentication attempts
        
        # Try regular CLI-based auth if direct auth failed or we're in a GUI environment
        if cli_available or sf_path:
            print("\n================== Attempting Token Refresh ==================\n")
            token, instance_url = refresh_salesforce_token(org_alias=org_alias)
            
            if token:
                print(f"\n✅ Authentication successful!")
                print(f"New token: {token[:10]}... (Token refreshed successfully)")
                print(f"Instance URL: {instance_url}")
                
                # Test sample query
                print("\n================== Testing API Query ==================\n")
                query = "SELECT Id, Name FROM Account LIMIT 5"
                result = salesforce_request('get', '/services/data/v57.0/query', 
                                           params={'q': query}, 
                                           headers={"Authorization": f"Bearer {token}"})
                
                if result and 'records' in result:
                    print(f"✅ API query successful! Found {len(result.get('records', []))} records")
                else:
                    print("❌ API query failed")
            else:
                print("\n❌ Authentication failed")
                print("Please run one of these commands manually to authenticate:")
                print(f"  1. sf org login web --instance-url os.environ.get('SALESFORCE_INSTANCE_URL') --alias {org_alias} --set-default")
                print("  2. Or add SALESFORCE_USERNAME, SALESFORCE_PASSWORD, and SALESFORCE_SECURITY_TOKEN to your .env file")
        else:
            print("\n❌ Salesforce CLI not configured properly")
            print("Please check the issues above and try again")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        print_diagnostics() 