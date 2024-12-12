import os
import shutil
import logging
import platform

# Configure logging for detailed output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_cache_directories():
    """
    Deletes the cache directories for HuggingFace and Torch.
    Handles different operating systems (Windows, macOS, Linux).
    """
    # Determine the user's home directory in a cross-platform way
    home_dir = os.path.expanduser('~')
    logger.info(f"[clear_cache] Home directory resolved to: {home_dir}")

    # Define cache directory paths
    cache_dirs = {
        'huggingface': '',
        'torch': ''
    }

    # Handle different operating systems
    system = platform.system()
    if system == 'Windows':
        # On Windows, cache directories might be in a different location
        cache_dirs['huggingface'] = os.path.join(home_dir, '.cache', 'huggingface')
        cache_dirs['torch'] = os.path.join(home_dir, '.cache', 'torch')
        cache_dirs['temp'] = os.path.join(home_dir, 'AppData', 'Local', 'Temp', 'tmpu3hj57lo')
    else:
        # For Unix-like systems (Linux, macOS)
        cache_dirs['huggingface'] = os.path.join(home_dir, '.cache', 'huggingface')
        cache_dirs['torch'] = os.path.join(home_dir, '.cache', 'torch')

    # Iterate over cache directories and delete if they exist
    for name, cache_dir in cache_dirs.items():
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"[clear_cache] Deleted {name} cache directory: {cache_dir}")
            except Exception as e:
                logger.error(f"[clear_cache] Failed to delete {name} cache directory: {cache_dir}. Error: {str(e)}")
        else:
            logger.info(f"[clear_cache] {name} cache directory does not exist: {cache_dir}")

if __name__ == '__main__':
    delete_cache_directories()
