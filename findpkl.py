import boto3
import os
from dotenv import load_dotenv
import logging
import tempfile
from pathlib import Path
import glob
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_pkl_files():
    """
    Search S3 bucket and local temp directory for .pkl files and return their locations.
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Bucket name
        bucket_name = "generate-input-f5bef08a-9228-4f8c-a550-56d842b94088"
        
        # List to store pkl file information
        pkl_files = []
        
        # First, check S3
        logger.info("[findpkl] Searching S3 bucket...")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.pkl'):
                    size_mb = obj['Size'] / (1024 * 1024)
                    last_modified = obj['LastModified']
                    
                    pkl_files.append({
                        'location': f"s3://{bucket_name}/{key}",
                        'key': key,
                        'size_mb': round(size_mb, 2),
                        'last_modified': last_modified.strftime('%Y-%m-%d %H:%M:%S'),
                        'source': 'S3'
                    })
        
        # Then, check local temp directory
        logger.info("[findpkl] Searching local temp directory...")
        temp_dir = tempfile.gettempdir()
        logger.info(f"[findpkl] Temp directory location: {temp_dir}")
        
        # Search for .pkl files in temp directory and its subdirectories
        for pkl_path in glob.glob(os.path.join(temp_dir, '**', '*.pkl'), recursive=True):
            size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
            last_modified = os.path.getmtime(pkl_path)
            
            pkl_files.append({
                'location': pkl_path,
                'key': os.path.basename(pkl_path),
                'size_mb': round(size_mb, 2),
                'last_modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified)),
                'source': 'Local Temp'
            })
            
            logger.info(f"[findpkl] Found PKL file:")
            logger.info(f"[findpkl] Location: {pkl_path}")
            logger.info(f"[findpkl] Size: {round(size_mb, 2)} MB")
            logger.info(f"[findpkl] Last Modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified))}")
            logger.info("-" * 80)
        
        return pkl_files
        
    except Exception as e:
        logger.error(f"[findpkl] Error searching for PKL files: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        pkl_files = find_pkl_files()
        
        if not pkl_files:
            logger.info("[findpkl] No PKL files found in S3 or temp directory")
        else:
            logger.info(f"[findpkl] Found {len(pkl_files)} PKL files in total")
            
            # Print summary grouped by source
            logger.info("\nSummary of PKL files found:")
            
            # Group by source
            s3_files = [f for f in pkl_files if f['source'] == 'S3']
            local_files = [f for f in pkl_files if f['source'] == 'Local Temp']
            
            if s3_files:
                logger.info("\nS3 Files:")
                for file in s3_files:
                    logger.info(f"\nFile: {file['key']}")
                    logger.info(f"Size: {file['size_mb']} MB")
                    logger.info(f"Last Modified: {file['last_modified']}")
            
            if local_files:
                logger.info("\nLocal Temp Files:")
                for file in local_files:
                    logger.info(f"\nFile: {file['key']}")
                    logger.info(f"Size: {file['size_mb']} MB")
                    logger.info(f"Last Modified: {file['last_modified']}")
                    logger.info(f"Full Path: {file['location']}")
                
    except Exception as e:
        logger.error(f"[findpkl] Script execution failed: {str(e)}")
        exit(1)
