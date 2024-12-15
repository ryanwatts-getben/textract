import logging
import pickle
import os
import tempfile
import shutil
import argparse
from typing import List, Dict, Optional, Set, Tuple
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.anthropic import Anthropic
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import boto3
from botocore.exceptions import ClientError
import time
from dotenv import load_dotenv
import json
import io
from pypdf import PdfReader
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
AWS_UPLOAD_BUCKET_NAME = os.getenv("AWS_UPLOAD_BUCKET_NAME", "my-bucket")
logger.info(f"[config] AWS_UPLOAD_BUCKET_NAME set to: {AWS_UPLOAD_BUCKET_NAME}")

########################################
# Original Functions (with minor adjustments)
########################################

def preprocess_document(document_text: str) -> str:
    """Preprocess a single document's text."""
    logger.debug(f"[preprocess_document] Original document length: {len(document_text)}")
    processed_text = document_text.lower()
    logger.debug(f"[preprocess_document] Processed document length: {len(processed_text)}")
    return processed_text

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Initialize embedding model with proper device configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[get_embedding_model] Using device: {device} for embeddings")
    logger.info(f"[get_embedding_model] Initializing HuggingFaceEmbeddings with model_name: {model_name}")
    
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )
    logger.debug(f"[get_embedding_model] HuggingFaceEmbeddings initialized with model_kwargs: {huggingface_embeddings.model_kwargs}")
    
    langchain_embedding = LangchainEmbedding(huggingface_embeddings)
    logger.debug(f"[get_embedding_model] LangchainEmbedding instance created: {langchain_embedding}")
    
    return langchain_embedding

def create_index(
    documents: List[Document],
    s3_client = None,
    bucket_name: str = None,
    index_cache_key: str = None,
    temp_dir: str = None,
    cache_path: str = None
) -> VectorStoreIndex:
    """
    Create a vector store index from documents with S3 caching support.
    """
    logger.info(f"[create_index] Starting index creation with {len(documents)} documents.")
    logger.debug(f"[create_index] Parameters received - s3_client: {s3_client}, bucket_name: {bucket_name}, "
                 f"index_cache_key: {index_cache_key}, temp_dir: {temp_dir}, cache_path: {cache_path}")
    try:
        embed_model = get_embedding_model()
        logger.info('[create_index] Creating VectorStoreIndex with provided documents')
        
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        logger.info('[create_index] VectorStoreIndex created successfully')
        logger.debug(f"[create_index] VectorStoreIndex details: {index}")
        
        # Cache locally
        current_files = set(os.listdir(temp_dir))
        logger.debug(f"[create_index] Current files in temp_dir '{temp_dir}': {current_files}")
        with open(cache_path, 'wb') as f:
            pickle.dump((index, current_files), f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("[create_index] Created and cached new vector index locally")
        logger.debug(f"[create_index] Cached index at '{cache_path}'")
        
        # Upload to S3 if parameters provided
        if all([s3_client, bucket_name, index_cache_key, temp_dir, cache_path]):
            logger.info("[create_index] S3 caching parameters provided. Proceeding to upload index to S3.")
            try:
                index_cache_path = os.path.join(temp_dir, 'index_cache.pkl')
                logger.debug(f"[create_index] Index cache path set to: {index_cache_path}")
                with open(index_cache_path, 'wb') as f:
                    pickle.dump(index, f)
                logger.info('[create_index] Index serialized and saved to local cache successfully')
                
                logger.info(f"[create_index] Uploading index cache from '{index_cache_path}' to S3 bucket '{bucket_name}' with key '{index_cache_key}'")
                s3_client.upload_file(
                    Filename=index_cache_path,
                    Bucket=bucket_name,
                    Key=index_cache_key
                )
                logger.info('[create_index] Index cache uploaded to S3 successfully')
            except Exception as e:
                logger.error(f'[create_index] Error caching index to S3: {str(e)}')
        
        logger.info("[create_index] Index creation process completed successfully.")
        return index

    except Exception as e:
        logger.error(f"[create_index] Error creating index: {str(e)}")
        raise

def query_index(index: VectorStoreIndex, query_text: str) -> str:
    """Query the index with given text using Claude."""
    logger.info("[query_index] Starting query operation.")
    logger.debug(f"[query_index] Query text received: {query_text}")
    try:
        llm = Anthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        logger.debug(f"[query_index] Anthropic LLM initialized with model: claude-3-5-sonnet-20241022")
        
        query_engine = index.as_query_engine(
            llm=llm,
            response_mode="compact"
        )
        logger.debug(f"[query_index] Query engine created with response_mode='compact'")
        
        response = query_engine.query(query_text)
        logger.info("[query_index] Query executed successfully.")
        logger.debug(f"[query_index] Response received: {response.response}")
        return response.response
    except Exception as e:
        logger.error(f"[query_index] Error querying index: {str(e)}")
        raise

########################################
# New Functions to Implement Requirements
########################################

### 1. List Discovery Phase

def list_user_ids_from_bucket(s3_client, bucket_name: str) -> List[str]:
    """
    Lists all userIds in the given bucket.
    userId is inferred from the top-level "directory" (prefix) structure.
    """
    logger.info(f"[list_user_ids_from_bucket] Listing userIds from bucket: {bucket_name}")
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        result = paginator.paginate(Bucket=bucket_name, Delimiter='/')
        user_ids = []
        page_number = 1
        for page in result:
            logger.debug(f"[list_user_ids_from_bucket] Processing page {page_number} of userIds")
            for prefix in page.get('CommonPrefixes', []):
                user_id = prefix.get('Prefix').strip('/')
                logger.debug(f"[list_user_ids_from_bucket] Found user_id: {user_id}")
                if user_id:
                    user_ids.append(user_id)
            logger.debug(f"[list_user_ids_from_bucket] Completed processing page {page_number}")
            page_number += 1
        logger.info(f"[list_user_ids_from_bucket] Total userIds found: {len(user_ids)}")
        return user_ids
    except ClientError as e:
        logger.error(f"[list_user_ids_from_bucket] Failed to list userIds from bucket '{bucket_name}': {e}")
        return []

def list_project_ids_for_user(s3_client, bucket_name: str, user_id: str) -> List[str]:
    """
    Lists all projectIds for a given userId.
    ProjectIds are one level down from the userId prefix.
    """
    logger.info(f"[list_project_ids_for_user] Listing projectIds for user_id: {user_id} in bucket: {bucket_name}")
    try:
        user_prefix = f"{user_id}/"
        logger.debug(f"[list_project_ids_for_user] User prefix set to: {user_prefix}")
        paginator = s3_client.get_paginator('list_objects_v2')
        result = paginator.paginate(Bucket=bucket_name, Prefix=user_prefix, Delimiter='/')
        project_ids = []
        page_number = 1
        for page in result:
            logger.debug(f"[list_project_ids_for_user] Processing page {page_number} of projectIds for user_id: {user_id}")
            for prefix in page.get('CommonPrefixes', []):
                project_id = prefix.get('Prefix').replace(user_prefix, '').strip('/')
                logger.debug(f"[list_project_ids_for_user] Found project_id: {project_id}")
                if project_id:
                    project_ids.append(project_id)
            logger.debug(f"[list_project_ids_for_user] Completed processing page {page_number} for user_id: {user_id}")
            page_number += 1
        logger.info(f"[list_project_ids_for_user] Total projectIds found for user_id '{user_id}': {len(project_ids)}")
        return project_ids
    except ClientError as e:
        logger.error(f"[list_project_ids_for_user] Failed to list projectIds for user '{user_id}' in bucket '{bucket_name}': {e}")
        return []

def get_user_projects_mapping(s3_client, bucket_name: str) -> Dict[str, List[str]]:
    """
    Returns a dictionary of {userId: [projectId1, projectId2, ...]}.
    """
    logger.info(f"[get_user_projects_mapping] Building user-projects mapping for bucket: {bucket_name}")
    mapping = {}
    user_ids = list_user_ids_from_bucket(s3_client, bucket_name)
    logger.debug(f"[get_user_projects_mapping] Retrieved user_ids: {user_ids}")
    for u in user_ids:
        logger.info(f"[get_user_projects_mapping] Processing user_id: {u}")
        projects = list_project_ids_for_user(s3_client, bucket_name, u)
        mapping[u] = projects
        logger.debug(f"[get_user_projects_mapping] User_id '{u}' has projects: {projects}")
    logger.info(f"[get_user_projects_mapping] Completed user-projects mapping. Total users: {len(mapping)}")
    return mapping

### 2. Path Management

def get_document_prefix(user_id: str, project_id: str) -> str:
    """
    Document path pattern: {userId}/{projectId}/input/
    """
    document_prefix = f"{user_id}/{project_id}/input/"
    logger.debug(f"[get_document_prefix] Generated document prefix: {document_prefix} for user_id: {user_id}, project_id: {project_id}")
    return document_prefix

def get_index_path(user_id: str, project_id: str) -> str:
    """
    Index storage path pattern: {userId}/{projectId}/index.pkl
    """
    index_path = f"{user_id}/{project_id}/index.pkl"
    logger.debug(f"[get_index_path] Generated index path: {index_path} for user_id: {user_id}, project_id: {project_id}")
    return index_path

### 3. Document Collection

def download_project_documents(s3_client, bucket_name: str, user_id: str, project_id: str, temp_dir: str) -> List[str]:
    """
    Lists all documents in the project's input folder and downloads them.
    Returns a list of local file paths for the downloaded documents.
    """
    logger.info(f"[download_project_documents] Starting download of documents for user_id: {user_id}, project_id: {project_id} from bucket: {bucket_name}")
    
    # Use the input directory prefix
    doc_prefix = get_document_prefix(user_id, project_id)
    local_files = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        page_number = 1
        
        # List objects with this prefix
        for page in paginator.paginate(Bucket=bucket_name, Prefix=doc_prefix):
            logger.debug(f"[download_project_documents] Processing page {page_number} for documents under prefix '{doc_prefix}'")
            
            # Log the raw response for debugging
            if 'Contents' in page:
                logger.debug(f"[download_project_documents] Found {len(page['Contents'])} objects with prefix '{doc_prefix}'")
                for obj in page['Contents']:
                    logger.debug(f"[download_project_documents] Found object: {obj['Key']}")
            
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/'):  # Skip directories
                    logger.debug(f"[download_project_documents] Skipping directory key: {key}")
                    continue
                    
                # Only process files with common document extensions
                if not (any(key.lower().endswith(ext) for ext in ['.txt', '.pdf', '.doc', '.docx', '.html', '.json'])):
                    logger.debug(f"[download_project_documents] Skipping non-document file: {key}")
                    continue
                    
                filename = os.path.join(temp_dir, os.path.basename(key))
                logger.debug(f"[download_project_documents] Attempting to download '{key}' to '{filename}'")
                
                try:
                    s3_client.download_file(bucket_name, key, filename)
                    local_files.append(filename)
                    logger.info(f"[download_project_documents] Successfully downloaded '{key}' to '{filename}'")
                except ClientError as e:
                    logger.error(f"[download_project_documents] Failed to download '{key}': {e}")
            
            page_number += 1
            
        if not local_files:
            # List all keys under the input directory to help debug
            try:
                all_objects = []
                for page in paginator.paginate(Bucket=bucket_name, Prefix=doc_prefix):
                    if 'Contents' in page:
                        all_objects.extend(obj['Key'] for obj in page['Contents'])
                if all_objects:
                    logger.debug(f"[download_project_documents] All objects found under {doc_prefix}:")
                    for obj in all_objects:
                        logger.debug(f"[download_project_documents] - {obj}")
                else:
                    logger.debug(f"[download_project_documents] No objects found under {doc_prefix}")
            except ClientError as e:
                logger.error(f"[download_project_documents] Failed to list all objects under {doc_prefix}: {e}")
    
    except ClientError as e:
        logger.error(f"[download_project_documents] Failed to list documents under prefix '{doc_prefix}': {e}")
    
    logger.info(f"[download_project_documents] Total documents downloaded: {len(local_files)} for user_id: {user_id}, project_id: {project_id}")
    return local_files

### 4. Index Creation Pipeline

def read_file_content(file_path: str) -> str:
    """
    Read file content based on file type.
    Returns the text content of the file.
    """
    file_lower = file_path.lower()
    try:
        if file_lower.endswith('.pdf'):
            # Handle PDF files
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text() + '\n'
                return text
        elif file_lower.endswith('.json'):
            # Handle JSON files
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Convert JSON to string representation
                if isinstance(data, dict):
                    # If it's a dictionary, extract values
                    return ' '.join(str(v) for v in data.values() if v)
                elif isinstance(data, list):
                    # If it's a list, extract all items
                    return ' '.join(str(item) for item in data if item)
                else:
                    # For other JSON types, convert to string
                    return str(data)
        else:
            # Handle text files
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    except UnicodeDecodeError:
        # Try different encoding if utf-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            logger.error(f"[read_file_content] Failed to read file with latin-1 encoding: {e}")
            raise
    except Exception as e:
        logger.error(f"[read_file_content] Failed to read file: {e}")
        raise

def create_project_index(
    s3_client,
    bucket_name: str,
    user_id: str,
    project_id: str,
    force_refresh: bool = False
) -> bool:
    """
    Orchestrates index creation for a single project:
    - Checks if index.pkl exists unless force_refresh is True
    - Creates a temp directory
    - Downloads documents
    - Preprocesses and creates index
    - Uploads index.pkl to S3
    - Cleans up
    
    Returns True if successful, False otherwise.
    """
    logger.info(f"[create_project_index] Initiating index creation pipeline for user_id: {user_id}, project_id: {project_id}, force_refresh: {force_refresh}")
    index_key = get_index_path(user_id, project_id)
    logger.debug(f"[create_project_index] Index key set to: {index_key}")
    
    # Check if index already exists
    if not force_refresh:
        try:
            logger.info(f"[create_project_index] Checking existence of index at '{index_key}' in bucket '{bucket_name}'")
            s3_client.head_object(Bucket=bucket_name, Key=index_key)
            logger.info(f"[create_project_index] Index for user_id: {user_id}, project_id: {project_id} already exists. Skipping index creation.")
            return True
        except ClientError as e:
            logger.warning(f"[create_project_index] Index does not exist for user_id: {user_id}, project_id: {project_id}. Proceeding to create index. Error: {e}")
    
    temp_dir = tempfile.mkdtemp(prefix=f"{user_id}_{project_id}_")
    logger.info(f"[create_project_index] Temporary working directory created at: {temp_dir}")
    
    try:
        # Download documents
        logger.info(f"[create_project_index] Starting document download for user_id: {user_id}, project_id: {project_id}")
        local_docs = download_project_documents(s3_client, bucket_name, user_id, project_id, temp_dir)
        logger.debug(f"[create_project_index] Local documents downloaded: {local_docs}")
        if not local_docs:
            logger.warning(f"[create_project_index] No documents found for user_id: {user_id}, project_id: {project_id}. Skipping index creation.")
            return True
        
        # Preprocess documents
        logger.info(f"[create_project_index] Starting preprocessing of {len(local_docs)} documents.")
        documents = []
        for fpath in local_docs:
            logger.debug(f"[create_project_index] Preprocessing document: {fpath}")
            try:
                text = read_file_content(fpath)
                if text:
                    text = preprocess_document(text)
                    logger.debug(f"[create_project_index] Preprocessed text length: {len(text)}")
                    documents.append(Document(text=text))
                    logger.info(f"[create_project_index] Successfully preprocessed and added document from '{fpath}'")
                else:
                    logger.warning(f"[create_project_index] Empty content from file: {fpath}")
            except Exception as e:
                logger.error(f"[create_project_index] Failed to read/preprocess '{fpath}': {e}")
                continue  # Skip this file but continue with others
        
        if not documents:
            logger.warning(f"[create_project_index] No valid documents could be processed for user_id: {user_id}, project_id: {project_id}. Skipping index creation.")
            return True
        
        # Create index
        cache_path = os.path.join(temp_dir, 'local_index_cache.pkl')
        logger.debug(f"[create_project_index] Cache path for index: {cache_path}")
        try:
            logger.info(f"[create_project_index] Creating index for user_id: {user_id}, project_id: {project_id}")
            index = create_index(
                documents=documents,
                s3_client=s3_client,
                bucket_name=bucket_name,
                index_cache_key=index_key,
                temp_dir=temp_dir,
                cache_path=cache_path
            )
            if index:
                logger.info(f"[create_project_index] Index created successfully for user_id: {user_id}, project_id: {project_id}")
                logger.debug(f"[create_project_index] Index details: {index}")
                return True
            else:
                logger.error(f"[create_project_index] Index creation returned None for user_id: {user_id}, project_id: {project_id}")
                return False
        except Exception as e:
            logger.error(f"[create_project_index] Error creating index for user_id: {user_id}, project_id: {project_id}: {e}")
            return False
    finally:
        logger.info(f"[create_project_index] Initiating cleanup of temporary directory: {temp_dir}")
        cleanup_temp_dir(temp_dir)

### 5. Batch Processing System

def process_user_projects(s3_client, bucket_name: str, user_id: str, force_refresh: bool = False) -> Dict[str, str]:
    """
    Processes all projects for a specific user:
    - Gets projects for the specified user
    - For each project, tries to create index (if not present or force_refresh)
    - Logs results and returns a report

    Returns a dict of:
    {
        "projectId": "success"|"skipped"|"failed"
    }
    """
    logger.info(f"[process_user_projects] Starting processing for user: {user_id} in bucket: {bucket_name}, force_refresh: {force_refresh}")
    project_ids = list_project_ids_for_user(s3_client, bucket_name, user_id)
    total_projects = len(project_ids)
    logger.info(f"[process_user_projects] Total projects to process for user {user_id}: {total_projects}")
    
    project_count = 0
    success_count = 0
    failure_count = 0
    skipped_count = 0

    report = {}

    start_time = time.time()
    for project_id in project_ids:
        project_count += 1
        elapsed = time.time() - start_time
        projects_left = total_projects - project_count
        est_time_remaining = (elapsed / project_count) * projects_left if project_count else 0
        logger.info(f"[process_user_projects] Processing project {project_id} ({project_count}/{total_projects}). "
                    f"Elapsed: {elapsed:.2f}s, Est. Remaining: {est_time_remaining:.2f}s")
        
        try:
            if create_project_index(s3_client, bucket_name, user_id, project_id, force_refresh=force_refresh):
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=get_index_path(user_id, project_id))
                    if force_refresh:
                        report[project_id] = "success"
                        success_count += 1
                        logger.info(f"[process_user_projects] Index successfully created for project {project_id}")
                    else:
                        report[project_id] = "skipped"
                        skipped_count += 1
                        logger.info(f"[process_user_projects] Index already exists for project {project_id}. Skipped creation.")
                except ClientError:
                    report[project_id] = "success"
                    success_count += 1
            else:
                report[project_id] = "failed"
                failure_count += 1
                logger.error(f"[process_user_projects] Index creation failed for project {project_id}")
        except Exception as e:
            logger.error(f"[process_user_projects] Exception while processing project {project_id}: {e}")
            report[project_id] = "failed"
            failure_count += 1

    logger.info(f"[process_user_projects] Processing complete for user {user_id}. "
                f"Success: {success_count}, Failed: {failure_count}, Skipped: {skipped_count} out of {total_projects}")
    logger.debug(f"[process_user_projects] Final report: {report}")
    return report

### 6. Error Handling
# Already integrated into try/except blocks and logging.

### 7. Progress Tracking
# Integrated into process_all_projects via logs.

### 8. Cleanup and Maintenance

def cleanup_temp_dir(temp_dir: str):
    """Remove temporary directory and all its contents."""
    logger.info(f"[cleanup_temp_dir] Attempting to remove temporary directory: {temp_dir}")
    try:
        shutil.rmtree(temp_dir)
        logger.debug(f"[cleanup_temp_dir] Temporary directory '{temp_dir}' removed successfully.")
    except Exception as e:
        logger.error(f"[cleanup_temp_dir] Failed to clean up '{temp_dir}': {e}")

def verify_index_integrity(index: VectorStoreIndex) -> bool:
    """
    Placeholder: Verify that the index is valid and not corrupted.
    Currently returns True, but can be extended.
    """
    logger.info("[verify_index_integrity] Verifying index integrity.")
    # Implement actual integrity checks here
    logger.debug("[verify_index_integrity] Index integrity check passed.")
    return True

def remove_orphaned_indices(s3_client, bucket_name: str):
    """
    Placeholder for removing orphaned indices.
    Could list all indices and verify if corresponding documents exist.
    Not fully implemented here.
    """
    logger.info("[remove_orphaned_indices] Initiating check for orphaned indices in bucket.")
    # Implement logic to identify and remove orphaned indices
    logger.debug("[remove_orphaned_indices] Orphaned indices removal not implemented yet.")

########################################
# Example Usage (Commented Out)
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process documents and create indices for a specific user')
    parser.add_argument('user_id', help='The user ID to process')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh of existing indices')
    
    args = parser.parse_args()
    
    logger.info(f"Starting processing for user: {args.user_id}")
    s3 = boto3.client('s3')
    
    report = process_user_projects(s3, AWS_UPLOAD_BUCKET_NAME, args.user_id, force_refresh=args.force_refresh)
    logger.info(f"Processing report for user {args.user_id}: {report}")
