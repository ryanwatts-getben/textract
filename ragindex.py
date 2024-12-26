import logging
import pickle
import os
import tempfile
import shutil
import argparse
from typing import List, Dict, Optional, Set, Tuple
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import boto3
from botocore.exceptions import ClientError
import time
from dotenv import load_dotenv
import json
import io
from pypdf import PdfReader

# Import configurations
from app_rag_disease_config import (
    LOG_CONFIG,
    SUPPORTED_EXTENSIONS,
    EMBEDDING_MODEL_CONFIG,
    RAGDocumentConfig,
    VECTOR_STORE_CONFIG,
    FALLBACK_CONFIG,
    STORAGE_CONFIG,
    RAG_ERROR_MESSAGES,
    CATEGORY_PREFIXES
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
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

def get_embedding_model(model_name: str = EMBEDDING_MODEL_CONFIG["model_name"]):
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
        
        # Configure settings for batching and chunking
        Settings.chunk_size = VECTOR_STORE_CONFIG["chunk_size"]
        Settings.chunk_overlap = VECTOR_STORE_CONFIG["chunk_overlap"]
        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(
            chunk_size=VECTOR_STORE_CONFIG["chunk_size"],
            chunk_overlap=VECTOR_STORE_CONFIG["chunk_overlap"]
        )
        Settings.embed_batch_size = VECTOR_STORE_CONFIG["embed_batch_size"]
        
        index = VectorStoreIndex.from_documents(
            documents=documents,
            show_progress=VECTOR_STORE_CONFIG["show_progress"]
        )
        logger.info('[create_index] VectorStoreIndex created successfully')
        logger.debug(f"[create_index] VectorStoreIndex details: {index}")
        
        # Cache locally
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(index, f, protocol=STORAGE_CONFIG["pickle_protocol"])
            logger.info("[create_index] Created and cached new vector index locally")
            logger.debug(f"[create_index] Cached index at '{cache_path}'")
        
        # Upload to S3 if parameters provided
        if all([s3_client, bucket_name, index_cache_key, temp_dir]):
            try:
                index_cache_path = os.path.join(temp_dir, STORAGE_CONFIG["cache_filename"])
                with open(index_cache_path, 'wb') as f:
                    pickle.dump(index, f, protocol=STORAGE_CONFIG["pickle_protocol"])
                logger.info('[create_index] Index serialized and saved to local cache successfully')
                
                s3_client.upload_file(
                    Filename=index_cache_path,
                    Bucket=bucket_name,
                    Key=index_cache_key
                )
                logger.info('[create_index] Index cache uploaded to S3 successfully')
            except Exception as e:
                logger.error(RAG_ERROR_MESSAGES["index_creation_error"].format(error=str(e)))
        
        return index

    except Exception as e:
        logger.error(RAG_ERROR_MESSAGES["index_creation_error"].format(error=str(e)))
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
            logger.error(RAG_ERROR_MESSAGES["file_read_error"].format(file_path=file_path, error=str(e)))
            raise
    except Exception as e:
        logger.error(RAG_ERROR_MESSAGES["file_read_error"].format(file_path=file_path, error=str(e)))
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
            logger.warning(RAG_ERROR_MESSAGES["index_not_found"].format(user_id=user_id, project_id=project_id, error=str(e)))
    
    temp_dir = tempfile.mkdtemp(prefix=f"{user_id}_{project_id}_")
    logger.info(f"[create_project_index] Temporary working directory created at: {temp_dir}")
    
    try:
        # Download documents
        logger.info(f"[create_project_index] Starting document download for user_id: {user_id}, project_id: {project_id}")
        local_docs = download_project_documents(s3_client, bucket_name, user_id, project_id, temp_dir)
        logger.debug(f"[create_project_index] Local documents downloaded: {local_docs}")
        if not local_docs:
            logger.warning(RAG_ERROR_MESSAGES["no_documents"].format(user_id=user_id, project_id=project_id))
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
                    logger.warning(RAG_ERROR_MESSAGES["empty_document"].format(file_path=fpath))
            except Exception as e:
                logger.error(RAG_ERROR_MESSAGES["processing_error"].format(file_path=fpath, error=str(e)))
                continue  # Skip this file but continue with others
        
        if not documents:
            logger.warning(RAG_ERROR_MESSAGES["no_valid_documents"].format(user_id=user_id, project_id=project_id))
            return True
        
        # Create index
        cache_path = os.path.join(temp_dir, STORAGE_CONFIG["cache_filename"])
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
                logger.error(RAG_ERROR_MESSAGES["index_creation_failed"].format(user_id=user_id, project_id=project_id))
                return False
        except Exception as e:
            logger.error(RAG_ERROR_MESSAGES["index_creation_error"].format(error=str(e)))
            return False
    finally:
        logger.info(f"[create_project_index] Initiating cleanup of temporary directory: {temp_dir}")
        cleanup_temp_dir(temp_dir)

### 5. Batch Processing System

def process_user_projects(s3_client, bucket_name: str, user_id: str, force_refresh: bool = False) -> Dict[str, str]:
    """Process all projects for a given user."""
    logger.info(f"[process_user_projects] Processing projects for user: {user_id}")
    results = {}
    
    try:
        # List all projects for the user
        projects = list_project_ids_for_user(s3_client, bucket_name, user_id)
        logger.info(f"[process_user_projects] Found {len(projects)} projects for user {user_id}")
        
        for project_id in projects:
            logger.info(f"[process_user_projects] Found active project: {project_id}")
            
            # Check if index exists
            index_key = f"{user_id}/{project_id}/index.pkl"
            try:
                s3_client.head_object(Bucket=bucket_name, Key=index_key)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.warning(f"[process_user_projects] Project {project_id} has no index")
                    continue
                else:
                    logger.error(f"[process_user_projects] Error checking index for project {project_id}: {str(e)}")
                    continue
                    
            results[project_id] = "active"
            
        return results
        
    except Exception as e:
        logger.error(f"[process_user_projects] Error processing user projects: {str(e)}")
        return results

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
        logger.error(RAG_ERROR_MESSAGES["cleanup_error"].format(temp_dir=temp_dir, error=str(e)))

def verify_index_integrity(index: VectorStoreIndex) -> bool:
    """
    Verify that the index is valid and not corrupted.
    Currently returns True, but can be extended.
    """
    logger.info("[verify_index_integrity] Verifying index integrity.")
    try:
        # Basic integrity checks
        if not index or not hasattr(index, 'docstore'):
            logger.error(RAG_ERROR_MESSAGES["invalid_index"])
            return False
            
        # Check if index has documents
        doc_count = len(list(index.docstore.docs.values()))
        if doc_count == 0:
            logger.warning(RAG_ERROR_MESSAGES["empty_index"])
            return False
            
        logger.info(f"[verify_index_integrity] Index contains {doc_count} documents")
        return True
        
    except Exception as e:
        logger.error(RAG_ERROR_MESSAGES["index_verification_error"].format(error=str(e)))
        return False

def remove_orphaned_indices(s3_client, bucket_name: str):
    """
    Remove orphaned indices that don't have corresponding documents.
    """
    logger.info("[remove_orphaned_indices] Initiating check for orphaned indices in bucket.")
    try:
        # List all indices
        paginator = s3_client.get_paginator('list_objects_v2')
        indices = []
        for page in paginator.paginate(Bucket=bucket_name, Prefix=''):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('index.pkl'):
                    indices.append(obj['Key'])
        
        logger.info(f"[remove_orphaned_indices] Found {len(indices)} indices to check")
        
        # Check each index
        for index_key in indices:
            try:
                # Extract user_id and project_id from key
                parts = index_key.split('/')
                if len(parts) >= 2:
                    user_id = parts[0]
                    project_id = parts[1]
                    
                    # Check if documents exist
                    doc_prefix = get_document_prefix(user_id, project_id)
                    response = s3_client.list_objects_v2(
                        Bucket=bucket_name,
                        Prefix=doc_prefix,
                        MaxKeys=1
                    )
                    
                    if 'Contents' not in response:
                        logger.warning(RAG_ERROR_MESSAGES["orphaned_index"].format(
                            index_key=index_key,
                            user_id=user_id,
                            project_id=project_id
                        ))
                        # Delete orphaned index
                        s3_client.delete_object(Bucket=bucket_name, Key=index_key)
                        logger.info(f"[remove_orphaned_indices] Deleted orphaned index: {index_key}")
                        
            except Exception as e:
                logger.error(RAG_ERROR_MESSAGES["orphan_check_error"].format(
                    index_key=index_key,
                    error=str(e)
                ))
                continue
                
    except Exception as e:
        logger.error(RAG_ERROR_MESSAGES["orphan_removal_error"].format(error=str(e)))

########################################
# Example Usage (Commented Out)
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process documents and create indices for a specific user')
    parser.add_argument('user_id', help='The user ID to process')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh of existing indices')
    parser.add_argument('--cleanup', action='store_true', help='Remove orphaned indices')
    
    args = parser.parse_args()
    
    logger.info(f"Starting processing for user: {args.user_id}")
    s3 = boto3.client('s3')
    
    if args.cleanup:
        remove_orphaned_indices(s3, AWS_UPLOAD_BUCKET_NAME)
    
    report = process_user_projects(s3, AWS_UPLOAD_BUCKET_NAME, args.user_id, force_refresh=args.force_refresh)
    logger.info(f"Processing report for user {args.user_id}: {report}")
