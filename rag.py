# Standard library imports
import logging
import pickle
import os
import tempfile
import shutil
import argparse
from typing import List, Dict, Optional, Set, Tuple, Any
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.node_parser import SentenceSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from pathlib import Path
import torch
import boto3
from botocore.exceptions import ClientError
import time
from dotenv import load_dotenv
import json
import io
import gc
from pypdf import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports
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

def preprocess_document(document_text: str) -> str:
    """Preprocess a single document's text."""
    return document_text.lower()

def get_embedding_model(model_name: str = EMBEDDING_MODEL_CONFIG["model_name"]):
    """Initialize the embedding model with proper device configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[rag] Attempting to load embedding model: {model_name} on {device}")

    try:
        # Create HuggingFaceEmbeddings from langchain
        base_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        
        # Wrap with LangchainEmbedding for LlamaIndex compatibility
        embedding_model = LangchainEmbedding(
            langchain_embeddings=base_embeddings
        )

        # Get embedding dimension
        test_embedding = embedding_model.get_text_embedding("test")
        embedding_dimension = len(test_embedding)

        logger.info(f"[rag] Embedding model initialized with {model_name}")
        logger.info(f"[rag] Embedding dimension: {embedding_dimension}")

        return embedding_model, embedding_dimension

    except Exception as e:
        logger.error(f"[rag] Error loading model {model_name}: {str(e)}")
        raise

def process_single_document(doc: Document) -> Optional[Document]:
    """Process a single document with optimized settings."""
    try:
        if not doc.text.strip():
            return None
            
        # Process text in chunks for memory efficiency
        processed_text = ""
        chunk_size = RAGDocumentConfig.CHUNK_SIZE
        text = doc.text
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            processed_chunk = preprocess_document(chunk)
            processed_text += processed_chunk
            
        if not processed_text:
            return None
            
        return Document(
            text=processed_text,
            metadata={
                **doc.metadata,
                'processed_timestamp': time.time()
            }
        )
        
    except Exception as e:
        logger.error(RAG_ERROR_MESSAGES["processing_error"].format(file_path=doc.metadata.get('source', 'unknown'), error=str(e)))
        return None

class IndexManager:
    def __init__(self, storage_dir: Optional[str] = None):
        self.indexes: Dict[str, VectorStoreIndex] = {}
        self.batch_size = RAGDocumentConfig.BATCH_SIZE
        self.chunk_size = RAGDocumentConfig.CHUNK_SIZE_TOKENS
        self.max_nodes_per_batch = RAGDocumentConfig.MAX_NODES_PER_BATCH
        self.timeout = RAGDocumentConfig.PROCESSING_TIMEOUT
        self.progress: Dict[str, float] = {}
        self.storage_dir = storage_dir or os.path.join(tempfile.gettempdir(), 'index_storage')
        self.temp_dirs: List[str] = []
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)

    def _create_temp_dir(self) -> str:
        """Create a temporary directory and track it for cleanup."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def cleanup(self):
        """Clean up temporary directories and free memory."""
        try:
            # Clean up temporary directories
            for temp_dir in self.temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            self.temp_dirs.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("[rag] Cleanup completed successfully")
        except Exception as e:
            logger.error(f"[rag] Error during cleanup: {str(e)}")

    def _get_index_path(self, category: str) -> str:
        """Get the storage path for a category's index."""
        return os.path.join(self.storage_dir, f"index_{category}")

    def save_indexes(self):
        """Persist indexes to storage."""
        try:
            for category, index in self.indexes.items():
                if index:
                    index_path = self._get_index_path(category)
                    os.makedirs(index_path, exist_ok=True)
                    index.storage_context.persist(persist_dir=index_path)
                    logger.info(f"[rag] Saved index for category {category}")
            
            # Save index metadata
            metadata = {
                "categories": list(self.indexes.keys()),
                "timestamp": time.time()
            }
            metadata_path = os.path.join(self.storage_dir, STORAGE_CONFIG["metadata_filename"])
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f, protocol=STORAGE_CONFIG["pickle_protocol"])
                
        except Exception as e:
            logger.error(f"[rag] Error saving indexes: {str(e)}")
            raise

    def _get_index_category(self, doc: Document) -> str:
        """Determine index category based on first three letters of filename."""
        try:
            filename = Path(doc.metadata.get('source', '')).name
            prefix = filename[:3].lower()
            if prefix in RAGDocumentConfig.VALID_CATEGORIES:
                return prefix
            logger.warning(f"[rag] Unknown prefix {prefix} for file {filename}, using 'other'")
            return 'other'
        except Exception as e:
            logger.error(f"[rag] Error determining category: {str(e)}")
            return 'other'

    def _categorize_documents(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by their prefix category."""
        categorized: Dict[str, List[Document]] = {
            category: [] for category in RAGDocumentConfig.VALID_CATEGORIES
        }
        
        for doc in documents:
            category = self._get_index_category(doc)
            categorized[category].append(doc)
            
        # Log document distribution
        for category, docs in categorized.items():
            logger.info(f"[rag] Category {CATEGORY_PREFIXES[category]}: {len(docs)} documents")
            
        return categorized

def create_index(documents: List[Document], embed_model: Any = None, batch_size: int = 10, chunk_size: int = 512) -> VectorStoreIndex:
    """Create a new vector store index from the provided documents."""
    try:
        logger.info(f"[create_index] Starting index creation with {len(documents)} documents.")
        
        if embed_model is None:
            embed_model = get_embedding_model()
        
        # Configure LLM to use Anthropic
        llm = Anthropic(
            model=os.getenv('ANTHROPIC_MODEL', 'claude-2'),
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
        # Configure global settings to use Anthropic
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        logger.info("[create_index] Creating VectorStoreIndex with Anthropic LLM")
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            llm=llm,  # Explicitly pass the LLM
            show_progress=True
        )
        logger.info("[create_index] VectorStoreIndex created successfully")

        # Cache the index locally
        try:
            temp_dir = tempfile.mkdtemp()
            cache_path = os.path.join(temp_dir, 'vector_index.pkl')
            
            # Save locally first
            with open(cache_path, 'wb') as f:
                pickle.dump(index, f)
            logger.info("[create_index] Created and cached new vector index locally")

            # Upload to S3
            try:
                s3_client = boto3.client('s3')
                bucket_name = os.getenv('AWS_UPLOAD_BUCKET_NAME')
                if bucket_name:
                    s3_client.upload_file(cache_path, bucket_name, '00000/22222/index.pkl')
                    logger.info("[create_index] Index serialized and saved to local cache successfully")
                    logger.info("[create_index] Index cache uploaded to S3 successfully")
            except Exception as s3_error:
                logger.error(f"[create_index] Error uploading to S3: {str(s3_error)}")
                # Continue even if S3 upload fails
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
        except Exception as cache_error:
            logger.error(f"[create_index] Error caching index: {str(cache_error)}")
            # Continue even if caching fails
            
        return index

    except Exception as e:
        logger.error(f"[create_index] Error creating index: {str(e)}")
        raise

def process_chunk_with_fallback(chunk: List[Document], embed_model) -> Optional[VectorStoreIndex]:
    """Fallback processing for chunks that fail normal processing."""
    try:
        logger.info("[rag] Attempting fallback processing with smaller batches")
        
        # Process one document at a time
        indexes = []
        for doc in chunk:
            try:
                single_index = VectorStoreIndex.from_documents(
                    [doc],
                    embed_model=embed_model,
                    show_progress=FALLBACK_CONFIG["show_progress"],
                    use_async=FALLBACK_CONFIG["use_async"]
                )
                indexes.append(single_index)
                logger.info("[rag] Successfully processed single document in fallback mode")
            except Exception as e:
                logger.error(RAG_ERROR_MESSAGES["fallback_error"].format(error=str(e)))
                continue
                
            # Clear memory after each document
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
        # Merge all successful indexes
        if indexes:
            logger.info(f"[rag] Merging {len(indexes)} indexes from fallback processing")
            final_index = indexes[0]
            for idx, index in enumerate(indexes[1:], 1):
                final_index = final_index.merge(index)
                logger.info(f"[rag] Merged fallback index {idx+1}/{len(indexes)}")
            return final_index
            
        return None
        
    except Exception as e:
        logger.error(RAG_ERROR_MESSAGES["fallback_error"].format(error=str(e)))
        return None

def process_document(file_path: str) -> Optional[Document]:
    """Process a single document file."""
    try:
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        if file_extension not in SUPPORTED_EXTENSIONS:
            logger.warning(RAG_ERROR_MESSAGES["unsupported_file"].format(extension=file_extension))
            return None
            
        with open(file_path, 'rb') as f:
            content = f.read()
            
        text = extract_text_from_file(content, file_extension)
        
        if not text.strip():
            logger.warning(RAG_ERROR_MESSAGES["no_text_content"].format(file_path=file_path))
            return None
            
        return Document(
            text=text,
            metadata={
                'file_name': os.path.basename(file_path),
                'file_type': file_extension,
                'source': file_path
            }
        )
        
    except Exception as e:
        logger.error(RAG_ERROR_MESSAGES["processing_error"].format(file_path=file_path, error=str(e)))
        return None

def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from a file, supporting different file types.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The extracted text.
    """
    path = Path(file_path)
    if path.suffix.lower() == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except Exception as e:
            logger.error(f"[rag] Error reading text file {file_path}: {e}")
            return ''
    elif path.suffix.lower() == '.pdf':
        # Use PyPDF to extract text from PDF files
        try:
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                text = ''
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text
        except Exception as e:
            logger.error(f"[rag] Error extracting text from PDF {file_path}: {e}")
            return ''
    else:
        logger.warning(f"[rag] Unsupported file type for {file_path}, skipping.")
        return ''