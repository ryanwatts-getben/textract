import logging
import pickle
import os
from typing import List, Set, Tuple
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.anthropic import Anthropic
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_document(document_text: str) -> str:
    """Preprocess a single document's text."""
    return document_text.lower()

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Initialize embedding model with proper device configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[rag] Using device: {device} for embeddings")

    # Initialize HuggingFace embeddings
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )

    # Wrap with LangchainEmbedding for compatibility with LlamaIndex
    return LangchainEmbedding(huggingface_embeddings)

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
    
    Args:
        documents: List of Document objects to index
        s3_client: Boto3 S3 client for cache operations
        bucket_name: S3 bucket name for cache storage
        index_cache_key: S3 key for the cache file
        temp_dir: Local temporary directory for cache operations
        cache_path: Local path for the cache file
    """
    try:
        # Get the embedding model
        embed_model = get_embedding_model()

        logger.info('[rag] Creating VectorStoreIndex with provided documents')
        try:
            index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
            logger.info('[rag] Index created successfully')
        except Exception as e:
            logger.error(f"[rag] Error creating index: {str(e)}")
            raise

        # If S3 caching is enabled, save the index        
        
        current_files = set(os.listdir(temp_dir))
        with open(cache_path, 'wb') as f:
            # Store as tuple of (index, files)
            pickle.dump((index, current_files), f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info("[rag] Created and cached new vector index")
        
        if all([s3_client, bucket_name, index_cache_key, temp_dir, cache_path]):
            try:
                # Save index to local cache first
                index_cache_path = os.path.join(temp_dir, 'index_cache.pkl')
                logger.debug(f'[rag] Saving index to local cache at "{index_cache_path}"')
                with open(index_cache_path, 'wb') as f:
                    pickle.dump(index, f)
                logger.info('[rag] Index serialized and saved to local cache successfully')

                # Upload to S3
                logger.debug(f'[rag] Uploading index cache to S3 bucket "{bucket_name}" with key "{index_cache_key}"')
                s3_client.upload_file(
                    Filename=index_cache_path,
                    Bucket=bucket_name,
                    Key=index_cache_key
                )
                logger.info('[rag] Index cache uploaded to S3 successfully')
            except Exception as e:
                logger.error(f'[rag] Error caching index to S3: {str(e)}')
                # Continue even if caching fails
                pass

        return index

    except Exception as e:
        logger.error(f"[rag] Error creating index: {str(e)}")
        raise
def query_index(index: VectorStoreIndex, query_text: str) -> str:
    """Query the index with given text using Claude."""
    try:
        # Initialize Claude
        llm = Anthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )

        # Create query engine with Claude
        query_engine = index.as_query_engine(
            llm=llm,
            response_mode="compact"
        )
        
        response = query_engine.query(query_text)
        return response.response
    except Exception as e:
        logger.error(f"[app] Error querying index: {str(e)}")
        raise
    
def preprocess_document(document_text: str) -> str:
    """Preprocess a single document's text."""
    preprocessed_text = document_text.lower()
    return preprocessed_text