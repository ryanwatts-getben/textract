import logging
import os
import pickle
from typing import List, Set, Tuple
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.anthropic import Anthropic
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_document(document_text: str) -> str:
    """Preprocess a single document's text."""
    preprocessed_text = document_text.lower()
    return preprocessed_text

def load_documents(directory: str = "./data") -> List[Document]:
    """Load and preprocess documents from the specified directory."""
    try:
        if not os.path.exists(directory):
            logger.error("[app] Data directory not found")
            raise FileNotFoundError(f"Directory {directory} does not exist")

        loader = SimpleDirectoryReader(input_dir=directory)
        documents = loader.load_data()
        logger.info(f"[app] Successfully loaded {len(documents)} documents")

        for doc in documents:
            doc.text = preprocess_document(doc.text)
        logger.info(f"[app] Successfully preprocessed {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"[app] Error loading documents: {str(e)}")
        raise

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Initialize embedding model with proper device configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[app] Using device: {device} for embeddings")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )

def create_index(
    documents: List[Document],
    cache_path: str = "./index_cache.pkl",
    data_directory: str = "./data"
) -> VectorStoreIndex:
    """Create a vector store index from documents, with caching."""
    try:
        cache_valid = False
        if os.path.exists(cache_path):
            current_files = set(os.listdir(data_directory))
            
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if isinstance(cached_data, tuple):
                        index, cached_files = cached_data
                        if current_files == cached_files:
                            logger.info("[app] Using cached index")
                            cache_valid = True
                            return index
            except Exception as e:
                logger.warning(f"[app] Cache load failed, rebuilding index: {str(e)}")
        
        # Create new index if cache is invalid or doesn't exist
        embed_model = get_embedding_model()
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model
        )
        
        # Store index and current files list
        current_files = set(os.listdir(data_directory))
        with open(cache_path, 'wb') as f:
            # Store as tuple of (index, files)
            pickle.dump((index, current_files), f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info("[app] Created and cached new vector index")
        return index
        
    except Exception as e:
        logger.error(f"[app] Error creating index: {str(e)}")
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


