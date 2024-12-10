import logging
import pickle
import os
from typing import List, Set, Tuple
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.langchain import LangchainEmbedding

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

def create_index(documents: List[Document]) -> VectorStoreIndex:
    """Create a vector store index from documents."""
    try:
        # Validate documents
        valid_documents = []
        for doc in documents:
            if not doc.text.strip():
                logger.warning("[rag] Skipping empty document")
                continue
            logger.info(f"[rag] Adding document with {len(doc.text)} characters")
            valid_documents.append(doc)
        
        if not valid_documents:
            raise ValueError("No valid documents to index")
            
        logger.info(f"[rag] Creating index with {len(valid_documents)} valid documents")
        
        # Get the embedding model
        embed_model = get_embedding_model()
        
        # Create the index
        index = VectorStoreIndex.from_documents(
            valid_documents,
            embed_model=embed_model,
            show_progress=True  # This will show a progress bar
        )
        
        # Verify index creation
        node_count = len(index.docstore.docs)
        logger.info(f"[rag] Created index with {node_count} nodes")
        
        if node_count == 0:
            raise ValueError("Index created but contains no nodes")
            
        return index

    except Exception as e:
        logger.error(f"[rag] Error creating index: {str(e)}")
        raise