# Standard library imports
import logging
import pickle
import os
from typing import List, Dict, Optional
from pathlib import Path
import gc
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# Third-party imports
import torch

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

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

    # Initialize HuggingFace embeddings with updated import
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )

    # Wrap with LangchainEmbedding for compatibility with LlamaIndex
    return LangchainEmbedding(huggingface_embeddings)

def process_single_document(doc: Document, chunk_size: int = 512) -> List[Optional[Document]]:
    """Process a single document with chunking."""
    try:
        if not doc.text.strip():
            logger.warning("[rag] Skipping empty document")
            return []
            
        processed_text = preprocess_document(doc.text)
        if not processed_text:
            return []
            
        # Split into chunks
        chunks = []
        words = processed_text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1  # +1 for space
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        # Create documents from chunks
        return [
            Document(
                text=chunk,
                metadata={
                    **doc.metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
    except Exception as e:
        logger.error(f"[rag] Error processing document: {str(e)}")
        return []

class IndexManager:
    def __init__(self, storage_dir: Optional[str] = None):
        self.indexes: Dict[str, VectorStoreIndex] = {}
        self.batch_size = 100000  # Reduce from 1000000 to 100000
        self.chunk_size = 512     # Add chunk size control
        self.max_nodes_per_batch = 500  # Add node limit per batch
        self.timeout = 300  # 5 minute timeout per batch
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
            metadata_path = os.path.join(self.storage_dir, "index_metadata.pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)
                
        except Exception as e:
            logger.error(f"[rag] Error saving indexes: {str(e)}")
            raise

    def load_indexes(self) -> bool:
        """Load indexes from storage if they exist."""
        try:
            metadata_path = os.path.join(self.storage_dir, "index_metadata.pkl")
            if not os.path.exists(metadata_path):
                return False

            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            for category in metadata["categories"]:
                index_path = self._get_index_path(category)
                if os.path.exists(index_path):
                    storage_context = StorageContext.from_defaults(persist_dir=index_path)
                    self.indexes[category] = load_index_from_storage(storage_context)
                    logger.info(f"[rag] Loaded index for category {category}")

            return True
        except Exception as e:
            logger.error(f"[rag] Error loading indexes: {str(e)}")
            return False

    def merge_indexes(self) -> Optional[VectorStoreIndex]:
        """Merge all indexes into a single index with memory management."""
        try:
            if not self.indexes:
                return None

            # Create a temporary directory for merging
            merge_temp_dir = self._create_temp_dir()
            
            merged_index = None
            total_indexes = len(self.indexes)
            
            for idx, (category, index) in enumerate(self.indexes.items(), 1):
                if not index:
                    continue
                    
                logger.info(f"[rag] Merging index {idx}/{total_indexes} from category {category}")
                
                if not merged_index:
                    merged_index = index
                else:
                    # Save current state to temporary storage
                    temp_path = os.path.join(merge_temp_dir, f"temp_merge_{idx}")
                    os.makedirs(temp_path, exist_ok=True)
                    
                    # Merge indexes with memory management
                    merged_nodes = merged_index.docstore.docs.copy()
                    merged_nodes.update(index.docstore.docs)
                    
                    # Clear memory of individual indexes
                    del merged_index
                    del index
                    gc.collect()
                    
                    # Create new merged index
                    merged_index = VectorStoreIndex.from_documents(
                        [Document(text=node.text, metadata=node.metadata) 
                         for node in merged_nodes.values()],
                        embed_model=get_embedding_model(),
                        show_progress=True
                    )
                    
                    # Clear memory
                    del merged_nodes
                    gc.collect()
            
            return merged_index
            
        except Exception as e:
            logger.error(f"[rag] Error merging indexes: {str(e)}")
            return None
        finally:
            self.cleanup()

    def create_index_batch(self, documents: List[Document], category: str) -> Optional[VectorStoreIndex]:
        """Create index for a batch of documents with memory management."""
        temp_dir = None
        try:
            temp_dir = self._create_temp_dir()
            total_docs = len(documents)
            self.progress[category] = 0.0
            
            processed_docs = []
            total_processed = 0
            
            # Process documents in smaller batches
            batch_size = 10  # Process 10 documents at a time
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_doc = {
                        executor.submit(
                            process_single_document, 
                            doc, 
                            self.chunk_size
                        ): doc for doc in batch
                    }
                    
                    for future in as_completed(future_to_doc):
                        try:
                            doc_chunks = future.result(timeout=self.timeout)
                            processed_docs.extend([chunk for chunk in doc_chunks if chunk])
                            total_processed += 1
                            self.progress[category] = (total_processed / total_docs) * 100
                            logger.info(f"[rag] {category} index progress: {self.progress[category]:.2f}%")
                        except TimeoutError:
                            logger.warning(f"[rag] Processing timeout for document in {category}")
                            continue
                        except Exception as e:
                            logger.error(f"[rag] Error processing document: {str(e)}")
                
                # Create intermediate index every N chunks to manage memory
                if len(processed_docs) >= self.max_nodes_per_batch:
                    if category not in self.indexes:
                        self.indexes[category] = VectorStoreIndex.from_documents(
                            processed_docs,
                            embed_model=get_embedding_model(),
                            show_progress=True
                        )
                    else:
                        # Merge with existing index
                        temp_index = VectorStoreIndex.from_documents(
                            processed_docs,
                            embed_model=get_embedding_model(),
                            show_progress=True
                        )
                        self.indexes[category].merge(temp_index)
                    
                    processed_docs = []
                    gc.collect()
            
            # Process any remaining documents
            if processed_docs:
                if category not in self.indexes:
                    self.indexes[category] = VectorStoreIndex.from_documents(
                        processed_docs,
                        embed_model=get_embedding_model(),
                        show_progress=True
                    )
                else:
                    temp_index = VectorStoreIndex.from_documents(
                        processed_docs,
                        embed_model=get_embedding_model(),
                        show_progress=True
                    )
                    self.indexes[category].merge(temp_index)
            
            return self.indexes.get(category)
            
        except Exception as e:
            logger.error(f"[rag] Failed to create index for {category}: {str(e)}")
            return None
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def _get_index_category(self, doc: Document) -> str:
        """Determine index category based on first three letters of filename."""
        try:
            filename = Path(doc.metadata.get('source', '')).name
            prefix = filename[:3].lower()
            if prefix in ['icd', 'der', 'sct']:
                return prefix
            logger.warning(f"[rag] Unknown prefix {prefix} for file {filename}, using 'other'")
            return 'other'
        except Exception as e:
            logger.error(f"[rag] Error determining category: {str(e)}")
            return 'other'

    def _categorize_documents(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by their prefix category."""
        categorized: Dict[str, List[Document]] = {
            'icd': [], 'der': [], 'sct': [], 'other': []
        }
        
        for doc in documents:
            category = self._get_index_category(doc)
            categorized[category].append(doc)
            
        # Log document distribution
        for category, docs in categorized.items():
            logger.info(f"[rag] Category {category}: {len(docs)} documents")
            
        return categorized

    def initialize_indexes(self, documents: List[Document]) -> None:
        """Initialize all indexes with progress monitoring."""
        try:
            # Group documents by category
            categorized_docs = self._categorize_documents(documents)
            
            # Create indexes for each category
            for category, docs in categorized_docs.items():
                if not docs:
                    logger.info(f"[rag] Skipping empty category: {category}")
                    continue
                    
                try:
                    start_time = time.time()
                    self.indexes[category] = self.create_index_batch(docs, category)
                    duration = time.time() - start_time
                    
                    if self.indexes[category]:
                        logger.info(f"[rag] Successfully created index for {category} in {duration:.2f}s")
                    else:
                        logger.error(f"[rag] Failed to create index for {category}")
                        
                except Exception as e:
                    logger.error(f"[rag] Error creating index for {category}: {str(e)}")

        except Exception as e:
            logger.error(f"[rag] Error in initialize_indexes: {str(e)}")
            raise

    def get_progress(self) -> Dict[str, float]:
        """Get current progress for all index creation tasks."""
        return self.progress.copy()

def create_index(documents: List[Document], max_workers: int = 10, persist: bool = True) -> VectorStoreIndex:
    """Create indexes using the IndexManager with persistence and memory management."""
    with IndexManager() as index_manager:
        try:
            if not documents:
                raise ValueError("No documents provided to index")
            
            # Try to load existing indexes
            if persist and index_manager.load_indexes():
                logger.info("[rag] Loaded existing indexes")
            else:
                # Create new indexes
                index_manager.initialize_indexes(documents)
                
                if persist:
                    index_manager.save_indexes()
            
            # Merge indexes
            combined_index = index_manager.merge_indexes()
            
            if not combined_index:
                raise ValueError("No valid indexes created")
            
            return combined_index

        except Exception as e:
            logger.error(f"[rag] Error creating indexes: {str(e)}")
            raise