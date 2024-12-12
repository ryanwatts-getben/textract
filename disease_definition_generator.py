import logging
import os
import json
import boto3
import tempfile
import pickle
from tqdm import tqdm

from flask import request, jsonify
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from botocore.exceptions import ClientError

# Import Hugging Face embedding and LLM models
from llama_index.core import (
    load_index_from_storage, 
    VectorStoreIndex, 
    Document, 
    StorageContext,
    PromptTemplate,
    Settings
)
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from llama_index.llms.anthropic import Anthropic
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Pydantic models for structured output
class Symptom(BaseModel):
    name: str
    commonality: Optional[str] = Field(None, description="VERY_COMMON, COMMON, UNCOMMON, or RARE")
    severity: Optional[str] = Field(None, description="MILD, MODERATE, SEVERE, or CRITICAL")

class LabResult(BaseModel):
    name: str
    unit: Optional[str] = None
    normalRange: Optional[str] = None
    significance: Optional[str] = None

class DiagnosticProcedure(BaseModel):
    name: str
    accuracy: Optional[float] = Field(None, ge=0, le=1)
    invasiveness: Optional[str] = Field(None, description="NON_INVASIVE, MINIMALLY_INVASIVE, or INVASIVE")

class Treatment(BaseModel):
    name: str
    type: str = Field(description="MEDICATION, SURGERY, THERAPY, LIFESTYLE, or OTHER")
    effectiveness: Optional[float] = Field(None, ge=0, le=1)

class Prognosis(BaseModel):
    survivalRate: Optional[float] = Field(None, ge=0, le=1)
    chronicityLevel: Optional[str] = Field(None, description="ACUTE, SUBACUTE, or CHRONIC")
    qualityOfLifeImpact: Optional[str] = Field(None, description="MILD, MODERATE, or SEVERE")

class DiseaseDefinition(BaseModel):
    name: str
    icd10: List[str]
    description: str
    symptoms: List[str]
    commonTests: List[str]
    treatments: List[str]
    complications: List[str]
    relatedConditions: List[str]
    
# Define the get_embedding_model function before it's used
def get_embedding_model(model_name: str = "BioBERT-mnli-snli-scinli-scitail-mednli-stsb"):
    """Initialize the embedding model with proper device configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[disease_definition_generator] Attempting to load embedding model: {model_name} on {device}")
    try:
        # Initialize HuggingFaceEmbeddings
        huggingface_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )

        # Get the embedding dimension
        embedding_dimension = huggingface_embeddings.model.get_sentence_embedding_dimension()

        # Wrap with LangchainEmbedding
        embedding_model = LangchainEmbedding(huggingface_embeddings)
        # Attach attributes for easy access
        embedding_model.model_name = model_name
        embedding_model.embedding_dimension = embedding_dimension

        logger.info(f"[disease_definition_generator] Successfully loaded embedding model: {model_name}")
        logger.info(f"[disease_definition_generator] Embedding dimension: {embedding_dimension}")

        return embedding_model
    except Exception as e:
        logger.error(f"[disease_definition_generator] Error loading model {model_name}: {str(e)}")
        raise

# Singleton class for the Embedding Model
class EmbeddingModelSingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if EmbeddingModelSingleton._instance is None:
            # Initialize the embedding model using the defined function
            EmbeddingModelSingleton._instance = get_embedding_model()
        return EmbeddingModelSingleton._instance

# Initialize Singleton Embedding Model
embedding_model_instance = EmbeddingModelSingleton.get_instance()
logger.info(f"[disease_definition_generator] Using Singleton embedding model: {embedding_model_instance.model_name}")

class S3ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._pbar = tqdm(total=self._size, unit='B', unit_scale=True, desc=f"[disease_definition_generator] Uploading {os.path.basename(filename)}")

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        self._pbar.update(bytes_amount)
        if self._seen_so_far >= self._size:
            self._pbar.close()

class DiseaseDefinitionEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("[disease_definition_generator] Creating new DiseaseDefinitionEngine instance")
            cls._instance = super(DiseaseDefinitionEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Only initialize once
        if self._initialized:
            logger.debug("[disease_definition_generator] Using existing DiseaseDefinitionEngine instance")
            return
            
        logger.info("[disease_definition_generator] Initializing DiseaseDefinitionEngine...")

        # Load environment variables
        self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        if not self.ANTHROPIC_API_KEY:
            logger.error("[disease_definition_generator] ANTHROPIC_API_KEY environment variable is not set")
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        # Initialize Hugging Face embedding model using Langchain
        self.embed_model = get_embedding_model()
        logger.info(f"[disease_definition_generator] Embedding model initialized with {self.embed_model.model_name}")

        embedding_dimension = self.embed_model.embedding_dimension
        logger.info(f"[disease_definition_generator] Embedding dimension: {embedding_dimension}")

        # Initialize Anthropic LLM
        self.llm = Anthropic(
            api_key=self.ANTHROPIC_API_KEY,
            model="claude-3-5-sonnet-20241022",
            temperature=0,
            max_tokens=2000
        )
        logger.info("[disease_definition_generator] LLM predictor initialized with Claude")

        # Configure global settings before loading index
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        logger.info("[disease_definition_generator] Global settings configured with embedding model and LLM")

        # Initialize S3 client and bucket name
        self.s3_client = boto3.client('s3')
        self.AWS_UPLOAD_BUCKET_NAME = "generate-input-f5bef08a-9228-4f8c-a550-56d842b94088"

        # Try to load the index from S3
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                index_cache_path = os.path.join(temp_dir, 'index.pkl')
                
                # Try to download index from S3
                try:    
                    self.download_index_from_s3(index_cache_path)
                    
                    # Load the index directly from pickle
                    logger.info("[disease_definition_generator] Loading index from downloaded file")
                    with open(index_cache_path, 'rb') as f:
                        self.index = pickle.load(f)
                    
                    # Verify the index is loaded with documents
                    if hasattr(self.index, 'docstore') and self.index.docstore.docs:
                        logger.info("[disease_definition_generator] Index loaded successfully with documents")
                    else:
                        logger.warning("[disease_definition_generator] Loaded index has no documents - a new index needs to be created")
                        self.index = None
                except ClientError as e:
                    logger.info(f"[disease_definition_generator] Failed to download index from S3: {str(e)} - a new index needs to be created")
                    self.index = None
                except FileNotFoundError:
                    logger.info("[disease_definition_generator] No existing index found - a new index needs to be created")
                    self.index = None
                except Exception as e:
                    logger.warning(f"[disease_definition_generator] Error loading index: {str(e)} - a new index needs to be created")
                    self.index = None

        except Exception as e:
            logger.error(f"[disease_definition_generator] Unexpected error during initialization: {str(e)}")
            self.index = None

        if self.index is None:
            logger.info("[disease_definition_generator] No valid index available - system will need to create a new one")

        # Define the prompt template
        self.prompt_template = (
            "Please provide a detailed definition of the disease '{disease_name}' in the following JSON format:\n\n"
            "{\n"
            '  "name": "Disease Name",\n'
            '  "icd10": ["ICD Code 1", "ICD Code 2"],\n'
            '  "description": "Brief description",\n'
            '  "symptoms": [\n'
            '    {\n'
            '      "name": "Symptom Name",\n'
            '      "commonality": "RARE|COMMON|VERY_COMMON",\n'
            '      "severity": "MILD|MODERATE|SEVERE"\n'
            '    }\n'
            '  ],\n'
            '  "commonTests": ["Test 1", "Test 2"],\n'
            '  "treatments": [\n'
            '    {\n'
            '      "name": "Treatment Name",\n'
            '      "effectiveness": 0.0\n'
            '    }\n'
            '  ],\n'
            '  "complications": ["Complication 1", "Complication 2"],\n'
            '  "relatedConditions": ["Condition 1", "Condition 2"]\n'
            '}\n\n'
            "Ensure that all fields are provided and follow the specified types."
        )

        self._initialized = True

    def generate_definition(self, disease_name: str) -> DiseaseDefinition:
        """Generates a structured definition for a given disease"""
        try:
            logger.info(f"[disease_definition_generator] Generating definition for: {disease_name}")

            # Format the prompt with the disease name first
            formatted_prompt = self.prompt_template.format(disease_name=disease_name)

            # Check if index exists
            if self.index is None:
                logger.warning("[disease_definition_generator] No index available - using direct LLM query")
                # Query the LLM directly without using an index
                response = self.llm.complete(formatted_prompt)
                response_text = response.text if hasattr(response, 'text') else response.response
            else:
                # Use index-based query if available
                query_engine = self.index.as_query_engine(
                    llm=Settings.llm,
                    embed_model=self.embed_model,
                    response_mode="tree_summarize",
                    similarity_top_k=3,
                    verbose=True
                )
                response = query_engine.query(formatted_prompt)
                response_text = response.response

            # Parse the response into our Pydantic model
            try:
                # Attempt to parse the response as JSON
                definition_dict = json.loads(response_text)
                definition = DiseaseDefinition(**definition_dict)
                logger.info(f"[disease_definition_generator] Successfully generated definition for {disease_name}")
                return definition

            except json.JSONDecodeError as e:
                logger.error(f"[disease_definition_generator] Failed to parse response as JSON: {str(e)}")
                logger.error(f"[disease_definition_generator] Raw response: {response_text}")
                raise ValueError(f"Invalid response format: {str(e)}")
            except Exception as e:
                logger.error(f"[disease_definition_generator] Failed to create DiseaseDefinition: {str(e)}")
                raise ValueError(f"Failed to create disease definition: {str(e)}")

        except Exception as e:
            logger.error(f"[disease_definition_generator] Error generating definition for {disease_name}: {str(e)}")
            raise

    def save_index_to_s3(self, index_cache_path: str):
        """Save index to S3 with progress bar."""
        try:
            logger.info("[disease_definition_generator] Uploading index to S3")
            self.s3_client.upload_file(
                index_cache_path,
                self.AWS_UPLOAD_BUCKET_NAME,
                "11111/22222/index.pkl",
                Callback=S3ProgressPercentage(index_cache_path)
            )
            logger.info("[disease_definition_generator] Successfully uploaded index to S3")
        except ClientError as e:
            logger.error(f"[disease_definition_generator] Failed to upload index to S3: {str(e)}")
            raise

    def download_index_from_s3(self, index_cache_path: str):
        """Download index from S3 with progress bar."""
        try:
            # Get file size first
            response = self.s3_client.head_object(
                Bucket=self.AWS_UPLOAD_BUCKET_NAME,
                Key="11111/22222/index.pkl"
            )
            file_size = response['ContentLength']

            # Setup progress bar
            pbar = tqdm(total=file_size, unit='B', unit_scale=True, 
                       desc="[disease_definition_generator] Downloading index.pkl")

            # Define callback
            def download_progress(chunk):
                pbar.update(chunk)

            # Download with progress
            self.s3_client.download_file(
                self.AWS_UPLOAD_BUCKET_NAME,
                "11111/22222/index.pkl",
                index_cache_path,
                Callback=download_progress
            )
            pbar.close()
            logger.info("[disease_definition_generator] Successfully downloaded index from S3")
        except ClientError as e:
            logger.error(f"[disease_definition_generator] Failed to download index from S3: {str(e)}")
            raise

# At the bottom of the file, replace:
# engine = DiseaseDefinitionEngine()

# With:
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = DiseaseDefinitionEngine()
    return _engine

# Export the singleton instance
engine = get_engine()

def generate_multiple_definitions(disease_names: List[str]) -> Dict[str, Any]:
    """
    Generate definitions for multiple diseases.
    
    Args:
        disease_names: List of disease names to generate definitions for
        
    Returns:
        Dictionary mapping disease names to their definitions or errors
    """
    logger.info("[disease_definition_generator] Processing multiple disease definitions")
    
    if not isinstance(disease_names, list) or not all(isinstance(name, str) for name in disease_names):
        logger.error("[disease_definition_generator] Invalid disease names format")
        raise ValueError("diseaseNames must be a list of strings")

    results = {}
    for disease in disease_names:
        logger.info(f"[disease_definition_generator] Generating definition for disease: {disease}")
        try:
            definition = engine.generate_definition(disease)
            results[disease] = definition.dict()
            logger.info(f"[disease_definition_generator] Definition generated for {disease}")
        except Exception as e:
            logger.error(f"[disease_definition_generator] Error generating definition for {disease}: {str(e)}")
            results[disease] = {'error': str(e)}

    return results
try:
    engine = DiseaseDefinitionEngine()
    logger.info("[disease_definition_generator] DiseaseDefinitionEngine initialized successfully")
except Exception as e:
    logger.error("[disease_definition_generator] Engine initialization failed: %s", str(e))
    raise

logger.info(f"[disease_definition_generator] Attributes of self.embed_model: {dir(self.embed_model)}")
logger.info(f"[disease_definition_generator] Type of self.embed_model: {type(self.embed_model)}")


