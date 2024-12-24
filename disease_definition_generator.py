import logging
import os
import json
import boto3
import tempfile
import csv
import io
import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path

from flask import request, jsonify
from pydantic import BaseModel, Field
from botocore.exceptions import ClientError
from pypdf import PdfReader
from docx.document import Document as DocxDocument
from defusedxml import ElementTree as DefusedET

# Import Hugging Face embedding and LLM models
from llama_index.core import (
    load_index_from_storage, 
    VectorStoreIndex, 
    Document, 
    StorageContext,
    PromptTemplate,
    Settings
)
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.llms.anthropic import Anthropic
import torch
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.langchain import LangchainEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported file types
SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.json', '.xml', '.xsd', '.doc', '.docx', '.csv', '.rrf'}

def extract_text_from_file(content: bytes, file_extension: str) -> str:
    """
    Extract text from various file types.
    
    Args:
        content (bytes): The binary content of the file
        file_extension (str): The file extension (including the dot)
        
    Returns:
        str: The extracted text
    """
    try:
        if file_extension == '.txt':
            return content.decode('utf-8', errors='ignore')
            
        elif file_extension == '.pdf':
            with io.BytesIO(content) as pdf_buffer:
                reader = PdfReader(pdf_buffer)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text() or '')
                return '\n'.join(text)
                
        elif file_extension == '.docx':
            with io.BytesIO(content) as docx_buffer:
                doc = DocxDocument(docx_buffer)
                return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
                
        elif file_extension == '.csv':
            text_data = content.decode('utf-8', errors='ignore')
            csv_reader = csv.reader(text_data.splitlines())
            return '\n'.join(','.join(row) for row in csv_reader)
            
        elif file_extension in {'.xml', '.xsd'}:
            try:
                root = DefusedET.fromstring(content.decode('utf-8', errors='ignore'))
                text_content = []
                
                def extract_text_from_element(element, path=""):
                    current_path = f"{path}/{element.tag}" if path else element.tag
                    
                    # Add element text if present
                    if element.text and element.text.strip():
                        text_content.append(f"{current_path}: {element.text.strip()}")
                    
                    # Process attributes
                    for key, value in element.attrib.items():
                        text_content.append(f"{current_path}[@{key}]: {value}")
                    
                    # Process child elements
                    for child in element:
                        extract_text_from_element(child, current_path)
                
                extract_text_from_element(root)
                return '\n'.join(text_content)
                
            except Exception as xml_e:
                logger.error(f"[disease_definition_generator] Error parsing XML/XSD content: {str(xml_e)}")
                return content.decode('utf-8', errors='ignore')
                
        elif file_extension == '.rrf':
            # RRF (Rich Release Format) files are typically pipe-delimited
            text_data = content.decode('utf-8', errors='ignore')
            return '\n'.join(line for line in text_data.splitlines())
            
        elif file_extension == '.json':
            try:
                json_data = json.loads(content.decode('utf-8', errors='ignore'))
                if isinstance(json_data, dict):
                    return '\n'.join(f"{k}: {v}" for k, v in json_data.items())
                elif isinstance(json_data, list):
                    return '\n'.join(str(item) for item in json_data)
                else:
                    return str(json_data)
            except json.JSONDecodeError:
                return content.decode('utf-8', errors='ignore')
        
        else:
            logger.warning(f"[disease_definition_generator] Unsupported file extension: {file_extension}")
            return content.decode('utf-8', errors='ignore')
            
    except Exception as e:
        logger.error(f"[disease_definition_generator] Error extracting text from {file_extension} file: {str(e)}")
        return ""

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
    alternateNames: Optional[List[str]] = []
    icd10: Optional[str] = None
    isGlobal: Optional[bool] = None
    symptoms: List[Symptom]
    labResults: Optional[List[LabResult]] = []
    diagnosticProcedures: List[DiagnosticProcedure]
    treatments: Optional[List[Treatment]] = []
    prognosis: Optional[Prognosis] = None

def get_embedding_model(model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"):
    """Initialize the embedding model with proper device configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[disease_definition_generator] Using device: {device} for embeddings")

    try:
        # Set multiprocessing start method to 'fork' on Unix systems
        if hasattr(torch.multiprocessing, 'set_start_method'):
            try:
                torch.multiprocessing.set_start_method('fork', force=True)
            except RuntimeError:
                pass  # Method already set

        # Create HuggingFaceEmbeddings with specific configurations for BioBERT
        base_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                'device': str(device),
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )
        
        # Wrap with LangchainEmbedding for LlamaIndex compatibility
        embedding_model = LangchainEmbedding(
            langchain_embeddings=base_embeddings
        )

        # Get embedding dimension
        test_embedding = embedding_model.get_text_embedding("test")
        embedding_dimension = len(test_embedding)

        logger.info(f"[disease_definition_generator] Successfully loaded BioBERT model")
        logger.info(f"[disease_definition_generator] Embedding dimension: {embedding_dimension}")

        return embedding_model, embedding_dimension

    except Exception as e:
        logger.error(f"[disease_definition_generator] Error loading BioBERT model: {str(e)}")
        raise

class DiseaseDefinitionEngine:
    """Engine for generating disease definitions using LLM and embeddings."""
    
    # Class variables for singleton pattern
    _instance = None
    _initialized = False
    _embed_model = None
    _embedding_dimension = None
    _llm = None
    _prompt_template = None
    _index = None

    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            logger.info("[disease_definition_generator] Creating new DiseaseDefinitionEngine instance")
            cls._instance = super(DiseaseDefinitionEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the engine components if not already initialized."""
        if self._initialized:
            logger.debug("[disease_definition_generator] Using existing DiseaseDefinitionEngine instance")
            return

        logger.info("[disease_definition_generator] Initializing DiseaseDefinitionEngine")
        try:
            # Load environment variables
            self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
            if not self.ANTHROPIC_API_KEY:
                logger.error("[disease_definition_generator] ANTHROPIC_API_KEY environment variable is not set")
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

            # Initialize embedding model only if not already initialized
            if self._embed_model is None:
                self._embed_model, self._embedding_dimension = get_embedding_model()
                logger.info(f"[disease_definition_generator] Embedding model initialized")
                logger.info(f"[disease_definition_generator] Embedding dimension: {self._embedding_dimension}")

            # Initialize LLM predictor only if not already initialized
            if self._llm is None:
                self._llm = Anthropic(
                    api_key=self.ANTHROPIC_API_KEY,
                    model="claude-3-5-sonnet-20241022",
                    temperature=0,
                    max_tokens=2000
                )
                logger.info("[disease_definition_generator] LLM predictor initialized with Claude")

            # Configure settings
            Settings.embed_model = self._embed_model
            Settings.llm = self._llm
            logger.info("[disease_definition_generator] Global settings configured")

            # Initialize prompt template
            self._prompt_template = (
                "Please provide a detailed definition of the disease '{disease_name}' in the following JSON format. "
                "Make sure to include all required fields and follow the exact format:\n\n"
                "{{\n"
                '  "name": "{disease_name}",\n'
                '  "alternateNames": ["string"],\n'
                '  "icd10": "A00.0",\n'
                '  "isGlobal": true,\n'
                '  "symptoms": [\n'
                '    {{\n'
                '      "name": "string",\n'
                '      "commonality": "VERY_COMMON",\n'
                '      "severity": "MILD"\n'
                '    }}\n'
                '  ],\n'
                '  "labResults": [\n'
                '    {{\n'
                '      "name": "string",\n'
                '      "unit": "string",\n'
                '      "normalRange": "string",\n'
                '      "significance": "string"\n'
                '    }}\n'
                '  ],\n'
                '  "diagnosticProcedures": [\n'
                '    {{\n'
                '      "name": "string",\n'
                '      "accuracy": 0.95,\n'
                '      "invasiveness": "NON_INVASIVE"\n'
                '    }}\n'
                '  ],\n'
                '  "treatments": [\n'
                '    {{\n'
                '      "name": "string",\n'
                '      "type": "MEDICATION",\n'
                '      "effectiveness": 0.85\n'
                '    }}\n'
                '  ],\n'
                '  "prognosis": {{\n'
                '    "survivalRate": 0.8,\n'
                '    "chronicityLevel": "ACUTE",\n'
                '    "qualityOfLifeImpact": "MILD"\n'
                '  }}\n'
                "}}\n\n"
                "Please ensure the response is valid JSON and includes accurate medical information for {disease_name}. "
                "The values should be based on current medical knowledge and research."
            )
            logger.info("[disease_definition_generator] Prompt template initialized")

            self._initialized = True
            logger.info("[disease_definition_generator] DiseaseDefinitionEngine initialized successfully")

        except Exception as e:
            logger.error(f"[disease_definition_generator] Engine initialization failed: {str(e)}")
            raise

    @property
    def embed_model(self):
        """Get the embedding model."""
        return self._embed_model

    @property
    def embedding_dimension(self):
        """Get the embedding dimension."""
        return self._embedding_dimension

    @property
    def llm(self):
        """Get the LLM model."""
        return self._llm

    @property
    def prompt_template(self):
        """Get the prompt template."""
        return self._prompt_template

    def load_index(self) -> Optional[VectorStoreIndex]:
        """Load the index from storage if it exists."""
        try:
            # S3 bucket and path configuration
            BUCKET_NAME = "generate-input-f5bef08a-9228-4f8c-a550-56d842b94088"
            INDEX_KEY = "00000/22222/index.pkl"
            
            # Create a temporary directory for downloading the index
            temp_dir = tempfile.mkdtemp()
            temp_index_path = os.path.join(temp_dir, 'index.pkl')
            
            try:
                # Download index from S3
                logger.info(f"[disease_definition_generator] Attempting to load index from s3://{BUCKET_NAME}/{INDEX_KEY}")
                s3_client = boto3.client('s3')
                s3_client.download_file(BUCKET_NAME, INDEX_KEY, temp_index_path)
                
                # Check CUDA availability
                import torch
                if not torch.cuda.is_available():
                    logger.info("[disease_definition_generator] CUDA not available, using CPU")
                    # Load with CPU mapping
                    index = torch.load(temp_index_path, map_location='cpu')
                else:
                    # Load normally
                    index = torch.load(temp_index_path)
                    
                logger.info("[disease_definition_generator] Index loaded successfully from S3")
                return index
                
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.warning("[disease_definition_generator] Index file not found in S3")
                else:
                    logger.error(f"[disease_definition_generator] Error accessing S3: {str(e)}")
                return None
                
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"[disease_definition_generator] Error loading index: {str(e)}")
            return None

    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the index, loading it if necessary."""
        if self._index is None:
            self._index = self.load_index()
        return self._index

    def generate_definition(self, disease_name: str) -> DiseaseDefinition:
        """Generates a structured definition for a given disease"""
        try:
            logger.info(f"[disease_definition_generator] Generating definition for: {disease_name}")

            # Format the prompt with the disease name
            formatted_prompt = self.prompt_template.format(disease_name=disease_name)
            
            # Get response either from index or directly from LLM
            index = self.get_index()
            if index is not None:
                logger.info("[disease_definition_generator] Using index for query")
                query_engine = index.as_query_engine(
                    llm=Settings.llm,
                    response_mode="compact",
                    similarity_top_k=3,
                    verbose=True
                )
                response = query_engine.query(formatted_prompt)
                response_text = response.response
            else:
                logger.info("[disease_definition_generator] Index not available, querying LLM directly")
                response_text = Settings.llm.complete(formatted_prompt).text

            # Parse the response into our Pydantic model
            try:
                # Attempt to parse the response as JSON
                logger.debug(f"[disease_definition_generator] Raw response: {response_text}")
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

# Initialize the engine at startup (but don't load index yet)
try:
    engine = DiseaseDefinitionEngine()
    logger.info("[disease_definition_generator] DiseaseDefinitionEngine initialized successfully")
except Exception as e:
    logger.error("[disease_definition_generator] Engine initialization failed: %s", str(e))
    raise

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