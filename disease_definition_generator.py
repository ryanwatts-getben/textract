import logging
import os
import json
import boto3
import tempfile
import pickle
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
    logger.info(f"[disease_definition_generator] Attempting to load embedding model: {model_name} on {device}")

    try:
        # Initialize HuggingFaceEmbedding from llama_index
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            embed_batch_size=32,
            normalize=True
        )

        # Get embedding dimension by generating a test embedding
        test_embedding = embedding_model.get_text_embedding("test")
        embedding_dimension = len(test_embedding)

        logger.info(f"[disease_definition_generator] Successfully loaded embedding model: {model_name}")
        logger.info(f"[disease_definition_generator] Embedding dimension: {embedding_dimension}")

        # Return both the embedding model and the embedding dimension
        return embedding_model, embedding_dimension

    except Exception as e:
        logger.error(f"[disease_definition_generator] Error loading model {model_name}: {str(e)}")
        raise
class DiseaseDefinitionEngine:
    _instance = None  # Class variable to hold the singleton instance

    def __new__(cls):
        if cls._instance is None:
            logger.info("[disease_definition_generator] Creating new DiseaseDefinitionEngine instance")
            cls._instance = super(DiseaseDefinitionEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            logger.debug("[disease_definition_generator] Using existing DiseaseDefinitionEngine instance")
            return

        logger.info("[disease_definition_generator] Initializing DiseaseDefinitionEngine...")

        # Load environment variables
        self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        if not self.ANTHROPIC_API_KEY:
            logger.error("[disease_definition_generator] ANTHROPIC_API_KEY environment variable is not set")
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        # Initialize embedding model
        self.embed_model, self.embedding_dimension = get_embedding_model()
        logger.info(f"[disease_definition_generator] Embedding model initialized with {self.embed_model.model_name}")
        logger.info(f"[disease_definition_generator] Embedding dimension: {self.embedding_dimension}")

        # Initialize LLM predictor
        self.llm = Anthropic(
            api_key=self.ANTHROPIC_API_KEY,
            model="claude-3-5-sonnet-20241022",
            temperature=0,
            max_tokens=2000
        )
        logger.info("[disease_definition_generator] LLM predictor initialized with Claude")

        # Configure settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        logger.info("[disease_definition_generator] Global settings configured with embedding model and LLM")

        # Attempt to load the index
        try:
            self.index = self.load_index()
            logger.info("[disease_definition_generator] Index loaded successfully")
        except Exception as e:
            logger.warning(f"[disease_definition_generator] No index was found: {str(e)}")
            self.index = None  # Handle absence of index gracefully

        # Initialize prompt template
        self.prompt_template = (
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

        # Set initialization flag
        self._initialized = True
        logger.info("[disease_definition_generator] DiseaseDefinitionEngine initialized successfully")

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
                
                # Load the index from the temporary file
                with open(temp_index_path, 'rb') as f:
                    index = pickle.load(f)
                    
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

    def save_index(self, index: VectorStoreIndex) -> bool:
        """Save the index to storage."""
        try:
            # Define the storage path
            storage_path = os.path.join(os.path.dirname(__file__), 'storage')
            
            # Create storage directory if it doesn't exist
            os.makedirs(storage_path, exist_ok=True)
            
            # Save the index
            index.storage_context.persist(persist_dir=storage_path)
            logger.info(f"[disease_definition_generator] Index saved successfully to: {storage_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"[disease_definition_generator] Error saving index: {str(e)}")
            return False

    def generate_definition(self, disease_name: str) -> DiseaseDefinition:
        """Generates a structured definition for a given disease"""
        try:
            logger.info(f"[disease_definition_generator] Generating definition for: {disease_name}")

            # Format the prompt with the disease name
            formatted_prompt = self.prompt_template.format(disease_name=disease_name)
            
            # Get response either from index or directly from LLM
            if self.index is not None:
                logger.info("[disease_definition_generator] Using index for query")
                query_engine = self.index.as_query_engine(
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

# Initialize the engine at startup
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

try:
    engine = DiseaseDefinitionEngine()
    logger.info("[disease_definition_generator] DiseaseDefinitionEngine initialized successfully")
except Exception as e:
    logger.error("[disease_definition_generator] Engine initialization failed: %s", str(e))
    raise