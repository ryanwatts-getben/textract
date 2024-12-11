import logging
import os
import json
import boto3
import tempfile
import pickle

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
from langchain.embeddings.huggingface import HuggingFaceEmbeddings #type: ignore
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
    alternateNames: Optional[List[str]] = []
    icd10: Optional[str] = None
    isGlobal: Optional[bool] = None
    symptoms: List[Symptom]
    labResults: Optional[List[LabResult]] = []
    diagnosticProcedures: List[DiagnosticProcedure]
    treatments: Optional[List[Treatment]] = []
    prognosis: Optional[Prognosis] = None

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
class DiseaseDefinitionEngine:
    def __init__(self):
        logger.info("[disease_definition_generator] Initializing DiseaseDefinitionEngine...")

        # Load environment variables
        self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        if not self.ANTHROPIC_API_KEY:
            logger.error("[disease_definition_generator] ANTHROPIC_API_KEY environment variable is not set")
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        # Initialize Hugging Face embedding model using Langchain
        self.embed_model = get_embedding_model()
        logger.info("[disease_definition_generator] Embedding model initialized with all-MiniLM-L6-v2")

        # Initialize Anthropic LLM
        self.llm = Anthropic(
            api_key=self.ANTHROPIC_API_KEY,  # Changed from ANTHROPIC_API_KEY parameter
            model="claude-3-5-sonnet-20241022",
            temperature=0,
            max_tokens=2000
        )
        logger.info("[disease_definition_generator] LLM predictor initialized with Claude")

        # Configure global settings before loading index
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        logger.info("[disease_definition_generator] Global settings configured with embedding model and LLM")

        # Define the prompt template
        self.prompt_template = PromptTemplate(
            template=(
                "Given the disease name '{disease_name}', provide a comprehensive medical definition "
                "and analysis in JSON format that matches the following structure:\n"
                "{\n"
                '  "name": "string",\n'
                '  "alternateNames": ["string"],\n'
                '  "icd10": "string",\n'
                '  "isGlobal": boolean,\n'
                '  "symptoms": [\n'
                '    {\n'
                '      "name": "string",\n'
                '      "commonality": "VERY_COMMON|COMMON|UNCOMMON|RARE",\n'
                '      "severity": "MILD|MODERATE|SEVERE|CRITICAL"\n'
                '    }\n'
                '  ],\n'
                '  "labResults": [\n'
                '    {\n'
                '      "name": "string",\n'
                '      "unit": "string",\n'
                '      "normalRange": "string",\n'
                '      "significance": "string"\n'
                '    }\n'
                '  ],\n'
                '  "diagnosticProcedures": [\n'
                '    {\n'
                '      "name": "string",\n'
                '      "accuracy": float,\n'
                '      "invasiveness": "NON_INVASIVE|MINIMALLY_INVASIVE|INVASIVE"\n'
                '    }\n'
                '  ],\n'
                '  "treatments": [\n'
                '    {\n'
                '      "name": "string",\n'
                '      "type": "MEDICATION|SURGERY|THERAPY|LIFESTYLE|OTHER",\n'
                '      "effectiveness": float\n'
                '    }\n'
                '  ],\n'
                '  "prognosis": {\n'
                '    "survivalRate": float,\n'
                '    "chronicityLevel": "ACUTE|SUBACUTE|CHRONIC",\n'
                '    "qualityOfLifeImpact": "MILD|MODERATE|SEVERE"\n'
                '  }\n'
                "}\n\n"
                "Ensure all information is medically accurate and provide the response in valid JSON format. "
                "Use the exact structure shown above. For numerical values like accuracy and effectiveness, "
                "use values between 0 and 1. For survival rate, use a decimal between 0 and 1."
            )
        )

        # Initialize S3 client and bucket name
        self.s3_client = boto3.client('s3')
        self.AWS_UPLOAD_BUCKET_NAME = "generate-input-f5bef08a-9228-4f8c-a550-56d842b94088"

        # Load the index from S3
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                index_cache_path = os.path.join(temp_dir, 'index.pkl')
                
                # Download index from S3
                logger.info("[disease_definition_generator] Downloading index from S3")
                self.s3_client.download_file(
                    self.AWS_UPLOAD_BUCKET_NAME,
                    "00000/11111/index.pkl",
                    index_cache_path
                )
                
                # Load the index directly from pickle
                logger.info("[disease_definition_generator] Loading index from downloaded file")
                with open(index_cache_path, 'rb') as f:
                    self.index = pickle.load(f)
                
                # Verify the index is loaded with documents
                if hasattr(self.index, 'docstore') and self.index.docstore.docs:
                    logger.info("[disease_definition_generator] Index loaded successfully with documents")
                else:
                    raise ValueError("Loaded index has no documents")

        except ClientError as e:
            logger.error("[disease_definition_generator] Failed to download index from S3: %s", str(e))
            raise RuntimeError(f"Failed to download index: {str(e)}")
        except Exception as e:
            logger.error("[disease_definition_generator] Failed to load index: %s", str(e))
            raise RuntimeError(f"Failed to load index: {str(e)}")

    def generate_definition(self, disease_name: str) -> DiseaseDefinition:
        """Generates a structured definition for a given disease"""
        try:
            logger.info(f"[disease_definition_generator] Generating definition for: {disease_name}")

            # Format the prompt with the disease name
            formatted_prompt = self.prompt_template.format(disease_name=disease_name)
            
            # Create a query engine using the loaded index
            query_engine = self.index.as_query_engine(
                llm=Settings.llm,
                embed_model=Settings.embed_model,
                response_mode="tree_summarize",
                similarity_top_k=3,  # Adjust based on requirements
                verbose=True
            )

            # Query the index with the formatted prompt
            response = query_engine.query(formatted_prompt)

            # Parse the response into our Pydantic model
            try:
                # Attempt to parse the response as JSON
                definition_dict = json.loads(response.response)
                definition = DiseaseDefinition(**definition_dict)
                logger.info(f"[disease_definition_generator] Successfully generated definition for {disease_name}")
                return definition

            except json.JSONDecodeError as e:
                logger.error(f"[disease_definition_generator] Failed to parse response as JSON: {str(e)}")
                logger.error(f"[disease_definition_generator] Raw response: {response.response}")
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