import logging
import os
import json
import boto3
import tempfile
import csv
import io
import shutil
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import platform
import pickle

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
        # Set multiprocessing start method based on platform
        if platform.system() != 'Windows' and hasattr(torch.multiprocessing, 'set_start_method'):
            try:
                torch.multiprocessing.set_start_method('fork', force=True)
            except RuntimeError:
                pass  # Method already set
        elif platform.system() == 'Windows' and hasattr(torch.multiprocessing, 'set_start_method'):
            try:
                torch.multiprocessing.set_start_method('spawn', force=True)
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
    """Singleton class for disease definition generation and document scanning."""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            logger.info("[disease_definition_generator] Creating new DiseaseDefinitionEngine instance")
            cls._instance = super(DiseaseDefinitionEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            logger.info("[disease_definition_generator] Initializing DiseaseDefinitionEngine")
            try:
                self._initialize()
                DiseaseDefinitionEngine._initialized = True
            except Exception as e:
                logger.error(f"[disease_definition_generator] Engine initialization failed: {str(e)}")
                raise

    def _initialize(self):
        """Initialize the engine components."""
        try:
            # Initialize embedding model
            self._embed_model, self._embedding_dimension = get_embedding_model()
            logger.info("[disease_definition_generator] Engine initialized successfully")
        except Exception as e:
            logger.error(f"[disease_definition_generator] Engine initialization failed: {str(e)}")
            raise

    def scan_documents(self, documents: List[str], disease_criteria: Dict) -> Dict:
        """
        Scan documents for disease criteria using the new scoring system.
        """
        try:
            # Perform the scan using the new scoring system
            scan_results = scan_for_disease(documents, disease_criteria, self._embed_model)
            
            # Format the response
            response = format_scan_response(
                scan_results=scan_results,
                mass_tort_name=disease_criteria.get('mass_tort_name', 'Unknown Mass Tort'),
                disease_name=disease_criteria.get('disease_name', 'Unknown Disease')
            )
            
            logger.info(f"[disease_definition_generator] Document scan completed with confidence: {scan_results['overall_confidence']}")
            return response

        except Exception as e:
            logger.error(f"[disease_definition_generator] Error scanning documents: {str(e)}")
            raise

    def get_embedding_model(self):
        """Return the initialized embedding model."""
        return self._embed_model

    def get_embedding_dimension(self):
        """Return the embedding dimension."""
        return self._embedding_dimension

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

@dataclass
class FindingMatch:
    """Represents a matched finding in a document with confidence scoring."""
    text: str  # The matched text from the document
    score: float  # Confidence score (0-1)
    finding_type: str  # 'symptom', 'lab_result', 'procedure', 'risk_factor'
    finding_name: str  # The actual name of the finding that matched
    document: str  # Source document name/id
    page: int  # Page number where the match was found
    source_type: str  # Type of document (e.g., 'medical_record', 'incident_report')

def determine_source_type(document_name: str) -> str:
    """Determine the source type based on the document name."""
    document_name_lower = document_name.lower()
    if any(term in document_name_lower for term in ['report', 'assessment', 'record', 'note']):
        return 'medical_record'
    elif 'incident' in document_name_lower:
        return 'incident_report'
    else:
        return 'other'

def calculate_category_scores(matches: List[FindingMatch], threshold: float) -> Dict[str, float]:
    """Calculate average scores for each category of findings, only using scores above threshold."""
    category_scores = {}
    for category in ['symptoms', 'lab_results', 'procedures', 'risk_factors']:
        # Only use matches above threshold
        valid_matches = [m for m in matches if m.finding_type == category and m.score >= threshold]
        if valid_matches:
            category_scores[category] = sum(m.score for m in valid_matches) / len(valid_matches)
        else:
            category_scores[category] = 0.0
    return category_scores

def create_scoring_summary(matches: List[FindingMatch], disease_criteria: Dict) -> Dict:
    """Create a detailed scoring summary."""
    threshold = disease_criteria.get('scoring_model', {}).get('confidence_threshold', 0.7)
    weights = disease_criteria.get('scoring_model', {}).get('weights', {
        'symptoms': 0.4,
        'lab_results': 0.3,
        'procedures': 0.2,
        'risk_factors': 0.1
    })
    
    # Calculate category scores using only findings above threshold
    category_scores = calculate_category_scores(matches, threshold)
    
    return {
        'total_findings': len(matches),
        'findings_above_threshold': len([m for m in matches if m.score >= threshold]),
        'threshold': threshold,
        'weights': weights,
        'category_scores': category_scores
    }

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split document text into overlapping chunks for better context preservation.
    Returns list of chunks with their metadata.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        # Calculate the page number (this should be replaced with actual page tracking)
        estimated_page = (i // 3000) + 1  # Rough estimate, should be replaced with actual page numbers
        
        chunks.append({
            'text': chunk_text,
            'start_idx': i,
            'end_idx': i + len(chunk_words),
            'page': estimated_page
        })
    
    return chunks

def score_finding(text: str, finding: str, finding_type: str, document_info: Dict[str, Any], embed_model: Any) -> FindingMatch:
    """Score an individual finding against a piece of text using the provided embedding model."""
    try:
        # Normalize text and finding
        text_lower = text.lower()
        finding_lower = finding.lower()

        # Check for exact or near-exact matches first
        if finding_lower in text_lower:
            confidence = 1.0
            matched_text = text[text_lower.find(finding_lower):text_lower.find(finding_lower) + len(finding_lower)]
            logger.debug(f"[disease_definition_generator] Exact match found for '{finding}' with confidence: 1.0")
        else:
            # Get embeddings for both text and finding
            text_embedding = embed_model.get_text_embedding(text_lower)
            finding_embedding = embed_model.get_text_embedding(finding_lower)

            # Convert to numpy arrays for calculation
            text_embedding_np = np.array(text_embedding)
            finding_embedding_np = np.array(finding_embedding)

            # Calculate cosine similarity
            confidence = float(np.dot(text_embedding_np, finding_embedding_np) / 
                         (np.linalg.norm(text_embedding_np) * np.linalg.norm(finding_embedding_np)))
            
            # Find the best matching context
            words = text.split()
            best_context = text
            if len(words) > 20:  # If text is long, find the most relevant context
                for i in range(len(words) - 10):
                    context = ' '.join(words[i:i+20])
                    context_embedding = embed_model.get_text_embedding(context.lower())
                    context_similarity = float(np.dot(np.array(context_embedding), finding_embedding_np) / 
                                         (np.linalg.norm(np.array(context_embedding)) * np.linalg.norm(finding_embedding_np)))
                    if context_similarity > confidence:
                        confidence = context_similarity
                        best_context = context

            matched_text = best_context

        # Apply type-specific boosts
        if finding_type == 'procedures' and any(proc.lower() in text_lower for proc in ['mri', 'ct scan', 'eeg', 'examination']):
            confidence = min(1.0, confidence * 1.2)
        elif finding_type == 'lab_results' and any(term in text_lower for term in ['test', 'level', 'count', 'scale']):
            confidence = min(1.0, confidence * 1.15)

        logger.debug(f"[disease_definition_generator] Scored finding '{finding}' with confidence: {confidence}")

        return FindingMatch(
            text=matched_text,
            score=confidence,
            finding_type=finding_type,
            finding_name=finding,
            document=document_info.get('name', 'Unknown Document'),
            page=document_info.get('page', 1),
            source_type=determine_source_type(document_info.get('name', ''))
        )
    except Exception as e:
        logger.error(f"[disease_definition_generator] Error scoring finding: {str(e)}")
        raise

def calculate_overall_confidence(matches: List[FindingMatch], disease_criteria: Dict) -> float:
    """Calculate overall disease confidence based on weighted finding scores."""
    try:
        threshold = disease_criteria.get('scoring_model', {}).get('confidence_threshold', 0.7)
        weights = disease_criteria.get('scoring_model', {}).get('weights', {
            'symptoms': 0.4,
            'lab_results': 0.3,
            'procedures': 0.2,
            'risk_factors': 0.1
        })

        # Calculate category scores using only findings above threshold
        category_scores = calculate_category_scores(matches, threshold)

        # Calculate weighted overall score
        overall_score = sum(
            category_scores[category] * weight
            for category, weight in weights.items()
        )

        logger.info(f"[disease_definition_generator] Category scores: {category_scores}")
        logger.info(f"[disease_definition_generator] Overall confidence: {overall_score}")

        return overall_score

    except Exception as e:
        logger.error(f"[disease_definition_generator] Error calculating overall confidence: {str(e)}")
        raise

def scan_for_disease(documents: List[Dict[str, Any]], disease_criteria: Dict, embed_model: Any) -> Dict:
    """Scan documents for disease criteria and return detailed match information."""
    try:
        matches = []
        characteristics = {
            'symptoms': [],
            'lab_results': [],
            'procedures': [],
            'risk_factors': []
        }

        # Process each type of finding
        for finding_type in ['symptoms', 'lab_results', 'procedures', 'risk_factors']:
            findings = disease_criteria.get(finding_type, [])
            for finding in findings:
                best_match = None
                best_score = 0.0

                for doc in documents:
                    doc_info = {
                        'name': doc.get('name', 'Unknown Document'),
                        'page': doc.get('page', 1)
                    }

                    # Split document into smaller chunks for more accurate matching
                    chunks = split_into_chunks(doc['text'], chunk_size=500, overlap=100)
                    
                    for chunk in chunks:
                        match = score_finding(
                            chunk['text'], 
                            finding, 
                            finding_type, 
                            {'name': doc_info['name'], 'page': chunk['page']},
                            embed_model
                        )
                        if match.score > best_score:
                            best_score = match.score
                            best_match = match

                if best_match:
                    matches.append(best_match)
                    characteristics[finding_type].append({
                        'name': finding,
                        'confidence': round(best_match.score * 100, 1),  # Convert to percentage
                        'text': best_match.text,
                        'document': best_match.document,
                        'page': best_match.page
                    })

        # Calculate overall confidence using only matches above threshold
        threshold = disease_criteria.get('scoring_model', {}).get('confidence_threshold', 0.7)
        valid_matches = [m for m in matches if m.score >= threshold]
        
        # Calculate category scores
        category_scores = calculate_category_scores(matches, threshold)
        
        # Calculate overall confidence
        weights = disease_criteria.get('scoring_model', {}).get('weights', {
            'symptoms': 0.4,
            'lab_results': 0.3,
            'procedures': 0.2,
            'risk_factors': 0.1
        })
        
        overall_confidence = sum(
            category_scores[cat] * weight
            for cat, weight in weights.items()
        )

        results = {
            'matches': matches,
            'characteristics': characteristics,
            'overall_confidence': round(overall_confidence * 100, 1),  # Convert to percentage
            'disease_criteria': disease_criteria,
            'category_scores': {
                cat: round(score * 100, 1)  # Convert to percentage
                for cat, score in category_scores.items()
            }
        }

        logger.info(f"[disease_definition_generator] Completed disease scan with {len(matches)} matches")
        logger.info(f"[disease_definition_generator] Found {len(valid_matches)} matches above threshold")
        logger.info(f"[disease_definition_generator] Overall confidence: {results['overall_confidence']}%")
        
        return results

    except Exception as e:
        logger.error(f"[disease_definition_generator] Error in disease scan: {str(e)}")
        raise

def format_scan_response(scan_results: Dict, mass_tort_name: str, disease_name: str) -> Dict:
    """Format scan results for frontend consumption with exact structure match."""
    try:
        matches = scan_results['matches']
        characteristics = scan_results['characteristics']
        
        # Create the response structure
        response = {
            'status': 'success',
            'results': [{
                'mass_tort_name': mass_tort_name,
                'diseases': [{
                    'disease_name': disease_name,
                    'overall_confidence': scan_results['overall_confidence'],
                    'matches': [
                        {
                            'finding_type': match.finding_type,
                            'finding_name': match.finding_name,
                            'text': match.text,
                            'score': match.score,
                            'document': match.document,
                            'page': match.page,
                            'source_type': match.source_type
                        } for match in matches
                    ],
                    'characteristics': characteristics,
                    'scoring_summary': create_scoring_summary(matches, scan_results.get('disease_criteria', {}))
                }]
            }]
        }
        
        return response
        
    except Exception as e:
        logger.error(f"[disease_definition_generator] Error formatting scan response: {str(e)}")
        raise

def create_index(documents: List[Document], cache_dir: Optional[str] = None) -> VectorStoreIndex:
    """Create a new vector store index from the provided documents."""
    try:
        logger.info(f"[create_index] Starting index creation with {len(documents)} documents.")
        
        # Initialize embedding model
        embed_model = get_embedding_model()
        logger.info("[create_index] Creating VectorStoreIndex with provided documents")
        
        # Create the index
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            show_progress=True
        )
        logger.info("[create_index] VectorStoreIndex created successfully")

        # Cache the index locally if cache_dir is provided
        if cache_dir:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(cache_dir, 'vector_index.pkl')
                
                # Save locally first
                with open(cache_path, 'wb') as f:
                    pickle.dump(index, f)
                logger.info("[create_index] Created and cached new vector index locally")

                # Upload to S3 if configured
                try:
                    s3_client = boto3.client('s3')
                    bucket_name = os.getenv('AWS_UPLOAD_BUCKET_NAME')
                    if bucket_name:
                        s3_client.upload_file(cache_path, bucket_name, '00000/22222/index.pkl')
                        logger.info("[create_index] Index uploaded to S3 successfully")
                except Exception as s3_error:
                    logger.error(f"[create_index] Error uploading to S3: {str(s3_error)}")
                    # Continue even if S3 upload fails
                
            except Exception as cache_error:
                logger.error(f"[create_index] Error caching index: {str(cache_error)}")
                # Continue even if caching fails
        
        logger.info("[create_index] Index creation completed, proceeding with document scanning")
        return index

    except Exception as e:
        logger.error(f"[create_index] Error creating index: {str(e)}")
        raise

def scan_documents(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scan documents for disease criteria.
    """
    try:
        logger.info("[disease_definition_generator] Starting document scan")
        
        # Extract necessary data
        documents = data.get('documents', [])
        disease_criteria = data.get('disease_criteria', {})
        
        if not documents:
            logger.error("[disease_definition_generator] No documents provided for scanning")
            return {
                'status': 'error',
                'message': 'No documents provided for scanning'
            }
            
        # Create or load index
        try:
            index = create_index(documents)
            logger.info("[disease_definition_generator] Index ready for scanning")
        except Exception as e:
            logger.error(f"[disease_definition_generator] Error preparing index: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error preparing document index: {str(e)}'
            }
        
        # Perform the scan using the index
        try:
            scan_results = scan_for_disease(documents, disease_criteria, index.embed_model)
            logger.info("[disease_definition_generator] Document scan completed successfully")
            
            # Format the response
            response = format_scan_response(
                scan_results=scan_results,
                mass_tort_name=disease_criteria.get('mass_tort_name', 'Unknown Mass Tort'),
                disease_name=disease_criteria.get('disease_name', 'Unknown Disease')
            )
            
            return response
            
        except Exception as e:
            logger.error(f"[disease_definition_generator] Error during document scan: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error scanning documents: {str(e)}'
            }
            
    except Exception as e:
        logger.error(f"[disease_definition_generator] Error in scan_documents: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }