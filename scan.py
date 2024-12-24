import json
import logging
import os
from typing import Dict, List, Optional, TypedDict, Union
from uuid import UUID
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.anthropic import Anthropic
import tempfile
import pickle
import time

# Import configurations
from app_rag_disease_config import (
    AWS_UPLOAD_BUCKET_NAME,
    LOG_CONFIG,
    LLM_MODEL,
    QUERY_ENGINE_CONFIG,
    RAG_ERROR_MESSAGES,
    STORAGE_CONFIG
)

# Import from ragindex
from ragindex import create_project_index

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)

# Type definitions
class Symptom(TypedDict):
    id: str
    name: str

class LabResult(TypedDict):
    id: str
    name: str
    range: str

class DiagnosticProcedure(TypedDict):
    id: str
    name: str

class RiskFactor(TypedDict):
    id: str
    name: str

class ScoringModel(TypedDict):
    id: str
    name: Optional[str]

class Disease(TypedDict):
    id: str
    name: str
    symptoms: List[Symptom]
    labResults: List[LabResult]
    diagnosticProcedures: List[DiagnosticProcedure]
    riskFactors: List[RiskFactor]
    scoringModel: Optional[ScoringModel]

class MassTort(TypedDict):
    id: str
    officialName: str
    diseases: List[Disease]

class ScanInput(TypedDict):
    userId: str
    projectId: str
    massTorts: List[MassTort]

class ScanResult(TypedDict):
    diseaseId: str
    diseaseName: str
    massTortId: str
    massTortName: str
    matchedSymptoms: List[str]
    matchedLabResults: List[str]
    matchedProcedures: List[str]
    matchedRiskFactors: List[str]
    confidence: float
    relevantExcerpts: List[str]

# Initialize S3 client
s3_client = boto3.client('s3')

def load_index(user_id: str, project_id: str) -> Optional[VectorStoreIndex]:
    """Load the index for a given user and project. If index doesn't exist, attempt to create it."""
    try:
        index_key = f"{user_id}/{project_id}/index.pkl"
        logger.info(f"[scan] Loading index from s3://{AWS_UPLOAD_BUCKET_NAME}/{index_key}")
        
        # Create a temporary file with a unique name
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Try to download existing index
            s3_client.download_file(AWS_UPLOAD_BUCKET_NAME, index_key, temp_filename)
            
            # Load the index using pickle
            with open(temp_filename, 'rb') as f:
                index = pickle.load(f)
            
            logger.info("[scan] Successfully loaded existing index")
            return index
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info("[scan] Index not found, attempting to create new index")
                try:
                    # Attempt to create new index
                    index = create_project_index(
                        s3_client=s3_client,
                        bucket_name=AWS_UPLOAD_BUCKET_NAME,
                        user_id=user_id,
                        project_id=project_id,
                        force_refresh=True
                    )
                    if index:
                        logger.info("[scan] Successfully created new index")
                        return index
                    else:
                        logger.error("[scan] Failed to create new index")
                        return None
                except Exception as create_error:
                    logger.error(f"[scan] Error creating new index: {str(create_error)}")
                    return None
            else:
                logger.error(f"[scan] Error accessing S3: {str(e)}")
                return None
                
    except Exception as e:
        logger.error(f"[scan] Error in load_index: {str(e)}")
        return None
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                logger.debug(f"[scan] Cleaned up temporary file: {temp_filename}")
        except Exception as e:
            logger.warning(f"[scan] Failed to clean up temporary file: {str(e)}")

def query_index(index: VectorStoreIndex, query: str) -> Dict:
    """Query the index with the given query text."""
    try:
        logger.info(f"[scan] Initializing query with length: {len(query)} characters -> {query}")
        
        llm = Anthropic(
            model=LLM_MODEL,
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        logger.debug(f"[scan] Initialized Anthropic LLM with model: {LLM_MODEL}")
        
        logger.info("[scan] Creating query engine with configured parameters")
        query_engine = index.as_query_engine(
            llm=llm,
            **QUERY_ENGINE_CONFIG
        )
        
        logger.info("[scan] Executing query against index")
        start_time = time.time()
        response = query_engine.query(query)
        query_time = time.time() - start_time
        logger.info(f"[scan] Query completed in {query_time:.2f} seconds")
        
        # Extract and log relevant excerpts
        excerpts = [str(node.node.text) for node in response.source_nodes]
        logger.info(f"[scan] Found {len(excerpts)} relevant excerpts")
        
        return {
            'response': response.response,
            'relevantExcerpts': excerpts
        }
    except Exception as e:
        logger.error(f"[scan] Error querying index: {str(e)}")
        return {'response': '', 'relevantExcerpts': []}

def analyze_disease(disease: Dict, index: VectorStoreIndex, mass_tort_name: str) -> Dict:
    """Analyze a single disease against the document index."""
    try:
        logger.info(f"[scan] Starting analysis of disease '{disease.get('name', '')}' for mass tort '{mass_tort_name}'")
        
        # Ensure disease is a dictionary and has required fields
        if not isinstance(disease, dict):
            raise ValueError("Disease must be a dictionary")

        disease_name = disease.get('name')
        if not disease_name:
            raise ValueError("Disease must have a name")

        # Log the raw data for debugging
        logger.debug(f"[scan] Raw disease data: {json.dumps(disease, indent=2)}")

        # Extract and validate disease characteristics
        symptoms = disease.get('symptoms', [])
        lab_results = disease.get('labResults', [])
        procedures = disease.get('diagnosticProcedures', [])
        risk_factors = disease.get('riskFactors', [])

        logger.info(f"[scan] Processing disease characteristics for {disease_name}:")
        logger.info(f"[scan] - Symptoms: {len(symptoms)}")
        logger.info(f"[scan] - Lab Results: {len(lab_results)}")
        logger.info(f"[scan] - Procedures: {len(procedures)}")
        logger.info(f"[scan] - Risk Factors: {len(risk_factors)}")

        # Extract names safely, handling both dictionary and string formats
        def safe_get_name(item) -> Optional[str]:
            if isinstance(item, dict):
                name = item.get('name', '').strip()
                return name if name else None
            elif isinstance(item, str):
                name = item.strip()
                return name if name else None
            elif isinstance(item, set):
                # Convert set to string and clean it
                name = str(item).strip('{}').strip()
                return name if name else None
            else:
                logger.warning(f"[scan] Unexpected item type: {type(item)}")
                return None

        # Extract names from each category and filter out None/empty values
        symptom_names = [name for s in symptoms if (name := safe_get_name(s))]
        lab_result_names = [name for l in lab_results if (name := safe_get_name(l))]
        procedure_names = [name for p in procedures if (name := safe_get_name(p))]
        risk_factor_names = [name for r in risk_factors if (name := safe_get_name(r))]

        # Log extracted names for debugging
        logger.debug(f"[scan] Extracted symptom names: {symptom_names}")
        logger.debug(f"[scan] Extracted lab result names: {lab_result_names}")
        logger.debug(f"[scan] Extracted procedure names: {procedure_names}")
        logger.debug(f"[scan] Extracted risk factor names: {risk_factor_names}")

        # Build search queries with proper formatting
        queries = []
        if symptom_names:
            formatted_symptoms = ', '.join(f'"{s}"' for s in symptom_names)
            queries.append(f"symptoms including {formatted_symptoms}")
        if lab_result_names:
            formatted_labs = ', '.join(f'"{l}"' for l in lab_result_names)
            queries.append(f"lab results including {formatted_labs}")
        if procedure_names:
            formatted_procedures = ', '.join(f'"{p}"' for p in procedure_names)
            queries.append(f"procedures including {formatted_procedures}")
        if risk_factor_names:
            formatted_risks = ', '.join(f'"{r}"' for r in risk_factor_names)
            queries.append(f"risk factors including {formatted_risks}")

        # Combine queries with proper context
        if queries:
            combined_query = f"Find evidence related to {disease_name} with the following characteristics: " + " AND ".join(queries)
        else:
            combined_query = f"Find evidence related to {disease_name}"
            
        logger.info(f"[scan] Generated query: {combined_query}")

        # Query the index
        query_engine = index.as_query_engine()
        response = query_engine.query(combined_query)

        # Process response
        matches = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                match = {
                    'text': node.text,
                    'score': float(node.score) if hasattr(node, 'score') else 0.0,
                    'document': node.metadata.get('file_name', 'Unknown')
                }
                matches.append(match)

        logger.info(f"[scan] Found {len(matches)} matches for {disease_name}")

        # Return analysis results
        return {
            'disease_name': disease_name,
            'matches': matches,
            'query': combined_query,
            'total_matches': len(matches),
            'characteristics': {
                'symptoms': symptom_names,
                'lab_results': lab_result_names,
                'procedures': procedure_names,
                'risk_factors': risk_factor_names
            }
        }

    except Exception as e:
        logger.error(f"[scan] Error analyzing disease {disease.get('name', 'Unknown')}: {str(e)}")
        logger.error(f"[scan] Error details:", exc_info=True)
        return {
            'disease_name': disease.get('name', 'Unknown'),
            'error': str(e),
            'matches': [],
            'total_matches': 0
        }

def store_scan_results(user_id: str, project_id: str, mass_tort_id: str, disease_id: str, results: ScanResult):
    """Store scan results in S3."""
    try:
        key = f"{user_id}/{project_id}/{mass_tort_id}/{disease_id}/scanresults.json"
        logger.info(f"[scan] Storing scan results at s3://{AWS_UPLOAD_BUCKET_NAME}/{key}")
        
        s3_client.put_object(
            Bucket=AWS_UPLOAD_BUCKET_NAME,
            Key=key,
            Body=json.dumps(results, indent=2),
            ContentType='application/json'
        )
        return True
    except Exception as e:
        logger.error(f"[scan] Error storing scan results: {str(e)}")
        return False

def check_user_projects(user_id: str, mass_tort_id: str) -> Dict:
    """Check if user has active projects for the given mass tort."""
    try:
        # List objects in the user's directory
        s3_client = boto3.client('s3')
        prefix = f"{user_id}/"
        
        logger.info(f"[scan] Checking projects for user {user_id} and mass tort {mass_tort_id}")
        
        try:
            response = s3_client.list_objects_v2(
                Bucket=AWS_UPLOAD_BUCKET_NAME,
                Prefix=prefix,
                Delimiter='/'
            )
            
            # Get project directories
            projects = []
            for obj in response.get('CommonPrefixes', []):
                project_id = obj.get('Prefix', '').strip('/').split('/')[-1]
                
                # Check if this project has an index
                index_key = f"{user_id}/{project_id}/index.pkl"
                try:
                    s3_client.head_object(Bucket=AWS_UPLOAD_BUCKET_NAME, Key=index_key)
                    projects.append(project_id)
                    logger.info(f"[scan] Found active project: {project_id}")
                except Exception:
                    logger.warning(f"[scan] Project {project_id} has no index")
                    continue
            
            if not projects:
                logger.error(f"[scan] No active projects found for user {user_id}")
                return {
                    'status': 'error',
                    'message': 'No active projects found',
                    'projects': []
                }
            
            logger.info(f"[scan] Found {len(projects)} active projects")
            return {
                'status': 'success',
                'message': f'Found {len(projects)} active projects',
                'projects': projects
            }
            
        except Exception as e:
            logger.error(f"[scan] Error listing projects: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error listing projects: {str(e)}',
                'projects': []
            }
            
    except Exception as e:
        logger.error(f"[scan] Error checking user projects: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'projects': []
        }

def scan_documents(input_data: Dict) -> Dict:
    """
    Scan documents for disease matches.
    
    Args:
        input_data: Dictionary containing scan parameters
        
    Returns:
        Dict containing scan results
    """
    try:
        user_id = input_data.get('userId')
        project_id = input_data.get('projectId')
        mass_torts = input_data.get('massTorts', [])

        if not user_id or not project_id:
            raise ValueError("Missing required userId or projectId")

        logger.info(f"[scan] Starting scan for user {user_id}")
        
        # Check for active projects first
        projects_check = check_user_projects(user_id, project_id)
        if projects_check['status'] == 'error':
            raise ValueError(projects_check['message'])
        
        logger.info(f"[scan] Processing {len(mass_torts)} mass torts")

        start_time = time.time()
        total_diseases = 0
        successful_diseases = 0
        errors = []

        # Load index using the existing load_index function
        logger.info("[scan] Loading index from S3")
        index = load_index(user_id, project_id)
        if not index:
            raise Exception("Failed to load or create index")
        logger.info("[scan] Successfully loaded index")

        # Process each mass tort
        results = []
        for mass_tort in mass_torts:
            mass_tort_name = mass_tort.get('officialName', 'Unknown Mass Tort')
            diseases = mass_tort.get('diseases', [])
            total_diseases += len(diseases)

            logger.info(f"[scan] Processing mass tort: {mass_tort_name} ({len(diseases)} diseases)")
            mass_tort_start = time.time()

            # Analyze each disease
            disease_results = []
            for i, disease in enumerate(diseases, 1):
                logger.info(f"[scan] Analyzing disease {i}/{len(diseases)}: {disease.get('name', 'Unknown')}")
                result = analyze_disease(disease, index, mass_tort_name)
                
                if 'error' not in result:
                    successful_diseases += 1
                else:
                    errors.append(f"{result['disease_name']}: {result.get('error')}")
                
                disease_results.append(result)

            mass_tort_time = time.time() - mass_tort_start
            logger.info(f"[scan] Completed mass tort {mass_tort_name} in {mass_tort_time:.2f} seconds")

            # Add results for this mass tort
            results.append({
                'mass_tort_name': mass_tort_name,
                'diseases': disease_results,
                'processing_time': mass_tort_time
            })

        total_time = time.time() - start_time
        logger.info(f"[scan] Scan completed in {total_time:.2f} seconds. Processed {successful_diseases}/{total_diseases} diseases successfully")

        if successful_diseases < total_diseases:
            error_message = f"Successfully processed {successful_diseases}/{total_diseases} diseases with {len(errors)} errors"
            if errors:
                error_message += f": {'; '.join(errors)}"
            raise Exception(error_message)

        return {
            'status': 'success',
            'message': f'Successfully processed {successful_diseases} diseases',
            'results': results,
            'processing_time': total_time,
            'total_diseases': total_diseases,
            'successful_diseases': successful_diseases
        }

    except Exception as e:
        logger.error(f"[scan] Error during scan: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'results': results if 'results' in locals() else [],
            'processing_time': time.time() - start_time if 'start_time' in locals() else 0,
            'total_diseases': total_diseases if 'total_diseases' in locals() else 0,
            'successful_diseases': successful_diseases if 'successful_diseases' in locals() else 0
        }

# Export the main function and types
__all__ = [
    'scan_documents',
    'ScanInput',
    'ScanResult',
    'MassTort',
    'Disease',
    'Symptom',
    'LabResult',
    'DiagnosticProcedure',
    'RiskFactor',
    'ScoringModel'
]
