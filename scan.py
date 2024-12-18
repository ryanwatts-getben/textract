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

def analyze_disease(index: VectorStoreIndex, disease: Disease, mass_tort: MassTort) -> ScanResult:
    """Analyze a single disease against the index."""
    start_time = time.time()
    logger.info(f"[scan] Starting analysis of disease '{disease['name']}' for mass tort '{mass_tort['officialName']}'")
    
    # Log disease characteristics
    logger.info(f"[scan] Disease characteristics:"
                f"\n - Symptoms: {len(disease['symptoms'])}"
                f"\n - Lab Results: {len(disease['labResults'])}"
                f"\n - Diagnostic Procedures: {len(disease['diagnosticProcedures'])}"
                f"\n - Risk Factors: {len(disease['riskFactors'])}")
    
    # Build query from disease details
    query_parts = [
        f"Find evidence related to {disease['name']} with the following characteristics:",
        "Symptoms: " + ", ".join(s['name'] for s in disease['symptoms']),
        "Lab Results: " + ", ".join(f"{l['name']} ({l['range']})" for l in disease['labResults']),
        "Diagnostic Procedures: " + ", ".join(p['name'] for p in disease['diagnosticProcedures']),
        "Risk Factors: " + ", ".join(r['name'] for r in disease['riskFactors'])
    ]
    query = "\n".join(query_parts)
    logger.debug(f"[scan] Generated query:\n{query}")
    
    # Query the index
    logger.info("[scan] Querying index for disease evidence")
    result = query_index(index, query)
    
    # Analyze matches
    logger.info("[scan] Analyzing matches in response")
    matched_symptoms = []
    matched_lab_results = []
    matched_procedures = []
    matched_risk_factors = []
    
    response_text = result['response'].lower()
    
    # Process symptoms
    logger.info("[scan] Processing symptom matches")
    for symptom in disease['symptoms']:
        if symptom['name'].lower() in response_text:
            matched_symptoms.append(symptom['name'])
            logger.debug(f"[scan] Matched symptom: {symptom['name']}")
            
    # Process lab results
    logger.info("[scan] Processing lab result matches")
    for lab_result in disease['labResults']:
        if lab_result['name'].lower() in response_text:
            matched_lab_results.append(f"{lab_result['name']} ({lab_result['range']})")
            logger.debug(f"[scan] Matched lab result: {lab_result['name']}")
            
    # Process procedures
    logger.info("[scan] Processing diagnostic procedure matches")
    for procedure in disease['diagnosticProcedures']:
        if procedure['name'].lower() in response_text:
            matched_procedures.append(procedure['name'])
            logger.debug(f"[scan] Matched procedure: {procedure['name']}")
            
    # Process risk factors
    logger.info("[scan] Processing risk factor matches")
    for risk_factor in disease['riskFactors']:
        if risk_factor['name'].lower() in response_text:
            matched_risk_factors.append(risk_factor['name'])
            logger.debug(f"[scan] Matched risk factor: {risk_factor['name']}")
    
    # Calculate confidence
    total_items = len(disease['symptoms']) + len(disease['labResults']) + \
                 len(disease['diagnosticProcedures']) + len(disease['riskFactors'])
    matched_items = len(matched_symptoms) + len(matched_lab_results) + \
                   len(matched_procedures) + len(matched_risk_factors)
    confidence = matched_items / total_items if total_items > 0 else 0.0
    
    # Log match summary
    logger.info(f"[scan] Match summary for {disease['name']}:"
                f"\n - Symptoms: {len(matched_symptoms)}/{len(disease['symptoms'])}"
                f"\n - Lab Results: {len(matched_lab_results)}/{len(disease['labResults'])}"
                f"\n - Procedures: {len(matched_procedures)}/{len(disease['diagnosticProcedures'])}"
                f"\n - Risk Factors: {len(matched_risk_factors)}/{len(disease['riskFactors'])}"
                f"\n - Confidence Score: {confidence:.2%}")
    
    analysis_time = time.time() - start_time
    logger.info(f"[scan] Disease analysis completed in {analysis_time:.2f} seconds")
    
    return ScanResult(
        diseaseId=disease['id'],
        diseaseName=disease['name'],
        massTortId=mass_tort['id'],
        massTortName=mass_tort['officialName'],
        matchedSymptoms=matched_symptoms,
        matchedLabResults=matched_lab_results,
        matchedProcedures=matched_procedures,
        matchedRiskFactors=matched_risk_factors,
        confidence=confidence,
        relevantExcerpts=result['relevantExcerpts']
    )

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

def scan_documents(input_data: ScanInput) -> Dict:
    """
    Main scanning function that processes the input data according to the workflow.
    Returns:
    {
        'status': 'success' | 'error',
        'message': str,
        'results': List[ScanResult],  # Only present on success
        'errors': List[Dict]  # List of errors encountered during processing
    }
    """
    start_time = time.time()
    logger.info(f"[scan] Starting scan for user {input_data['userId']}")
    logger.info(f"[scan] Processing {len(input_data['massTorts'])} mass torts")
    
    results = []
    errors = []
    total_diseases = sum(len(mt['diseases']) for mt in input_data['massTorts'])
    processed_diseases = 0
    
    try:
        # Load the index
        logger.info("[scan] Loading index from S3")
        index = load_index(input_data['userId'], input_data['projectId'])
        if not index:
            error_msg = "Failed to load or create index"
            logger.error(f"[scan] {error_msg}")
            errors.append({
                'type': 'index_error',
                'message': error_msg,
                'userId': input_data['userId'],
                'projectId': input_data['projectId']
            })
            return {
                'status': 'error',
                'message': error_msg,
                'errors': errors
            }
            
        for mass_tort in input_data['massTorts']:
            mt_start_time = time.time()
            logger.info(f"[scan] Processing mass tort: {mass_tort['officialName']} "
                       f"({len(mass_tort['diseases'])} diseases)")
            
            for disease in mass_tort['diseases']:
                processed_diseases += 1
                disease_start_time = time.time()
                logger.info(f"[scan] Analyzing disease {processed_diseases}/{total_diseases}: "
                          f"{disease['name']}")
                
                try:
                    # Analyze disease
                    scan_result = analyze_disease(index, disease, mass_tort)
                    
                    # Store results
                    logger.info(f"[scan] Storing results for disease: {disease['name']}")
                    if store_scan_results(
                        input_data['userId'],
                        input_data['projectId'],
                        mass_tort['id'],
                        disease['id'],
                        scan_result
                    ):
                        results.append(scan_result)
                        disease_time = time.time() - disease_start_time
                        logger.info(f"[scan] Completed disease analysis in {disease_time:.2f} seconds")
                    else:
                        error_msg = f"Failed to store results for disease {disease['name']}"
                        logger.error(f"[scan] {error_msg}")
                        errors.append({
                            'type': 'storage_error',
                            'message': error_msg,
                            'diseaseId': disease['id'],
                            'massTortId': mass_tort['id']
                        })
                except Exception as e:
                    error_msg = f"Error analyzing disease {disease['name']}: {str(e)}"
                    logger.error(f"[scan] {error_msg}")
                    errors.append({
                        'type': 'analysis_error',
                        'message': error_msg,
                        'diseaseId': disease['id'],
                        'massTortId': mass_tort['id']
                    })
                    continue  # Continue with next disease
            
            mt_time = time.time() - mt_start_time
            logger.info(f"[scan] Completed mass tort {mass_tort['officialName']} "
                       f"in {mt_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"[scan] Scan completed in {total_time:.2f} seconds. "
                   f"Processed {len(results)}/{total_diseases} diseases successfully")
        
        # Determine overall status
        status = 'success' if len(results) > 0 else 'error'
        message = f"Successfully processed {len(results)}/{total_diseases} diseases"
        if errors:
            message += f" with {len(errors)} errors"
        
        return {
            'status': status,
            'message': message,
            'results': results,
            'errors': errors
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Error in scan process after {total_time:.2f} seconds: {str(e)}"
        logger.error(f"[scan] {error_msg}")
        errors.append({
            'type': 'scan_error',
            'message': error_msg
        })
        return {
            'status': 'error',
            'message': error_msg,
            'errors': errors
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
