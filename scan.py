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

# Create logger for this module
logger = logging.getLogger(__name__)

# Configure other loggers
logging.getLogger("openai").setLevel(logging.WARNING)  # Set OpenAI logger to WARNING level
logging.getLogger("httpx").setLevel(logging.WARNING)   # Set httpx logger to WARNING level
logging.getLogger("urllib3").setLevel(logging.WARNING) # Set urllib3 logger to WARNING level

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
            
            # Load the index using pickle with CPU device mapping
            with open(temp_filename, 'rb') as f:
                # Use map_location to handle CUDA/CPU device mapping
                index = pickle.load(f)
                if hasattr(index, 'to'):
                    index = index.to('cpu')
                elif hasattr(index, '_device'):
                    index._device = 'cpu'
            
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
        
        # Initialize Anthropic LLM
        llm = Anthropic(
            model=LLM_MODEL,
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        logger.debug(f"[scan] Initialized Anthropic LLM with model: {LLM_MODEL}")
        
        # Create query engine with explicit LLM configuration
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

def analyze_disease(index: VectorStoreIndex, disease_name: str, disease_criteria: Dict) -> Dict:
    """Analyze a single disease using the provided index."""
    try:
        logger.info(f"[scan] Starting analysis of disease '{disease_name}'")
        
        # Verify index is valid
        if not isinstance(index, VectorStoreIndex):
            logger.error("[scan] Invalid index object provided")
            raise ValueError("Invalid index object provided")
            
        # Log disease characteristics
        logger.info(f"[scan] Processing disease characteristics for {disease_name}:")
        logger.info(f"[scan] - Symptoms: {len(disease_criteria.get('symptoms', []))}")
        logger.info(f"[scan] - Lab Results: {len(disease_criteria.get('lab_results', []))}")
        logger.info(f"[scan] - Procedures: {len(disease_criteria.get('procedures', []))}")
        logger.info(f"[scan] - Risk Factors: {len(disease_criteria.get('risk_factors', []))}")

        # Generate query from disease criteria
        query = generate_disease_query(disease_name, disease_criteria)
        logger.info(f"[scan] Generated query: {query}")

        # Query the index using our configured function
        query_result = query_index(index, query)
        if not query_result['response']:
            logger.warning("[scan] No relevant information found in documents")
            return {
                'status': 'no_matches',
                'disease_name': disease_name,
                'message': 'No relevant information found in documents'
            }
        
        # Process response
        result = {
            'status': 'success',
            'disease_name': disease_name,
            'matches': process_response(query_result['response'], disease_criteria)
        }
        
        # Log the disease analysis result
        logger.info(f"[scan] Disease Analysis Result for '{disease_name}':")
        logger.info(json.dumps(result, indent=2))
        
        return result

    except Exception as e:
        error_result = {
            'status': 'error',
            'disease_name': disease_name,
            'error': str(e)
        }
        logger.error(f"[scan] Error analyzing disease {disease_name}: {str(e)}")
        logger.error("[scan] Error details:", exc_info=True)
        logger.error(json.dumps(error_result, indent=2))
        raise

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
                result = analyze_disease(index, disease.get('name', 'Unknown'), disease)
                
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

        response = {
            'status': 'success',
            'message': f'Successfully processed {successful_diseases} diseases',
            'results': results,
            'processing_time': total_time,
            'total_diseases': total_diseases,
            'successful_diseases': successful_diseases
        }

        # Log the response in a readable format
        logger.info("[scan] Frontend Response:")
        logger.info(json.dumps(response, indent=2))

        return response

    except Exception as e:
        error_response = {
            'status': 'error',
            'message': str(e),
            'results': results if 'results' in locals() else [],
            'processing_time': time.time() - start_time if 'start_time' in locals() else 0,
            'total_diseases': total_diseases if 'total_diseases' in locals() else 0,
            'successful_diseases': successful_diseases if 'successful_diseases' in locals() else 0
        }
        
        # Log the error response
        logger.error("[scan] Frontend Error Response:")
        logger.error(json.dumps(error_response, indent=2))
        
        return error_response

def generate_disease_query(disease_name: str, disease_criteria: Dict) -> str:
    """Generate a query string from disease criteria."""
    try:
        logger.info(f"[scan] Generating query for disease: {disease_name}")
        
        query_parts = [f"Find evidence related to {disease_name} with the following characteristics:"]
        
        # Process symptoms
        symptoms = disease_criteria.get('symptoms', [])
        if symptoms:
            symptom_names = [s.get('name', '') if isinstance(s, dict) else str(s) for s in symptoms]
            symptom_names = [s for s in symptom_names if s]  # Filter out empty strings
            if symptom_names:
                formatted_symptoms = ', '.join(f'"{s}"' for s in symptom_names)
                query_parts.append(f"symptoms including {formatted_symptoms}")
        
        # Process lab results
        lab_results = disease_criteria.get('labResults', [])
        if lab_results:
            lab_names = [l.get('name', '') if isinstance(l, dict) else str(l) for l in lab_results]
            lab_names = [l for l in lab_names if l]
            if lab_names:
                formatted_labs = ', '.join(f'"{l}"' for l in lab_names)
                query_parts.append(f"lab results including {formatted_labs}")
        
        # Process procedures
        procedures = disease_criteria.get('diagnosticProcedures', [])
        if procedures:
            procedure_names = [p.get('name', '') if isinstance(p, dict) else str(p) for p in procedures]
            procedure_names = [p for p in procedure_names if p]
            if procedure_names:
                formatted_procedures = ', '.join(f'"{p}"' for p in procedure_names)
                query_parts.append(f"procedures including {formatted_procedures}")
        
        # Process risk factors
        risk_factors = disease_criteria.get('riskFactors', [])
        if risk_factors:
            risk_names = [r.get('name', '') if isinstance(r, dict) else str(r) for r in risk_factors]
            risk_names = [r for r in risk_names if r]
            if risk_names:
                formatted_risks = ', '.join(f'"{r}"' for r in risk_names)
                query_parts.append(f"risk factors including {formatted_risks}")
        
        # Combine all parts with AND
        query = " AND ".join(query_parts)
        logger.info(f"[scan] Generated query: {query}")
        return query
        
    except Exception as e:
        logger.error(f"[scan] Error generating query: {str(e)}")
        raise

def process_response(response_text: str, disease_criteria: Dict) -> Dict:
    """Process the response text and match it against disease criteria with contextual analysis."""
    try:
        logger.info("[scan] Processing response text for medical evidence")
        
        matches = {
            'symptoms': [],
            'lab_results': [],
            'procedures': [],
            'risk_factors': [],
            'confidence_scores': {
                'symptoms': 0.0,
                'lab_results': 0.0,
                'procedures': 0.0,
                'risk_factors': 0.0
            },
            'overall_confidence': 0.0,
            'relevant_excerpts': []
        }
        
        # Split response into sentences for better context analysis
        sentences = [s.strip() for s in response_text.split('.') if s.strip()]
        
        def calculate_confidence(term: str, context: str) -> tuple[float, str]:
            """Calculate confidence score based on contextual evidence."""
            confidence = 0.0
            evidence = ""
            
            # Check for definitive language
            if any(phrase in context.lower() for phrase in [
                "diagnosed with", "confirmed", "tested positive for",
                "exhibits clear signs of", "demonstrates", "shows"
            ]):
                confidence += 0.4
                evidence = context
            
            # Check for temporal markers
            elif any(phrase in context.lower() for phrase in [
                "history of", "chronic", "recurring", "persistent",
                "ongoing", "frequently", "regularly"
            ]):
                confidence += 0.3
                evidence = context
            
            # Check for severity indicators
            elif any(phrase in context.lower() for phrase in [
                "severe", "significant", "substantial", "major",
                "acute", "critical", "concerning"
            ]):
                confidence += 0.25
                evidence = context
            
            # Check for symptom associations
            elif any(phrase in context.lower() for phrase in [
                "associated with", "linked to", "related to",
                "consistent with", "indicative of"
            ]):
                confidence += 0.2
                evidence = context
            
            # Check for measurement or quantification
            elif any(phrase in context.lower() for phrase in [
                "measured", "recorded", "quantified", "rated",
                "scored", "assessed at"
            ]):
                confidence += 0.35
                evidence = context
            
            # Check for medical professional observation
            elif any(phrase in context.lower() for phrase in [
                "doctor noted", "physician observed", "clinician reported",
                "medical record indicates", "specialist confirmed"
            ]):
                confidence += 0.45
                evidence = context
            
            # Basic mention without strong context
            elif term.lower() in context.lower():
                confidence += 0.15
                evidence = context
            
            # Add confidence if multiple symptoms are mentioned together
            symptom_count = sum(1 for s in disease_criteria.get('symptoms', []) 
                              if isinstance(s, dict) and s.get('name', '').lower() in context.lower())
            if symptom_count > 1:
                confidence += 0.1 * min(symptom_count - 1, 3)  # Cap at 3 additional symptoms
            
            # Normalize confidence to maximum of 1.0
            confidence = min(confidence, 1.0)
            
            return confidence, evidence
        
        # Process symptoms with context
        symptoms = disease_criteria.get('symptoms', [])
        for symptom in symptoms:
            symptom_name = symptom.get('name', '') if isinstance(symptom, dict) else str(symptom)
            max_confidence = 0.0
            best_evidence = ""
            
            # Look for the symptom in each sentence to find the best context
            for sentence in sentences:
                if symptom_name.lower() in sentence.lower():
                    confidence, evidence = calculate_confidence(symptom_name, sentence)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_evidence = evidence
            
            if max_confidence > 0:
                matches['symptoms'].append({
                    'name': symptom_name,
                    'confidence': max_confidence,
                    'excerpt': best_evidence.strip()
                })
        
        # Process lab results with context
        lab_results = disease_criteria.get('labResults', [])
        for lab in lab_results:
            lab_name = lab.get('name', '') if isinstance(lab, dict) else str(lab)
            max_confidence = 0.0
            best_evidence = ""
            
            for sentence in sentences:
                if lab_name.lower() in sentence.lower():
                    # Higher base confidence for lab results as they're more definitive
                    confidence, evidence = calculate_confidence(lab_name, sentence)
                    confidence += 0.2  # Lab results are more objective
                    confidence = min(confidence, 1.0)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_evidence = evidence
            
            if max_confidence > 0:
                matches['lab_results'].append({
                    'name': lab_name,
                    'confidence': max_confidence,
                    'excerpt': best_evidence.strip()
                })
        
        # Process procedures with context
        procedures = disease_criteria.get('diagnosticProcedures', [])
        for procedure in procedures:
            procedure_name = procedure.get('name', '') if isinstance(procedure, dict) else str(procedure)
            max_confidence = 0.0
            best_evidence = ""
            
            for sentence in sentences:
                if procedure_name.lower() in sentence.lower():
                    confidence, evidence = calculate_confidence(procedure_name, sentence)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_evidence = evidence
            
            if max_confidence > 0:
                matches['procedures'].append({
                    'name': procedure_name,
                    'confidence': max_confidence,
                    'excerpt': best_evidence.strip()
                })
        
        # Process risk factors with context
        risk_factors = disease_criteria.get('riskFactors', [])
        for risk in risk_factors:
            risk_name = risk.get('name', '') if isinstance(risk, dict) else str(risk)
            max_confidence = 0.0
            best_evidence = ""
            
            for sentence in sentences:
                if risk_name.lower() in sentence.lower():
                    confidence, evidence = calculate_confidence(risk_name, sentence)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_evidence = evidence
            
            if max_confidence > 0:
                matches['risk_factors'].append({
                    'name': risk_name,
                    'confidence': max_confidence,
                    'excerpt': best_evidence.strip()
                })
        
        # Calculate weighted category scores
        def calculate_category_score(matches_list: List[Dict]) -> float:
            if not matches_list:
                return 0.0
            # Weight higher confidence findings more heavily
            weights = [m['confidence'] for m in matches_list]
            scores = [m['confidence'] for m in matches_list]
            return sum(w * s for w, s in zip(weights, scores)) / sum(weights) if weights else 0.0
        
        # Update confidence scores for each category
        matches['confidence_scores']['symptoms'] = calculate_category_score(matches['symptoms'])
        matches['confidence_scores']['lab_results'] = calculate_category_score(matches['lab_results'])
        matches['confidence_scores']['procedures'] = calculate_category_score(matches['procedures'])
        matches['confidence_scores']['risk_factors'] = calculate_category_score(matches['risk_factors'])
        
        # Calculate overall confidence with weighted importance
        weights = {
            'symptoms': 0.4,      # Symptoms are important but subjective
            'lab_results': 0.3,   # Lab results are objective but may be routine
            'procedures': 0.2,    # Procedures indicate medical attention
            'risk_factors': 0.1   # Risk factors are important context
        }
        
        weighted_scores = []
        for category, weight in weights.items():
            score = matches['confidence_scores'][category]
            if score > 0:
                weighted_scores.append(score * weight)
        
        matches['overall_confidence'] = sum(weighted_scores) / sum(weights[cat] 
            for cat in ['symptoms', 'lab_results', 'procedures', 'risk_factors']
            if matches['confidence_scores'][cat] > 0) if weighted_scores else 0.0
        
        # Add relevant excerpts with confidence context
        matches['relevant_excerpts'] = []
        for category in ['symptoms', 'lab_results', 'procedures', 'risk_factors']:
            for match in matches[category]:
                excerpt = f"{match['name']} (Confidence: {match['confidence']:.2%}): {match['excerpt']}"
                matches['relevant_excerpts'].append(excerpt)
        
        logger.info(f"[scan] Processed response with {len(matches['relevant_excerpts'])} matches")
        logger.info(f"[scan] Overall confidence: {matches['overall_confidence']:.2%}")
        logger.debug(f"[scan] Match details: {json.dumps(matches, indent=2)}")
        
        return matches
        
    except Exception as e:
        logger.error(f"[scan] Error processing response: {str(e)}")
        logger.error("[scan] Error details:", exc_info=True)
        raise

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
