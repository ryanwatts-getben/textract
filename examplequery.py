import json
from typing import Dict, List
import logging
from uuid import UUID
from sqlalchemy import  text
from db import get_session
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)

def get_mass_tort_data(
    user_id: str,
    project_id: str,
    mass_tort_ids: List[str],
    db_update: bool = False
) -> Dict:
    """
    Fetch mass tort data with associated diseases and scoring models.
    
    Args:
        user_id (str): The user's ID
        project_id (str): The project's ID
        mass_tort_ids (List[str]): List of mass tort IDs to fetch
        db_update (bool): Whether to perform database updates (default: False)
    
    Returns:
        Dict: The formatted response containing mass tort data
    """
    try:
        with get_session() as db:
            # Build the SQL query to fetch mass tort data
            query = text("""
                WITH mass_tort_diseases AS (
                    SELECT 
                        mt.id as mass_tort_id,
                        mt."officialName" as official_name,
                        d.id as disease_id,
                        d.name as disease_name,
                        d.symptoms,
                        d."labResults" as lab_results,
                        d."diagnosticProcedures" as diagnostic_procedures,
                        d."riskFactors" as risk_factors,
                        sm.id as scoring_model_id,
                        sm."confidenceThreshold" as confidence_threshold
                    FROM "MassTort" mt
                    JOIN "MassTortDisease" mtd ON mt.id = mtd."massTortId"
                    JOIN "Disease" d ON mtd."diseaseId" = d.id
                    LEFT JOIN "ScoringModel" sm ON d."scoringModelId" = sm.id
                    WHERE mt.id::text = ANY(:mass_tort_ids)
                    AND (mt."userId"::text = :user_id OR mt."isGlobal" = true)
                )
                SELECT * FROM mass_tort_diseases
                ORDER BY mass_tort_id, disease_name
            """)
            
            # Execute the query with parameters
            result = db.execute(query, {
                'user_id': user_id,
                'mass_tort_ids': mass_tort_ids
            })
            
            # Initialize a dictionary to store mass tort data
            mass_torts_data = {}
            for row in result:
                mass_tort_id = str(row.mass_tort_id)  # Convert UUID to string
                
                # Initialize mass tort data if not exists
                if mass_tort_id not in mass_torts_data:
                    mass_torts_data[mass_tort_id] = {
                        'id': mass_tort_id,
                        'officialName': row.official_name,
                        'diseases': []
                    }
                
                # Prepare disease data
                disease = {
                    'id': str(row.disease_id),  # Convert UUID to string
                    'name': row.disease_name,
                    'symptoms': row.symptoms if row.symptoms else [],
                    'labResults': row.lab_results if row.lab_results else [],
                    'diagnosticProcedures': row.diagnostic_procedures if row.diagnostic_procedures else [],
                    'riskFactors': row.risk_factors if row.risk_factors else [],
                }
                
                # Include scoring model information if available
                if row.scoring_model_id:
                    disease['scoringModel'] = {
                        'id': str(row.scoring_model_id),  # Convert UUID to string
                        'confidenceThreshold': row.confidence_threshold
                    }
                
                # Append the disease to the mass tort's disease list
                mass_torts_data[mass_tort_id]['diseases'].append(disease)
                
                if db_update:
                    # Update the DiseaseProject table where projectId, diseaseId, and massTortId match
                    update_disease_project_status(
                        project_id=project_id,
                        mass_tort_id=mass_tort_id,
                        disease_id=str(row.disease_id),
                        status='PROCESSING'
                    )
                    
                    logger.info(
                        "[examplequery] Updated DiseaseProject status to PROCESSING for project %s, disease %s, mass tort %s",
                        project_id, row.disease_id, mass_tort_id
                    )
            
            # Format the final response
            response = {
                'userId': user_id,
                'projectId': project_id,
                'massTorts': list(mass_torts_data.values())
            }
            
            logger.info(
                "[examplequery] Successfully fetched data for %d mass torts",
                len(mass_torts_data)
            )
            return response
                
    except Exception as e:
        logger.error(
            "[examplequery] Error fetching mass tort data: %s",
            str(e)
        )
        raise

def get_disease_project_by_status(status: str = 'PENDING') -> Dict:
    """
    Fetch disease projects with PENDING status.
    
    Args:
        user_id (str): The user's ID
        project_id (str): The project's ID
    
    Returns:
        Dict: The formatted response containing pending disease projects
    """
    try:
        with get_session() as db:
            # Build the SQL query
            query = text("""
                SELECT 
                    dp.id as disease_project_id,
                    dp."status" as project_status,
                    dp."userId" as user_id,
                    dp."createdAt" as created_at,
                    dp."projectId" as project_id,
                    dp."massTortId" as mass_tort_id
                FROM "DiseaseProject" dp
                WHERE dp.status = :status
                ORDER BY dp."createdAt" ASC
                LIMIT 1
            """)
            
            # Execute the query with parameters
            db_result = db.execute(query, {
                'status': status,
            })

            row = db_result.fetchone()
            
            if row:
                disease_project = {
                    'id': str(row.disease_project_id),
                    'status': row.project_status,
                    'projectId': str(row.project_id),
                    'massTortId': str(row.mass_tort_id),
                    'userId': str(row.user_id)
                }

                logger.info(
                    "[get_first_pending_disease_project] Found %s disease project %s",
                    status, disease_project['id']
                )
                return disease_project

            logger.info(f"[get_first_pending_disease_project] No {status} disease projects found")
            return None

    except Exception as e:
        logger.error(
            "[get_pending_disease_projects] Error fetching pending disease projects: %s",
            str(e)
        )
        raise

def update_disease_project_status(
    project_id: str,
    mass_tort_id: str,
    disease_id: str,
    status: str
) -> None:
    """
    Update the DiseaseProject table to set the status to the provided status value and updatedAt to now
    where the given projectId, massTortId, and diseaseId are found.

    Args:
        project_id (str): The project's ID
        mass_tort_id (str): The mass tort's ID
        disease_id (str): The disease's ID
        status (str): The status to set (e.g., 'ERROR', 'PROCESSING', etc.)
    """
    try:
        with get_session() as db:
            # Build the SQL update query
            update_query = text("""
                UPDATE "DiseaseProject"
                SET status = :status, "updatedAt" = NOW()
                WHERE "projectId"::text = :project_id
                AND "massTortId"::text = :mass_tort_id
                AND "diseaseId"::text = :disease_id
            """)
            # Execute the update query with parameters
            db.execute(update_query, {
                'project_id': project_id,
                'mass_tort_id': mass_tort_id,
                'disease_id': disease_id,
                'status': status
            })
            logger.info(
                "[examplequery] Updated DiseaseProject status to %s for project %s, disease %s, mass tort %s",
                status, project_id, disease_id, mass_tort_id
            )
    except Exception as e:
        logger.error(
            "[examplequery] Error updating DiseaseProject status to %s: %s",
            status, str(e)
        )
        raise

def get_mass_torts_by_user_id(user_id: str) -> List[Dict]:
    """
    Fetch all active mass torts for a given user ID, including all related disease data.
    
    Args:
        user_id (str): The user's ID
    
    Returns:
        List[Dict]: List of mass torts with their associated diseases and related data
    """
    logger.info(f"[examplequery/get_mass_torts_by_user_id] starting for user {user_id}")
    try:
        with get_session() as db:
            # Build the SQL query to fetch mass tort data with all related information
            query = text("""
                WITH mass_tort_data AS (
                    SELECT 
                        mt.id as mass_tort_id,
                        mt."officialName" as official_name,
                        mt."isActive" as is_active,
                        d.id as disease_id,
                        d.name as disease_name,
                        d.symptoms as symptoms,
                        d."labResults" as lab_results,
                        d."diagnosticProcedures" as diagnostic_procedures,
                        d."riskFactors" as risk_factors,
                        sm.id as scoring_model_id,
                        sm."confidenceThreshold" as confidence_threshold
                    FROM "MassTort" mt
                    JOIN "MassTortDisease" mtd ON mt.id = mtd."massTortId"
                    JOIN "Disease" d ON mtd."diseaseId" = d.id
                    LEFT JOIN "ScoringModel" sm ON d."scoringModelId" = sm.id
                    WHERE (mt."userId"::text = :user_id OR mt."isGlobal" = true)
                    AND mt."isActive" = true
                )
                SELECT * FROM mass_tort_data
                ORDER BY mass_tort_id, disease_name
            """)
            
            # Execute the query with parameters
            results = db.execute(query, {'user_id': user_id})
            
            # Initialize a dictionary to store mass tort data
            mass_torts_data = {}
            
            for row in results:
                mass_tort_id = str(row.mass_tort_id)
                
                # Initialize mass tort data if not exists
                if mass_tort_id not in mass_torts_data:
                    mass_torts_data[mass_tort_id] = {
                        'id': mass_tort_id,
                        'officialName': row.official_name,
                        'isActive': row.is_active,
                        'massTortDiseases': []
                    }
                
                # Format disease data with all related information
                disease_data = {
                    'disease': {
                        'id': str(row.disease_id),
                        'name': row.disease_name,
                        'Symptom': row.symptoms if row.symptoms else [],
                        'LabResult': row.lab_results if row.lab_results else [],
                        'DiagnosticProcedure': row.diagnostic_procedures if row.diagnostic_procedures else [],
                        'RiskFactor': row.risk_factors if row.risk_factors else [],
                    }
                }
                
                # Add scoring model if available
                if row.scoring_model_id:
                    disease_data['disease']['scoringModel'] = {
                        'id': str(row.scoring_model_id),
                        'confidenceThreshold': row.confidence_threshold
                    }
                
                # Add the disease data to the mass tort
                mass_torts_data[mass_tort_id]['massTortDiseases'].append(disease_data)
            
            logger.info(
                "[examplequery] Successfully fetched %d active mass torts for user %s",
                len(mass_torts_data), user_id
            )
            
            return list(mass_torts_data.values())
            
    except Exception as e:
        logger.error(
            "[examplequery] Error fetching mass torts for user %s: %s",
            user_id, str(e)
        )
        raise

def get_disease_project(
    project_id: str,
    mass_tort_id: str,
    disease_id: str
) -> Dict:
    """
    Find an existing disease project by project ID, mass tort ID, and disease ID.
    
    Args:
        project_id (str): The project's ID
        mass_tort_id (str): The mass tort's ID
        disease_id (str): The disease's ID
    
    Returns:
        Dict: The disease project if found, None otherwise
    """
    try:
        with get_session() as db:
            # Build the SQL query
            query = text("""
                SELECT 
                    dp.id as disease_project_id,
                    dp."status" as project_status,
                    dp."createdAt" as created_at,
                    dp."updatedAt" as updated_at,
                    dp."projectId" as project_id,
                    dp."massTortId" as mass_tort_id,
                    dp."diseaseId" as disease_id
                FROM "DiseaseProject" dp
                WHERE dp."projectId"::text = :project_id
                AND dp."massTortId"::text = :mass_tort_id
                AND dp."diseaseId"::text = :disease_id
                LIMIT 1
            """)
            
            # Execute the query with parameters
            db_result = db.execute(query, {
                'project_id': project_id,
                'mass_tort_id': mass_tort_id,
                'disease_id': disease_id
            })

            row = db_result.fetchone()
            
            if row:
                disease_project = {
                    'id': str(row.disease_project_id),
                    'status': row.project_status,
                    'createdAt': row.created_at,
                    'updatedAt': row.updated_at,
                    'projectId': str(row.project_id),
                    'massTortId': str(row.mass_tort_id),
                    'diseaseId': str(row.disease_id)
                }

                logger.info(
                    "[examplequery] Found existing disease project %s",
                    disease_project['id']
                )
                return disease_project

            logger.info("[examplequery] No existing disease project found")
            return None

    except Exception as e:
        logger.error(
            "[examplequery] Error finding existing disease project: %s",
            str(e)
        )
        raise

def create_disease_project(
    name: str,
    project_id: str,
    disease_id: str,
    mass_tort_id: str,
    user_id: str,
    status: str = 'PENDING',
    score: float = None,
    confidence: float = None,
    law_firm: str = None,
    plaintiff: str = None,
    is_active: bool = True,
    match_count: int = 0,
    scoring_model_id: str = None,
    matched_symptoms: List[str] = None,
    matched_lab_results: List[str] = None,
    matched_procedures: List[str] = None,
    matched_risk_factors: List[str] = None,
    relevant_excerpts: List[str] = None,
    last_updated: datetime = None
) -> Dict:
    """
    Create a new DiseaseProject record.
    
    Args:
        name (str): Name of the disease project
        project_id (str): UUID of the associated project
        disease_id (str): UUID of the associated disease
        mass_tort_id (str): UUID of the associated mass tort
        user_id (str): UUID of the associated user
        status (str, optional): Project status. Defaults to 'PENDING'
        score (float, optional): Project score
        confidence (float, optional): Confidence score
        law_firm (str, optional): Associated law firm
        plaintiff (str, optional): Associated plaintiff
        is_active (bool, optional): Whether the project is active. Defaults to True
        match_count (int, optional): Number of matches. Defaults to 0
        scoring_model_id (str, optional): UUID of the associated scoring model
        matched_symptoms (List[str], optional): List of matched symptoms
        matched_lab_results (List[str], optional): List of matched lab results
        matched_procedures (List[str], optional): List of matched procedures
        matched_risk_factors (List[str], optional): List of matched risk factors
        relevant_excerpts (List[str], optional): List of relevant excerpts
        last_updated (datetime, optional): Last update timestamp
    
    Returns:
        Dict: The created disease project record
    """
    try:
        with get_session() as db:
            # Build the SQL insert query
            query = text("""
                INSERT INTO "DiseaseProject" (
                    name,
                    "projectId",
                    "diseaseId",
                    "massTortId",
                    "userId",
                    status,
                    score,
                    confidence,
                    "lawFirm",
                    plaintiff,
                    "isActive",
                    "matchCount",
                    "scoringModelId",
                    "matchedSymptoms",
                    "matchedLabResults",
                    "matchedProcedures",
                    "matchedRiskFactors",
                    "relevantExcerpts",
                    "lastUpdated",
                    "createdAt",
                    "updatedAt"
                ) VALUES (
                    :name,
                    :project_id,
                    :disease_id,
                    :mass_tort_id,
                    :user_id,
                    :status,
                    :score,
                    :confidence,
                    :law_firm,
                    :plaintiff,
                    :is_active,
                    :match_count,
                    :scoring_model_id,
                    :matched_symptoms,
                    :matched_lab_results,
                    :matched_procedures,
                    :matched_risk_factors,
                    :relevant_excerpts,
                    :last_updated,
                    NOW(),
                    NOW()
                )
                RETURNING *
            """)
            
            # Execute the query with parameters
            result = db.execute(query, {
                'name': name,
                'project_id': project_id,
                'disease_id': disease_id,
                'mass_tort_id': mass_tort_id,
                'user_id': user_id,
                'status': status,
                'score': score,
                'confidence': confidence,
                'law_firm': law_firm,
                'plaintiff': plaintiff,
                'is_active': is_active,
                'match_count': match_count,
                'scoring_model_id': scoring_model_id,
                'matched_symptoms': matched_symptoms or [],
                'matched_lab_results': matched_lab_results or [],
                'matched_procedures': matched_procedures or [],
                'matched_risk_factors': matched_risk_factors or [],
                'relevant_excerpts': relevant_excerpts or [],
                'last_updated': last_updated
            })
            
            row = result.fetchone()
            
            if row:
                disease_project = {
                    'id': str(row.id),
                    'name': row.name,
                    'projectId': str(row.projectId),
                    'diseaseId': str(row.diseaseId),
                    'massTortId': str(row.massTortId),
                    'userId': str(row.userId),
                    'status': row.status,
                    'score': row.score,
                    'confidence': row.confidence,
                    'lawFirm': row.lawFirm,
                    'plaintiff': row.plaintiff,
                    'isActive': row.isActive,
                    'matchCount': row.matchCount,
                    'scoringModelId': str(row.scoringModelId) if row.scoringModelId else None,
                    'matchedSymptoms': row.matchedSymptoms,
                    'matchedLabResults': row.matchedLabResults,
                    'matchedProcedures': row.matchedProcedures,
                    'matchedRiskFactors': row.matchedRiskFactors,
                    'relevantExcerpts': row.relevantExcerpts,
                    'lastUpdated': row.lastUpdated,
                    'createdAt': row.createdAt,
                    'updatedAt': row.updatedAt
                }
                
                logger.info(
                    "[examplequery] Created new disease project %s for project %s, disease %s, mass tort %s",
                    disease_project['id'], project_id, disease_id, mass_tort_id
                )
                return disease_project
                
            logger.error("[examplequery] Failed to create disease project")
            return None
            
    except Exception as e:
        logger.error(
            "[examplequery] Error creating disease project: %s",
            str(e)
        )
        raise

# Example usage
if __name__ == "__main__":
    # Test data
    test_user_id = "4e6f9de5-5128-4ff4-980c-2429e31ec2ec"
    test_project_id = "71577c8f-f322-4c0e-a5e6-73d6df3477e1"
    test_mass_tort_ids = ["02ccfcb0-1658-4242-a4c9-0b0d4de17e85"]
    
    try:
        result = get_mass_tort_data(test_user_id, test_project_id, test_mass_tort_ids)
        # result = get_disease_project_by_status('PENDING')
        print(json.dumps(result, indent=2, cls=UUIDEncoder))  # Use custom encoder
    except Exception as e:
        print(f"Error: {str(e)}")