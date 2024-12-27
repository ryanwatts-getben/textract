from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import json
from typing import Dict, List
import logging
from db import get_session
from uuid import UUID

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
    mass_tort_ids: List[str]
) -> Dict:
    """
    Fetch mass tort data with associated diseases and scoring models.
    
    Args:
        user_id (str): The user's ID
        project_id (str): The project's ID
        mass_tort_ids (List[str]): List of mass tort IDs to fetch
    
    Returns:
        Dict: The formatted response containing mass tort data
    """
    try:
        with get_session() as db:
            # Build the SQL query
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
            
            # Process the results
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
                
                # Add disease data
                disease = {
                    'id': str(row.disease_id),  # Convert UUID to string
                    'name': row.disease_name,
                    'symptoms': row.symptoms if row.symptoms else [],
                    'labResults': row.lab_results if row.lab_results else [],
                    'diagnosticProcedures': row.diagnostic_procedures if row.diagnostic_procedures else [],
                    'riskFactors': row.risk_factors if row.risk_factors else [],
                }
                
                # Add scoring model if exists
                if row.scoring_model_id:
                    disease['scoringModel'] = {
                        'id': str(row.scoring_model_id),  # Convert UUID to string
                        'confidenceThreshold': row.confidence_threshold
                    }
                
                mass_torts_data[mass_tort_id]['diseases'].append(disease)
            
            # Format the final response
            response = {
                'userId': user_id,
                'projectId': project_id,
                'massTorts': list(mass_torts_data.values())
            }
            
            logger.info(f"[examplequery] Successfully fetched data for {len(mass_torts_data)} mass torts")
            return response
            
    except Exception as e:
        logger.error(f"[examplequery] Error fetching mass tort data: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Test data
    test_user_id = "4e6f9de5-5128-4ff4-980c-2429e31ec2ec"
    test_project_id = "71577c8f-f322-4c0e-a5e6-73d6df3477e1"
    test_mass_tort_ids = ["02ccfcb0-1658-4242-a4c9-0b0d4de17e85"]
    
    try:
        result = get_mass_tort_data(test_user_id, test_project_id, test_mass_tort_ids)
        print(json.dumps(result, indent=2, cls=UUIDEncoder))  # Use custom encoder
    except Exception as e:
        print(f"Error: {str(e)}")