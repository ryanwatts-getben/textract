import os
import json
import logging
from typing import List, Literal
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Define the expected structure with Pydantic
class LabResult(BaseModel):
    name: str
    range: str

class Symptom(BaseModel):
    name: str

class DiagnosticProcedure(BaseModel):
    name: str

class RiskFactor(BaseModel):
    name: str

class ScoringModel(BaseModel):
    name: str
    symptomWeight: float = Field(ge=0, le=1)
    labResultWeight: float = Field(ge=0, le=1)
    diagnosticProcedureWeight: float = Field(ge=0, le=1)
    riskFactorWeight: float = Field(ge=0, le=1)
    confidenceThreshold: float = Field(ge=0, le=1)
    isGlobal: bool

class DiseaseRelationship(BaseModel):
    relatedDisease: str
    relationType: Literal["HIGHLY_RELATED", "COMORBIDITY", "SYMPTOM_INDICATOR", "HEREDITARY", "OTHER"]
    description: str

class Disease(BaseModel):
    name: str
    icd10: str
    isGlobal: bool
    cptCodes: List[str]
    symptoms: List[Symptom]
    labResults: List[LabResult]
    diagnosticProcedures: List[DiagnosticProcedure]
    riskFactors: List[RiskFactor]
    scoringModel: ScoringModel
    relationships: List[DiseaseRelationship]

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Initialize embedding model with proper device configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )
    return LangchainEmbedding(huggingface_embeddings)

def query_index_for_disease(index: VectorStoreIndex, disease_name: str) -> str:
    try:
        # Initialize the output parser with the Disease Pydantic model
        parser = PydanticOutputParser(Disease)
        
        # Load the disease schema as JSON
        disease_template = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Disease Definition",
    "description": "A comprehensive medical disease definition with associated metadata",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "The official medical name of the disease"
        },
        "icd10": {
            "type": "string",
            "description": "The valid ICD-10 code for the disease",
            "pattern": "^[A-Z][0-9][0-9A-Z](\\.?[0-9]{0,2})?$"
        },
        "isGlobal": {
            "type": "boolean",
            "description": "Indicates if the disease is globally recognized and documented"
        },
        "cptCodes": {
            "type": "array",
            "description": "List of valid CPT codes for procedures related to this disease",
            "items": {
                "type": "string",
                "pattern": "^[0-9]{5}$"
            }
        },
        "symptoms": {
            "type": "array",
            "description": "List of symptoms associated with the disease",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The medical term for the symptom"
                    }
                },
                "required": [
                    "name"
                ]
            }
        },
        "labResults": {
            "type": "array",
            "description": "List of laboratory tests relevant for diagnosis",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the laboratory test"
                    },
                    "range": {
                        "type": "string",
                        "description": "The normal range or expected values for the test"
                    }
                },
                "required": [
                    "name",
                    "range"
                ]
            }
        },
        "diagnosticProcedures": {
            "type": "array",
            "description": "List of diagnostic procedures used to identify the disease",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the diagnostic procedure"
                    }
                },
                "required": [
                    "name"
                ]
            }
        },
        "riskFactors": {
            "type": "array",
            "description": "List of factors that increase the risk of developing the disease",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the risk factor"
                    }
                },
                "required": [
                    "name"
                ]
            }
        },
        "scoringModel": {
            "type": "object",
            "description": "Weighted scoring model for disease probability assessment",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the scoring model"
                },
                "symptomWeight": {
                    "type": "number",
                    "description": "Weight assigned to symptoms in the scoring model",
                    "minimum": 0,
                    "maximum": 1
                },
                "labResultWeight": {
                    "type": "number",
                    "description": "Weight assigned to lab results in the scoring model",
                    "minimum": 0,
                    "maximum": 1
                },
                "diagnosticProcedureWeight": {
                    "type": "number",
                    "description": "Weight assigned to diagnostic procedures in the scoring model",
                    "minimum": 0,
                    "maximum": 1
                },
                "riskFactorWeight": {
                    "type": "number",
                    "description": "Weight assigned to risk factors in the scoring model",
                    "minimum": 0,
                    "maximum": 1
                },
                "confidenceThreshold": {
                    "type": "number",
                    "description": "Minimum confidence score required for positive identification",
                    "minimum": 0,
                    "maximum": 1
                },
                "isGlobal": {
                    "type": "boolean",
                    "description": "Indicates if the scoring model is globally validated"
                }
            },
            "required": [
                "name",
                "symptomWeight",
                "labResultWeight",
                "diagnosticProcedureWeight",
                "riskFactorWeight",
                "confidenceThreshold",
                "isGlobal"
            ]
        },
        "relationships": {
            "type": "array",
            "description": "List of relationships with other diseases",
            "items": {
                "type": "object",
                "properties": {
                    "relatedDisease": {
                        "type": "string",
                        "description": "The name of the related disease"
                    },
                    "relationType": {
                        "type": "string",
                        "enum": [
                            "HIGHLY_RELATED",
                            "COMORBIDITY",
                            "SYMPTOM_INDICATOR",
                            "HEREDITARY",
                            "OTHER"
                        ],
                        "description": "The type of relationship between the diseases"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the relationship"
                    }
                },
                "required": [
                    "relatedDisease",
                    "relationType",
                    "description"
                ]
            }
        }
    },
    "required": [
        "name",
        "icd10",
        "isGlobal",
        "cptCodes",
        "symptoms",
        "labResults",
        "diagnosticProcedures",
        "riskFactors",
        "scoringModel",
        "relationships"
    ]
}
        format_instructions = json.dumps(disease_template)
        
        # Define the system prompt with explicit instructions
        system_prompt = f"""
You are a medical data assistant that returns ONLY JSON. You have one task:

Output EXACTLY ONE JSON object that matches the provided schema. No explanation, no text outside the JSON object, no apologies, no markdown formatting. If uncertain, output an empty JSON object {{}}. The JSON must strictly conform to the schema below.

SCHEMA:
{format_instructions}

The disease: {disease_name}

EXAMPLE OF A VALID JSON RESPONSE (for a hypothetical disease):

{{
  "name": "Hypothetical Disease",
  "icd10": "A00",
  "isGlobal": true,
  "cptCodes": ["12345"],
  "symptoms": [{{ "name": "High fever" }}],
  "labResults": [{{ "name": "Test X", "range": "50-100" }}],
  "diagnosticProcedures": [{{ "name": "MRI Scan" }}],
  "riskFactors": [{{ "name": "Smoking" }}],
  "scoringModel": {{
    "name": "Hypothetical Model",
    "symptomWeight": 0.5,
    "labResultWeight": 0.5,
    "diagnosticProcedureWeight": 0.5,
    "riskFactorWeight": 0.5,
    "confidenceThreshold": 0.5,
    "isGlobal": true
  }},
  "relationships": [{{
    "relatedDisease": "Another Disease",
    "relationType": "COMORBIDITY",
    "description": "Often occurs together"
  }}]
}}

DO NOT USE THIS EXAMPLE TEXT OUTSIDE OF THIS EXAMPLE. NOW PRODUCE YOUR OWN JSON FOR THE GIVEN DISEASE.
"""

        # Configure the query engine without a prompt template
        query_engine = index.as_query_engine(
            llm=Anthropic(
                model="claude-3-5-sonnet-20241022",
                api_key=os.getenv('ANTHROPIC_API_KEY')
            ),
            embed_model=get_embedding_model(),
            system_prompt=system_prompt,
            output_parser=parser,
            structured_answer_filtering=True,
            similarity_top_k=5,
            response_kwargs={
                "temperature": 0.0,
                "max_tokens": 8192,  # Adjust if needed
            },
            verbose=False
        )
        
        # Execute the query with an empty string since the instructions are in the system prompt
        response = query_engine.query("")

        # Get the full response
        full_response = response.response

        # Attempt to parse the JSON response
        try:
            disease_data = Disease.model_validate_json(full_response)
            return disease_data.model_dump_json()
        except Exception as e:
            logger.error(f"[disease_definition_generator] Failed to validate response structure: {e}")
            raise ValueError(f"Generated response did not match required structure:\n{full_response}")

    except Exception as e:
        logger.error(f"[disease_definition_generator] Error generating disease definition: {str(e)}")
        raise
