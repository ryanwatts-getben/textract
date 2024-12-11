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
        parser = PydanticOutputParser(Disease)
        disease_template = {"name":"DISEASE_NAME","icd10":"ICD10_CODE","isGlobal":True,"cptCodes":["CPT_CODE_1","CPT_CODE_2"],"symptoms":[{"name":"SYMPTOM_1"},{"name":"SYMPTOM_2"}],"labResults":[{"name":"LAB_TEST_1","range":"RANGE_1"},{"name":"LAB_TEST_2","range":"RANGE_2"}],"diagnosticProcedures":[{"name":"PROCEDURE_1"},{"name":"PROCEDURE_2"}],"riskFactors":[{"name":"RISK_FACTOR_1"},{"name":"RISK_FACTOR_2"}],"scoringModel":{"name":"SCORING_MODEL","symptomWeight":0.3,"labResultWeight":0.3,"diagnosticProcedureWeight":0.2,"riskFactorWeight":0.2,"confidenceThreshold":0.7,"isGlobal":True},"relationships":[{"relatedDisease":"RELATED_DISEASE_1","relationType":"HIGHLY_RELATED","description":"RELATIONSHIP_DESCRIPTION_1"}]}
        format_instructions = json.dumps(disease_template)

        # Initialize variables for handling continued responses
        full_response = ""
        messages = []
        MAX_ITERATIONS = 5
        iteration_count = 0

        while True:
            iteration_count += 1
            if iteration_count > MAX_ITERATIONS:
                logger.error("[disease_definition_generator] Exceeded maximum number of iterations")
                raise ValueError("Exceeded maximum number of iterations for response generation")

            # Configure query engine with appropriate template
            query_engine = index.as_query_engine(
                llm=Anthropic(
                    model="claude-3-5-sonnet-20241022",
                    api_key=os.getenv('ANTHROPIC_API_KEY')
                ),
                embed_model=get_embedding_model(),
                prompt_template=f"""You are a medical knowledge base assistant. Generate a comprehensive disease definition in JSON format.

Requirements:
- The response must be valid JSON that matches the following structure exactly:
{format_instructions}
- Generate a structured disease definition for: {disease_name}
- Response must be valid JSON that can be parsed into the specified structure.""" if not messages else "Continue exactly where you left off, without repeating anything or starting over.",
                structured_answer_filtering=True,
                output_parser=parser,
                system_prompt=f"""You are a medical data assistant specialized in generating structured disease definitions.
Your responses must always be valid JSON that matches the required structure exactly.
Do not include any explanatory text, only return the JSON object. Do not preamble. JSON object = {format_instructions}""",
                similarity_top_k=5,
                response_kwargs={
                    "temperature": 0.0,
                    "max_tokens": 8192,
                },
                verbose=False
            )

            # Execute query
            response = query_engine.query(disease_name)
            current_response = response.response
            logger.info(f"[disease_definition_generator] Current response: {current_response}")
            # Add to full response
            full_response += current_response
            
            # Check if we need to continue
            stop_reason = getattr(response, 'stop_reason', None)
            logger.info(f"[disease_definition_generator] Stop reason: {stop_reason}")
            
            if stop_reason != 'max_tokens':
                break
                
            # Add to messages for context in next iteration
            messages.append({"role": "assistant", "content": current_response})
            logger.info(f"[disease_definition_generator] Continuing response due to max_tokens...")

        # Attempt to parse the complete response
        try:
            disease_data = Disease.model_validate_json(full_response)
            return disease_data.model_dump_json()
        except Exception as e:
            logger.error(f"[disease_definition_generator] Failed to validate response structure: {e}")
            raise ValueError(f"Generated response did not match required structure:\n{full_response}")

    except Exception as e:
        logger.error(f"[disease_definition_generator] Error generating disease definition: {str(e)}")
        raise
