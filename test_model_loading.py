import logging
from disease_definition_generator import get_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    embed_model = get_embedding_model()
    logger.info(f"Test model loading complete. Model name: {embed_model.model_name}") 