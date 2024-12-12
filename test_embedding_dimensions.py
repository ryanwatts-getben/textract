import logging
from llama_index.core import VectorStoreIndex, Document
from disease_definition_generator import get_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    embed_model = get_embedding_model()
    embedding_dimension = embed_model.client.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {embedding_dimension}")

    documents = [Document(text="Sample text about kidney cancer.")]
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    # Save the index
    index.storage_context.persist(persist_dir='./test_index')

    # Load the index
    storage_context = StorageContext.from_defaults(persist_dir='./test_index')
    loaded_index = load_index_from_storage(storage_context, embed_model=embed_model)

    # Verify the embedding dimension
    loaded_embedding_dimension = loaded_index._vector_store._dim
    logger.info(f"Loaded index embedding dimension: {loaded_embedding_dimension}")

    # Query the index
    query_engine = loaded_index.as_query_engine(embed_model=embed_model)
    response = query_engine.query("What is kidney cancer?")
    print(response.response)
    
    from transformers import AutoModel

    model_name = "BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    model = AutoModel.from_pretrained(model_name)
    print(f"Model loaded: {model_name}")
    print(f"Model loaded: {model}")
if __name__ == "__main__":
    main()
