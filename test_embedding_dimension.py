from langchain_huggingface import HuggingFaceEmbeddings

model_name = "BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name=model_name
)

embedding_dimension = huggingface_embeddings.model.get_sentence_embedding_dimension()
print(f"Embedding dimension: {embedding_dimension}") 