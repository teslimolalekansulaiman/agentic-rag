from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.core import StorageContext, VectorStoreIndex

from src.common.config import settings


def load_index():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", device="cpu")

    vstore = PGVectorStore.from_params(
        host=settings.PG_HOST,
        port=settings.PG_PORT,
        database=settings.PG_DB,
        user=settings.PG_USER,
        password=settings.PG_PASSWORD,
        table_name=settings.PG_VECTOR_TABLE,
        embed_dim=settings.EMBED_DIM,
        hybrid_search=True,
        text_search_config="english",
    )

    # ADD THIS: Load the document store
    doc_store = PostgresDocumentStore.from_params(
        host=settings.PG_HOST,
        port=settings.PG_PORT,
        database=settings.PG_DB,
        user=settings.PG_USER,
        password=settings.PG_PASSWORD,
        table_name=settings.PG_DOC_TABLE,
    )

    # Include doc_store in storage context
    storage = StorageContext.from_defaults(
        vector_store=vstore,
        docstore=doc_store  # Add this
    )

    return VectorStoreIndex.from_vector_store(
        vector_store=vstore,
        storage_context=storage,
        embed_model=embed_model
    )