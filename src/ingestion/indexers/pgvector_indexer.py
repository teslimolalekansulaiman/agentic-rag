import logging
from typing import List
from llama_index.core.schema import TextNode, Document, BaseNode
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.common.config import settings
from src.ingestion.postprocess.node_merging import _group_nodes_by_heading

logger = logging.getLogger(__name__)


class PGVectorIndexer:

    def __init__(self):

        self.vector_store = self._init_vector_store()
        self.doc_store = self._init_doc_store()
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.doc_store,
        )
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
        )

        # Set global embedding model
        Settings.embed_model = self.embed_model

    def _init_vector_store(self) -> PGVectorStore:
        """Initialize PostgreSQL vector store."""
        return PGVectorStore.from_params(
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

    def _init_doc_store(self) -> PostgresDocumentStore:
        """Initialize PostgreSQL document store."""
        return PostgresDocumentStore.from_params(
            host=settings.PG_HOST,
            port=settings.PG_PORT,
            database=settings.PG_DB,
            user=settings.PG_USER,
            password=settings.PG_PASSWORD,
            table_name=settings.PG_DOC_TABLE,
        )

    def index(self, nodes: List[BaseNode]) -> VectorStoreIndex:

        logger.info(f"🚀 Starting indexing process for {len(nodes)} nodes...")

        try:
            # Group nodes by heading and store as documents
            grouped_docs = _group_nodes_by_heading(nodes)
            if grouped_docs:
                self.doc_store.add_documents(grouped_docs)
                logger.info(f"✅ Indexed {len(nodes)} nodes into {len(grouped_docs)} documents")
            else:
                logger.warning("⚠️ No documents with headings found to group")
            # Create vector index from nodes
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                show_progress=True,
            )

            return index

        except Exception as e:
            logger.error(f"❌ Indexing failed: {e}", exc_info=True)
            raise
