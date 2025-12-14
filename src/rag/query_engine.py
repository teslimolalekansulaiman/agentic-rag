from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

from src.providers.llm_factory import get_llm
from src.common.config import Settings
from src.rag.head_merging import HeadingMergePostprocessor
from src.rag.index_utils import load_index
from src.rag.retrievers import build_retriever, build_reranker
from llama_index.core import Settings as LlamaSettings

LlamaSettings.llm = None
class RAGQueryEngine:
    def __init__(self):
        Settings.llm = get_llm()
        self.index = load_index()
        self.retriever = build_retriever(self.index, 10)
        self.reranker = build_reranker()
        # Initialize heading merger with your doc_store (from index.storage_context)
        self.heading_merger = HeadingMergePostprocessor(
            self.index.storage_context.docstore
        )

        # Create query engine directly from RetrieverQueryEngine
        self.qe = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            node_postprocessors=[self.reranker],
            response_mode=ResponseMode.NO_TEXT,
        )
        # Chat engine - returns text with conversational context
        self.chat_engine = self.index.as_chat_engine(
            chat_mode="condense_plus_context",
            node_postprocessors=[self.reranker, self.heading_merger],
            llm=Settings.llm,
            verbose=True,
            #filters=MetadataFilters(
                    #filters=[
                        #MetadataFilter(key="category", value="policy"),
                        #MetadataFilter(key="year", value=2024)
                    #]
            #)
        )

    def query(self, question: str):
        return self.qe.query(question)

    def chat(self, message):
        """Conversational query - maintains context"""
        return self.chat_engine.chat(message)
