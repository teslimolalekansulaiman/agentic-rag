import ssl
from pathlib import Path
from typing import List, Any

from src.ingestion.parsers.docling_parser import DoclingParser
from src.providers.llm_factory import get_llm
from src.ingestion.chunkers.hybrid_chunker import HybridChunker
from src.ingestion.postprocess.dedup import dedup
from src.ingestion.postprocess.metadata import base_metadata, extract_heading
from src.ingestion.indexers.pgvector_indexer import PGVectorIndexer
from src.common.logging import configure_logging
from src.common.config import settings
from llama_index.core.schema import TextNode
from llama_index.llms.ollama import Ollama

ssl._create_default_https_context = ssl._create_unverified_context
logger = configure_logging()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def table_to_markdown(table_item: dict) -> str:
    """Convert table grid -> markdown text."""
    grid = table_item.get("data", {}).get("grid", [])
    if not grid or not any(grid):  # empty table
        return ""   # or return a fallback like "Empty table"

    rows = [[cell.get("text", "").strip() for cell in row] for row in grid]

    if not rows or not rows[0]:  # no actual content
        return ""

    header = "| " + " | ".join(rows[0]) + " |"
    separator = "|" + "|".join(["---"] * len(rows[0])) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]

    return "\n".join([header, separator] + body)



def summarize_table(md_text: str, llm: Ollama) -> str:
    """Ask LLM to summarize the table into natural text."""
    prompt = f"""
    You are given a table in Markdown format:

    {md_text}

    Please summarize the key information in this table in 3-5 sentences,
    highlighting the most important details for search/retrieval.
    """
    resp = llm.complete(prompt)
    return resp.text.strip()


# ---------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------

def process_pdf_text_nodes(pdf: Path, ddoc, chunker, json_uri: str) -> List[TextNode]:
    """Extract and chunk text nodes from a PDF document."""
    nodes: List[TextNode] = []

    for order, c in enumerate(chunker.split(ddoc)):
        text = (chunker.contextualize(c) or "").strip()
        if not text:
            continue
        heading = extract_heading(c)
        bm = base_metadata(pdf, c, order, json_uri, None)
        meta = {**bm, "content_type": "text", "heading": heading}

        nodes.append(TextNode(text=text, metadata=meta))
    return nodes


def process_pdf_table_nodes(pdf: Path, ddoc, json_uri: str) -> List[TextNode]:
    """Extract and summarize tables from a PDF document."""
    llm = get_llm()
    tables = ddoc.export_to_dict().get("tables", [])
    nodes: List[TextNode] = []

    for idx, table in enumerate(tables):
        md_text = table_to_markdown(table)
        summary = summarize_table(md_text, llm)

        bm = base_metadata(pdf, None, idx, json_uri, None)
        meta = {
            **bm,
            "content_type": "table",
            "table_markdown": md_text,   # keep full table as metadata
        }

        nodes.append(TextNode(text=summary, metadata=meta))
    return nodes


def deduplicate_nodes(nodes: List[TextNode]) -> List[TextNode]:
    """Run deduplication and log stats."""
    before = len(nodes)
    nodes = dedup(nodes)
    after = len(nodes)
    logger.info(f"🧹 Deduplicated {before - after} duplicates, {after} nodes remain")
    return nodes


def index_nodes(nodes: List[TextNode]) -> None:
    """Index all nodes into PGVector."""
    indexer = PGVectorIndexer()
    indexer.index(nodes)

# ---------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------

def run_pipeline(input_dir: Path) -> List[Any]:
    logger.info(f"📂 Starting pipeline from: {input_dir}")

    parser = DoclingParser(artifacts_dir=settings.ARTIFACTS_DIR)
    chunker = HybridChunker()
    #post = SizePostPass(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    all_nodes: List[Any] = []

    pdfs: List[Path] = [
        p for p in Path(input_dir).rglob("*")
        if p.suffix.lower() == ".pdf"
    ]
    for pdf in pdfs:
        logger.info(f"⚙️ Processing {pdf.name}")
        ddoc, json_path = parser.parse(pdf)
        json_uri = f"file://{json_path.resolve()}"

        # Process text & table nodes
        all_nodes.extend(process_pdf_text_nodes(pdf, ddoc, chunker, json_uri))
        all_nodes.extend(process_pdf_table_nodes(pdf, ddoc, json_uri))

    # Post-processing
    all_nodes = deduplicate_nodes(all_nodes)
    index_nodes(all_nodes)
    for i, node in enumerate(all_nodes[:10]):
        print(f"\n[Node #{i + 1}]")
        print(f"ID: {node.node_id}")
        print(f"Heading: {node.metadata.get('heading') or 'NO HEADING'}")
        print(f"Content type: {node.metadata.get('content_type')}")
        print(f"Text length: {len(node.text)} chars")
        print(f"Preview: {node.text[:200]}...")  # first 200 chars for preview
        print(f"Metadata: {node.metadata}")


    logger.info("🎉 Pipeline completed.")
    return all_nodes
