from typing import Dict, Any
from pathlib import Path


def _docling_version() -> str | None:
    try:
        import docling  # type: ignore
        return getattr(docling, "__version__", None)
    except Exception:
        return None


def base_metadata(
        pdf_path: Path,
        chunk_obj: Any,
        order: int,
        json_uri: str,
        table_uri: str | None
) -> Dict[str, Any]:
    # Extract flexible fields
    kind = getattr(chunk_obj, "kind", None)
    markdown = getattr(chunk_obj, "markdown", None)
    pages = getattr(chunk_obj, "pages", None) or getattr(chunk_obj, "page_no", None)
    headers = getattr(chunk_obj, "headers", None) or getattr(chunk_obj, "headings", None)

    # Table preview: safe one-liner if table and has markdown
    table_preview = None
    if kind == "table" and markdown:
        lines = markdown.splitlines()
        if lines:
            table_preview = lines[0]

    return {
        "source": pdf_path.name,
        "uri_pdf": str(pdf_path),
        "uri_docling_json": json_uri,
        "kind": kind,
        "pages": pages,
        "section_headers": headers,
        "parent_block_id": getattr(chunk_obj, "id", None),
        "bbox": getattr(chunk_obj, "bbox", None),
        "position": {"order": order},
        "table_markdown_uri": table_uri,
        "table_preview": table_preview,
        "pipeline": {
            "docling_version": _docling_version(),
            "chunker": "HybridChunker",
            "postpass": "SentenceSplitter(900/150)",
        },
        # Optional: include a content type flag to distinguish text vs table
        "content_type": "table" if kind == "table" else "text",
    }

def extract_heading(chunk: Any) -> str | None:
    """Extract heading string from a Docling chunk."""
    heading = None
    if hasattr(chunk, "heading") and chunk.heading:
        heading = str(chunk.heading)
    elif hasattr(chunk, "headings") and chunk.headings:
        heading = " > ".join(chunk.headings)
    elif hasattr(chunk, "meta") and hasattr(chunk.meta, "headings") and chunk.meta.headings:
        heading = " > ".join(chunk.meta.headings)
    elif hasattr(chunk, "meta") and isinstance(chunk.meta, dict) and "headings" in chunk.meta:
        heading = " > ".join(chunk.meta["headings"])
    return heading