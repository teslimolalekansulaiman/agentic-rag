try:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker as _Chunker
except ImportError:  # fallback for older versions
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker as _Chunker


class HybridChunker:
    def __init__(self, **kwargs):
        # ✅ Instantiate the real Docling chunker, not this wrapper
        self._impl = _Chunker(**kwargs)

        # Debug: Check what methods are available
        print("Available methods:", [m for m in dir(self._impl) if not m.startswith("_")])

    def split(self, ddoc):
        """Try splitting with available method"""
        if hasattr(self._impl, "split"):
            return self._impl.split(ddoc)
        elif hasattr(self._impl, "chunk"):
            return self._impl.chunk(ddoc)
        elif hasattr(self._impl, "split_text"):
            return self._impl.split_text(ddoc)
        else:
            raise AttributeError(f"Chunker {type(self._impl)} has no split method")

    def contextualize(self, chunk) -> str:
        """Return heading-path + body text when available"""
        ctx_fn = getattr(self._impl, "contextualize", None)
        if callable(ctx_fn):
            return ctx_fn(chunk) or ""
        return getattr(chunk, "text", "") or ""
