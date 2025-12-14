import hashlib
from typing import List
from llama_index.core.schema import TextNode

def _norm(t: str) -> str:
    return " ".join(t.lower().split())

def dedup(nodes: List[TextNode]) -> List[TextNode]:
    seen = {}
    kept: List[TextNode] = []
    for n in nodes:
        key = hashlib.sha1(_norm(n.text).encode()).hexdigest()
        order = (n.metadata.get("position", {}) or {}).get("order", None)
        is_neighbor_overlap = (kept and order is not None and order == (kept[-1].metadata.get("position", {}) or {}).get("order", -2) + 1)
        if key in seen and not is_neighbor_overlap:
            k = seen[key]
            kp = kept[k].metadata
            src = n.metadata.get("source")
            if src:
                kp.setdefault("sources", set()).add(src)
            pgs = n.metadata.get("pages") or []
            if pgs:
                kp.setdefault("pages", set()).update(pgs)
            continue
        seen[key] = len(kept)
        kept.append(n)
    for n in kept:
        if isinstance(n.metadata.get("sources"), set):
            n.metadata["sources"] = sorted(n.metadata["sources"])
        if isinstance(n.metadata.get("pages"), set):
            n.metadata["pages"] = sorted(n.metadata["pages"])
    return kept
