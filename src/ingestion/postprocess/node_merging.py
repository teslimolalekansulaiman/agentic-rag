from collections import defaultdict
from typing import List
from llama_index.core.schema import TextNode, Document, BaseNode


def _group_nodes_by_heading(nodes: List[BaseNode]) -> List[Document]:
    grouped = defaultdict(list)

    # 1️⃣ Group nodes by heading
    for node in nodes:
        heading = node.metadata.get("heading")
        if heading:
            grouped[heading].append(node)

    docs = []

    # 2️⃣ Merge grouped nodes
    for heading, child_nodes in grouped.items():
        cleaned_texts = []

        for node in child_nodes:
            text = (node.text or "").strip()
            if heading and text.startswith(heading):
                text = text[len(heading):].strip()
            cleaned_texts.append(text)

        merged_text = f"{heading}\n" + "\n".join(cleaned_texts)

        # 4️⃣ Simplified children metadata (just node_ids)
        children_node_ids = [node.node_id for node in child_nodes]

        # 5️⃣ Construct the merged document
        doc = Document(
            text=merged_text.strip(),
            metadata={
                "heading": heading,
                "children": children_node_ids,  # ✅ Just IDs, not full metadata
                "num_children": len(child_nodes),
            },
            doc_id=heading,
        )
        docs.append(doc)

    return docs