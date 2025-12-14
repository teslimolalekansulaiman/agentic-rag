from llama_index.core.schema import NodeWithScore, TextNode


class HeadingMergePostprocessor:
    def __init__(self, doc_store):
        self.doc_store = doc_store

    def postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle=None) -> list[NodeWithScore]:
        merged_nodes = []
        seen_headings = set()

        for n in nodes:
            node = n.node
            heading = node.metadata.get("heading")

            # Skip if no heading
            if not heading:
                merged_nodes.append(n)
                continue

            # Skip if already merged
            if heading in seen_headings:
                continue

            try:
                # Get the parent document (method name depends on your docstore)
                parent_doc = self.doc_store.get_document(heading)  # or .get(heading)

                merged_nodes.append(
                    NodeWithScore(
                        node=TextNode(
                            text=parent_doc.text,
                            metadata={**parent_doc.metadata, "merged": True},
                        ),
                        score=n.score,
                    )
                )
                seen_headings.add(heading)
                print(f"Merged heading '{heading}' (score: {n.score:.3f})")

            except Exception as e:
                print(f"No merged doc for '{heading}': {e}, keeping original node")
                merged_nodes.append(n)

        print(f"Returned {len(merged_nodes)} nodes ({len(seen_headings)} merged)")
        return merged_nodes