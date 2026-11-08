"""
Recursive Character Text Splitter.
Splits text based on paragraph breaks, line breaks, and sentence boundaries
while maintaining contextual overlap and metadata.
"""

import uuid
from typing import List, Dict, Any
from ..models.schemas import DocumentChunk

class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Takes raw document dictionaries and returns a structured list of DocumentChunk objects.
        """
        all_chunks: List[DocumentChunk] = []

        for doc_idx, doc in enumerate(documents):
            text = doc["text"]
            base_meta = doc.get("metadata", {})
            source = base_meta.get("source", f"doc_{doc_idx}")
            raw_splits = self._split_text(text)

            for split_idx, split_text in enumerate(raw_splits):
                if not split_text.strip():
                    continue
                chunk_id = f"{source}_{base_meta.get('page', 1)}_{split_idx}_{uuid.uuid4().hex[:6]}"
                chunk_meta = dict(base_meta)
                chunk_meta["chunk_index"] = split_idx
                chunk_meta["char_length"] = len(split_text)

                # Approximate token count (approx 4 chars per token)
                approx_tokens = max(1, len(split_text) // 4)

                all_chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=source,
                    text=split_text.strip(),
                    metadata=chunk_meta,
                    token_count=approx_tokens
                ))

        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._recursive_split(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(s)

        # Merge splits respecting chunk_size and chunk_overlap
        merged = self._merge_splits(good_splits, separator)
        return merged

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for s in splits:
            _len = len(s)
            if total + _len + (len(separator) if len(current_doc) > 0 else 0) > self.chunk_size:
                if total > 0:
                    doc = separator.join(current_doc)
                    if doc.strip():
                        docs.append(doc)
                    # Handle overlap by keeping tail elements
                    while total > self.chunk_overlap or (
                        total + _len + (len(separator) if len(current_doc) > 0 else 0) > self.chunk_size and total > 0
                    ):
                        if not current_doc:
                            break
                        popped = current_doc.pop(0)
                        total -= len(popped) + (len(separator) if len(current_doc) > 0 else 0)
                current_doc.append(s)
                total = sum(len(x) for x in current_doc) + max(0, len(current_doc) - 1) * len(separator)
            else:
                current_doc.append(s)
                total += _len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc)
            if doc.strip():
                docs.append(doc)

        return docs
