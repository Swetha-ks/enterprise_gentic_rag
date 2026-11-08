"""
Document Loader module.
Supports loading documents from PDF, Markdown, Text, and JSON files with metadata preservation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

class DocumentLoader:
    @staticmethod
    def load_file(file_path: str | Path) -> List[Dict[str, Any]]:
        """
        Loads a file and returns a list of document page/section dictionaries:
        [{'text': ..., 'metadata': {'source': ..., 'page': ...}}]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return DocumentLoader._load_pdf(path)
        elif suffix in [".md", ".markdown", ".txt"]:
            return DocumentLoader._load_text(path)
        elif suffix == ".json":
            return DocumentLoader._load_json(path)
        else:
            # Fallback to UTF-8 text loader
            return DocumentLoader._load_text(path)

    @staticmethod
    def _load_pdf(path: Path) -> List[Dict[str, Any]]:
        import pypdf
        documents = []
        reader = pypdf.PdfReader(str(path))
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "file_path": str(path.resolve()),
                        "page": page_idx + 1,
                        "total_pages": len(reader.pages),
                        "file_type": "pdf"
                    }
                })
        return documents

    @staticmethod
    def _load_text(path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        return [{
            "text": content.strip(),
            "metadata": {
                "source": path.name,
                "file_path": str(path.resolve()),
                "page": 1,
                "total_pages": 1,
                "file_type": path.suffix.lower().replace(".", "")
            }
        }]

    @staticmethod
    def _load_json(path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                text = json.dumps(item, indent=2) if not isinstance(item, str) else item
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "file_path": str(path.resolve()),
                        "section_index": i,
                        "file_type": "json"
                    }
                })
        elif isinstance(data, dict):
            for key, val in data.items():
                text = f"## {key}\n{json.dumps(val, indent=2) if not isinstance(val, str) else val}"
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "file_path": str(path.resolve()),
                        "section_key": key,
                        "file_type": "json"
                    }
                })
        return documents
