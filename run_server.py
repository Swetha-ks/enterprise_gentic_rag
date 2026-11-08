"""
FastAPI Server Launcher.
Starts the Enterprise Agentic RAG REST API on http://localhost:8000
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

def main():
    print("[*] Starting Enterprise Agentic RAG API on http://localhost:8000")
    print("[*] OpenAPI Swagger Docs available at http://localhost:8000/docs")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
