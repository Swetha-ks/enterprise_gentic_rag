"""
Project Packaging Script.
Creates a clean, production-ready .zip archive of the enterprise-agentic-rag project,
excluding pycache, virtualenvs, and temporary files.
"""

import os
import zipfile
from pathlib import Path

def create_project_zip():
    project_dir = Path(__file__).resolve().parent
    parent_dir = project_dir.parent
    zip_output_path = parent_dir / "enterprise-agentic-rag.zip"

    ignored_dirs = {
        "__pycache__", ".git", ".pytest_cache", ".venv", "venv", "node_modules", ".idea", ".vscode"
    }
    ignored_extensions = {".pyc", ".pyo", ".pyd"}

    print(f"[*] Packaging project from: {project_dir}")
    print(f"[*] Destination archive:    {zip_output_path}")

    file_count = 0
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_dir):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in ignored_extensions or file.startswith("."):
                    if file != ".env.example":
                        continue

                # Relative path inside the zip archive
                rel_path = file_path.relative_to(parent_dir)
                zipf.write(file_path, arcname=str(rel_path))
                file_count += 1

    size_kb = zip_output_path.stat().st_size / 1024
    print(f"[+] Successfully packaged {file_count} files into '{zip_output_path.name}' ({size_kb:.1f} KB)")
    return zip_output_path

if __name__ == "__main__":
    create_project_zip()
