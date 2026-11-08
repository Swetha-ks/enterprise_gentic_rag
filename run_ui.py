"""
Streamlit UI Launcher.
Starts the Interactive Dashboard on http://localhost:8501
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    dashboard_path = PROJECT_ROOT / "frontend" / "dashboard.py"
    print(f"[*] Launching NexRAG Streamlit Dashboard from: {dashboard_path}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])

if __name__ == "__main__":
    main()
