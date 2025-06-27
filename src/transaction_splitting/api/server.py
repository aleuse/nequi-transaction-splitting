"""
Server script for running the Transaction Splitting Detection API.
"""
import uvicorn
import sys
from pathlib import Path

if __name__ == "__main__":
    # Add the project root to Python path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Configure uvicorn with import string for reload to work
    uvicorn.run(
        "src.transaction_splitting.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info",
        access_log=True,
        timeout_keep_alive=30
    ) 