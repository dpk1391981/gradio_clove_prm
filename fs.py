import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    "src/agents/rag.py",
    "src/agents/prm_sql.py",
    "src/agents/external.py",
    ".env",
    "requirements.txt",
    "README.md",
    "setup.py",
    "app.py",
    "vectordb_store.py"
    # "research/test.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for the file {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Created file: {filepath}")
            
    else:
        logging.info(f"File {filename} already exists")