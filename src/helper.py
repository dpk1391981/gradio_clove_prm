from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os
import subprocess
import shutil

# Extract data from pdf
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()

#split text
def text_split(extract_data):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(extract_data)

#embeddings
def download_huggingface_embedding(model_name = "all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def process_pdfs(uploaded_files):
    if not uploaded_files:
        return "No files uploaded."
    
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "pdfs")
    os.makedirs(data_dir, exist_ok=True)  # Ensure the Data directory exists

    processed_files = []
    skipped_files = []

    for uploaded_file in uploaded_files:
        file_name = os.path.basename(uploaded_file)
        file_path = os.path.join(data_dir, file_name)  # Ensure safe filename
        
        # Check if file already exists
        if os.path.exists(file_path):
            print(f"Skipping {file_name}, already exists.")
            skipped_files.append(file_name)
            continue  # Skip processing the duplicate file
        
        shutil.copy(uploaded_file, file_path)  # Copy new file
        processed_files.append(file_path)
        print(f"Saved file: {file_path}")

    # Execute vectordb_store.py only if new files were processed
    if processed_files:
        subprocess.run(["python", "vectordb_store.py"], check=True)

    # Proper return statement using Python's ternary operator
    return f"File already exists." if not processed_files else f"Processed {len(processed_files)} documents. VectorDB updated."

def few_shots():
    few_shots = [
        {
            "Question": "How many users do we have?",
            "SQLQuery": "SELECT COUNT(*) FROM users",
            "SQLResult": "Result of the SQL query",
            "Answer": "1118"
        },
        {
            "Question": "How many facilities or clinics do we have?",
            "SQLQuery": "SELECT COUNT(*) FROM facility",
            "SQLResult": "Result of the SQL query",
            "Answer": "62"
        },
        {
            "Question": "How many users are in Amanora clinics?",
            "SQLQuery": """
                SELECT COUNT(*) 
                FROM users  
                INNER JOIN facility ON users.facility_id = facility.id 
                WHERE facility.name = 'Amanora'
            """,
            "SQLResult": "Result of the SQL query",
            "Answer": "5"
        },
        {
            "Question": "How many receipts do we have for last year?",
            "SQLQuery": """
                SELECT COUNT(*) 
                FROM reciept 
                WHERE YEAR(rect_created_date) = YEAR(CURDATE()) - 1
            """,
            "SQLResult": "Result of the SQL query",
            "Answer": "480"
        },
        {
            "Question": "What is the cost of 'RCT with Rubber Dam - By consultant, using rotary files with endomotor, apex locator, permanent fill' treatment?",
            "SQLQuery": """
                SELECT trname, SUM(trprice) AS total_price 
                FROM treatment_master 
                WHERE trname LIKE '%RCT with Rubber Dam%' 
                GROUP BY trname
            """,
            "SQLResult": "Result of the SQL query",
            "Answer": "5000"
        },
        {
            "Question": "what is pubpid for patient deepak_17_pay_reco_to ?",
            "SQLQuery": """
                SELECT pubpid FROM patient_data WHERE fname = 'deepak_17_pay_reco_to' LIMIT 1;
            """,
            "SQLResult": "Result of the SQL query",
            "Answer": "5000"
        },
        {
            "Question": "what is pubpid for patient deepak_17_pay_reco_to ?",
            "SQLQuery": """
                SELECT pubpid FROM patient_data WHERE fname = 'deepak_17_pay_reco_to' LIMIT 1;
            """,
            "SQLResult": "Result of the SQL query",
            "Answer": "5000"
        },
        {
            "Question": "what is name and email for patient deepak_17_pay_reco_to ?",
            "SQLQuery": """
                SELECT pubpid FROM patient_data WHERE fname = 'deepak_17_pay_reco_to' LIMIT 1;
            """,
            "SQLResult": "Result of the SQL query",
            "Answer": "5000"
        }

    ]
    
    return few_shots
