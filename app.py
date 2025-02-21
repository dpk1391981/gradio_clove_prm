import gradio as gr
import cassio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from agents.sql_agent import sql_agent
from agents.rag_agent import retrieve
from agents.wikipedia import wiki_search
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal, Any
from typing_extensions import TypedDict
from gradiocallback import GradioCallbackHandler  # Import the custom handler
import tempfile

# Load environment variables
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID_MULTI_AGENT = os.getenv("ASTRA_DB_ID_MULTI_AGENT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Cassandra/AstraDB
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID_MULTI_AGENT)

# Database Configuration
MYSQL_HOST = "localhost"
MYSQL_USER = "genai"
MYSQL_PASS = "Genai123!"
MYSQL_DB = "offer_prm_uat"

# Data Model
class RoueQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search", "sql_agent"] = Field(
        ..., description="Choose to route it to Wikipedia, vectorstore, or a SQL agent."
    )

# Manage chat history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

session_store = {}
query_limit = 100

class GraphState(TypedDict):
    question: str
    llm: Any
    dbconfig: dict
    generation: str
    callbacks: Any
    documents: List[str]
    pdf_documents: Any
    get_session_history: Any
    session_id: Any
    agents: str  # <-- Add this


# Function to route question
def route_question(state):
    print("---ROUTE QUESTION---")
    print(state)

    # Ensure 'agents' key exists
    agents = state.get("agents", None)  
    if agents is None:
        raise KeyError("'agents' key is missing from state!")

    if agents == "RAG-PDFs":
        return "vectorstore"
    elif agents == "Wikipedia":
        return "wiki_search"

    return "sql_agent"

def process_pdfs(uploaded_files):
    pdf_documents = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Ensure uploaded_file is a dictionary with 'name' and 'data'
            if isinstance(uploaded_file, dict) and "data" in uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file["data"])  # Use data instead of read()
                    temp_file_path = temp_file.name
                
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                pdf_documents.extend(docs)

    return f"Processed {len(pdf_documents)} documents."


# Main AI Chat Function
def chat_with_ai(question, api_key_type, agents, uploaded_files):
    model = 'gpt-4o' if api_key_type == "Open API" else 'deepseek-r1:1.5b'
    api_key = OPENAI_API_KEY if api_key_type == "Open API" else GROQ_API_KEY
    llm = ChatOpenAI(api_key=api_key, model=model, temperature=0, streaming=True)
    pdf_documents = []
    
    if agents == 'RAG-PDFs' and uploaded_files:
        process_pdfs(uploaded_files)
    
    workflow = StateGraph(GraphState)
    workflow.add_node("sql_agent", sql_agent)
    if agents == "RAG-PDFs":
        workflow.add_node("retrieve", retrieve)
    elif agents == "Wikipedia":
        workflow.add_node("wiki_search", wiki_search)
    
    routeNode = {"sql_agent": "sql_agent"}
    if agents == "RAG-PDFs":
        routeNode["vectorstore"] = "retrieve"
    elif agents == "Wikipedia":
        routeNode["wiki_search"] = "wiki_search"
    
    workflow.add_conditional_edges(START, route_question, routeNode)
    workflow.add_edge("sql_agent", END)
    if agents == "RAG-PDFs":
        workflow.add_edge("retrieve", END)
    elif agents == "Wikipedia":
        workflow.add_edge("wiki_search", END)
    
    app = workflow.compile()
    inputs = {
    "question": question,
    "llm": llm,
    "dbconfig": {
        "host": MYSQL_HOST,
        "user": MYSQL_USER,
        "pass": MYSQL_PASS,
        "db_name": MYSQL_DB,
        "limit": query_limit,
    },
    "callbacks": GradioCallbackHandler(gr.update),  # Use the custom handler
    "pdf_documents": pdf_documents,
    "get_session_history": get_session_history,
    "session_id": "default_session",
    "agents": agents  # Ensure 'agents' is included
}

    
    response = ""
    for output in app.stream(inputs):
        for key, value in output.items():
            response += value['documents'].page_content + "\n"
    return response

# Gradio Interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# PRM AI Assistant")
        api_key_type = gr.Dropdown(["Deepseek ollama API", "Open API"], label="Select LLM API")
        agents = gr.Dropdown(["RAG-PDFs", "SQL", "Wikipedia"], label="Select Agent")
        uploaded_files = gr.File(label="Upload PDFs", file_types=[".pdf"], interactive=True)
        question = gr.Textbox(label="Ask a question")
        submit = gr.Button("Submit")
        output = gr.Textbox(label="Response")
        
        submit.click(chat_with_ai, inputs=[question, api_key_type, agents, uploaded_files], outputs=output)
    
    demo.launch()

gradio_interface()