import gradio as gr
import cassio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from src.agents.prm_sql import sql_agent
from src.agents.rag import retrieve
from src.agents.external import wiki_search
from langgraph.graph import StateGraph, START, END
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal, Any
from typing_extensions import TypedDict
from src.gradiocallback import GradioCallbackHandler  # Import the custom handler
from src.helper import process_pdfs

# Load environment variables
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID_MULTI_AGENT = os.getenv("ASTRA_DB_ID_MULTI_AGENT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#mysql setup
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASS = os.getenv("MYSQL_PASS")
MYSQL_DB = os.getenv("MYSQL_DB")

#Astra DB config
ASTRA_KEYSPACE = os.getenv("ASTRA_KEYSPACE")
ASTRA_TBL = os.getenv("ASTRA_TBL")

# Initialize Cassandra/AstraDB
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID_MULTI_AGENT)

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
    astraConfig: dict
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

pdf_documents = []

# Main AI Chat Function
def chat_with_ai(message_history, question, api_key_type, agents):
    try:
        model = 'gpt-4o' if api_key_type == "Open API" else 'deepseek-r1-distill-llama-70b'
        api_key = OPENAI_API_KEY if api_key_type == "Open API" else GROQ_API_KEY
        
        if not api_key:
            raise ValueError("API Key is missing. Please configure the correct API key.")
        try:
            llm = ChatOpenAI(api_key=api_key, model=model, temperature=0, streaming=True) if api_key_type == "Open API" else ChatGroq(groq_api_key=api_key, model=model, streaming=True)
        except Exception as e:
            return f"Error initializing LLM: {str(e)}"

        try:
            workflow = StateGraph(GraphState)
            workflow.add_node("sql_agent", sql_agent)
            
            if agents == "RAG-PDFs":
                workflow.add_node("retrieve", retrieve)
            elif agents == "Wikipedia":
                workflow.add_node("wiki_search", wiki_search)

            # Define routing
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
        except Exception as e:
            return f"Error setting up workflow: {str(e)}"

        inputs = {
            "question": message_history,
            "llm": llm,
            "dbconfig": {
                "host": MYSQL_HOST,
                "user": MYSQL_USER,
                "pass": MYSQL_PASS,
                "db_name": MYSQL_DB,
                "limit": query_limit,
            },
            "astraConfig": {
                "keyspace": ASTRA_KEYSPACE,
                "table": ASTRA_TBL
            },
            "callbacks": GradioCallbackHandler(gr.update),  # Use the custom handler
            "pdf_documents": pdf_documents,
            "get_session_history": get_session_history,
            "session_id": "default_session",
            "agents": agents  # Ensure 'agents' is included
        }

        response = ""
        try:
            for output in app.stream(inputs):
                for key, value in output.items():
                    response += value['documents'].page_content + "\n"
        except Exception as e:
            return f"Error during response generation: {str(e)}"

        return response

    except Exception as e:
        return f"Unexpected error: {str(e)}"


def toggle_upload(agent):
    return gr.update(visible=(agent == "RAG-PDFs"))

with gr.Blocks() as app:
    gr.Markdown("# PRM AI Assistant")

    with gr.Accordion("Additional Input", open=False):
        api_key_type = gr.Dropdown(["Open API", "Deepseek Ollama API"], label="Select LLM API")
        agents = gr.Dropdown(["RAG-PDFs", "SQL", "Wikipedia"], label="Select Agent")

    with gr.Row(visible=True) as file_upload_section:
        file_upload = gr.Files(file_types=[".pdf"], label="Upload PDFs")
        output_text = gr.Textbox(label="Status")

    file_upload.change(process_pdfs, inputs=file_upload, outputs=output_text)
    agents.change(toggle_upload, inputs=agents, outputs=file_upload_section)

    gr.ChatInterface(
        chat_with_ai,
        type="messages",
        additional_inputs=[api_key_type, agents]
    )

if __name__ == "__main__":
    app.launch()