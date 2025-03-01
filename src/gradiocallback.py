from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
import gradio as gr

class GradioCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for Gradio."""
    
    def __init__(self, update_fn):
        self.update_fn = update_fn  # Function to update the UI

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.update_fn("⏳ Processing your request...")

    def on_llm_new_token(self, token: str, **kwargs):
        self.update_fn(token, append=True)  # Append tokens to response

    def on_llm_end(self, response: LLMResult, **kwargs):
        self.update_fn("✅ Done!", append=True)

    def on_llm_error(self, error: Exception, **kwargs):
        self.update_fn(f"❌ Error: {str(error)}")
