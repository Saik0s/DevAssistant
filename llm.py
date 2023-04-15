from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_llm(temperature: float = 0, max_tokens: int = 500, model_name: str = "gpt-4", verbose: bool = True):
  from langchain.chat_models import ChatOpenAI
  return ChatOpenAI(temperature=temperature, max_tokens=max_tokens, model_name=model_name, verbose=verbose, request_timeout=180, max_retries=10)
