from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI


def create_llm(temperature: float = 0, max_tokens: int = 500, model_name: str = "gpt-3.5-turbo", verbose: bool = True):
    return ChatOpenAI(temperature=temperature, max_tokens=max_tokens, model_name=model_name, verbose=verbose, request_timeout=180, max_retries=10)
