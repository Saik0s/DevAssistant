from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_summarize_chain(llm: BaseLLM, verbose: bool = True):
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=verbose)
    text_splitter = RecursiveCharacterTextSplitter()

    def summarize_thoughts(thoughts):
        texts = text_splitter.split_text(thoughts)
        docs = [Document(page_content=t) for t in texts[:3]]
        return chain.run(docs)

    return summarize_thoughts


