from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_summarize_chain(verbose: bool = True):
    llm = ChatOpenAI(temperature=0, max_tokens=1000, verbose=verbose)
    chain = load_summarize_chain(llm, chain_type="refine")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=0)

    def summarize_text(text):
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts[:3]]
        return chain.run(docs)

    return summarize_text
