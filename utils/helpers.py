from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_summarize_chain(verbose: bool = True):
    llm = ChatOpenAI(temperature=0, max_tokens=750, verbose=verbose)
    chain = load_summarize_chain(llm, chain_type="refine", verbose=verbose)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=0)

    def summarize_text(text):
        if isinstance(text, str):
            texts = text_splitter.split_text(text)
            docs = [Document(page_content=t) for t in texts]
        elif isinstance(text, Document):
            docs = text_splitter.split_documents([text])
        else:
            print("Invalid input type. Expected str or Document.")
            texts = text_splitter.split_text(str(text))
            docs = [Document(page_content=t) for t in texts]

        if len(docs) == 0:
            print("No text to summarize.")
            return ""
        else:
            return chain.run(docs)

    return summarize_text
