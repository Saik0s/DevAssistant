from langchain.llms import BaseLLM
from langchain.text_splitter import NLTKTextSplitter, Document
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from typing import List
import faiss
import uuid


class MemoryModule:
    vectorstore: VectorStoreRetriever

    def __init__(self, llm: BaseLLM, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose
        embeddings_model = OpenAIEmbeddings()
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        self.vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}).as_retriever(
            search_kwargs={"k": 5}
        )

    def retrieve_related_information(self, query):
        try:
            search_results = self.vectorstore.get_relevant_documents(query)
            return "\n".join([doc.page_content for doc in search_results])[:10000]
        except Exception as e:
            print(f"An error occurred during similarity search: {e}")
            return ""

    def store_result(self, result: str, task: dict):
        self.vectorstore.add_documents([Document(page_content=result, metadata=task)])

    def store(self, text: str):
        self._add_to_vectorstore([text])
        self.context = text

    def _add_to_vectorstore(self, texts: List[str]):
        try:
            for text in texts:
                text_id = str(uuid.uuid4())
                splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=0)
                text_chunks = splitter.split_text(text)
                self.vectorstore.add_documents(
                    [Document(page_content=chunk, metadata={"text_id": text_id}) for chunk in text_chunks]
                )
        except Exception as e:
            print(f"An error occurred during adding documents to vectorstore: {e}")
