from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import PythonCodeTextSplitter, MarkdownTextSplitter, NLTKTextSplitter
from langchain.schema import Document
from typing import List

class MemoryModule:
    def __init__(self, collection_name: str):
        self.vectorstore = self.create_vectorstore(collection_name)

    def create_vectorstore(self, collection_name: str):
        embeddings = OpenAIEmbeddings()
        return Chroma(
            embedding_function=embeddings,
            persist_directory="chroma",
            collection_name=collection_name,
        )

    def store(self, text: str):
        self.add_to_vectorstore(self.vectorstore, [Document(text)])

    def add_to_vectorstore(self, vectorstore: Chroma, documents: List[Document]):
        doc_chunks = []
        for document in documents:
            filename = document.extra_info["file_name"] or ""
            if filename.endswith('.py'):
                splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=0)
            elif filename.endswith('.md'):
                splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
            else:
                splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=0)
            text_chunks = splitter.split_text(document.text)
            doc_chunks.extend(Document(text, doc_id=document.doc_id, extra_info=document.extra_info) for text in text_chunks)

        documents = [d.to_langchain_format() for d in doc_chunks]
        vectorstore.add_documents(documents)
        vectorstore.persist()

    def retrieve_related_information(self, query, top_k=5):
        search_results = self.vectorstore.similarity_search(query, k=top_k)
        return [result["document"] for result in search_results]

    def get_context_data(self):
        search_results = self.vectorstore.similarity_search(query="task results and project state", k=20)
        return [result["document"] for result in search_results]

