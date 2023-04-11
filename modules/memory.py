from langchain.llms import BaseLLM
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores import Chroma
from typing import List


class MemoryModule:
    def __init__(self, llm: BaseLLM, vectorstore: Chroma, verbose: bool = True):
        self.llm = llm
        self.vectorstore = vectorstore
        self.verbose = verbose

    def retrieve_related_information(self, query, top_k=5):
        try:
            search_results = self.vectorstore.similarity_search_with_score(
                query, k=top_k
            )
            return "\n".join(
                [f"{score}: {doc.page_content}" for doc, score in search_results]
            )
        except Exception as e:
            # print(f"An error occurred during similarity search: {e}")
            return ""

    def get_context(self):
        return self.context

    def get_objective(self):
        return self.objective

    def store(self, text: str):
        self._add_to_vectorstore(self.vectorstore, [text])
        self.context = text

    def _add_to_vectorstore(self, vectorstore: Chroma, texts: List[str]):
        for text in texts:
            splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=0)
            text_chunks = splitter.split_text(text)
            vectorstore.add_texts(text_chunks)
        vectorstore.persist()
