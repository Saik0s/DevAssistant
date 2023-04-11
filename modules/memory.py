from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import (
    NLTKTextSplitter,
)
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class MemoryModule:
    def __init__(
        self, collection_name: str, objective: str, chat_model: ChatOpenAI, max_tokens: int = 1000
    ):
        self.vectorstore = self._create_vectorstore(collection_name)
        self.objective = objective
        self.chat_model = chat_model
        self.project_summary = ""
        self.max_tokens = max_tokens

    def retrieve_related_information(self, query, top_k=5):
        try:
            search_results = self.vectorstore.similarity_search(query, k=top_k)
            return "\n".join([result.page_content for result in search_results])
        except Exception as e:
            print(f"An error occurred during similarity search: {e}")
            return ""

    def get_context(self):
        return self.project_summary

    def get_objective(self):
        return self.objective

    def store(self, text: str):
        self._add_to_vectorstore(self.vectorstore, [text])
        self.project_summary = text

    def _create_vectorstore(self, collection_name: str):
        embeddings = OpenAIEmbeddings()
        return Chroma(
            embedding_function=embeddings,
            persist_directory="chroma",
            collection_name=collection_name,
        )

    def _add_to_vectorstore(self, vectorstore: Chroma, texts: List[str]):
        for text in texts:
            splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=0)
            text_chunks = splitter.split_text(text)
            vectorstore.add_texts(text_chunks)
        vectorstore.persist()

    def _update_project_summary(self, text: str):
        template = "You are an intelligent AI that extracts the most crucial information to create a concise project summary. The current summary is:\n{current_summary}\n\nUpdate the summary while ensuring it stays within {max_tokens} tokens. The summary should provide a clear understanding of the current project stage and include all key details."

        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("New entry:\n{new_entry}")

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        chain = LLMChain(llm=self.chat_model, prompt=chat_prompt)

        updated_summary = chain.run(
            {
                "current_summary": self.project_summary,
                "new_entry": text,
                "max_tokens": self.max_tokens,
            }
        )
        self.project_summary = updated_summary
