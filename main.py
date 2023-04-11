from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from orchestrator import AgentOrchestrator
from typing import Optional

# import utils.debug

OBJECTIVE = (
    "The goal is to develop a user-friendly SwiftUI gallery screen that integrates The Composable "
    "Architecture from Point-Free, presenting a grid-like gallery of images fetched from an API. "
    "The feature should support pagination and search functionality, and enable users to like "
    "images while showing the like status and count for each image. By leveraging The Composable "
    "Architecture, the resulting feature will be modular, maintainable, and thoroughly tested, "
    "ensuring a smooth user experience, efficient memory usage, and adaptability across various "
    "screen sizes and orientations."
)

verbose = False
max_iterations: Optional[int] = None

llm = ChatOpenAI(temperature=0, max_tokens=1000, verbose=verbose)
# llm = OpenAI(temperature=0)

collection_name = "dev_assistant"
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma",
    collection_name=collection_name,
)

orchestrator = AgentOrchestrator.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)

async def run():
    orchestrator({"objective": OBJECTIVE})

import langchain_visualizer
langchain_visualizer.visualize(run)
