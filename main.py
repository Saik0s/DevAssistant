from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from orchestrator import AgentOrchestrator
from typing import Optional

# import utils.debug

OBJECTIVE = (
    "Write a program that takes 2 number as input and outputs the sum of the two numbers, save the program as sum.py. write tests for the program and run the tests, make sure the tests pass."
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
