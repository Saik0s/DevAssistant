import sys
import pinecone
import os
import langchain_visualizer
from typing import Optional
from orchestrator import AgentOrchestrator
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

if "--test" in sys.argv:
    OBJECTIVE = (
        "Create a simple project that uses next.js and python for the backend."
        " It should have a chat interface and subscribe to new messages in real time."
    )
else:
    OBJECTIVE = input("Please enter the objective: ")


####################################################################################################


openai_api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_environment = os.environ["PINECONE_ENVIRONMENT"]
pinecone_index_name = os.environ["PINECONE_INDEX_NAME"]

verbose = True
max_iterations: Optional[int] = None

collection_name = "dev_assistant"


####################################################################################################


if verbose:
    import utils.debug

llm = ChatOpenAI(temperature=0, max_tokens=1000, verbose=verbose)
# llm = OpenAI(temperature=0)

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index(
    pinecone_index_name, embeddings, namespace=collection_name
)

orchestrator = AgentOrchestrator.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)

async def run():
    orchestrator({"objective": OBJECTIVE})

langchain_visualizer.visualize(run)
