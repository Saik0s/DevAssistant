import sys
import pinecone
import os
import random
import string
import langchain_visualizer
from typing import Optional
from orchestrator import AgentOrchestrator
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings


if "--test" in sys.argv:
    OBJECTIVE = (
        "Write a program that takes 2 number as input and outputs the sum of the two numbers, save the program as sum.py. write tests for the program and run the tests, make sure the tests pass."
    )
else:
    OBJECTIVE = input("Please enter the objective: ")

verbose = "--verbose" in sys.argv


####################################################################################################


openai_api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_environment = os.environ["PINECONE_ENVIRONMENT"]
pinecone_index_name = os.environ["PINECONE_INDEX_NAME"]

max_iterations: Optional[int] = None

random_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))
collection_name = f"dev_assistant_{random_id}"
print(f"Using collection name: {collection_name}")


####################################################################################################


if verbose:
    import utils.debug

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index(pinecone_index_name, embeddings, namespace=collection_name)

orchestrator = AgentOrchestrator.from_llm(vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations)

if "--visualizer" in sys.argv:
    async def run():
        orchestrator({"objective": OBJECTIVE})
    langchain_visualizer.visualize(run)
else:
    orchestrator({"objective": OBJECTIVE})
