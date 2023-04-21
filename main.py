import argparse
import asyncio
import logging
import os
import random
import string
from typing import Optional

import pinecone
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from orchestrator import AgentOrchestrator
from utils.debug import enable_verbose_logging


def load_environment_variables():
    load_dotenv('.envrc')
    return {
        "openai_api_key": os.environ["OPENAI_API_KEY"],
        "pinecone_api_key": os.environ["PINECONE_API_KEY"],
        "pinecone_environment": os.environ["PINECONE_ENVIRONMENT"],
        "pinecone_index_name": os.environ["PINECONE_INDEX_NAME"],
    }


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--obj", type=str, help="Objective")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--visualizer", action="store_true", help="Enable visualizer")
    return parser.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level)


async def run_orchestrator(objective: str, vectorstore: Pinecone, verbose: bool, max_iterations: Optional[int]):
    orchestrator = AgentOrchestrator.from_llm(vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations)
    await orchestrator({"objective": objective})


async def run_visualizer(objective: str, vectorstore: Pinecone, verbose: bool, max_iterations: Optional[int]):
    async def run():
        await run_orchestrator(objective, vectorstore, verbose, max_iterations)

    langchain_visualizer.visualize(run)


def main():
    environment_variables = load_environment_variables()
    openai_api_key = environment_variables["openai_api_key"]
    pinecone_api_key = environment_variables["pinecone_api_key"]
    pinecone_environment = environment_variables["pinecone_environment"]
    pinecone_index_name = environment_variables["pinecone_index_name"]

    args = parse_arguments()
    verbose = args.verbose
    max_iterations = None

    if args.test:
        objective = "Write a program that takes 2 number as input and outputs the sum of the two numbers, save the program as sum.py. write tests for the program and run the tests, make sure the tests pass."
    else:
        objective = args.obj or input("Please enter the objective: ")

    if verbose:
        enable_verbose_logging()

    setup_logging(verbose)

    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    collection_name = f"dev_assistant_{random_id}"
    logging.info(f"Using collection name: {collection_name}")

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(pinecone_index_name, embeddings, namespace=collection_name)

    if args.visualizer:
        asyncio.run(run_visualizer(objective, vectorstore, verbose, max_iterations))
    else:
        asyncio.run(run_orchestrator(objective, vectorstore, verbose, max_iterations))


if __name__ == "__main__":
    main()
