import argparse
import asyncio
import logging
import os
from typing import Optional
import langchain_visualizer

from dotenv import load_dotenv
from orchestrator import AgentOrchestrator
from langchain.callbacks import SharedCallbackManager, OpenAICallbackHandler, StdOutCallbackHandler

os.environ["LANGCHAIN_HANDLER"] = "langchain"


def load_environment_variables():
    load_dotenv(".envrc")
    return {
        "openai_api_key": os.environ["OPENAI_API_KEY"],
    }


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--obj", type=str, help="Objective")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--visualizer", action="store_true", help="Enable visualizer")
    parser.add_argument("--max_iterations", type=int, help="Set maximum iterations")
    return parser.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level)
    SharedCallbackManager().add_handler(OpenAICallbackHandler())


async def run_orchestrator(objective: str, verbose: bool, max_iterations: Optional[int]):
    orchestrator = AgentOrchestrator.from_llm(verbose=verbose, max_iterations=max_iterations)
    orchestrator({"objective": objective})
    return


async def run_visualizer(objective: str, verbose: bool, max_iterations: Optional[int]):
    async def run():
        await run_orchestrator(objective, verbose, max_iterations)

    langchain_visualizer.visualize(run)


def main():
    _ = load_environment_variables()

    args = parse_arguments()
    verbose = args.verbose
    max_iterations = args.max_iterations

    if args.test:
        objective = "Write a program that takes 2 number as input and outputs the sum of the two numbers, save the program as sum.py. write tests for the program and run the tests, make sure the tests pass."
    else:
        objective = args.obj or input("Please enter the objective: ")

    setup_logging(verbose)

    if args.visualizer:
        asyncio.run(run_visualizer(objective, verbose, max_iterations))
    else:
        asyncio.run(run_orchestrator(objective, verbose, max_iterations))


if __name__ == "__main__":
    main()
