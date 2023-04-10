import os
from typing import Any, Dict, List, Union
from modules.perception import PerceptionModule
from modules.memory import MemoryModule
from modules.reasoning import ReasoningModule
from modules.execution import ExecutionModule
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import logging as log
from langchain.callbacks.shared import SharedCallbackManager
from langchain.callbacks.openai_info import OpenAICallbackHandler, LLMResult

os.environ["LANGCHAIN_HANDLER"] = "langchain"
log_format = "%(asctime)s - %(levelname)s - %(message)s"
log.basicConfig(level=log.DEBUG, format=log_format)

class DebugCallbackHandler(OpenAICallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts with colors."""
        print("\033[1;34m> Prompts:\033[0m")
        for prompt in prompts:
            print(f"\033[1;34m{prompt}\033[0m")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        OpenAICallbackHandler.on_llm_end(self, response, **kwargs)
        print(f"Total cost: {self.total_cost}")
        print("\033[1;32m> Response:\033[0m")
        print(f"\033[1;32m{response}\033[0m")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Print errors with colors."""
        print(f"\033[1;31mError: {error}\033[0m")



    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain and its outputs."""
        print("\n\033[1m> Finished chain.\033[0m")
        print("\033[1m> Chain outputs:\033[0m")
        print(outputs)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Print extensive information about the chain error."""
        print("\n\033[1;31m> Chain error occurred:\033[0m")
        print(f"\033[1;31mError type:\033[0m {type(error).__name__}")
        print(f"\033[1;31mError message:\033[0m {str(error)}")
        print("\033[1;31mError traceback:\033[0m")
        import traceback
        traceback.print_tb(error.__traceback__)

SharedCallbackManager().add_handler(DebugCallbackHandler())

# Define the overall objective
objective = (
    "The goal is to develop a user-friendly SwiftUI gallery screen that integrates The Composable "
    "Architecture from Point-Free, presenting a grid-like gallery of images fetched from an API. "
    "The feature should support pagination and search functionality, and enable users to like "
    "images while showing the like status and count for each image. By leveraging The Composable "
    "Architecture, the resulting feature will be modular, maintainable, and thoroughly tested, "
    "ensuring a smooth user experience, efficient memory usage, and adaptability across various "
    "screen sizes and orientations."
)

chat_llm = ChatOpenAI(temperature=0, model_name="gpt-4", max_tokens=1500, verbose=True)
llm = OpenAI(temperature=0, max_tokens=1500, verbose=True)

# Initialize the system components
memory_module = MemoryModule(collection_name="assist", objective=objective, chat_model=chat_llm)
perception_module = PerceptionModule(memory_module, chat_llm)
reasoning_module = ReasoningModule(llm=llm, objective=objective)
execution_module = ExecutionModule(llm=chat_llm, memory_module=memory_module)


def main():
    # Initialize tasks based on the objective
    reasoning_module.initialize_tasks()
    log.debug("\033[1;34mTasks initialized\033[0m")
    log.debug("\033[1;32mTask list:\033[0m")
    print(reasoning_module.get_task_list(), end="\n\n")

    # Main loop
    while True:
        # Get the current task from the ReasoningModule
        current_task = reasoning_module.get_current_task()
        log.debug("\033[1;32mCurrent task:\033[0m")
        print(current_task, end="\n\n")
        log.debug("\033[1;32mTask list:\033[0m")
        print(reasoning_module.get_task_list(), end="\n\n")

        # If there are no more tasks, the objective is reached, and the loop can break
        if not current_task:
            log.info("\033[1;33mObjective reached!\033[0m")
            break

        # Process the current task using PerceptionModule
        processed_task = perception_module.process_task(current_task)
        log.debug("\033[1;35mProcessed task:\033[0m")
        print(processed_task, end="\n\n")

        # Execute the task with the ExecutionModule
        execution_result = execution_module.execute(processed_task)
        log.debug("\033[1;36mExecution result:\033[0m")
        print(execution_result, end="\n\n")

        # Process the execution result using PerceptionModule before storing it in the MemoryModule
        processed_execution_result = perception_module.process_text(execution_result)
        log.debug("\033[1;35mProcessed execution result:\033[0m")
        print(processed_execution_result, end="\n\n")

        # MemoryModule stores the new data
        memory_module.store(processed_execution_result)
        log.debug("\033[1;34mStored processed execution result in MemoryModule\033[0m")

        # Advance to the next task
        reasoning_module.advance_to_next_task()
        log.debug("\033[1;32mAdvanced to the next task\033[0m")

        # ReasoningModule analyzes the stored data and updates the task priorities and generates new tasks
        reasoning_module.update_tasks(memory_module.get_context())
        log.debug("\033[1;33mUpdated tasks based on stored data\033[0m")

if __name__ == "__main__":
    main()

