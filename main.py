from typing import Any, Dict, List, Union
from modules.perception import PerceptionModule
from modules.memory import MemoryModule
from modules.reasoning import ReasoningModule
from modules.execution import ExecutionModule
from modules.learning import LearningModule
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import logging as log
from langchain.callbacks.shared import SharedCallbackManager
from langchain.callbacks.openai_info import OpenAICallbackHandler, LLMResult
from collections import deque
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain.agents import Tool
from langchain.llms.base import BaseLLM
from langchain.vectorstores import VectorStore

from utils.helpers import print_objective, print_next_task, print_processed_task, print_task_list, print_task_result, print_end

import utils.debug

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
max_iterations: Optional[int] = None

chat_llm = ChatOpenAI(temperature=0, max_tokens=1500, verbose=True)
# llm = OpenAI(temperature=0, max_tokens=1500, verbose=True)

# Initialize the system components
memory_module = MemoryModule(collection_name="assist", objective=objective, chat_model=chat_llm)
perception_module = PerceptionModule(memory_module, chat_llm)
learning_module = LearningModule(chat_llm, objective)
reasoning_module = ReasoningModule(llm=chat_llm, objective=objective, memory_module=memory_module)
execution_module = ExecutionModule(llm=chat_llm, objective=objective, memory_module=memory_module)

current_iteration = 0

def main():
    print_objective(objective)

    # Initialize tasks based on the objective
    reasoning_module.initialize_tasks()

    # Main loop
    while True:

        # Get the current task from the ReasoningModule
        current_task = reasoning_module.get_current_task()
        print_task_list(reasoning_module.completed_task_list, reasoning_module.task_list)

        # If there are no more tasks, the objective is reached, and the loop can break
        if not current_task or (max_iterations and current_iteration > max_iterations):
            log.info("\033[1;33mObjective reached!\033[0m")
            break

        # Process the current task using PerceptionModule
        processed_task = perception_module.process_task(current_task)
        print_processed_task(processed_task)

        # Execute the task with the ExecutionModule
        execution_result = execution_module.execute(processed_task["task_name"], processed_task["context"])
        print_task_result(execution_result)

        # Process the execution result using PerceptionModule before storing it in the MemoryModule
        processed_execution_result = perception_module.process_text(execution_result)
        print_task_result(processed_execution_result)

        new_memory = learning_module.learn_from(
            memory=memory_module.get_context(),
            observation=processed_execution_result,
            completed_tasks=reasoning_module.get_completed_task_list(),
            pending_tasks=[t["task_name"] for t in reasoning_module.get_task_list()],
        )

        # MemoryModule stores the new data
        memory_module.store(new_memory)
        print("\033[1;34mSaved new memory\033[0m")

        # Advance to the next task
        reasoning_module.advance_to_next_task()
        print("\033[1;32mAdvanced to the next task\033[0m")

        # ReasoningModule analyzes the stored data and updates the task priorities and generates new tasks
        reasoning_module.update_tasks()
        print("\033[1;33mUpdated tasks based on stored data\033[0m")

    final_answer = execution_module.execute("Provide the final answer", memory_module.get_context())
    print_end(final_answer)

if __name__ == "__main__":
    main()
