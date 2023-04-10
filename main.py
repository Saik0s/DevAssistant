import os
# os.environ["LANGCHAIN_HANDLER"] = "langchain"

from modules.perception import PerceptionModule
from modules.memory import MemoryModule
from modules.reasoning import ReasoningModule
from modules.execution import ExecutionModule
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import logging as log
from pprint import pprint

log_format = "%(asctime)s - %(levelname)s - %(message)s"
log.basicConfig(level=log.DEBUG, format=log_format)

chat_llm = ChatOpenAI(temperature=0, max_tokens=1000, verbose=True)
llm = OpenAI(temperature=0, max_tokens=1000, verbose=True)

# Initialize the system components
memory_module = MemoryModule(collection_name="assist", chat_model=chat_llm)
perception_module = PerceptionModule(memory_module)
reasoning_module = ReasoningModule(llm=llm)
execution_module = ExecutionModule(chat_llm=chat_llm, llm=llm, memory_module=memory_module)

# Define the overall objective
objective = "Create a SwiftUI gallery screen with pagination and a search bar."

# Initialize tasks based on the objective
reasoning_module.initialize_tasks(objective)

# Main loop
while True:
    # Get the current task from the ReasoningModule
    current_task = reasoning_module.get_current_task()
    print("Current task:")
    pprint(current_task)
    print("Task list:")
    pprint(reasoning_module.get_task_list())

    # If there are no more tasks, the objective is reached, and the loop can break
    if not current_task:
        log.info("Objective reached!")
        break

    # Process the current task using PerceptionModule
    processed_task = perception_module.process_task(current_task)

    # Execute the task with the ExecutionModule
    execution_result = execution_module.execute(processed_task)

    # Process the execution result using PerceptionModule before storing it in the MemoryModule
    processed_execution_result = perception_module.process_text(execution_result)

    # MemoryModule stores the new data
    memory_module.store(processed_execution_result)

    # Advance to the next task
    reasoning_module.advance_to_next_task()

    # ReasoningModule analyzes the stored data and updates the task priorities or generates new tasks
    reasoning_module.update_tasks(memory_module.get_context_data(), objective)

log.debug("Finishing the script")
