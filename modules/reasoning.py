from typing import Dict, List
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import BaseLLM
from typing import List, Dict
from pydantic import Field
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool
from langchain.llms.base import BaseLLM
from typing import List, Dict
from langchain import LLMChain, PromptTemplate
from langchain.llms.base import BaseLLM
from modules.memory import MemoryModule
from utils.helpers import *
from collections import deque
from modules.execution_tools import get_tools

class ReasoningModule:
    def __init__(self, llm, objective, memory_module: MemoryModule, verbose: bool = True):
        self.task_list = deque()
        self.completed_task_list = deque()
        self.objective = objective
        self.memory_module = memory_module
        self.task_creation_chain = TaskCreationChain.from_llm(llm, objective, get_tools(llm, memory_module), verbose)
        self.task_prioritization_chain = TaskPrioritizationChain.from_llm(llm, objective, verbose)

    def initialize_tasks(self):
        initial_tasks = self.task_creation_chain.generate_tasks()
        self._set_tasks(initial_tasks)

    def get_task_list(self):
        return list(self.task_list)

    def get_completed_task_list(self):
        return list(self.completed_task_list)

    def get_current_task(self):
        return self.task_list[0] if self.task_list else None

    def advance_to_next_task(self):
        completed_task = self.task_list.pop(0)
        self.completed_task_list.append(completed_task)

    def update_tasks(self):
        self._set_tasks(self.get_task_list())

    def _set_tasks(self, updated_tasks):
        result = self.task_prioritization_chain.prioritize_tasks(len(self.get_completed_task_list()), self.get_completed_task_list(), updated_tasks, self.memory_module.get_context())
        self.task_list = result

task_creation_template = """You are a task creation AI tasked with generating a full, exhaustive list of tasks to accomplish the following objective: {objective}.
The AI system that will execute these tasks will have access to the following tools:
{tool_strings}
Each task may only use a single tool, but not all tasks need to use one. The task should not specify the tool. The final task should achieve the objective.
Each task will be performed by a capable agent, do not break the problem down into too many tasks.
Aim to keep the list short, and never generate more than 5 tasks. Your response should be each task in a separate line, one line per task.
Use the following format:
1. First task
2. Second task
"""

task_creation_prompt = lambda objective: PromptTemplate(
        template=task_creation_template,
        partial_variables={"objective": objective},
        input_variables=["tool_strings"],
        )


class TaskCreationChain(LLMChain):
    tool_strings: str

    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, tools: List[Tool] , verbose: bool = True):
        tool_strings = "\n".join([f"> {tool.name}: {tool.description}" for tool in tools])
        return cls(prompt=task_creation_prompt(objective), llm=llm, verbose=verbose, tool_strings=tool_strings)

    def generate_tasks(self) -> List[Dict]:
        response = self.run(tool_strings=self.tool_strings)
        return parse_task_list(response)

task_prioritisation_template = """Assistant is a task reviewing and prioritization AI, tasked with cleaning the formatting of and reprioritizing the following tasks: {pending_tasks}.
Assistant is provided with the list of completed tasks, the current pending tasks, and the information context that has been generated so far by the system.

Assistant will decide if the current completed tasks and context are enough to generate a final answer. If this is the case, Assistant will notify this using this exact format:
Review: Can answer

Assistant will never generate the final answer.
If there is not enough information to answer, Assistant will generate a new list of tasks. The tasks will be ordered by priority, with the most important task first. The tasks will be numbered, starting with {next_task_id}. The following format will be used:
Review: Must continue
#. First task
#. Second task

Assistant will use the current pending tasks to generate this list, but it may remove tasks that are no longer required, or add new ones if strictly required.

The ultimate objective is: {objective}.
The following tasks have already been completed: {completed_tasks}.
This is the information context generated so far:
{context}
"""

task_prioritization_prompt = lambda objective: PromptTemplate(
        template=task_prioritisation_template,
        partial_variables={"objective": objective},
        input_variables=["completed_tasks", "pending_tasks", "context", "next_task_id"],
        )

class TaskPrioritizationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True):
        return cls(prompt=task_prioritization_prompt(objective), llm=llm, verbose=verbose)

    def prioritize_tasks(self, this_task_id: int, completed_tasks: List[str], pending_tasks: List[Dict], context: str) -> List[Dict]:
        pending_tasks = [t["task_name"] for t in pending_tasks]
        next_task_id = this_task_id + 1
        response = self.run(completed_tasks=completed_tasks, pending_tasks=pending_tasks, context=context, next_task_id=next_task_id)
        return parse_task_list(response)
