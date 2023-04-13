from collections import deque
from langchain import LLMChain, PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from modules.memory import MemoryModule
from typing import Dict, List


class ReasoningModule:
    first_task = "Analyze objective"

    def __init__(self, llm, memory_module: MemoryModule, verbose: bool = True):
        self.task_list = deque()
        self.completed_task_list = deque()
        self.memory_module = memory_module
        self.task_creation_chain = TaskCreationChain.from_llm(llm, verbose)
        self.task_prioritization_chain = TaskPrioritizationChain.from_llm(llm, verbose)

    def initialize_tasks(self):
        self.task_list.append({"task_id": 1, "task_name": self.first_task})

    def update_tasks(self, task: dict, result: dict):
        incomplete_tasks = [t["task_name"] for t in self.task_list]
        task_description = task["task_name"]
        incomplete_tasks = "\n".join(incomplete_tasks)
        objective = self.memory_module.objective
        response = self.task_creation_chain.run(
            result=result, task_description=task_description, incomplete_tasks=incomplete_tasks, objective=objective
        )
        new_tasks = response.split("\n")
        new_tasks = [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]
        this_task_id = int("".join(filter(str.isdigit, task["task_id"]))) if isinstance(task["task_id"], str) else task["task_id"]
        task_id_counter = this_task_id
        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            self.task_list.append(task)
        self.task_list = deque(self.prioritize_tasks(this_task_id))

    def prioritize_tasks(self, this_task_id: int) -> List[Dict]:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in self.task_list]
        task_names = "\n".join(task_names)
        objective = self.memory_module.objective
        next_task_id = this_task_id + 1
        response = self.task_prioritization_chain.run(task_names=task_names, next_task_id=next_task_id, objective=objective)
        new_tasks = response.split("\n")
        prioritized_task_list = []
        for task_string in new_tasks:
            if not task_string.strip():
                continue
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
        return prioritized_task_list


class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective}.\n"
            "The last completed task has the result: {result}.\n"
            "This result was based on this task description: {task_description}.\n"
            "These are incomplete tasks: \n{incomplete_tasks}\n"
            "Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks.\n"
            "Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=["result", "task_description", "incomplete_tasks", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: \n{task_names}\n"
            "Consider the ultimate objective of your team: {objective}\n"
            "Do not remove any tasks. Return the result as a numbered list, like:\n"
            "1. First task\n"
            "2. Second task\n"
            "Start the task list with number {next_task_id}.\n"
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
