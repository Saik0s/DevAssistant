from collections import deque
from langchain import LLMChain, PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from modules.memory import MemoryModule
from typing import Dict, List

class ReasoningModule:
    def __init__(self, llm, memory_module: MemoryModule, verbose: bool = True):
        self.task_list = deque()
        self.completed_task_list = deque()
        self.memory_module = memory_module
        self.task_creation_chain = TaskCreationChain.from_llm(llm, verbose)
        self.task_prioritization_chain = TaskPrioritizationChain.from_llm(llm, verbose)

    def initialize_tasks(self):
        first_task = "Do a research for the most optimal way to achieve the objective. max 500 characters"
        self.task_list.append({"task_id": 1, "task_name": first_task})

    def update_tasks(self, task: dict, result: dict):
        incomplete_tasks = [t["task_name"] for t in self.task_list]
        task_description = task["task_name"]
        incomplete_tasks = ", ".join(incomplete_tasks)
        if len(self.task_list) == 0:
            incomplete_tasks = "all"
        objective = self.memory_module.objective
        response = self.task_creation_chain.run(result=result, task_description=task_description, incomplete_tasks=incomplete_tasks, objective=objective)
        new_tasks = response.strip().split('\n')
        new_tasks = [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]
        this_task_id = int(task["task_id"])
        task_id_counter = this_task_id

        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            self.task_list.append(new_task)

        self.task_list = deque(
            self.prioritize_tasks(this_task_id)
        )

    def prioritize_tasks(self, this_task_id: int) -> List[Dict]:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in self.task_list]
        objective = self.memory_module.objective
        next_task_id = this_task_id + 1
        response = self.task_prioritization_chain.run(task_names=task_names, next_task_id=next_task_id, objective=objective)
        new_tasks = response.split('\n')
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
            "running on the AI system following the instructions: use WBS format with 3 levels depth to create a task list, focus on the objective, all tasks must be minimal and nessessary to achieve the ultimate goal. Not doing anything far from ultimate goal. keep time-value ratio balance. not spend more than 30 percent in research, use authorized tools. limitation at 100 tassk,"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Ouput format: a task list as an WBS template, no any explanation needed"
            " #task number. task name"
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
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
