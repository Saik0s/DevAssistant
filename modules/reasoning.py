from typing import Dict, List
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import BaseLLM

class ReasoningModule:
    def __init__(self, llm, objective):
        self.task_list = []
        self.completed_task_list = []
        self.objective = objective
        self.task_creation_chain = TaskCreationChain.from_llm(llm, objective)
        self.task_prioritization_chain = TaskPrioritizationChain.from_llm(llm, objective)

    def initialize_tasks(self):
        initial_tasks = self._analyze_objective_with_llm()
        self._set_tasks(initial_tasks)

    def _analyze_objective_with_llm(self):
        result = self.task_creation_chain.get_next_task({}, "", [])
        return [{"task_id": i, "task_name": task["task_name"]} for i, task in enumerate(result)]

    def prioritize_tasks(self, tasks):
        result = self.task_prioritization_chain.prioritize_tasks(0, tasks)
        return [{"task_id": task["task_id"], "task_name": task["task_name"]} for task in result]

    def get_task_list(self):
        return self.task_list

    def get_current_task(self):
        return self.task_list[0] if self.task_list else None

    def advance_to_next_task(self):
        completed_task = self.task_list.pop(0)
        self.completed_task_list.append(completed_task)

    def update_tasks(self, data: str):
        result = self.task_creation_chain.get_next_task(data, self.get_current_task(), self.task_list)
        updated_tasks = [{"task_id": i, "task_name": task["task_name"]} for i, task in enumerate(result)]
        self._set_tasks(updated_tasks)

    def _set_tasks(self, updated_tasks):
        self.task_list = updated_tasks
        self.task_list = self.prioritize_tasks(self.task_list)
        self.task_list = [task for task in self.task_list if task]

class TaskCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = """
        You are a task creation AI that generates detailed and specific tasks based on the following objective: {objective}.
        The last completed task produced the result: {result}.
        This result was obtained from the task description: {task_description}.
        The following tasks are still incomplete: {incomplete_tasks}.
        Considering the result, create new tasks for the AI system to complete, ensuring they do not overlap with the incomplete tasks.
        Each task should be straightforward, easy to complete in one go, and specific.
        Return the tasks as an array.
        """.strip()
        prompt = PromptTemplate(
            template=task_creation_template,
            partial_variables={"objective": objective},
            input_variables=["result", "task_description", "incomplete_tasks"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def get_next_task(self, result: Dict, task_description: str, task_list: List[Dict]) -> List[Dict]:
        """Get the next task."""
        incomplete_tasks = ", ".join([task["task_name"] for task in task_list])
        response = self.run(result=result, task_description=task_description, incomplete_tasks=incomplete_tasks)
        new_tasks = response.split('\n')
        return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]

class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = """
        You are a task prioritization AI responsible for refining and prioritizing
        the following tasks: {task_names}.
        Keep in mind the ultimate objective of your team: {objective}.
        Do not remove any tasks. Return the result as a numbered list, like:
        #. First task
        #. Second task
        Ensure that each task is clear, specific, and can be completed in a single attempt.
        Start the task list with number {next_task_id}.
        """.strip()
        prompt = PromptTemplate(
            template=task_prioritization_template,
            partial_variables={"objective": objective},
            input_variables=["task_names", "next_task_id"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def prioritize_tasks(self, this_task_id: int, task_list: List[Dict]) -> List[Dict]:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in task_list]
        next_task_id = this_task_id + 1
        response = self.run(task_names=task_names, next_task_id=next_task_id)
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

