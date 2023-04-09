from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI


class ReasoningModule:
    def __init__(self, llm):
        self.task_list = []
        self.completed_task_list = []
        self.llm_chain_task_analysis = self.create_llm_chain_task_analysis(llm)
        self.llm_chain_task_prioritization = self.create_llm_chain_task_prioritization(llm)
        self.llm_chain_task_update = self.create_llm_chain_task_update(llm)

    def create_llm_chain_task_analysis(self, llm):
        template = """Given the objective: '{objective}', provide a list of tasks that should be completed to achieve this objective.

Tasks:"""
        prompt_template = PromptTemplate(
            input_variables=["objective"], template=template
        )
        return LLMChain(llm=llm, prompt=prompt_template, output_key="tasks")

    def create_llm_chain_task_prioritization(self, llm):
        template = """Given the objective: '{objective}', prioritize the following tasks in order of importance:

Tasks:
{tasks}

Prioritized tasks:"""
        prompt_template = PromptTemplate(
            input_variables=["objective", "tasks"], template=template
        )
        return LLMChain(llm=llm, prompt=prompt_template, output_key="prioritized_tasks")

    def create_llm_chain_task_update(self, llm):
        template = """Given the objective: '{objective}', the current tasks, and new data, suggest any updates to the task list:

Current tasks:
{current_tasks}

New data:
{data}

Updated tasks:"""
        prompt_template = PromptTemplate(
            input_variables=["current_tasks", "data", "objective"], template=template
        )
        return LLMChain(llm=llm, prompt=prompt_template, output_key="updated_tasks")

    def initialize_tasks(self, objective):
        # Analyze the objective and generate a list of tasks
        initial_tasks = self.analyze_objective_with_llm(objective)

        self._set_tasks(initial_tasks, objective)

    def analyze_objective_with_llm(self, objective):
        # Use LLMChain to analyze the objective and generate a list of tasks
        result = self.llm_chain_task_analysis({"objective": objective})
        return result["tasks"].split("\n")

    def prioritize_tasks(self, tasks, objective):
        # Use LLMChain to prioritize tasks
        tasks_string = "\n".join(tasks).strip()
        result = self.llm_chain_task_prioritization({"tasks": tasks_string, "objective": objective})
        return result["prioritized_tasks"].split("\n")

    def get_task_list(self):
        return self.task_list

    def get_current_task(self):
        return self.task_list[0] if self.task_list else None

    def advance_to_next_task(self):
        completed_task = self.task_list.pop(0)
        self.completed_task_list.append(completed_task)

    def update_tasks(self, data: str, objective):
        # Use LLMChain to update tasks based on new data
        current_tasks_string = "\n".join(self.task_list).strip()
        result = self.llm_chain_task_update(
            {"current_tasks": current_tasks_string, "data": data, "objective": objective}
        )
        updated_tasks = result["updated_tasks"].split("\n")
        self._set_tasks(updated_tasks, objective)

    def _set_tasks(self, updated_tasks, objective):
        # Replace the current task list with the updated tasks
        self.task_list = updated_tasks

        self.task_list = self.prioritize_tasks(self.task_list, objective)

        # Remove empty tasks from the task list
        self.task_list = [task for task in self.task_list if task]
