from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI


class ReasoningModule:
    def __init__(self):
        self.task_list = []
        self.llm_chain_task_analysis = self.create_llm_chain_task_analysis()
        self.llm_chain_task_prioritization = self.create_llm_chain_task_prioritization()
        self.llm_chain_task_update = self.create_llm_chain_task_update()

    def create_llm_chain_task_analysis(self):
        llm = OpenAI(temperature=0.7)
        template = """Given the objective: '{objective}', provide a list of tasks that should be completed to achieve this objective.

Tasks:"""
        prompt_template = PromptTemplate(
            input_variables=["objective"], template=template
        )
        return LLMChain(llm=llm, prompt=prompt_template, output_key="tasks")

    def create_llm_chain_task_prioritization(self):
        llm = OpenAI(temperature=0.7)
        template = """Given the objective: '{objective}', prioritize the following tasks in order of importance:

Tasks:
{tasks}

Prioritized tasks:"""
        prompt_template = PromptTemplate(
            input_variables=["objective", "tasks"], template=template
        )
        return LLMChain(llm=llm, prompt=prompt_template, output_key="prioritized_tasks")

    def create_llm_chain_task_update(self):
        llm = OpenAI(temperature=0.7)
        template = """Given the objective: '{objective}', the current tasks, and new data, suggest any updates to the task list:

Current tasks:
{current_tasks}

New data:
{data}

Updated tasks:"""
        prompt_template = PromptTemplate(
            input_variables=["current_tasks", "data"], template=template
        )
        return LLMChain(llm=llm, prompt=prompt_template, output_key="updated_tasks")

    def initialize_tasks(self, objective):
        # Analyze the objective and generate a list of tasks
        initial_tasks = self.analyze_objective_with_llm(objective)

        # Prioritize the tasks based on their importance, dependencies, etc.
        prioritized_tasks = self.prioritize_tasks(initial_tasks)

        # Set the initial task list
        self.task_list = prioritized_tasks

    def analyze_objective_with_llm(self, objective):
        # Use LLMChain to analyze the objective and generate a list of tasks
        result = self.llm_chain_task_analysis({"objective": objective})
        return result["tasks"].split("\n")

    def prioritize_tasks(self, tasks):
        # Use LLMChain to prioritize tasks
        tasks_string = "\n".join(tasks)
        result = self.llm_chain_task_prioritization({"tasks": tasks_string})
        return result["prioritized_tasks"].split("\n")

    def get_task_list(self):
        return self.task_list

    def get_current_task(self):
        return self.task_list[0] if self.task_list else None

    def advance_to_next_task(self):
        self.task_list.pop(0)

    def update_tasks(self, data):
        # Use LLMChain to update tasks based on new data
        current_tasks_string = "\n".join(self.task_list)
        data_string = "\n".join([f"{key}: {value}" for key, value in data.items()])
        result = self.llm_chain_task_update(
            {"current_tasks": current_tasks_string, "data": data_string}
        )
        updated_tasks = result["updated_tasks"].split("\n")

        # Replace the current task list with the updated tasks
        self.task_list = updated_tasks

        self.task_list = self.prioritize_tasks(self.task_list)
