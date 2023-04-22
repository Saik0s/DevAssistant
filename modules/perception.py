from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from modules.execution_tools import get_tools
from modules.memory import MemoryModule


class PerceptionModule:
    def __init__(self, llm: BaseLLM, memory_module: MemoryModule, verbose: bool = True):
        self.task_enhancement_chain = TaskEnhancementChain.from_llm(llm, verbose)
        self.memory_module = memory_module

    def process_task(self, task):  # sourcery skip: avoid-builtin-shadow
        id = task["task_id"]
        name = task["task_name"]
        summary = self.memory_module.retrieve_related_information(name)
        objective = self.memory_module.objective
        name = self.task_enhancement_chain.run(objective=objective, context=summary, task=name)
        if "Task: " in name:
            name = name.split("Task: ", 1)[1].strip()
        return {"task_id": id, "task_name": name}

    def process_result(self, text):
        # TODO: Add processing for text
        return text


class TaskEnhancementChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        template = (
            "You are an task improver Assistant for an autonomous agent.\n"
            "Autonomous agent has limited access to authorized tools and resources such as internet, shell, filesystem.\n"
            "Always make sure that tasks are actionable and achievable by task driven autonomous agent with limited access to resources. \n"
            "Consider the ultimate objective of your team: {objective}\n"
            "Task related context: \n{context}\n"
            "Task: {task}.\n"
            "Now write only one sentence that includes this task and description of how end result should look like.\n"
        )
        prompt = PromptTemplate(
            template=template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
