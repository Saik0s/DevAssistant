from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
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
        return {"task_id": id, "task_name": name}

    def process_result(self, text):
        # TODO: Add processing for text
        return text

class TaskEnhancementChain(LLMChain):

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        template = (
            "You are an task explainer AI tasked with preparing a task for an autonomous agent."
            "Consider the ultimate objective of your team: {objective}."
            "Task related context: {context}."
            "Task to improve: {task}."
            "Please rewrite task to be self contained and include all relevant information in as concise as possible way."
        )
        prompt = PromptTemplate(
            template=template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
