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
            "As a Task Improver Assistant, ensure tasks are actionable and achievable for an autonomous agent with limited resources.\n"
            "Ultimate objective: {objective}\n"
            "Context: {context}\n\n"
            "Task: {task}\n\n"
            "Now write a single sentence describing the task and the expected end result. Phrase it to be call to action instead of description.\n"
        )
        prompt = PromptTemplate(
            template=template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

