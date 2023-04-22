from langchain import LLMChain, PromptTemplate
from langchain.llms.base import BaseLLM
from modules.memory import MemoryModule
from typing import List


class LearningModule:
    def __init__(self, llm: BaseLLM, memory_module: MemoryModule, verbose: bool = True):
        self.memory_module = memory_module
        self.learning_chain = LearningChain.from_llm(llm, verbose)

    def learn_from(self, observation: str, completed_tasks: List[dict], pending_tasks: List[dict]) -> str:
        memory = self.memory_module.retrieve_related_information(observation)
        objective = self.memory_module.objective
        return self.learning_chain.run(
            completed_tasks=completed_tasks,
            pending_tasks=pending_tasks,
            last_output=observation,
            context=memory,
            objective=objective,
        )


learning_template = (
    "As LearningAssistant, analyze the project to efficiently achieve the objective: {objective}.\n\n"
    "Completed tasks: {completed_tasks}\n"
    "Last task output: {last_output}\n"
    "Pending tasks: {pending_tasks}\n"
    "Context: {context}\n\n"
    "Provide concise instructions for behavior adjustments to improve system efficiency."
)

learning_prompt = PromptTemplate(
    template=learning_template,
    input_variables=[
        "completed_tasks",
        "pending_tasks",
        "last_output",
        "context",
        "objective",
    ],
)


class LearningChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True):
        return cls(prompt=learning_prompt, llm=llm, verbose=verbose)
