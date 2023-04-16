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


learning_template = """You are LearningAssistant - AI specialized in information consolidation, part of a larger system that is solving a complex problem in multiple steps.

The ultimate objective is:
{objective}

Completed tasks:
{completed_tasks}

The last task output was:
{last_output}

The list of pending tasks:
{pending_tasks}

Current context:
{context}

Perform a detailed analysis of the current state of the project, taking into account all available data, and write concise and concrete instructions for behavior adjustments of the whole system that will help the system move more efficiently towards the objective.
"""

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
