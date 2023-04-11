from langchain import LLMChain, PromptTemplate
from langchain.llms.base import BaseLLM
from modules.memory import MemoryModule
from typing import List

class LearningModule:
    def __init__(self, llm: BaseLLM, memory_module: MemoryModule, verbose: bool = True):
        self.memory_module = memory_module
        self.learning_chain = LearningChain.from_llm(llm, verbose)

    def learn_from(
        self,
        observation: str,
        completed_tasks: List[dict],
        pending_tasks: List[dict]
    ) -> str:
        memory = self.memory_module.retrieve_related_information(observation)
        objective = self.memory_module.objective
        return self.learning_chain.run(
            completed_tasks=completed_tasks,
            pending_tasks=pending_tasks,
            last_output=observation,
            context=memory,
            objective=objective,
        )


learning_template = """LearningAssistant is an AI specialized in information consolidation, part of a larger system that is solving a complex problem in multiple steps. LearningAssistant is provided the current information context, and the result of the latest step, and updates the context incorporating the result.
LearningAssistant is also provided the list of completed and still pending tasks.
The rest of the system is provided the task lists and context in the same way, so the context should never contain the tasks themselves
The information context is the only persistent memory the system has, after every step, the context must be updated with all relevant informtion, such that the context contains all information needed to complete the objective.

The ultimate objective is: {objective}.
Completed tasks: {completed_tasks}
The last task output was:
{last_output}

The list of pending tasks: {pending_tasks}

Current context to update:
{context}

LearningAssistant will generate an updated context. This context will replace the current context. Context should be up to 500 characters.
This context should reflect changes to the large system that needs to be made to achieve objective more efficiently, and should not contain any information that is not relevant to the objective.
LearningAssistant: """

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
