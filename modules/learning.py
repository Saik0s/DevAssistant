from typing import List
from langchain import LLMChain, PromptTemplate
from langchain.llms.base import BaseLLM

class LearningModule:
    def __init__(self, llm, objective, verbose: bool = True):
        self.objective = objective
        self.learning_chain = LearningChain.from_llm(llm, objective, verbose)

    def learn_from(self, memory: str, observation: str, completed_tasks: List[str], pending_tasks: List[str]) -> str:
        return self.learning_chain.run(
                completed_tasks=completed_tasks,
                pending_tasks=pending_tasks,
                last_output=observation,
                context=memory
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

LearningAssistant will generate an updated context. This context will replace the current context.
LearningAssistant: """

learning_prompt = lambda objective: PromptTemplate(
        template=learning_template,
        partial_variables={"objective": objective},
        input_variables=["completed_tasks", "pending_tasks", "last_output", "context"],
        )

class LearningChain(LLMChain):

    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True):
        return cls(prompt=learning_prompt(objective), llm=llm, verbose=verbose)


