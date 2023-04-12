from langchain import LLMChain, PromptTemplate
from langchain.llms.base import BaseLLM
from modules.memory import MemoryModule
from typing import List

class EvaluationModule:
    def __init__(self, llm: BaseLLM, memory_module: MemoryModule, verbose: bool = True):
        self.memory_module = memory_module
        self.evaluate_chain = EvaluateChain.from_llm(llm, verbose)

    def evaluate_from(
        self,
        observation: str,
        completed_tasks: List[dict],
        pending_tasks: List[dict]
    ):
        memory = self.memory_module.retrieve_related_information(observation)
        objective = self.memory_module.objective
        response = self.evaluate_chain.run(
            completed_tasks=completed_tasks,
            pending_tasks=pending_tasks,
            last_output=observation,
            context=memory,
            objective=objective,
        )
        is_finished =  "YES" in response.strip().upper() 
        if is_finished:
            final_answer = response.replace("YES", "").replace('-', '').strip()
            return (is_finished, final_answer)

        return (is_finished, "")


learning_template = """EvaluationModuleAssistant is an AI specialized in evaluate task, part of a larger system that is solving a complex problem in multiple steps. EvaluationModuleAssistant is answer question about the overal result of whole system, from result of the last step and the overal context, and provide feedback for the system.
EvaluationModuleAssistant is also decide is the system archive its ultimate objective.
The rest of the system is provided the task lists and context contains all information needed to complete the objective.

The ultimate objective is: {objective}.
Completed tasks: {completed_tasks}
The last task output was:
{last_output}

The list of pending tasks: {pending_tasks}

Current context to update:
{context}

EvaluationModuleAssistant will anser question is that the system archive its ultimate objective, and provide final answer for the ultimate task.
Expected answer: YES - the final answer for the ultimate task, or NO
EvaluationModuleAssistant: """

evaluation_prompt = PromptTemplate(
    template=learning_template,
    input_variables=[
        "completed_tasks",
        "pending_tasks",
        "last_output",
        "context",
        "objective",
    ],
)


class EvaluateChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True):
        return cls(prompt=evaluation_prompt, llm=llm, verbose=verbose)
