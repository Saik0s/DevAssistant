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
    ):
        memory = self.memory_module.retrieve_related_information(observation)
        objective = self.memory_module.objective
        response = self.evaluate_chain.run(
            last_output=observation,
            context=memory,
            objective=objective,
        )
        is_finished =  "YES" in response.strip().upper()
        if is_finished:
            final_answer = response.replace("YES", "").replace('-', '').strip()
            return (is_finished, final_answer)

        return (is_finished, "")


evaluation_template = (
    "You are an Evaluation Assistant that checks if the ultimate objective was reached.\n\n"
    "Last task output: {last_output}\n"
    "Context: {context}\n\n"
    "Evaluate provided information and determine if agent achieved this objective: {objective}\n"
    "Make sure that objective is achieved and expected result is delivered and not only planned.\n"
    "If the objective is achieved, provide the final answer; otherwise, answer NO.\n"
    "Expected answer: YES - the final answer for the ultimate task, or NO\n"
    "EvaluationModuleAssistant:"
)

evaluation_prompt = PromptTemplate(
    template=evaluation_template,
    input_variables=[
        "last_output",
        "context",
        "objective",
    ],
)

class EvaluateChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True):
        return cls(prompt=evaluation_prompt, llm=llm, verbose=verbose)
