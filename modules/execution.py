from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from rich import print

from langchain.agents.agent import Agent
from langchain.agents.agent import AgentExecutor
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.schema import AgentAction
from langchain.tools.base import BaseTool
from langchain.output_parsers import GuardrailsOutputParser
from langchain.prompts.chat import *

from modules.execution_tools import get_tools, tree_tool
from modules.memory import MemoryModule
from utils.helpers import create_summarize_chain


#################################################################################################
### Guardrails Schema
#################################################################################################

rail_spec = """
<rail version="0.1">

<output>
    <choice name="action" on-fail-choice="reask">
{tool_strings_spec}
        <case name="final" description="Use it when you have a final answer">
            <string name="input" description="the final answer to the original input question" />
        </case>
    </choice>
</output>


<instructions>
You are Execution Assistant performing tasks within larger workflows and only capable of communicating with valid JSON, and no other text.

@json_suffix_prompt_examples
</instructions>


<prompt>
You focus on the given task to achieve this objective: {{{{objective}}}}.

Take into account these previously completed tasks and project context:
{{{{context}}}}

Current working directory tree:
{{{{dir_tree}}}}

Your task: {{{{input}}}}

@complete_json_suffix_v2
@xml_prefix_prompt

{{output_schema}}

{{{{agent_scratchpad}}}}
</prompt>


</rail>
"""

#################################################################################################
### ExecutionModule
#################################################################################################


class ExecutionModule:
    def __init__(self, llm: BaseLLM, memory_module: MemoryModule, verbose: bool = True):
        self.memory_module = memory_module
        tools = get_tools(llm, memory_module)
        agent = ExecutionAgent.from_llm_and_tools(llm=llm, tools=tools, verbose=verbose)
        agent.max_tokens = 4000
        self.agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)

    def execute(self, task: Dict[str, Any]) -> Union[str, Document]:
        task_name = task["task_name"]
        objective = self.memory_module.objective
        context = self.memory_module.retrieve_related_information(task_name)
        dir_tree = tree_tool().func("")
        for i in range(3):
            try:
                return self.agent.run(
                    {
                        "input": task_name,
                        "objective": objective,
                        "context": context,
                        "dir_tree": dir_tree,
                    }
                )
            except ValueError:
                print(f"Value error running executor agent. Will retry {2-i} times")
        return "Failed to execute task."


#################################################################################################
### ExecutionAgent
#################################################################################################


class ExecutionAgent(Agent):
    """An agent designed to execute a single task within a larger workflow."""

    max_tokens: int = 4000
    output_parser: GuardrailsOutputParser

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    def _construct_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
        agent_scratchpad = super()._construct_scratchpad(intermediate_steps)
        if not isinstance(agent_scratchpad, str):
            raise ValueError("agent_scratchpad should be of type string.")
        if agent_scratchpad:
            return (
                f"This was your previous work "
                f"(but I haven't seen any of it! I only see what "
                f"you return as final answer):\n{agent_scratchpad}"
            )
        else:
            return agent_scratchpad

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        print("=============================")
        print(llm_output)
        response = self.output_parser.parse(llm_output)
        print(response)
        print("=============================")
        return response["action"], response["input"]

    def get_full_inputs(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> Dict[str, Any]:
        inputs = super().get_full_inputs(intermediate_steps, **kwargs)
        prompts, stop = self.llm_chain.prep_prompts([inputs])
        prompts = [prompt.to_string() for prompt in prompts]
        full_prompt = "\n".join(prompts)

        if self.llm_chain.llm.get_num_tokens(full_prompt) > self.max_tokens - self.llm_chain.llm.max_tokens:
            summarize_chain = create_summarize_chain(verbose=self.llm_chain.verbose)
            summarize_tuple = lambda tup: (tup[0], summarize_chain(tup[1]))
            intermediate_steps = list(map(summarize_tuple, intermediate_steps))
            inputs = super().get_full_inputs(intermediate_steps, **kwargs)

        return inputs

    @classmethod
    def create_prompt(cls, output_parser: GuardrailsOutputParser) -> BasePromptTemplate:
        messages = [
            SystemMessagePromptTemplate.from_template(output_parser.guard.instructions.source),
            HumanMessagePromptTemplate.from_template(output_parser.guard.base_prompt),
        ]
        prompt = ChatPromptTemplate.from_messages(messages=messages)
        print(prompt.messages)
        return prompt

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLLM,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> Agent:
        cls._validate_tools(tools)
        tool_strings_spec = "\n".join(
            [
                f'<case name="{tool.name}" description="{tool.description}"><string name="input"/></case>'
                for tool in tools
            ]
        )
        complete_rail_spec = rail_spec.format(tool_strings_spec=tool_strings_spec)
        output_parser = GuardrailsOutputParser.from_rail_string(complete_rail_spec)
        prompt = cls.create_prompt(output_parser)
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=output_parser,
            **kwargs,
        )

    @property
    def _stop(self) -> List[str]:
        return ["Observation:"]

    @property
    def _agent_type(self) -> str:
        raise ValueError
