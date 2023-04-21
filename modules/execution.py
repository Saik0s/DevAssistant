from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import re
from rich import print

from langchain import OpenAI
from langchain.agents.agent import Agent
from langchain.agents.agent import AgentExecutor
from langchain.agents import AgentType
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, BaseMessage, AgentFinish
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools.base import BaseTool
from langchain.output_parsers import GuardrailsOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
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
    </choice>
</output>


<instructions>
You are Execution Assistant performing tasks within larger workflows and only capable of communicating with valid JSON, and no other text.
You focus on the given task to achieve this objective: {{objective}}.

Take into account these previously completed tasks and project context:
{{context}}

Current working directory tree:
{{dir_tree}}

@json_suffix_prompt_examples
</instructions>


<prompt>
Your task: {{input}}

@complete_json_suffix_v2
@xml_prefix_prompt

{output_schema}

{{agent_scratchpad}}
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
        mrkl = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
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
                return self.agent.run({"input": task_name, "objective": objective, "context": context, "dir_tree": dir_tree})
            except ValueError:
                print(f"Value error running executor agent. Will retry {2-i} times")
        return "Failed to execute task."


#################################################################################################
### ExecutionAgent
#################################################################################################

class ExecutionAgent(Agent):
    """An agent designed to execute a single task within a larger workflow."""

    ai_prefix: str = "Assistant"
    max_tokens: int = 4000
    output_parser: GuardrailsOutputParser

    @property
    def _agent_type(self) -> str:
        return "autonomous"

    @property
    def observation_prefix(self) -> str:
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        return "Thought:"

    @property
    def finish_tool_name(self) -> str:
        return self.ai_prefix

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        self.output_parser.parse(llm_output)

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
    def from_llm_and_tools(
        cls,
        llm: BaseLLM,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> "ExecutionAgent":
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        tool_strings_spec = "\n".join([f'<string name="{tool.name}" description="{tool.description}" on-fail-valid-choices="reask" />' for tool in tools])
        complete_rail_spec = rail_spec.format(tool_strings_spec=tool_strings_spec)
        output_parser = GuardrailsOutputParser.from_rail_string(complete_rail_spec)
        prompt = PromptTemplate(
            template=output_parser.guard.base_prompt,
            input_variables=output_parser.guard.prompt.variable_names,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, output_parser=output_parser, **kwargs)


################

FINAL_ANSWER_ACTION = "Final Answer:"


class ChatAgent(Agent):
    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
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

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        if FINAL_ANSWER_ACTION in text:
            return "Final Answer", text.split(FINAL_ANSWER_ACTION)[-1].strip()
        try:
            _, action, _ = text.split("```")
            response = json.loads(action.strip())
            return response["action"], response["action_input"]

        except Exception:
            raise ValueError(f"Could not parse LLM output: {text}")

    @property
    def _stop(self) -> List[str]:
        return ["Observation:"]

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        input_variables: Optional[List[str]] = None,
    ) -> BasePromptTemplate:
        tool_strings_spec = "\n".join([f'<string name="{tool.name}" description="{tool.description}" on-fail-valid-choices="reask" />' for tool in tools])
        complete_rail_spec = rail_spec.format(tool_strings_spec=tool_strings_spec)
        output_parser = GuardrailsOutputParser.from_rail_string(complete_rail_spec)
        prompt = PromptTemplate(
            template=output_parser.guard.base_prompt,
            input_variables=output_parser.guard.prompt.variable_names,
        )
        messages = [
            SystemMessagePromptTemplate.from_template(template),
            HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}"),
        ]
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)

    @property
    def _agent_type(self) -> str:
        raise ValueError
