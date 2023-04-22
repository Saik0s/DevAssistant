import json
import platform
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from rich import print

from langchain.agents import Agent, AgentExecutor
from langchain.callbacks import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.output_parsers import GuardrailsOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AgentAction, AgentFinish, BaseLanguageModel, OutputParserException
from langchain.tools.base import BaseTool

from modules.execution_tools import get_tools, tree_tool
from modules.memory import MemoryModule

import platform
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from langchain.agents.agent import AgentOutputParser
from langchain.schema import BaseLanguageModel
from langchain.agents.agent import Agent, AgentExecutor
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.schema import AgentAction
from langchain.tools.base import BaseTool
from langchain.output_parsers import GuardrailsOutputParser
from langchain.prompts.chat import *
import json
from typing import Union
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from typing import Any, List, Optional, Sequence, Tuple

from pydantic import Field

from langchain.agents.agent import Agent, AgentOutputParser
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AgentAction, BaseLanguageModel
from langchain.tools import BaseTool


# Define the Guardrails Schema for the Execution Assistant
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
You are a Task Driven Autonomous Agent running on {operating_system} only capable of communicating with valid JSON, and no other text.
You only give final answer when task is completed. You should always evaluate and see if additional actions are required.
</instructions>


<prompt>
You focus on the given task to achieve this objective: {{{{objective}}}}.

Take into account these previously completed tasks and project context:
{{{{context}}}}

Current working directory tree:
{{{{dir_tree}}}}

Your task: {{{{input}}}}

{{output_schema}}
@json_suffix_prompt_examples
{{{{agent_scratchpad}}}}
</prompt>


</rail>
"""

# ExecutionModule class for executing tasks within a larger workflow
class ExecutionModule:
    def __init__(self, llm: BaseLLM, memory_module: MemoryModule, verbose: bool = True):
        self.memory_module = memory_module
        tools = get_tools(llm, memory_module)
        agent = ExecutionAgent.from_llm_and_tools(llm=llm, tools=tools, verbose=verbose)
        self.agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)

    # Execute a given task and return the result
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

# ExecutionAgent class for executing a single task within a larger workflow
FINAL_ANSWER_ACTION = "final"

class ExecutionOutputParser(GuardrailsOutputParser, AgentOutputParser):
    def get_format_instructions(self) -> str:
        return self.guard.instructions.source

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # sourcery skip: avoid-builtin-shadow
        try:
            result = json.loads(text)
            action = result["action"]
            input = result[action]
        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text}") from e
        if FINAL_ANSWER_ACTION in action:
            return AgentFinish(
                {"output": input}, text
            )
        return AgentAction(action, input, text)

class ExecutionAgent(Agent):
    output_parser: ExecutionOutputParser = Field(default_factory=ExecutionOutputParser)
    """An agent designed to execute a single task within a larger workflow."""

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    # Construct the agent scratchpad based on intermediate steps
    def _construct_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
        agent_scratchpad = super()._construct_scratchpad(intermediate_steps)
        if not isinstance(agent_scratchpad, str):
            raise ValueError("agent_scratchpad should be of type string.")
        if agent_scratchpad:
            return (
                f"This was your previous work " f"(but I haven't seen any of it! I only see what " f"you return as final answer):\n{agent_scratchpad}"
            )
        else:
            return agent_scratchpad

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        ExecutionOutputParser()

    # Create the prompt for the ExecutionAgent
    @classmethod
    def create_prompt(cls, output_parser: GuardrailsOutputParser) -> BasePromptTemplate:
        messages = [
            SystemMessagePromptTemplate.from_template(output_parser.guard.instructions.source),
            HumanMessagePromptTemplate.from_template(output_parser.guard.base_prompt),
        ]
        return ChatPromptTemplate.from_messages(messages=messages)

    # Initialize the ExecutionAgent with LLM and tools
    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[ExecutionOutputParser] = None,
        **kwargs: Any,
    ) -> Agent:
        cls._validate_tools(tools)
        tool_strings_spec = "\n".join([f'<case name="{tool.name}" description="{tool.description}"><string name="input"/></case>' for tool in tools])
        print(tool_strings_spec)
        operating_system = platform.platform()
        complete_rail_spec = rail_spec.format(tool_strings_spec=tool_strings_spec, operating_system=operating_system)
        output_parser = ExecutionOutputParser.from_rail_string(complete_rail_spec)
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
