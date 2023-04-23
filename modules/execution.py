from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import platform
from pydantic import Field
from langchain.agents import Agent, AgentExecutor
from langchain.agents.agent import AgentOutputParser
from langchain.callbacks import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.output_parsers import GuardrailsOutputParser
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import AgentAction, AgentFinish, BaseLanguageModel
from langchain.tools import BaseTool
from langchain.prompts.chat import MessagesPlaceholder
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain.schema import OutputParserException
from modules.execution_tools import GuardRailTool, get_tools, tree
from modules.memory import MemoryModule
from rich import print

# Define the Guardrails Schema for the Execution Assistant
rail_spec = """
<rail version="0.1">

<output>
    <choice name="action" description="Action that you want to take, mandatory field" on-fail-choice="reask" required="true">
{tool_strings_spec}
        <case name="final">
            <object name="final" >
            <string name="action_input" description="the final answer to the original input question" required="true" />
            </object>
        </case>
    </choice>
</output>


<instructions>
You are a Task Driven Autonomous Agent running on {operating_system} only capable of communicating with valid JSON, and no other text.
You only give final answer when task is completed. You should always evaluate and see if additional actions are required.

@complete_json_suffix_v2
</instructions>

<prompt>
Finish the following task.

Task: {{{{input}}}}

Always answer with a valid JSON of a single action and nothing else.

@json_suffix_prompt_v2_wo_none
</prompt>


</rail>
"""


class ExecutionModule:
    def __init__(
        self,
        llm: BaseLLM,
        memory_module: MemoryModule,
        verbose: bool = True,
    ):
        self.memory_module = memory_module
        tools = get_tools(llm, memory_module)
        agent = ExecutionAgent.from_llm_and_tools(llm=llm, tools=tools, verbose=verbose)
        self.agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)

    def execute(self, task: Dict[str, Any]) -> Union[str, Document]:
        task_name = task["task_name"]
        objective = self.memory_module.objective
        context = self.memory_module.retrieve_related_information(task_name)
        dir_tree = tree() or "No directory tree available"
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


FINAL_ANSWER_ACTION = "final"


class ExecutionOutputParser(GuardrailsOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # sourcery skip: avoid-builtin-shadow
        try:
            result = json.loads(text)
            action = result["action"]
            input = text
        except Exception as e:
            print(e)
            raise OutputParserException(f"Could not parse LLM output: {text}\nerror: {e}")
        if FINAL_ANSWER_ACTION in action:
            return AgentFinish({"output": input}, text)
        return AgentAction(action, input, text)


class ExecutionAgent(Agent):
    output_parser: ExecutionOutputParser = Field(default_factory=ExecutionOutputParser)

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    def _construct_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        if not intermediate_steps:
            return []
        last_three_steps = intermediate_steps[-3:]
        thoughts: List[BaseMessage] = []
        for action, observation in last_three_steps:
            thoughts.extend(
                (
                    AIMessage(content=action.log),
                    HumanMessage(content=f"Tool Response: {observation}"),
                )
            )

        thoughts.append(
            HumanMessage(
                content=("Evaluate the above information and determine if task is complete. Take appropriate actions.",)
            )
        )
        return thoughts

    def get_full_inputs(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        return kwargs | new_inputs

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        ExecutionOutputParser()

    @classmethod
    def create_prompt(cls, output_parser: GuardrailsOutputParser) -> BasePromptTemplate:
        messages = [
            SystemMessagePromptTemplate.from_template(output_parser.guard.instructions.source),
            HumanMessagePromptTemplate.from_template("Ultimate objective: {objective}"),
            HumanMessagePromptTemplate.from_template("Previously completed tasks and project context: {context}"),
            HumanMessagePromptTemplate.from_template("Working directory tree: {dir_tree}"),
            HumanMessagePromptTemplate.from_template(output_parser.guard.base_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad"),

        ]
        return ChatPromptTemplate.from_messages(messages=messages)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[GuardRailTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[ExecutionOutputParser] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Agent:
        cls._validate_tools(tools)
        tool_strings_spec = "\n".join(
            [
                f'<case name="{tool.name}" description="{tool.description}"><object name="{tool.name}">'
                + "".join(
                    [
                        f'<string name="{arg_key}" description="{arg_value}" required="true"/>'
                        for arg_key, arg_value in tool.input_args.items()
                    ]
                )
                + "</object></case>"
                for tool in tools
            ]
        )
        operating_system = platform.platform()
        complete_rail_spec = rail_spec.format(tool_strings_spec=tool_strings_spec, operating_system=operating_system)
        output_parser = ExecutionOutputParser.from_rail_string(complete_rail_spec)
        prompt = cls.create_prompt(output_parser)
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
            verbose=verbose,
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
