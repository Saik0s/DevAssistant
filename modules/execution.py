from __future__ import annotations
import json
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import platform
from langchain import OpenAI
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
from langchain.agents.chat.base import ChatAgent
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
            <string name="action_input" description="Detailed final answer to the original input question together with summary of used actions and results of used actions"/>
            </object>
        </case>
    </choice>
</output>


<instructions>
You are a helpful Task Driven Autonomous Agent running on {operating_system} only capable of communicating with valid JSON, and no other text.
You should always respond with one of the provided actions and corresponding to this action input. If you don't know what to do, you should decide by yourself.
You can take as many actions as you want, but you should always return a valid JSON that follows the schema and only one action at a time.

@complete_json_suffix_v2
</instructions>

<prompt>
Ultimate objective: {{{{objective}}}}
Previously completed tasks and project context: {{{{context}}}}
Working directory tree: {{{{dir_tree}}}}

Finish the following task.

Task: {{{{input}}}}

Choose one of the available actions and return a JSON that follows the correct schema.

{{{{agent_scratchpad}}}}
</prompt>

</rail>
"""
# Objective, Context and Directory tree are sent in separate messages, so they are not included in this prompt


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
            except Exception as e:
                print(traceback.format_exc())
                print(f"Exception running executor agent. Will retry {2-i} times")
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
            # Retry once by directly calling LLM and asking to extract action and action input as a json
            try:
                print(f"---\nCould not parse LLM output: {text}\nerror: {e}\nRetrying...\n---")
                llm = OpenAI(temperature=0)
                text = llm(f"{self.guard.instructions.source}\n\nExtract and return action and other fields in json format from this: {text}")
                result = json.loads(text)
                action = result["action"]
                input = text
            except Exception as e2:
                raise OutputParserException(f"---\nCould not parse LLM output: {text}\nerror: {e2}\n---") from e2
        if FINAL_ANSWER_ACTION in action:
            if "action_input" in result:
                action_input = result["action_input"]
            elif action in input and isinstance(input[action], dict) and "action_input" in input[action]:
                action_input = input[action]["action_input"]
            elif action in input:
                action_input = str(input[action])
            else:
                action_input = str(input)
            return AgentFinish({"output": action_input}, text)

        return AgentAction(action, input, text)

class ExecutionAgent(ChatAgent):
    output_parser: ExecutionOutputParser = Field(default_factory=ExecutionOutputParser)

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Result of Action JSON: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Action JSON:"

    def _construct_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""
        return super()._construct_scratchpad(intermediate_steps[-3:])

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
            HumanMessagePromptTemplate.from_template(output_parser.guard.base_prompt),
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
                        f'<string name="{arg_key}" description="{arg_value}"/>'
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
        return ["Result of Action JSON:"]

    @property
    def _agent_type(self) -> str:
        raise ValueError
