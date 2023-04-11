from langchain.agents.agent import AgentExecutor
from langchain.agents.chat.base import ChatAgent
import re
from typing import Any, List, Optional, Sequence, Tuple
from langchain.agents.agent import Agent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from typing import List, Optional, Sequence
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, Tool
from langchain.llms.base import BaseLLM

from .execution_tools import get_tools

class ExecutionModule:
    def __init__(self, llm, objective, memory_module, verbose: bool = True):
        self.llm = llm
        self.objective = objective
        self.memory_module = memory_module
        self.tools = self._initialize_tools()
        self.agent = self._create_agent(verbose)

    def execute(self, task_name, context):
        for i in range(3):
            try:
                return self.agent.run({"input": task_name, "context": context})
            except ValueError:
                print(f"Value error running executor agent. Will retry {2-i} times")
        return "Failed to execute task."

    def _initialize_tools(self):
        return get_tools(self.llm, self.memory_module)

    def _create_agent(self, verbose: bool):
        agent = ExecutionAgent.from_llm_and_tools(llm=self.llm, tools=self.tools, objective=self.objective, verbose=verbose)
        return AgentExecutor.from_agent_and_tools(agent=agent, tools=self.tools, verbose=verbose)


PREFIX = """ExecutionAssistant is a general purpose AI model trained by OpenAI.

ExecutionAssistant is tasked with executing a single task within the context of a larger workflow trying to accomplish the following objective: {objective}. It should focus only on the current task, and doesn't attempt to perform further work.

ExecutionAssistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions.

Overall, ExecutionAssistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics.
ExecutionAssistant is not having a conversation with a user, but rather producing the output of executing a task within a larger workflow.

TOOLS:
------

ExecutionAssistant has access to the following tools:"""
FORMAT_INSTRUCTIONS = """
Thought Process:
----------------

ExecutionAssistant always uses the following thought process and format to execute its tasks:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When ExecutionAssistant has a response to say to the Human, or if it doesn't need to use a tool, it always uses the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```"""

SUFFIX = """Begin!

Current context:
{context}

Current task: {input}
{agent_scratchpad}"""

class ExecutionAgent(Agent):
    """An agent designed to execute a single task within a larger workflow."""

    ai_prefix: str = "Assistant"

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

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = "AI",
        human_prefix: str = "Human",
        objective: Optional[str] = None,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        prefix = prefix.format(objective=objective)
        format_instructions = format_instructions.format(tool_names=tool_names, ai_prefix=ai_prefix, human_prefix=human_prefix)
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        input_variables = ["input", "context", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        if f"{self.ai_prefix}:" in llm_output:
            return self.ai_prefix, llm_output.split(f"{self.ai_prefix}:")[-1].strip()
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, llm_output)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1)
        action_input = match.group(2)
        return action.strip(), action_input.strip(" ").strip('"')

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLLM,
        tools: Sequence[BaseTool],
        objective: Optional[str] = None,
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = "Assistant",
        human_prefix: str = "Human",
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            ai_prefix=ai_prefix,
            human_prefix=human_prefix,
            prefix=prefix,
            suffix=suffix,
            objective=objective,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager, # type: ignore
        )
        tool_names = [tool.name for tool in tools]
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, ai_prefix=ai_prefix, **kwargs)



