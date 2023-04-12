from .execution_tools import get_tools
from langchain import OpenAI
from langchain.agents.agent import Agent
from langchain.agents.agent import AgentExecutor
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, BaseMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools.base import BaseTool
from modules.memory import MemoryModule
from typing import Any, List, Optional, Sequence, Tuple
from typing import Any, List, Optional, Sequence, Tuple, Union
import re

class ExecutionModule:
    def __init__(self, llm, memory_module: MemoryModule, verbose: bool = True):
        self.memory_module = memory_module
        tools = get_tools(llm, memory_module)
        agent = ExecutionAgent.from_llm_and_tools(llm=llm, tools=tools, verbose=verbose)
        self.agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)

    def execute(self, task):
        task_name = task["task_name"]
        objective = self.memory_module.objective
        context = self.memory_module.retrieve_related_information(task_name)
        for i in range(3):
            try:
                return self.agent.run({"input": task_name, "objective": objective, "context": context})
            except ValueError:
                print(f"Value error running executor agent. Will retry {2-i} times")
        return "Failed to execute task."


PREFIX = """ExecutionAssistant is a versatile AI model developed by OpenAI, which excels in executing a wide range of tasks within the context of a larger workflow.
To ensure a focused and effective outcome, ExecutionAssistant concentrates solely on the current task, without attempting to perform further work. It is constantly learning, improving, and evolving to offer accurate and informative results.
ExecutionAssistant is not engaging in a conversation but rather producing the output of executing a task within a larger workflow trying to accomplish the following objective: {objective}. It should focus only on the current task, and doesn't attempt to perform further work.

Its primary goal is to complete the task at hand, leveraging its own knowledge and the following tools:

"""
FORMAT_INSTRUCTIONS = """

ExecutionAssistant follows this thought process and format:

1. Determine if a tool is needed.
   - If yes, choose an action from the available tools and provide the necessary input.
   - If no, proceed to step 2.

2. Observe the result of the action or provide a response to the user.

Please remember to provide the current context and task when using ExecutionAssistant.
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

Do everything that is needed to complete this task.
Task: {input}

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
            [f"  - '{tool.name}': {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names, ai_prefix=ai_prefix, human_prefix=human_prefix)
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        input_variables = ["input", "objective", "context", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        if f"{self.ai_prefix}:" in llm_output:
            return self.ai_prefix, llm_output.split(f"{self.ai_prefix}:")[-1].strip()
        regex = r"Action: (.*?)[\n]*Action Input: ([.\n]*)"
        match = re.search(regex, llm_output)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match[1]
        action_input = match[2]
        if action:
            slugs = llm_output.split("Action Input:")
            action_input =  slugs[-1]
            
        return action.strip(), action_input.strip(" ").strip('"')

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"

        if len(thoughts) > 8000:
            print(">>>>>Thoughts too long, summarizing<<<<<")
            chain = load_summarize_chain(OpenAI(temperature=0, max_tokens=2000), chain_type="map_reduce")
            text_splitter = CharacterTextSplitter()
            texts = text_splitter.split_text(thoughts)
            docs = [Document(page_content=t) for t in texts[:3]]
            thoughts = chain.run(docs)

        return thoughts

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLLM,
        tools: Sequence[BaseTool],
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



