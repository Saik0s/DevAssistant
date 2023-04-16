from utils.helpers import create_summarize_chain
from modules.execution_tools import get_tools
from langchain import OpenAI
from langchain.agents.agent import Agent
from langchain.agents.agent import AgentExecutor
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, BaseMessage, AgentFinish
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools.base import BaseTool
from modules.memory import MemoryModule
from typing import Any, Dict, List, Optional, Sequence, Tuple
from typing import Any, List, Optional, Sequence, Tuple, Union
import re


class ExecutionModule:
    def __init__(self, llm, memory_module: MemoryModule, verbose: bool = True):
        self.memory_module = memory_module
        tools = get_tools(llm, memory_module)
        agent = ExecutionAgent.from_llm_and_tools(llm=llm, tools=tools, verbose=verbose)
        agent.max_tokens = 4000
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


PREFIX = """You are ExecutionAssistant, an AI model by OpenAI, performing tasks within larger workflows.
Continuously learning and improving, you focuse on the current task to achieve the objective: {objective}, without attempting further work.

Your primary goal is to complete tasks using its knowledge and these tools:
"""
FORMAT_INSTRUCTIONS = """

You follow this thought process:

1. Determine if a tool is needed.
   - If yes, choose an action from the available tools and provide the necessary input.
   - If no, proceed to step 2.

2. Observe the result of the action or provide a response to the user.

You alway use the following thought format to execute your tasks:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the User, or if you don't need to use a tool, you always use this format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""
SUFFIX = """
--------------------

Take into account these previously completed tasks and project context:
{context}

Your task: {input}

Now take as many steps as you need to make sure you have completed the task.
You will continue to execute the task until it is complete.
Be self critical about the way you move towards achieving objective.
Use available tools extensively. Heavily use file system for project state management.
Always make sure that task is fully completed before moving to the next one.

{agent_scratchpad}"""


class ExecutionAgent(Agent):
    """An agent designed to execute a single task within a larger workflow."""

    ai_prefix: str = "Assistant"
    max_tokens: int = 4000

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
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        tool_strings = "\n".join([f"  - '{tool.name}': {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names, ai_prefix=ai_prefix, human_prefix=human_prefix)
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        input_variables = ["input", "objective", "context", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        if f"{self.ai_prefix}:" in llm_output:
            return self.ai_prefix, llm_output.split(f"{self.ai_prefix}:")[-1].strip()
        regex = r"Action: (.*?)[\n]*Action (i|I)nput: ((.|\n)*)"
        match = re.search(regex, llm_output)

        if not match:
            print(f"Could not parse LLM output: `{llm_output}`")
            return "Could not parse action input", llm_output

        action = match[1]
        action_input = match[2]

        print("\n\033[1;34mAction:\033[0m", action, "\n\033[1;34mAction Input:\033[0m", action_input, "\n")

        return action.strip(), action_input.strip(" ").strip('"')

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
            callback_manager=callback_manager,  # type: ignore
        )
        tool_names = [tool.name for tool in tools]
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, ai_prefix=ai_prefix, **kwargs)
