from langchain import BasePromptTemplate
from langchain.agents import (
    AgentExecutor,
    ConversationalChatAgent,
    AgentOutputParser,
)
from typing import Any, Optional, Sequence
from langchain.agents.agent import Agent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import (
    BaseLanguageModel,
)
from langchain.tools.base import BaseTool


from typing import Any, Optional, Sequence
from .execution_tools import get_tools


class ExecutionModule:
    def __init__(self, llm, memory_module):
        self.llm = llm
        self.memory_module = memory_module
        self.tools = self._initialize_tools(llm, memory_module)

    def execute(self, task_info):
        agent = TaskExecutionAgent.create_from_llm_and_tools(
            llm=self.chat_llm, tools=self.tools, verbose=True
        )
        agent_executor = AgentExecutor.create_from_agent_and_tools(
            agent=agent, tools=self.tools, verbose=True
        )
        return agent_executor.run(
            {
                "objective": self.memory_module.get_objective(),
                "context": self.memory_module.get_context(),
                "task": task_info,
            }
        )

    def _initialize_tools(self, llm, memory_module):
        return get_tools(llm, memory_module)


class TaskExecutionAgent(ConversationalChatAgent):
    @property
    def _agent_type(self) -> str:
        return "TaskExecutionAgent"

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
    ) -> BasePromptTemplate:
        system_message = """
            You are an AI developed by OpenAI. Your purpose is to complete tasks by performing actions and going through the necessary steps."
        """

        tool_strings = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        execution_template = f"""
            Performs one task based on the following objective: {{objective}}.
            Take into account these previously completed tasks context: {{context}}.
            Available tools that you can use:
            {tool_strings}
            Use one of the following response formats:

            **Option 1:**
            If you want to use a tool, use this format:

            ```json
            {{{{
                "action": string, // The action to take. Must be one of {tool_names}
                "action_input": string // The input to the action
            }}}}
            ```

            **Option 2:**
            If you want to respond with a final answer, use this format:

            ```json
            {{{{
                "action": "Final Answer",
                "action_input": string // You should put what you want to return to use here
            }}}}
            ```

            Your task: {{task}}
        """

        input_variables = (["objective", "context", "task", "agent_scratchpad"],)
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(execution_template),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @classmethod
    def create_from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> Agent:
        cls._validate_tools(tools)
        _output_parser = AgentOutputParser()
        prompt = cls.create_prompt(tools)
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )
