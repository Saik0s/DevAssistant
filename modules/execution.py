from langchain import BasePromptTemplate, OpenAI, PromptTemplate
from langchain.agents.conversational_chat.base import (
    ConversationalChatAgent,
    BaseOutputParser,
)
from typing import Any, Dict, Optional, Sequence
from langchain.agents.agent import Agent, AgentExecutor
from langchain.agents.chat.base import ChatAgent
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
import json
from typing import Any, List, Optional, Sequence, Tuple

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
    AgentAction,
    AIMessage,
    BaseLanguageModel,
    BaseMessage,
    BaseOutputParser,
    HumanMessage,
)

from typing import Any, Optional, Sequence
from .execution_tools import get_tools


class ExecutionModule:
    def __init__(self, llm, memory_module):
        self.llm = llm
        self.memory_module = memory_module
        self.tools = self._initialize_tools(llm, memory_module)

    def execute(self, task_info):
        agent = ChatAgent.from_llm_and_tools(
            llm=self.llm, tools=self.tools, verbose=True
        )
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=self.tools, verbose=True
        )
        prompt = "Performs one task based on the following objective: {objective}\n\nTake into account these previously completed tasks context: {context}\n\n{task}".format(
            objective=self.memory_module.get_objective(),
            context=self.memory_module.get_context(),
            task=task_info,
        )
        return agent_executor.run(prompt)

    def _initialize_tools(self, llm, memory_module):
        return get_tools(llm, memory_module)
