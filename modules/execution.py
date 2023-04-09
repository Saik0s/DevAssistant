from langchain.agents import (
    AgentExecutor,
    ConversationalChatAgent,
)

class ExecutionModule:
    def __init__(self, chat_llm, llm, memory_module):
        self.chat_llm = chat_llm
        self.llm = llm
        self.memory_module = memory_module
        self.tools = self._create_tools(chat_llm, memory_module)

    def execute(self, task_info):
        agent = ConversationalChatAgent.from_llm_and_tools(
            llm=self.chat_llm,
            tools=self.tools,
            verbose=True
        )
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=self.tools, verbose=True
        )
        return agent_executor.run({"input": task_info, "chat_history": []})

    def _create_tools(self, llm, memory_module):
        from .execution_tools import get_tools
        return get_tools(llm, memory_module)
