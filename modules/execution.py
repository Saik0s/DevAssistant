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
        result = agent_executor.run({"input": task_info, "chat_history": []})
        return result

    def _create_tools(self, llm, memory_module):
        from .execution_tools import get_tools
        return get_tools(llm, memory_module)


# SYSTEM_MESSAGE = """Assistant is an AI who performs one task based on the following objective: {{objective}}. It can use tools. It always responds in json format."""

# INSTRUCTIONS = """
# Available tools:

# {tools}
# Final Answer: Use this if you want to respond directly to the human, input is what you want to return to user

# RESPONSE FORMAT INSTRUCTIONS
# ----------------------------
# Use only supported tools from this list: {tool_names}, Final Answer.
# When responding, please always output a response in this json format:

# {{{{
#     "question": string,
#     "thought": string,
#     "action": {{{{
#         "name": string,
#         "input": string
#     }}}},
#     "observation": string,
#     "thought": string,
#     "final_answer": string
# }}}}
# """

# USER_INPUT = """
# USER'S INPUT
# --------------------
# Take into account previously completed tasks and current progress towards objective.
# Here is the user's input (remember to respond with a JSON object, and NOTHING else):

# {{{{task_info}}}}"""

# TEMPLATE_TOOL_RESPONSE = """TOOL RESPONSE:
# ---------------------
# {observation}

# USER'S INPUT
# --------------------

# Okay, so what is the response to my last comment? If using information obtained from the tools you must mention it explicitly without mentioning the tool names - I have forgotten all TOOL RESPONSES! Remember to respond with a JSON object containing a single action and action_input, and NOTHING else."""


# class CustomOutputParser(BaseOutputParser):
#     def parse(self, text: str) -> Any:
#         cleaned_output = text.strip()
#         response = json.loads(cleaned_output)
#         return {"action": response["action"]["name"], "action_input": response["action"]["input"]}


# class CustomExecutionAgent(ConversationalChatAgent):
#     @classmethod
#     def create_prompt(
#         cls,
#         tools: Sequence[BaseTool],
#         input_variables: Optional[List[str]] = None,
#     ) -> BasePromptTemplate:
#         tool_strings = "\n".join(
#             [f"> {tool.name}: {tool.description}" for tool in tools]
#         )
#         tool_names = ", ".join([tool.name for tool in tools])
#         final_prompt = INSTRUCTIONS.format(
#             tool_names=tool_names, tools=tool_strings
#         )
#         if input_variables is None:
#             input_variables = ["task_info", "chat_history", "agent_scratchpad"]
#         messages = [
#             SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE),
#             HumanMessagePromptTemplate.from_template(USER_INPUT),
#             MessagesPlaceholder(variable_name="chat_history"),
#             HumanMessagePromptTemplate.from_template(final_prompt),
#             MessagesPlaceholder(variable_name="agent_scratchpad"),
#         ]
#         return ChatPromptTemplate(input_variables=input_variables, messages=messages)

#     def _construct_scratchpad(
#         self, intermediate_steps: List[Tuple[AgentAction, str]]
#     ) -> List[BaseMessage]:
#         """Construct the scratchpad that lets the agent continue its thought process."""
#         thoughts: List[BaseMessage] = []
#         for action, observation in intermediate_steps:
#             thoughts.append(AIMessage(content=action.log))
#             human_message = HumanMessage(
#                 content=TEMPLATE_TOOL_RESPONSE.format(observation=observation)
#             )
#             thoughts.append(human_message)
#         return thoughts

#     @classmethod
#     def from_llm_and_tools(
#         cls,
#         llm: BaseLanguageModel,
#         tools: Sequence[BaseTool],
#         callback_manager: Optional[BaseCallbackManager] = None,
#         system_message: str = SYSTEM_MESSAGE,
#         human_message: str = INSTRUCTIONS,
#         input_variables: Optional[List[str]] = None,
#         output_parser: Optional[BaseOutputParser] = None,
#         **kwargs: Any,
#     ) -> Agent:
#         """Construct an agent from an LLM and tools."""
#         cls._validate_tools(tools)
#         _output_parser = output_parser or CustomOutputParser()
#         prompt = cls.create_prompt(
#             tools,
#             system_message=system_message,
#             human_message=human_message,
#             input_variables=input_variables,
#             output_parser=_output_parser,
#         )
#         llm_chain = LLMChain(
#             llm=llm,
#             prompt=prompt,
#             callback_manager=callback_manager,
#         )
#         tool_names = [tool.name for tool in tools]
#         return cls(
#             llm_chain=llm_chain,
#             allowed_tools=tool_names,
#             output_parser=_output_parser,
#             **kwargs,
#         )
