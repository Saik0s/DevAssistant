from langchain import BasePromptTemplate, OpenAI, PromptTemplate
from langchain.agents.conversational_chat.base import (
    ConversationalChatAgent,
    BaseOutputParser,
)
from typing import Any, Dict, Optional, Sequence
from langchain.agents.agent import Agent, AgentExecutor
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
        agent = TaskExecutionAgent.create_from_llm_and_tools(
            llm=self.llm, tools=self.tools, verbose=True
        )
        agent_executor = AgentExecutor.from_agent_and_tools(
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
You are an AI developed by OpenAI. Your purpose is to complete tasks by performing actions and going through the necessary steps.
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

{{task}}
"""

        input_variables = ["objective", "context", "task", "agent_scratchpad"]
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(execution_template),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[BaseMessage]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts: List[BaseMessage] = []
        combined_content_length = 0
        for index, (action, observation) in enumerate(intermediate_steps):
            if index < len(intermediate_steps) - 2:
                action_log = "action text truncated"
            else:
                action_log = action.log
            thoughts.append(AIMessage(content=action_log))
            human_message = HumanMessage(
                content="""TOOL RESPONSE:
---------------------
{observation}

USER'S INPUT
--------------------

Remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else.
""".format(observation=observation)
            )
            thoughts.append(human_message)
            combined_content_length += len(action_log) + len(human_message.content)

        if combined_content_length > 8000:
            print("\033[95mContent length is too big and will be compressed.\033[0m")
            # TODO: Improve or at least extract this chain
            llm = OpenAI(temperature=0)
            prompt = PromptTemplate(
                input_variables=["text"],
                template="Summarize the following thoughts in a compressed form as much as possible, but without losing any details. Text: {text}",
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            combined_content = "\n".join([thought.content for thought in thoughts])
            summary = chain.run(combined_content)
            thoughts = [AIMessage(content=summary)]

        return thoughts

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        try:
            response = self.output_parser.parse(llm_output)
            return response["action"], response["action_input"]
        except Exception as e:
            print(f"Error occurred while extracting tool and input: {e}")
            print(f"LLM output: {llm_output}")
            return None



    def _fix_text(self, text: str) -> str:
        """Fix the text."""
        return text + "\n\n" + "Couldn't extract action and action input."

    def _get_next_action(self, full_inputs: Dict[str, str]) -> AgentAction:
        full_output = self.llm_chain.predict(**full_inputs)
        parsed_output = self._extract_tool_and_input(full_output)
        while parsed_output is None:
            full_output = self._fix_text(full_output)
            full_inputs["agent_scratchpad"].append(AIMessage(content=full_output))
            output = self.llm_chain.predict(**full_inputs)
            full_output += output
            parsed_output = self._extract_tool_and_input(full_output)
        return AgentAction(
            tool=parsed_output[0], tool_input=parsed_output[1], log=full_output
        )

    async def _aget_next_action(self, full_inputs: Dict[str, str]) -> AgentAction:
        full_output = await self.llm_chain.apredict(**full_inputs)
        parsed_output = self._extract_tool_and_input(full_output)
        while parsed_output is None:
            full_output = self._fix_text(full_output)
            full_inputs["agent_scratchpad"].append(AIMessage(content=full_output))
            output = await self.llm_chain.apredict(**full_inputs)
            full_output += output
            parsed_output = self._extract_tool_and_input(full_output)
        return AgentAction(
            tool=parsed_output[0], tool_input=parsed_output[1], log=full_output
        )

    @classmethod
    def create_from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> Agent:
        cls._validate_tools(tools)
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
            output_parser=NewAgentOutputParser(),
            **kwargs,
        )

class NewAgentOutputParser(BaseOutputParser):
    def parse(self, text: str) -> Any:
        cleaned_output = text.strip()
        print("__________")
        print(cleaned_output)
        print("__________")

        import re, json
        # Regex to extract the JSON string with action and action_input
        json_regex = r"(?:(?:```|```json)|)\s*({\s*\"action\"\s*:\s*\"[^\"]+\",\s*\"action_input\"\s*:\s*\"[^\"]+\"\s*})\s*(?:```|)"
        if json_match := re.search(json_regex, cleaned_output):
            json_string = json_match.group(1)
            print(json_string)
            response = json.loads(json_string)
            return {"action": response["action"], "action_input": response["action_input"]}
        else:
            print("No JSON string found using regex, trying to parse as JSON directly")


        if "```json" in cleaned_output:
            _, cleaned_output = cleaned_output.split("```json")
        if "```" in cleaned_output:
            cleaned_output, _ = cleaned_output.split("```")
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json") :]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[len("```") :]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[: -len("```")]
        cleaned_output = cleaned_output.strip()
        response = json.loads(cleaned_output)
        return {"action": response["action"], "action_input": response["action_input"]}
