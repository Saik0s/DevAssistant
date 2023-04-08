import json
from typing import Any
from langchain.agents import Tool
import os
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.agents import initialize_agent

from llama_index import GPTIndexMemory, GPTListIndex
from llama_index import Document, download_loader
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
import re
from langchain.schema import BaseOutputParser

import os
from pathlib import Path
from pprint import pprint
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re

PREFIX_PATH = f"{str(Path(__file__).resolve().parent)}/runs/test_output/"


def create_file_with_content(input_str: str) -> str:
    try:
        input_lines = input_str.split("\n")
        path = PREFIX_PATH + input_lines[0].replace("..", "")
        content = "\n".join(input_lines[1:])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            file.write(content)

        return f"Created file at {path}"
    except Exception as e:
        print("create_file_with_content error", e)
        return str(e)


def create_file_tool() -> Tool:
    return Tool(
        name="create file",
        description="Create or overwrite one file with action_input first line is the relative path, the rest is the content",
        func=create_file_with_content,
    )


def create_folder(path: str) -> str:
    try:
        path = PREFIX_PATH + path.replace("..", "")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.mkdir(path)
        return f"Created folder at {path}"
    except Exception as e:
        print("create_folder error", e)
        return str(e)


def create_folder_tool() -> Tool:
    return Tool(
        name="create folder",
        description="Create a folder, requires relative path to the folder",
        func=create_folder,
    )


def read_web_readability(url: str):
    try:
        ReadabilityWebPageReader = download_loader("ReadabilityWebPageReader")
        loader = ReadabilityWebPageReader()
        return loader.load_data(url=url)
    except Exception as e:
        print("read_web_readability error", e)
        return []


def create_web_readability_tool():
    def tool_func(query: str):
        documents = read_web_readability(query)
        llm = OpenAI(temperature=0, max_tokens=500)
        chat_llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=1000)
        text_splitter = CharacterTextSplitter()
        docs = []
        from langchain.docstore.document import Document

        for document in documents:
            texts = text_splitter.split_text(document.text)
            docs.extend(
                [Document(page_content=t, metadata=document.extra_info) for t in texts]
            )

        from langchain.chains.summarize import load_summarize_chain

        chain = load_summarize_chain(llm, chain_type="map_reduce")
        return chain.run(docs)

    return Tool(
        name="Web Readability",
        func=tool_func,
        description="Useful for reading a web page summary, input should be a url",
        return_direct=True,
    )

class NewAgentOutputParser(BaseOutputParser):
    def get_format_instructions(self) -> str:
        return """RESPONSE FORMAT INSTRUCTIONS
----------------------------
Only one action at a time is supported.
When responding to me please, please output a response in one of two formats:

**Option 1:**
Use this if you want the human to use a tool.
Text snippet with start and end lines formatted in the following schema:

******start
action_name: string // The action to take. Must be one of {tool_names}
action_input: string // The input to the action, must be string type
******end

**Option #2:**
Use this if you want to respond directly to the human. Text snippet with start and end lines formatted in the following schema:

******start
action_name: Final Answer
action_input: string // You should put what you want to return to use here, must be string type
******end

"""

    def parse(self, text: str) -> Any:
        print("-" * 20)
        cleaned_output = text.strip()
        print(cleaned_output)

        print("+" * 20)
        regex = r"action_name: (.*?)[\n]*action_input:[\s]*([\d\D]*)"
        match = re.search(regex, cleaned_output, re.DOTALL)
        if not match:
            regex = r'"action_name": "(.*?)",[\s\n]*"action_input": "(.*)"'
            match = re.search(regex, cleaned_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{cleaned_output}`")
            action = match.group(1)
            action_input = match.group(2)
            cleaned_output = f'{"action_name": "{action}", "action_input": "{action_input}"}'
            json_obj = json.loads(cleaned_output)
            action = json_obj["action"]
            action_input = json_obj["action_input"]

        else:
            action = match.group(1).strip()
            action_input = match.group(2).strip(" ").strip('"').replace("******end", "")
        return {"action": action, "action_input": action_input}
