import os
import re
import shutil
from pathlib import Path
from typing import List
from langchain.utilities import BashProcess
from langchain.agents import Tool, load_tools

from modules.memory import MemoryModule

PREFIX_PATH = f"{str(Path(__file__).resolve().parent)}/runs/test_output/"

def get_tools(llm, memory_module: MemoryModule) -> List[Tool]:
  tools = load_tools(["searx-search"], llm=llm,
                  searx_host="http://localhost:8080", unsecure=True)
  return tools + [
      write_tool(),
      read_tool(),
      tree_tool(),
      mkdir_tool(),
      replace_content_tool(),
      copy_tool(),
      move_tool(),
      delete_tool(),
      append_tool(),
      search_memory_tool(memory_module),
  ]

def search_memory_tool(memory_module: MemoryModule) -> Tool:
    def search_memory(input_str: str) -> str:
        try:
          return memory_module.retrieve_related_information(input_str, top_k=20)
        except Exception as e:
            return str(e)
    return Tool(
        name="search memory",
        description="Search through memory for completed tasks.",
        func=search_memory,
    )


def write_tool() -> Tool:
    def write_file(input_str: str) -> str:
        try:
            input_lines = input_str.split("\n")
            path = PREFIX_PATH + input_lines[0].replace("..", "")
            content = "\n".join(input_lines[1:])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as file:
                file.write(content)
            return f"Written content to file at {path}"
        except Exception as e:
            return str(e)
    return Tool(
        name="write",
        description="Write content to a file. First line is the relative path, the rest is the content.",
        func=write_file,
    )

def read_tool() -> Tool:
    def read_file(input_str: str) -> str:
        try:
            path = PREFIX_PATH + input_str.replace("..", "")
            with open(path, "r") as file:
                content = file.read()
            return content
        except Exception as e:
            return str(e)
    return Tool(
        name="read",
        description="Read content from a file. Input is the relative path.",
        func=read_file,
    )

def tree_tool() -> Tool:
    def tree(input_str: str) -> str:
        try:
            bash = BashProcess()
            return bash.run(f"tree {PREFIX_PATH}")
        except Exception as e:
            return str(e)
    return Tool(
        name="tree",
        description="Display the directory tree.",
        func=tree,
    )

def mkdir_tool() -> Tool:
    def make_directory(input_str: str) -> str:
        try:
            path = PREFIX_PATH + input_str.replace("..", "")
            os.makedirs(path, exist_ok=True)
            return f"Created directory at {path}"
        except Exception as e:
            return str(e)
    return Tool(
        name="mkdir",
        description="Create a new directory. Input is the relative path.",
        func=make_directory,
    )

def replace_content_tool() -> Tool:
    def replace_content(input_str: str) -> str:
        try:
            input_lines = input_str.split("\n")
            path = PREFIX_PATH + input_lines[0].replace("..", "")
            pattern = input_lines[1]
            replacement = input_lines[2]

            with open(path, "r") as file:
                content = file.read()

            content = re.sub(pattern, replacement, content)

            with open(path, "w") as file:
                file.write(content)

            return f"Replaced content in file at {path}"
        except Exception as e:
            return str(e)
    return Tool(
        name="replace content",
        description="Replace content in a file using regex. Input is the relative path, pattern, and replacement.",
        func=replace_content,
    )

def copy_tool() -> Tool:
    def copy_file(input_str: str) -> str:
        try:
            input_lines = input_str.split("\n")
            src_path = PREFIX_PATH + input_lines[0].replace("..", "")
            dest_path = PREFIX_PATH + input_lines[1].replace("..", "")
            shutil.copy(src_path, dest_path)
            return f"Copied file from {src_path} to {dest_path}"
        except Exception as e:
            return str(e)
    return Tool(
        name="copy",
        description="Copy a file. Input is the source and destination relative paths.",
        func=copy_file,
    )

def move_tool() -> Tool:
    def move_file(input_str: str) -> str:
        try:
            input_lines = input_str.split("\n")
            src_path = PREFIX_PATH + input_lines[0].replace("..", "")
            dest_path = PREFIX_PATH + input_lines[1].replace("..", "")
            shutil.move(src_path, dest_path)
            return f"Moved file from {src_path} to {dest_path}"
        except Exception as e:
            return str(e)
    return Tool(
        name="move",
        description="Move a file. Input is the source and destination relative paths.",
        func=move_file,
    )

def delete_tool() -> Tool:
    def delete_file(input_str: str) -> str:
        try:
            path = PREFIX_PATH + input_str.replace("..", "")
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            else:
                return "Invalid path."
            return f"Deleted {path}"
        except Exception as e:
            return str(e)
    return Tool(
        name="delete",
        description="Delete a file or directory. Input is the relative path.",
        func=delete_file,
    )

def append_tool() -> Tool:
    def append_file(input_str: str) -> str:
        try:
            input_lines = input_str.split("\n")
            path = PREFIX_PATH + input_lines[0].replace("..", "")
            content = "\n".join(input_lines[1:])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as file:
                file.write(content)
            return f"Appended content to file at {path}"
        except Exception as e:
            return str(e)
    return Tool(
        name="append",
        description="Append content to a file. First line is the relative path, the rest is the content.",
        func=append_file,
    )
