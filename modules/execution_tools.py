import shutil
import re
import os
from typing import List
from pathlib import Path
from langchain.utilities import BashProcess
from langchain.tools.python.tool import PythonREPLTool
from modules.memory import MemoryModule
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer
from llama_index import GPTSimpleVectorIndex
from llama_index import download_loader
from langchain.vectorstores import DeepLake
from langchain.utilities import BashProcess
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import Tool, load_tools
from langchain.agents.tools import BaseTool
from datetime import datetime

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
PREFIX_PATH = f"{str(Path(__file__).resolve().parent.parent)}/runs/test_output_{current_datetime}/"


def get_tools(llm, memory_module: MemoryModule) -> List[Tool]:
    def wrap_tool_with_try_catch(tool: BaseTool) -> Tool:
        def wrapped_tool(input_str: str) -> str:
            try:
                return tool._run(input_str)
            except Exception as e:
                return f"Error occurred while executing tool {tool.name}: {str(e)}"

        return Tool(name=f"{tool.name}", func=wrapped_tool, description=f"{tool.description}")

    python_tool = PythonREPLTool()
    python_tool.name = "python_repl"

    tools = [wrap_tool_with_try_catch(python_tool)]

    mkdir_tool()("")

    # tools = []
    return tools + [
        bash_tool(),
        google_search_tool,
        # write_tool(),
        # read_tool(),
        # tree_tool(),
        # mkdir_tool(),
        #   replace_content_tool(),
        #   copy_tool(),
        # move_tool(),
        # delete_tool(),
        # append_tool(),
        # search_memory_tool(memory_module),
        # read_web_readability_tool(),
        # github_tool(),
        # read_remote_depth_tool(),
        # apply_patch_tool(),
        # read_web_unstructured_tool(),
        # bf4_qa_tool(),
        # directory_qa_tool(),
    ]


def parse_lines(input_str):  # sourcery skip: raise-specific-error
    """
    This function takes an input string and splits it into lines based on newline characters.
    It also removes any leading and trailing whitespace from each line and filters out empty lines.
    If the input string starts and ends with triple backticks, they are removed as well.

    Args:
        input_str (str): The input string to be parsed.

    Returns:
        list: A list of parsed lines.

    Raises:
        Exception: If the parsing fails, an exception is raised with an error message.
    """
    lines = [line for line in re.split(r"\\n|\n", input_str.strip('"')) if line.strip()]

    if not lines:
        raise Exception(f"Parsing lines for {input_str} failed, don't use this tool")

    if lines[0] == "```" and lines[-1] == "```":
        lines = lines[1:-1]

    return lines


google_search_tool = Tool(
    name="google_search",
    description="This is Google. Use this tool to search the internet. Input should be a string",
    func=lambda query: BashProcess().run(f"cd {os.path.dirname(os.path.realpath(__file__))}/tools && node google.js \"{query}\""),
)


def bf4_qa_tool() -> Tool:
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

    def query_website(input_str: str) -> str:
        try:
            input_lines = parse_lines(input_str)
            url = input_lines[0]
            question = input_lines[1]

            loader = BeautifulSoupWebReader()
            documents = loader.load_data(urls=[url])
            index = GPTSimpleVectorIndex(documents)
            return index.query(question)
        except Exception as e:
            return f"Error occurred while executing query_website: {str(e)}"

        return wrapped_query_website(question)

    return Tool(
        name="qa_about_website",
        func=query_website,
        description=f"Useful when you want answer questions about the text on websites. Input format: url\\nquestion.",
    )


def directory_qa_tool() -> Tool:
    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

    def query_local_directory(q: str) -> str:
        try:
            loader = SimpleDirectoryReader(PREFIX_PATH, recursive=True, exclude_hidden=True)
            documents = loader.load_data()
            index = GPTSimpleVectorIndex(documents)
            return index.query(q)
        except Exception as e:
            return str(e)

    return Tool(
        name="qa_about_local_directory",
        func=query_local_directory,
        description="Useful when you want answer questions about the files in your local directory.",
    )


def bash_tool() -> Tool:
    bash = BashProcess()

    def wrapped_func(command):
        try:
            # Check if command uses sudo
            if "sudo" in command:
                return "Error: Command cannot use sudo"
            if "apt" in command or "apt-get" in command:
                return "Error: Command cannot use apt or apt-get"

            # Check if command tries to do anything outside of PREFIX_PATH
            if any(arg.startswith("/") for arg in command.split()):
                return "Error: Command cannot access files outside of current work directory"
            return bash.run(f"cd {PREFIX_PATH} && {command}")
        except Exception as e:
            return str(e)

    return Tool(name="BASH", description="Executes bash commands and returns the output", func=wrapped_func)


def github_tool() -> Tool:
    def load_github_repo(input_str: str) -> str:
        try:
            input_lines = parse_lines(input_str)

            url = input_lines[0]
            branch = input_lines[1]
            question = input_lines[2]

            # Create a unique identifier for the repository
            repo_id = f"{url}-{branch}"

            # Check if the vector is already stored locally
            local_vector_path = f"{PREFIX_PATH}/vectors/{repo_id}"

            embeddings = OpenAIEmbeddings()
            if os.path.exists(local_vector_path):
                db = DeepLake(
                    dataset_path=local_vector_path,
                    read_only=True,
                    embedding_function=embeddings,
                )
            else:
                # Load repository by URL and branch
                GithubRepoLoader = download_loader("GithubRepoLoader")
                loader = GithubRepoLoader(url=url, branch=branch)
                documents = loader.load_data()
                docs = [doc.to_langchain_format() for doc in documents]

                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(docs)

                db = DeepLake.from_documents(texts, embeddings, dataset_path=local_vector_path)

            retriever = db.as_retriever()
            retriever.search_kwargs["distance_metric"] = "cos"
            retriever.search_kwargs["fetch_k"] = 100
            retriever.search_kwargs["maximal_marginal_relevance"] = True
            retriever.search_kwargs["k"] = 20

            model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

            result = qa({"question": question, "chat_history": []})

            return result["answer"]

        except Exception as e:
            return str(e)

    return Tool(
        name="qa_github_repository",
        description="Load a GitHub repository by URL and ask a provided question. Input format: url\\nbranch\\nquestion.",
        func=load_github_repo,
    )


def read_remote_depth_tool() -> Tool:
    def read_remote_depth(input_str: str) -> str:
        try:
            input_lines = parse_lines(input_str)
            url = input_lines[0]
            depth = int(input_lines[1])
            query = input_lines[2]
            RemoteDepthReader = download_loader("RemoteDepthReader")
            loader = RemoteDepthReader(depth=depth, domain_lock=True)
            documents = loader.load_data(url=url)
            index = GPTSimpleVectorIndex.from_documents(documents)
            return index.query(query, optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.5))
        except Exception as e:
            return str(e)

    return Tool(
        name="read_remote_depth",
        description="Read data from a remote url with a specified depth and answers provided question. Input is the url, depth and question separated by a new line character. Depth is an integer. Example: url\n2\nQuestion",
        func=read_remote_depth,
    )


def read_web_unstructured_tool() -> Tool:
    def read_web_unstructured(url: str) -> str:
        try:
            UnstructuredURLLoader = download_loader("UnstructuredURLLoader")
            urls = [url]
            loader = UnstructuredURLLoader(urls=urls, continue_on_failure=False, headers={"User-Agent": ""})
            return loader.load()
        except Exception as e:
            return str(e)

    return Tool(
        name="read_webpage_unstructured",
        description="Read unstructured data from a webpage. Input is the url.",
        func=read_web_unstructured,
    )


def read_web_readability_tool() -> Tool:
    def read_web_readability(url: str) -> str:
        try:
            ReadabilityWebPageReader = download_loader("ReadabilityWebPageReader")
            loader = ReadabilityWebPageReader()
            return loader.load_data(url=url)
        except Exception as e:
            return str(e)

    return Tool(
        name="read_webpage",
        description="Useful when you need to get text content from the webpage. Input is the url.",
        func=read_web_readability,
    )


def search_memory_tool(memory_module: MemoryModule) -> Tool:
    def search_memory(input_str: str) -> str:
        try:
            return memory_module.retrieve_related_information(input_str, top_k=20)
        except Exception as e:
            return str(e)

    return Tool(
        name="search_memory",
        description="Search through your memory of completed tasks and research results. Input is a search query.",
        func=search_memory,
    )


def write_tool() -> Tool:
    def write_file(input_str: str) -> str:
        try:
            input_lines = parse_lines(input_str)
            path = PREFIX_PATH + input_lines[0].replace("..", "")
            content = "\n".join(input_lines[1:])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as file:
                file.write(content)
            return f"Written content to file at {input_lines[0]}"
        except Exception as e:
            return str(e)

    return Tool(
        name="write_file",
        description="Write content to a file. Input first line is the relative path, the rest is the content.",
        func=write_file,
    )


def apply_patch_tool() -> Tool:
    def apply_patch(input_str: str) -> str:
        try:
            patch_file = f"{PREFIX_PATH}temp_patch_file.patch"
            with open(patch_file, "w") as file:
                file.write(input_str)

            bash = BashProcess()
            result = bash.run(f"cd {PREFIX_PATH} && patch -p1 -u -f -i temp_patch_file.patch")

            os.remove(patch_file)
            return f"Patch applied:\n{result}"
        except Exception as e:
            return str(e)

    return Tool(
        name="apply_patch",
        description="Apply a patch to the current folder. Input is the patch file content.",
        func=apply_patch,
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
        name="read_file",
        description="Read content from a file. Input is the relative path.",
        func=read_file,
    )


def tree_tool() -> Tool:
    def tree(input_str: str) -> str:
        try:
            bash = BashProcess()
            return bash.run(f"cd {PREFIX_PATH} && tree --noreport")
        except Exception as e:
            return str(e)

    return Tool(
        name="directory_tree",
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
        name="make_directory",
        description="Create a new directory. Input is the relative path.",
        func=make_directory,
    )


def replace_content_tool() -> Tool:
    def replace_content(input_str: str) -> str:
        try:
            input_lines = parse_lines(input_str)
            path = PREFIX_PATH + input_lines[0].replace("..", "")
            pattern = input_lines[1]
            replacement = input_lines[2]

            with open(path, "r") as file:
                content = file.read()

            content = re.sub(pattern, replacement, content)

            with open(path, "w") as file:
                file.write(content)

            return f"Replaced content in file at {input_lines[0]}"
        except Exception as e:
            return str(e)

    return Tool(
        name="replace_content",
        description="Replace content in a file using regex. Input is the relative path, pattern, and replacement separated by new lines.",
        func=replace_content,
    )


def copy_tool() -> Tool:
    def copy_file(input_str: str) -> str:
        try:
            input_lines = parse_lines(input_str)
            src_path = PREFIX_PATH + input_lines[0].replace("..", "")
            dest_path = PREFIX_PATH + input_lines[1].replace("..", "")
            shutil.copy(src_path, dest_path)
            return f"Copied file from {input_lines[0]} to {input_lines[1]}"
        except Exception as e:
            return str(e)

    return Tool(
        name="copy_file",
        description="Copy a file. Input is the source and destination relative paths separated by a new line.",
        func=copy_file,
    )


def move_tool() -> Tool:
    def move_file(input_str: str) -> str:
        try:
            input_lines = parse_lines(input_str)
            src_path = PREFIX_PATH + input_lines[0].replace("..", "")
            dest_path = PREFIX_PATH + input_lines[1].replace("..", "")
            shutil.move(src_path, dest_path)
            return f"Moved file from {input_lines[0]} to {input_lines[1]}"
        except Exception as e:
            return str(e)

    return Tool(
        name="move_file",
        description="Move a file. Input is the source and destination relative paths separated by a new line.",
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
            return f"Deleted {input_str}"
        except Exception as e:
            return str(e)

    return Tool(
        name="delete_file",
        description="Delete a file or directory. Input is the relative path.",
        func=delete_file,
    )


def append_tool() -> Tool:
    def append_file(input_str: str) -> str:
        try:
            input_lines = parse_lines(input_str)
            path = PREFIX_PATH + input_lines[0].replace("..", "")
            content = "\n".join(input_lines[1:])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as file:
                file.write(content)
            return f"Appended content to file at {input_lines[0]}"
        except Exception as e:
            return str(e)

    return Tool(
        name="append_to_file",
        description="Append content to a file. Input first line is the relative path, the rest is the content.",
        func=append_file,
    )
