import shutil
import re
import os
from typing import Dict, List
from pathlib import Path
from langchain.utilities import BashProcess
from langchain.tools.python.tool import PythonREPLTool
from modules.memory import MemoryModule
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
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


class GuardRailTool:
    def __init__(self, child_tool: BaseTool, args: Dict[str, str]):
        super().__init__()
        self.name = child_tool.name
        self.description = child_tool.description
        self.child_tool = child_tool
        self.args = args

    def _run(self, input_str: str) -> str:
        try:
            return self.child_tool.run(input_str)
        except Exception as e:
            return f"Error occurred while executing tool {self.name}: {str(e)}"


def get_tools(llm, memory_module: MemoryModule) -> List[Tool]:
    def wrap_tool_with_try_catch(tool: BaseTool) -> Tool:
        def wrapped_tool(input_str: str) -> str:
            try:
                return tool._run(input_str)
            except Exception as e:
                return f"Error occurred while executing tool {tool.name}: {str(e)}"

        return Tool(name=f"{tool.name}", func=wrapped_tool, description=f"{tool.description}")


    tools = [wrap_tool_with_try_catch(python_tool)]

    mkdir_tool()("")

    # tools = []
    return tools + [
        bash_tool,
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


BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")


def query_website(url: str, question: str) -> str:
    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls=[url])
    index = GPTSimpleVectorIndex(documents)
    return index.query(question)


def query_local_directory(q: str) -> str:
    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
    loader = SimpleDirectoryReader(PREFIX_PATH, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    index = GPTSimpleVectorIndex(documents)
    return index.query(q)


def bash_func(command):
    bash = BashProcess()
    # Check if command uses sudo
    if "sudo" in command:
        return "Error: Command cannot use sudo"
    if "apt" in command or "apt-get" in command:
        return "Error: Command cannot use apt or apt-get"

    # Check if command tries to do anything outside of PREFIX_PATH
    if any(arg.startswith("/") for arg in command.split()):
        return "Error: Command cannot access files outside of current work directory"
    return bash.run(f"cd {PREFIX_PATH} && {command}")


def load_github_repo(url: str, branch: str, question: str) -> str:
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


def read_remote_depth(url: str, depth: int, query: str) -> str:
    RemoteDepthReader = download_loader("RemoteDepthReader")
    loader = RemoteDepthReader(depth=depth, domain_lock=True)
    documents = loader.load_data(url=url)
    index = GPTSimpleVectorIndex.from_documents(documents)
    return index.query(query, optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.5))


def read_web_unstructured(url: str) -> str:
    UnstructuredURLLoader = download_loader("UnstructuredURLLoader")
    urls = [url]
    loader = UnstructuredURLLoader(urls=urls, continue_on_failure=False, headers={"User-Agent": ""})
    return loader.load()


def read_web_readability(url: str) -> str:
    ReadabilityWebPageReader = download_loader("ReadabilityWebPageReader")
    loader = ReadabilityWebPageReader()
    return loader.load_data(url=url)



def write_file(relative_path: str, content: str) -> str:
    path = PREFIX_PATH + relative_path.replace("..", "")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        file.write(content)
    return f"Written content to file at {relative_path}"


def apply_patch(patch_content: str) -> str:
    patch_file = f"{PREFIX_PATH}temp_patch_file.patch"
    with open(patch_file, "w") as file:
        file.write(patch_content)

    bash = BashProcess()
    result = bash.run(f"cd {PREFIX_PATH} && patch -p1 -u -f -i temp_patch_file.patch")

    os.remove(patch_file)
    return f"Patch applied:\n{result}"


def read_file(relative_path: str) -> str:
    path = PREFIX_PATH + relative_path.replace("..", "")
    with open(path, "r") as file:
        content = file.read()
    return content


def tree() -> str:
    bash = BashProcess()
    return bash.run(f"cd {PREFIX_PATH} && tree --noreport")


def make_directory(relative_path: str) -> str:
    path = PREFIX_PATH + relative_path.replace("..", "")
    os.makedirs(path, exist_ok=True)
    return f"Created directory at {path}"


def replace_content(relative_path: str, pattern: str, replacement: str) -> str:
    path = PREFIX_PATH + relative_path.replace("..", "")

    with open(path, "r") as file:
        content = file.read()

    content = re.sub(pattern, replacement, content)

    with open(path, "w") as file:
        file.write(content)

    return f"Replaced content in file at {relative_path}"


def copy_file(source_path: str, destination_path: str) -> str:
    src_path = PREFIX_PATH + source_path.replace("..", "")
    dest_path = PREFIX_PATH + destination_path.replace("..", "")
    shutil.copy(src_path, dest_path)
    return f"Copied file from {source_path} to {destination_path}"


def move_file(source_path: str, destination_path: str) -> str:
    src_path = PREFIX_PATH + source_path.replace("..", "")
    dest_path = PREFIX_PATH + destination_path.replace("..", "")
    shutil.move(src_path, dest_path)
    return f"Moved file from {source_path} to {destination_path}"


def delete_file(relative_path: str) -> str:
    path = PREFIX_PATH + relative_path.replace("..", "")
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        return "Invalid path."
    return f"Deleted {relative_path}"


def append_file(relative_path: str, content: str) -> str:
    path = PREFIX_PATH + relative_path.replace("..", "")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as file:
        file.write(content)
    return f"Appended content to file at {relative_path}"


google_search_tool = GuardRailTool(
    child_tool=Tool(
        name="google_search",
        func=lambda query: BashProcess().run(
            f'cd {os.path.dirname(os.path.realpath(__file__))}/tools && node google.js "{query}"'
        ),
        description="This is Google. Use this tool to search the internet. Input should be a string",
    ),
    args={"action_input": "The search query to be passed to Google."},
)

bash_tool = GuardRailTool(
    child_tool=Tool(
        name="bash",
        func=bash_func,
        description="Execute a bash command within the current work directory. Input is the command.",
    ),
    args={"action_input": "The bash command to execute."},
)

python_tool = GuardRailTool(child_tool=PythonREPLTool(), args={})
python_tool.name = "python_repl"


bf4_qa_tool = GuardRailTool(
    child_tool=Tool(
        name="qa_about_website",
        func=query_website,
        description=(
            "Useful when you want answer questions about the text on websites. " "Input format: url\\nquestion."
        ),
    ),
    args={
        "url": "The URL of the website to query.",
        "question": "The question to ask about the content of the website.",
    },
)

directory_qa_tool = GuardRailTool(
    child_tool=Tool(
        name="qa_about_local_directory",
        func=query_local_directory,
        description=("Useful when you want answer questions about the files in your local directory."),
    ),
    args={"action_input": "The question to ask about the content of the local directory."},
)

github_tool = GuardRailTool(
    child_tool=Tool(
        name="qa_github_repository",
        func=load_github_repo,
        description="Load a GitHub repository by URL and ask a provided question. Input format: url\\nbranch\\nquestion.",
    ),
    args={
        "url": "The URL of the GitHub repository.",
        "branch": "The branch of the repository to load.",
        "question": "The question to ask about the content of the repository.",
    },
)

read_remote_depth_tool = GuardRailTool(
    child_tool=Tool(
        name="read_remote_depth",
        func=read_remote_depth,
        description="Read data from a remote url with a specified depth and answers provided question. Input is the url, depth and question separated by a new line character. Depth is an integer. Example: url\n2\nQuestion",
    ),
    args={
        "url": "The URL of the website to load.",
        "depth": "The depth to load content from the website.",
        "question": "The question to ask about the content of the website.",
    },
)

read_web_unstructured_tool = GuardRailTool(
    child_tool=Tool(
        name="read_webpage_unstructured",
        func=read_web_unstructured,
        description="Read unstructured data from a webpage. Input is the url.",
    ),
    args={"action_input": "The URL of the webpage to load unstructured data from."},
)

read_web_readability_tool = GuardRailTool(
    child_tool=Tool(
        name="read_webpage",
        func=read_web_readability,
        description="Useful when you need to get text content from the webpage. Input is the url.",
    ),
    args={"action_input": "The URL of the webpage to load text content from."},
)


def search_memory_factory(memory_module: MemoryModule):
    def search_memory(input_str: str) -> str:
        return memory_module.retrieve_related_information(input_str, top_k=20)

    return GuardRailTool(
        child_tool=Tool(
            name="search_memory",
            func=search_memory,
            description="Search through your memory of completed tasks and research results. Input is a search query.",
        ),
        args={"action_input": "The search query to search through memory."},
    )


write_tool = GuardRailTool(
    child_tool=Tool(
        name="write_file",
        func=write_file,
        description="Write content to a file. Input first line is the relative path, the rest is the content.",
    ),
    args={
        "relative_path": "The relative path of the file to write content to.",
        "content": "The content to write to the file.",
    },
)

apply_patch_tool = GuardRailTool(
    child_tool=Tool(
        name="apply_patch",
        func=apply_patch,
        description="Apply a patch to the current folder. Input is the patch file content.",
    ),
    args={"action_input": "The content of the patch file to apply."},
)

read_tool = GuardRailTool(
    child_tool=Tool(
        name="read_file",
        func=read_file,
        description="Read content from a file. Input is the relative path.",
    ),
    args={"action_input": "The relative path of the file to read content from."},
)

tree_tool = GuardRailTool(
    child_tool=Tool(
        name="directory_tree",
        func=tree,
        description="Display the directory tree.",
    ),
    args={"action_input": "No input required."},
)

mkdir_tool = GuardRailTool(
    child_tool=Tool(
        name="make_directory",
        func=make_directory,
        description="Create a new directory. Input is the relative path.",
    ),
    args={"action_input": "The relative path of the directory to create."},
)

replace_content_tool = GuardRailTool(
    child_tool=Tool(
        name="replace_content",
        func=replace_content,
        description="Replace content in a file using regex. Input is the relative path, pattern, and replacement separated by new lines.",
    ),
    args={
        "relative_path": "The relative path of the file to replace content in.",
        "pattern": "The regex pattern to match content to replace.",
        "replacement": "The replacement content for the matched pattern.",
    },
)

copy_tool = GuardRailTool(
    child_tool=Tool(
        name="copy_file",
        func=copy_file,
        description="Copy a file. Input is the source and destination relative paths separated by a new line.",
    ),
    args={
        "source_path": "The source relative path of the file to copy.",
        "destination_path": "The destination relative path to copy the file to.",
    },
)

move_tool = GuardRailTool(
    child_tool=Tool(
        name="move_file",
        func=move_file,
        description="Move a file. Input is the source and destination relative paths separated by a new line.",
    ),
    args={
        "source_path": "The source relative path of the file to move.",
        "destination_path": "The destination relative path to move the file to.",
    },
)

delete_tool = GuardRailTool(
    child_tool=Tool(
        name="delete_file",
        func=delete_file,
        description="Delete a file or directory. Input is the relative path.",
    ),
    args={"action_input": "The relative path of the file or directory to delete."},
)

append_tool = GuardRailTool(
    child_tool=Tool(
        name="append_to_file",
        func=append_file,
        description="Append content to a file. Input first line is the relative path, the rest is the content.",
    ),
    args={
        "relative_path": "The relative path of the file to append content to.",
        "content": "The content to append to the file.",
    },
)
