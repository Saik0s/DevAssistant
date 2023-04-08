import os
from typing import List
from langchain import PromptTemplate
from llama_index import Document, download_loader
from llama_index import GithubRepositoryReader, TwitterTweetReader
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType, LLMSingleActionAgent
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index import SimpleDirectoryReader, Document
from llama_index.utils import globals_helper
from langchain.text_splitter import NLTKTextSplitter, SpacyTextSplitter, RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, load_tools, ConversationalChatAgent, AgentExecutor
from langchain.agents.chat.base import ChatAgent
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

def read_github(repo, branch):
    owner, repo = repo.split("/")
    documents = GithubRepositoryReader(
        github_token=os.environ.get("GITHUB_TOKEN"),
        owner=owner,
        repo=repo,
    ).load_data(branch=branch)
    return documents


def read_twitter(handles):
    reader = TwitterTweetReader(os.environ.get("TWITTER_TOKEN"), num_tweets=3)
    documents = reader.load_data(handles)
    return documents


def read_reddit(subreddits, search_keys, post_limit):
    RedditReader = download_loader("RedditReader")
    loader = RedditReader()
    documents = loader.load_data(
        subreddits=subreddits, search_keys=search_keys, post_limit=post_limit
    )
    return documents


def gmail():
    GmailReader = download_loader("GmailReader")
    loader = GmailReader(query="from: me label:inbox")
    documents = loader.load_data()
    return documents


def gpt_read_repo(repo_path: str, preamble_str: str = ""):
    GPTRepoReader = download_loader("GPTRepoReader")
    loader = GPTRepoReader()
    documents = loader.load_data(
        repo_path=repo_path, preamble_str=preamble_str, extensions=[".swift"]
    )
    return documents


def read_remote_depth(url: str, depth: int = 1):
    RemoteDepthReader = download_loader("RemoteDepthReader")
    loader = RemoteDepthReader()
    documents = loader.load_data(url=url)
    return documents





def read_web_unsctructured(url: str):
    UnstructuredURLLoader = download_loader("UnstructuredURLLoader")
    urls = [url]
    loader = UnstructuredURLLoader(
        urls=urls, continue_on_failure=False, headers={"User-Agent": "value"}
    )
    documents = loader.load()
    return documents


def read_directory(path: str):
    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
    loader = SimpleDirectoryReader(path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    return documents


def create_vectorstore(collection_name: str):
    print("Loading vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="chroma",
        collection_name=collection_name,
    )
    print("Done loading vector store.")
    return vectorstore

def add_to_vectorstore(vectorstore: Chroma, documents: List[Document]):
    print("Adding documents to vector store...")
    doc_chunks = []
    for document in documents:
        nltk_splitter = NLTKTextSplitter()
        text_chunks = nltk_splitter.split_text(document.text)
        doc_chunks.extend(Document(text, doc_id=document.doc_id, extra_info=document.extra_info) for text in text_chunks)

    documents = [d.to_langchain_format() for d in doc_chunks]
    vectorstore.add_documents(documents)
    vectorstore.persist()
    print("Done adding documents to vector store.")


def create_index_data_retriever_tool(vectorstore: Chroma):
    # chat_llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=1000)
    chat_llm = ChatOpenAI(temperature=0, max_tokens=1000)

    from langchain.prompts.chat import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )

    system_template = """Use the following pieces of context to perform what user asked. Ignore the context if you don't think it's relevant.
    ----------------
    {context}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

    # prompt_template = """Use the following pieces of context to answer the users question or request. Ignore the context if you don't think it's relevant.

    # {context}

    # Question: {question}
    # Helpful Answer:"""
    # PROMPT = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "question"]
    # )
    # llm = OpenAI(temperature=0, max_tokens=500)
    # qa = RetrievalQA.from_llm(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(k=2),
    #     verbose=True,
    #     prompt=PROMPT,
    # )

    qa = RetrievalQA.from_llm(
        llm=chat_llm,
        retriever=vectorstore.as_retriever(k=2),
        verbose=True,
        prompt=CHAT_PROMPT,
    )

    return Tool(
        name="SubAssistant",
        func=qa.run,
        description="useful for when you need help with writing swift code for ios app. Input should be a properly defined coding task.",
    )


def create_agent(vectorstore: Chroma, with_qa: bool = False):
    llm = OpenAI(temperature=0, max_tokens=500)
    chat_llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2000)
    # chat_llm = ChatOpenAI(temperature=0, max_tokens=1000)

    tools = []
    tools = load_tools(["searx-search"], llm=chat_llm,
                    searx_host="http://localhost:8080", unsecure=True)
    # if with_qa:
    #     tools.append(create_index_data_retriever_tool(vectorstore=vectorstore))

    from tools import create_file_tool, create_folder_tool, create_web_readability_tool
    tools.append(create_file_tool())
    tools.append(create_folder_tool())
    # tools.append(create_web_readability_tool())

    # chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1000)
    # return initialize_agent(
    #     tools,
    #     chat_llm,
    #     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    # )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    from tools import NewAgentOutputParser

    # prompt = CustomPromptTemplate(
    #     template=custom_template,
    #     tools=tools,
    #     # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    #     # This includes the `intermediate_steps` variable because that is needed
    #     input_variables=["input", "intermediate_steps"]
    # )

    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=chat_llm, tools=tools, memory=memory, output_parser=NewAgentOutputParser(), verbose=True,
        human_message="""TOOLS
------
Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:

{{tools}}

{format_instructions}

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a single action in a format I defined, and NOTHING else):

{{{{input}}}}"""
        )

    # prompt = ChatAgent.create_prompt(
    #     tools,
    #     format_instructions=custom_format_template,
    # )
    # llm_chain = LLMChain(llm=llm, prompt=prompt)
    # tool_names = [tool.name for tool in tools]
    # agent = ChatAgent(
    #     llm_chain=llm_chain,
    #     allowed_tools=tool_names,
    #     format
    #     )
    # agent = LLMSingleActionAgent(
    #     llm_chain=llm_chain,
    #     output_parser=CustomOutputParser(),
    #     stop=["\nObservation:"],
    #     allowed_tools=tool_names
    # )

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
    )

def create_simple_chain():
    print("Creating simple chain...")
    # llm = OpenAI(temperature=0, max_tokens=500, verbose=True)
    # return LLMChain.from_string(llm=llm, template="")
    return OpenAI(temperature=0, max_tokens=500)
