import os
from llama_index import download_loader
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
from langchain.agents import AgentType

def read_github(repo, branch):
    owner, repo = repo.split("/")
    documents = GithubRepositoryReader(
        github_token=os.environ.get("GITHUB_TOKEN"),
        owner=owner,
        repo=repo,
        use_parser=False,
        verbose=False,
    ).load_data(branch=branch)
    return documents


def read_twitter(handles):
    reader = TwitterTweetReader(os.environ.get("TWITTER_TOKEN"), num_tweets=3)
    documents = reader.load_data(handles)
    return documents

def read_reddit(subreddits, search_keys, post_limit):
    RedditReader = download_loader('RedditReader')
    loader = RedditReader()
    documents = loader.load_data(subreddits=subreddits, search_keys=search_keys, post_limit=post_limit)
    return documents

def gmail():
    GmailReader = download_loader('GmailReader')
    loader = GmailReader(query="from: me label:inbox")
    documents = loader.load_data()
    return documents

def gpt_read_repo(repo_path: str, preamble_str: str = ""):
    GPTRepoReader = download_loader("GPTRepoReader")
    loader = GPTRepoReader()
    documents = loader.load_data(repo_path=repo_path, preamble_str=preamble_str, extensions=[".swift"])
    return documents

def read_remote_depth(url: str, depth: int = 1):
    RemoteDepthReader = download_loader("RemoteDepthReader")
    loader = RemoteDepthReader()
    documents = loader.load_data(url=url)
    return documents

def read_web_readability(url: str):
    ReadabilityWebPageReader = download_loader("ReadabilityWebPageReader")
    loader = ReadabilityWebPageReader()
    documents = loader.load_data(url=url)
    return documents

def read_web_unsctructured(url: str):
    UnstructuredURLLoader = download_loader("UnstructuredURLLoader")
    urls = [url]
    loader = UnstructuredURLLoader(urls=urls, continue_on_failure=False, headers={"User-Agent": "value"})
    documents = loader.load()
    return documents

def read_directory(path: str):
    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
    loader = SimpleDirectoryReader(path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    return documents

def create_vectorstore():
    print("Loading vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma")
    print("Done loading vector store.")
    return vectorstore

def create_index_data_retriever_tool():
    vectorstore = create_vectorstore()
    chat_llm_4 = ChatOpenAI(model_name='gpt-4', temperature=0, max_tokens=1500)
    qa = RetrievalQA.from_llm(llm=chat_llm_4, retriever=vectorstore.as_retriever(), verbose=True)

    return Tool(
        name = "Developer QA System",
        func=qa.run,
        description="useful for when you need to answer questions about software development. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before."
    )

def create_agent():
    tools = [create_index_data_retriever_tool()]
    chat_llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, max_tokens=1500)
    agent = initialize_agent(tools, chat_llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent

