import sys
from index_chat import create_agent, create_vectorstore, read_github, add_to_vectorstore
from langchain.vectorstores import Chroma

def loada_data(vectorstore: Chroma):
    print("Loading data...")
    documents = []
    # documents = read_github("getcursor/cursor", "main")
    # add_to_vectorstore(vectorstore, documents=documents)
    # documents = read_github("Torantulino/AI-Functions", "master")
    # add_to_vectorstore(vectorstore, documents=documents)
    # documents = read_github("Torantulino/Auto-GPT", "master")
    # add_to_vectorstore(vectorstore, documents=documents)
    # documents = read_github("jerryjliu/llama_index", "main")
    # add_to_vectorstore(vectorstore, documents=documents)
    documents = read_github("yoheinakajima/babyagi", "main")
    add_to_vectorstore(vectorstore, documents=documents)
    documents = read_github("jerryjliu/llama_index", "main")
    add_to_vectorstore(vectorstore, documents=documents)
    print("Done loading data.")

vectorstore = create_vectorstore(collection_name = "browse1")

if "--load" in sys.argv:
    loada_data(vectorstore)
    exit()

if "--prompt" in sys.argv:
    query = sys.argv[sys.argv.index("--prompt") + 1]
else:
    query = input("Type prompt:\n")
    print("-" * 80)

print("Thinking...")
result = create_agent(vectorstore, with_qa=True).run(query)
print("Done thinking.")
print(result)
