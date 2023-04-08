import sys
from index_chat import create_agent, create_vectorstore, read_github, add_to_vectorstore

def loada_data():
    print("Loading data...")
    documents = []
    documents += read_github("pointfreeco/swift-composable-architecture", "main")
    print("Done loading data.")
    return documents

vectorstore = create_vectorstore(collection_name = "langchain")

if "--load" in sys.argv:
    add_to_vectorstore(vectorstore, loada_data())

if "--prompt" in sys.argv:
    query = sys.argv[sys.argv.index("--prompt") + 1]
else:
    query = input("Type prompt:\n")
    print("-" * 80)

print("Thinking...")
result = create_agent(vectorstore).run(query)
print("Done thinking.")
print(result)
