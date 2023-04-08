import sys
from index_chat import create_agent, create_vectorstore, read_github

def loada_data():
    print("Loading data...")
    documents = []
    documents += read_github("pointfreeco/swift-composable-architecture", "main")
    print("Done loading data.")
    return [d.to_langchain_format() for d in documents]

if "--load" in sys.argv:
    vectorstore = create_vectorstore()
    vectorstore.add_documents(loada_data())
    vectorstore.persist()

if "--prompt" in sys.argv:
    query = sys.argv[sys.argv.index("--prompt") + 1]
else:
    query = input("Type prompt:\n")
    print("-" * 80)

print("Thinking...")
result = create_agent().run(query)
print("Done thinking.")
print(result)
