#!/usr/bin/env python3
import argparse
import os
import time
from collections import deque
from typing import Dict, List
import os
import datetime
from pprint import pprint


from index_chat import create_agent, create_vectorstore, read_github, create_simple_chain

# Set API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--objective', type=str, help='the objective of the project')
parser.add_argument('--first_task', type=str, help='the first task of the project')
parser.add_argument('--collection_name', type=str, help='the name of the collection to use')

args = parser.parse_args()

# Project config
OBJECTIVE = args.objective or input("Please enter the project objective: ")
YOUR_FIRST_TASK = args.first_task or input("Please enter the first task: ")
COLLECTION_NAME = args.collection_name or input("Please enter the collection name: ")

#Print OBJECTIVE
print("\033[96m\033[1m"+"\n*****OBJECTIVE*****\n"+"\033[0m\033[0m")
print(OBJECTIVE)

# Create VectorStore
vectorstore = create_vectorstore(collection_name = COLLECTION_NAME)
for _ in range(10):
    vectorstore.add_texts(texts=["none"], metadatas=[{"task": "none", "result": "none"}])
vectorstore.persist()
qa_vectorstore = create_vectorstore(collection_name = "langchain")
simple_chain = create_simple_chain()

# Task list
task_list = deque([])

def add_task(task: Dict):
    task_list.append(task)

def similarity_search(vectorstore, query: str, k: int = 5):
    query = query.replace("\n", " ")
    return vectorstore.similarity_search(query, k=k)

# Create runs folder if it doesn't exist
if not os.path.exists("runs"):
    os.makedirs("runs")

now = datetime.datetime.now()
file_name = f"runs/{now.strftime('%Y-%m-%d_%H-%M-%S')}.md"
with open(file_name, "w") as f:
    f.write(OBJECTIVE)
    f.write("\n\n")

def openai_call(prompt: str, use_qa: bool = True):
    global file_name
    try:
        result = create_agent(qa_vectorstore).run(prompt) if use_qa else simple_chain.__call__(prompt)

        # Write prompt and result to markdown file with current time as name
        with open(file_name, "a") as f:
            f.write(f"\n\n## Prompt\n\n{prompt}\n\n## Result\n\n{result}")

    except Exception as e:
        print("*********************************")
        print(f"prompt: {prompt}\nuse_qa: {use_qa}")
        print(f"An error occurred 1: {e}")
        pprint(e)
        result = simple_chain.__call__(prompt)

    return result

def task_creation_agent(objective: str, result: Dict, task_description: str, task_list: List[str]):
    prompt = f"You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}, The last completed task has the result: {result}. This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array."
    response = openai_call(prompt, use_qa=False)
    new_tasks = response.split('\n')
    return [{"task_name": task_name} for task_name in new_tasks]

def prioritization_agent(this_task_id: int):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = this_task_id + 1
    prompt = f"""You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}. Consider the ultimate objective of your team:{OBJECTIVE}. Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
    response = openai_call(prompt, use_qa=False)
    new_tasks = response.split('\n')
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})

def execution_agent(objective: str, task: str) -> str:
    context = context_agent(query=objective, n=5)
    prompt = f"You are an AI who performs one task based on the following objective: {objective}.\nTake into account these previously completed tasks: {context}\nYour task: {task}\nResponse:"
    return openai_call(prompt)

def context_agent(query: str, n: int):
    try:
        results = similarity_search(vectorstore, query, k=n)
    except Exception as e:
        print(f"An error occurred 2: {e}")
        pprint(e)
        results = []

    pprint(results)

    return [(str(item.metadata['task'])) for item in results]

# Add the first task
first_task = {
    "task_id": 1,
    "task_name": YOUR_FIRST_TASK
}

add_task(first_task)

# Main loop
task_id_counter = 1
while True:
    if task_list:
        # Print the task list
        print("\033[95m\033[1m"+"\n*****TASK LIST*****\n"+"\033[0m\033[0m")
        for t in task_list:
            print(str(t['task_id'])+": "+t['task_name'])

        # Step 1: Pull the first task
        task = task_list.popleft()
        print("\033[92m\033[1m"+"\n*****NEXT TASK*****\n"+"\033[0m\033[0m")
        print(task)
        print(str(task['task_id'])+": "+task['task_name'])

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE, task["task_name"])
        this_task_id = int(float(task["task_id"]))
        print("\033[93m\033[1m"+"\n*****TASK RESULT*****\n"+"\033[0m\033[0m")
        pprint(result)

        # Step 2: Enrich result and store in VectorStore
        enriched_result = {'data': result}  # This is where you should enrich the result if needed
        result_id = f"result_{task['task_id']}"
        vectorstore.add_texts([enriched_result['data']], metadatas=[{"task": task['task_name'], "result": result}])

    # Step 3: Create new tasks and reprioritize task list
    new_tasks = task_creation_agent(OBJECTIVE,enriched_result, task["task_name"], [t["task_name"] for t in task_list])

    for new_task in new_tasks:
        task_id_counter += 1
        new_task.update({"task_id": task_id_counter})
        add_task(new_task)
    prioritization_agent(this_task_id)

    time.sleep(1)  # Sleep before checking the task list again
