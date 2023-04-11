def print_objective(objective):
    print("\033[1;32m*****Objective*****:\033[0m")
    print(objective)

def print_task_list(completed_task_list, task_list):
    print("\033[1;32m*****Completed Tasks*****:\033[0m")
    for task in completed_task_list:
        print(str(task["task_id"]) + ": " + task["task_name"])
    print("\033[1;32m*****Task List*****:\033[0m")
    for task in task_list:
        print(str(task["task_id"]) + ": " + task["task_name"])

def print_next_task(task):
    print("\033[1;32m*****Next Task*****:\033[0m")
    print(str(task["task_id"]) + ": " + task["task_name"])

def print_processed_task(task):
    print("\033[1;32m*****Processed Task Context*****:\033[0m")
    print(task["context"])
    print("\033[1;32m*****Processed Task*****:\033[0m")
    print(task["task_name"])

def print_task_result(result):
    print("\033[1;32m*****Task Result*****:\033[0m")
    print(result)

def print_end(final_result):
    print("\033[1;32m*****Task End*****:\033[0m")
    print(final_result)

def parse_task_list(response):
    new_tasks = response.split('\n')
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip(): continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list
