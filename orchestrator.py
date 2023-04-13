from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain.vectorstores import Pinecone
from modules.execution import ExecutionModule
from modules.learning import LearningModule
from modules.memory import MemoryModule
from modules.perception import PerceptionModule
from modules.reasoning import ReasoningModule
from typing import Any, Dict, List, Optional


class AgentOrchestrator(Chain):
    memory_module: MemoryModule
    perception_module: PerceptionModule
    learning_module: LearningModule
    reasoning_module: ReasoningModule
    execution_module: ExecutionModule

    max_iterations: Optional[int] = None

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.reasoning_module.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    def print_end(self, final_result):
        print("\033[1;32m*****Task End*****:\033[0m")
        print(final_result)

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.memory_module.objective = inputs["objective"]
        self.reasoning_module.initialize_tasks()

        num_iters = 0
        while True:
            if self.reasoning_module.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.reasoning_module.task_list.popleft()
                self.print_next_task(task)

                # Process the current task using PerceptionModule
                processed_task = self.perception_module.process_task(task)
                self.print_next_task(processed_task)

                # Step 2: Execute the task
                execution_result = self.execution_module.execute(processed_task)
                self.print_task_result(execution_result)

                self.memory_module.store_result(execution_result, processed_task)
                print("\033[1;34mSaved new result\033[0m")

                # Process the execution result using PerceptionModule before storing it in the MemoryModule
                processed_execution_result = self.perception_module.process_result(execution_result)
                self.print_task_result(processed_execution_result)

                new_memory = self.learning_module.learn_from(
                    observation=processed_execution_result,
                    completed_tasks=list(self.reasoning_module.completed_task_list),
                    pending_tasks=list(self.reasoning_module.task_list),
                )

                # Step 3: Store the result in Memory
                self.memory_module.store(new_memory)
                print("\033[1;34mSaved new memory\033[0m")

                # Step 4: Create new tasks and reprioritize task list
                self.reasoning_module.update_tasks(processed_task, processed_execution_result)
                print("\033[1;33mUpdated tasks based on stored data\033[0m")

            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print("\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m")
                break

        final_answer = self.execution_module.execute({"task_id": 999, "task_name": "Provide the final answer"})
        self.print_end(final_answer)

        return {}

    @classmethod
    def from_llm(cls, llm: BaseLLM, vectorstore: Pinecone, verbose: bool = False, **kwargs) -> "AgentOrchestrator":
        memory_module = MemoryModule(llm, vectorstore=vectorstore, verbose=verbose)
        perception_module = PerceptionModule(llm, memory_module=memory_module, verbose=verbose)
        learning_module = LearningModule(llm, memory_module=memory_module, verbose=verbose)
        reasoning_module = ReasoningModule(llm, memory_module=memory_module, verbose=verbose)
        execution_module = ExecutionModule(llm, memory_module=memory_module, verbose=verbose)

        return cls(
            memory_module=memory_module,
            perception_module=perception_module,
            reasoning_module=reasoning_module,
            learning_module=learning_module,
            execution_module=execution_module,
            **kwargs
        )
