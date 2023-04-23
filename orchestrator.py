from langchain import OpenAI
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from modules.evaluation import EvaluationModule
from langchain.vectorstores import Pinecone
from modules.execution import ExecutionModule
from modules.learning import LearningModule
from modules.memory import MemoryModule
from modules.perception import PerceptionModule
from modules.reasoning import ReasoningModule
from typing import Any, Dict, List, Optional
from colorama import Fore, Back, Style
import rich

from utils.llm import create_llm


class AgentOrchestrator(Chain):
    memory_module: MemoryModule
    perception_module: PerceptionModule
    learning_module: LearningModule
    reasoning_module: ReasoningModule
    execution_module: ExecutionModule
    evaluation_module: EvaluationModule

    max_iterations: Optional[int] = None

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
                self.print_optimized_next_task(processed_task)

                # Step 2: Execute the task
                execution_result = self.execution_module.execute(processed_task)
                self.print_task_result(execution_result)
                self.reasoning_module.completed_task_list.append(task)

                self.memory_module.store_result(execution_result, processed_task)
                print(f"\n{Fore.LIGHTMAGENTA_EX}Saved new result to memory{Fore.RESET}")

                # # Process the execution result using PerceptionModule before storing it in the MemoryModule
                # processed_execution_result = self.perception_module.process_result(execution_result)
                # self.print_optimized_task_result(processed_execution_result)
                processed_execution_result = execution_result

                ## Evaluate the task result
                is_finished, final_answer = self.evaluation_module.evaluate_from(
                    observation=processed_execution_result,
                    completed_tasks=self.reasoning_module.completed_task_list,
                    pending_tasks=self.reasoning_module.task_list,
                )
                self.print_evaluated_task_result(is_finished, final_answer)

                if is_finished:
                    break

                # new_memory = self.learning_module.learn_from(
                #     observation=processed_execution_result,
                #     completed_tasks=list(self.reasoning_module.completed_task_list),
                #     pending_tasks=list(self.reasoning_module.task_list),
                # )
                # self.print_new_memory(new_memory)

                # # Step 3: Store the result in Memory
                # self.memory_module.store(new_memory)
                # print(f"\n{Fore.LIGHTMAGENTA_EX}Saved new learnings to memory{Fore.RESET}")

                # Step 4: Create new tasks and reprioritize task list
                self.reasoning_module.update_tasks(processed_task, processed_execution_result)
                print(f"\n{Fore.LIGHTMAGENTA_EX}Updated tasks based on stored data{Fore.RESET}")

            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print(f"\n{Fore.RED}\n*****TASK ENDING*****\n{Fore.RESET}")
                break

        self.print_end(final_answer)

        return {}

    @classmethod
    def from_llm(cls, verbose: bool = False, **kwargs) -> "AgentOrchestrator":
        llm = OpenAI(temperature=0, max_tokens=500, verbose=verbose, request_timeout=180, max_retries=10)
        exec_llm = create_llm(model_name="gpt-3.5-turbo", verbose=verbose)

        memory_module = MemoryModule(llm, verbose=verbose)
        perception_module = PerceptionModule(llm, memory_module=memory_module, verbose=verbose)
        learning_module = LearningModule(llm, memory_module=memory_module, verbose=verbose)
        reasoning_module = ReasoningModule(llm, memory_module=memory_module, verbose=verbose)
        execution_module = ExecutionModule(exec_llm, memory_module=memory_module, verbose=verbose)
        evaluation_module = EvaluationModule(llm, memory_module=memory_module, verbose=verbose)

        return cls(
            memory_module=memory_module,
            perception_module=perception_module,
            reasoning_module=reasoning_module,
            learning_module=learning_module,
            execution_module=execution_module,
            evaluation_module=evaluation_module,
            **kwargs,
        )

    def print_task_list(self):
        print(f"\n{Fore.BLUE}*****Completed*****{Fore.RESET}")
        rich.print(list(self.reasoning_module.completed_task_list))
        print(f"\n{Fore.GREEN}*****Pending*****{Fore.RESET}")
        rich.print(list(self.reasoning_module.task_list))

    def print_next_task(self, task: Dict):
        print(f"\n{Fore.LIGHTBLUE_EX}*****Next Task*****{Fore.RESET}")
        rich.print(task)

    def print_optimized_next_task(self, task: Dict):
        print(f"\n{Fore.LIGHTBLUE_EX}*****Optimized Next Task*****{Fore.RESET}")
        rich.print(task)

    def print_task_result(self, result: str):
        print(f"\n{Fore.LIGHTGREEN_EX}*****Task Result*****{Fore.RESET}")
        rich.print(result)

    def print_optimized_task_result(self, result: str):
        print(f"\n{Fore.LIGHTGREEN_EX}*****Optimized Task Result*****{Fore.RESET}")
        rich.print(result)

    def print_evaluated_task_result(self, is_finished: bool, result: str):
        print(f"\n{Fore.LIGHTCYAN_EX}*****Evaluated Task Result*****{Fore.RESET}")
        print(f"\n{Fore.LIGHTYELLOW_EX}Is finished: {is_finished}{Fore.RESET}")
        rich.print(result)

    def print_new_memory(self, new_memory: str):
        print(f"\n{Fore.LIGHTMAGENTA_EX}*****New Memory*****{Fore.RESET}")
        rich.print(new_memory)

    def print_end(self, final_result):
        print(f"\n{Fore.RED}*****End Result*****{Fore.RESET}")
        rich.print(final_result)
