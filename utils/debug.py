from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.shared import SharedCallbackManager
from langchain.callbacks.openai_info import OpenAICallbackHandler, LLMResult
import os

def enable_verbose_logging():
    os.environ["LANGCHAIN_HANDLER"] = "langchain"

    class DebugCallbackHandler(OpenAICallbackHandler):
        def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
            """Print out the prompts with colors."""
            # print("\033[1;34m> Prompts:\033[0m")
            # for prompt in prompts:
            #     print(f"\033[1;34m{prompt}\033[0m")
            pass

        def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
            OpenAICallbackHandler.on_llm_end(self, response, **kwargs)
            print(f"Total token: {self.total_tokens}")
            print("\033[1;32m> Response:\033[0m")
            print(f"\033[1;32m{response}\033[0m")

        def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
            """Print errors with colors."""
            print(f"\033[1;31mError: {error}\033[0m")

        def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
            """Print out that we finished a chain and its outputs."""
            print("\n\033[1m> Finished chain.\033[0m")
            print("\033[1m> Chain outputs:\033[0m")
            print(outputs)

        def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
            """Print extensive information about the chain error."""
            print("\n\033[1;31m> Chain error occurred:\033[0m")
            print(f"\033[1;31mError type:\033[0m {type(error).__name__}")
            print(f"\033[1;31mError message:\033[0m {str(error)}")
            print("\033[1;31mError traceback:\033[0m")
            import traceback

            traceback.print_tb(error.__traceback__)

        def on_tool_start(
            self,
            serialized: Dict[str, Any],
            input_str: str,
            **kwargs: Any,
        ) -> None:
            """Print out the log in specified color."""
            print(f"\033[1;35m> Tool start:\033[0m {serialized}")
            print(f"\033[1;35mInput string:\033[0m {input_str}")

        def on_tool_end(
            self,
            output: str,
            color: Optional[str] = None,
            observation_prefix: Optional[str] = None,
            llm_prefix: Optional[str] = None,
            **kwargs: Any,
        ) -> None:
            """If not the final action, print out observation."""
            if observation_prefix is not None:
                print(f"\033[1;35m{observation_prefix}\033[0m")
            print(f"\033[1;35m> Tool end:\033[0m {output}")

        def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
            """Do nothing."""
            print(f"\033[1;31m> Tool error occurred:\033[0m {error}")

    SharedCallbackManager().add_handler(DebugCallbackHandler())
