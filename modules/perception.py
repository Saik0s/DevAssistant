from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from modules.memory import MemoryModule

class PerceptionModule:
    def __init__(self, memory_module: MemoryModule, chat_model):
        self.memory_module = memory_module
        self.chat_model = chat_model

    def process_task(self, task):
        name = task["task_name"]
        summary = self.memory_module.retrieve_related_information(name)
        return {"task_name": name, "context": summary}

    def process_text(self, text):
        return self._process_text_nlp(text)

    def _process_text_nlp(self, text):
        return self._process_nlp(
            "You are a helpful AI that takes input text, summarizes and optimizes it to retain all essential details while reducing its length.",
            text,
        )

    def _process_task_nlp(self, text):
        return self._process_nlp(
            "You are a helpful AI that takes an input task, optimizes it to include all essential details, and reduces its length while maintaining its effectiveness. You should compress it as much as possible.",
            text,
        )

    def _process_nlp(self, arg0, text):
        template = arg0
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        chain = LLMChain(llm=self.chat_model, prompt=chat_prompt)
        return chain.run(text)
