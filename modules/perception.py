from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class PerceptionModule:
    def __init__(self, memory_module, chat_model=ChatOpenAI(temperature=0)):
        self.memory_module = memory_module
        self.chat_model = chat_model

    def process_task(self, text):
        # Process the task using NLP techniques or AI models
        processed_task = self._process_task_nlp(text)

        # Retrieve related information from the MemoryModule
        related_information = self.memory_module.retrieve_related_information(processed_task)

        return f"{related_information}\n\nTask: {processed_task}"

    def process_text(self, text):
        return self._process_text_nlp(text)

    def _process_text_nlp(self, text):
        return self._process_nlp(
            "You are a helpful AI that processes text and summarizes the main points.",
            text,
        )

    def _process_task_nlp(self, text):
        return self._process_nlp(
            "You are a helpful AI that processes task and add missing details.",
            text,
        )

    # TODO Rename this here and in `process_text_nlp` and `process_task_nlp`
    def _process_nlp(self, arg0, text):
        template = arg0
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        chain = LLMChain(llm=self.chat_model, prompt=chat_prompt)
        return chain.run(text)
