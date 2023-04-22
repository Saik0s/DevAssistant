from typing import Union
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def summarize_text(text: Union[str, Document], max_chars: int = 1000, verbose: bool = True) -> str:
    if not text:
        print("No text to summarize.")
        return ""

    prompt_template = (
        "Write a concise summary of the following:\n\n\n"
        '"{text}"'
        f"\n\n\nCONCISE SUMMARY up to {max_chars} characters:"
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    llm = OpenAI(temperature=0, verbose=verbose, max_tokens=1000)
    chain = load_summarize_chain(llm, chain_type="map_reduce", combine_prompt=prompt, verbose=verbose)
    text_splitter = RecursiveCharacterTextSplitter()

    texts = text_splitter.split_text(text) if isinstance(text, str) else text_splitter.split_documents([text])
    docs = [Document(page_content=t) for t in texts]

    return chain.run(docs)
