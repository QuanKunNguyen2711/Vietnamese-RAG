from typing import List
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import os

from rag.llm import LLM_chain
from rag.output_parser import CustomOutputParser
from rag.prompt import CONTEXT, PROMPT_TEMPLATE, QUESTION

load_dotenv()



class RAG:
    def __init__(self,
                 model_name: str = os.environ.get('LLM_MODEL'),
                 max_new_tokens: int = int(os.environ.get('MAX_NEW_TOKENS')),
                 promt_template: str = PROMPT_TEMPLATE,
                 input_variables: List[str] = [QUESTION, CONTEXT],
                ):
        
        self.prompt = PromptTemplate(
            input_variables=input_variables,
            template=promt_template,
            validate_template=False
        )
        self.llm = LLM_chain(model_name, max_new_tokens)
        self.output_parser = CustomOutputParser()
        self.docs = []
        
    def get_chain(self):
        llm = self.llm.get_llm_chain()
        return self.prompt | llm | self.output_parser
    

rag = RAG()
rag_chain = rag.get_chain()