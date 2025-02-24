from langchain_core.output_parsers import StrOutputParser

class CustomOutputParser(StrOutputParser):
    def __init__(self):
        super().__init__()
        
    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    
    def extract_answer(self, text: str, pattern: str = "<|im_start|>assistant") -> str:
        if pattern in text:
            parsed_response = text.split(pattern)[-1].strip()
            return parsed_response