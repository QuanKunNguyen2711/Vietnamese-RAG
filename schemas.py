from pydantic import BaseModel

class QuestionSchema(BaseModel):
    text: str