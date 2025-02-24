from typing import List
from langchain.embeddings.base import Embeddings
import numpy as np
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from dotenv import load_dotenv
import os

load_dotenv()

class CustomLangchainEmbeddingFunction(Embeddings):
    def __init__(self, 
                 model_name: str = os.environ.get('EMBED_MODEL'), 
                 max_seq_length: int = os.environ.get('MAX_EMBED_TOKENS')
                ):
        
        self.max_seq_length = max_seq_length
        self._model = SentenceTransformer(model_name)
        self._tokenizer = tokenize
        
    def __call__(self):
        return self
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        tokenized_texts = [self._tokenizer(text) for text in texts]
        # Generate embeddings
        embeddings = self._model.encode(tokenized_texts)
        return np.array(embeddings)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        embeddings = self._get_embeddings([text])
        return embeddings[0].tolist()