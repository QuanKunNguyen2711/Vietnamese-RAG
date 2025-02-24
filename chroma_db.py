import asyncio
from fastapi import Depends
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from typing import List
from dotenv import load_dotenv
from enum import Enum
import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber

from rag.embedding import CustomLangchainEmbeddingFunction
from rag.utils import get_surrounding_chunks, preprocess_text, process_table

load_dotenv()

class SearchTypeEnums(str, Enum):
    SIMILARITY = "similarity"
    SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"

class ChromaDB:
    def __init__(self,
                 embed_func:Embeddings = CustomLangchainEmbeddingFunction(),
                 persist_directory: str = os.environ.get('CHROMA_DB_PATH')
                ):
        
        self.embedding_function = embed_func
        self.persist_directory = persist_directory
        # self.docs = []
        
        
    def get_collection(self, db_str: str):
        return Chroma(
            collection_name=db_str,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function
        )
        
    def get_relevant_context(self,
                             query: str,
                             db_str: str,
                             search_type: str = SearchTypeEnums.SIMILARITY.value,
                             k: int = 1,
                             score_threshold: float = 0.6
                            ) -> str:
        
        retriever = self.get_retriever(db_str, search_type=search_type, top_k=k, score_threshold=score_threshold)
        retrieved_docs = retriever.invoke(query)
        all_docs = self.get_all_docs(db_str).get('documents', [])
        surrounding_chunks = get_surrounding_chunks(retrieved_docs, all_docs)
        
        return '\n'.join([context for chunk in surrounding_chunks for context in chunk])
        
        
    def get_retriever(self, 
                      db_str: str,
                      search_type: str = SearchTypeEnums.SIMILARITY.value,
                      top_k: int = 3,
                      score_threshold: float = 0.6
                      ):
        
        collection = self.get_collection(db_str)
        if search_type == SearchTypeEnums.SIMILARITY_SCORE_THRESHOLD.value:
            return collection.as_retriever(
                search_type=search_type,
                search_kwargs={"score_threshold": score_threshold}
            )
            
        return collection.as_retriever(
            search_type=search_type,
            search_kwargs={"k": top_k}
        )


    async def load_and_split_file(self, 
                                  file_path: str, 
                                  chunk_size=512,
                                  chunk_overlap=100
                                  ) -> List[Document]:
        # all_docs = []

        # # Open the PDF with pdfplumber
        # with pdfplumber.open(file_path) as pdf:
        #     for page_number, page in enumerate(pdf.pages):
        #         # Extract text
        #         text = page.extract_text()
        #         if text:
        #             # Add the page text as a document
        #             all_docs.append(Document(page_content=text, metadata={"page": page_number + 1, "type": "text", 'source': str(file_path)}))

        #         # Extract tables
        #         tables = page.extract_tables()
        #         for table_number, table in enumerate(tables):
        #             # Process each table
        #             table_text = process_table(table)
        #             if table_text:
        #                 # Add the table as a document with metadata
        #                 all_docs.append(Document(page_content=table_text, metadata={"page": page_number + 1, "table": table_number + 1, "type": "table", 'source': str(file_path)}))

        # # Split text documents
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap
        # )
        # split_docs = text_splitter.split_documents([doc for doc in all_docs if doc.metadata["type"] == "text"])
        # filtered_docs = []
        # for doc in split_docs:
        #     doc.page_content = preprocess_text(doc.page_content)
            
        #     min_len = 6
        #     if len(doc.page_content) < min_len:
        #         continue
            
        #     filtered_docs.append(doc)
        
        # # Add table documents (no splitting for tables to preserve structure)
        # final_docs = filtered_docs + [doc for doc in all_docs if doc.metadata["type"] == "table"]

        # return final_docs

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        
        filtered_docs = []
        for doc in split_docs:
            doc.page_content = preprocess_text(doc.page_content)
            
            min_len = 6
            if len(doc.page_content) < min_len:
                continue
            
            filtered_docs.append(doc)
            
        return filtered_docs
    
    
    async def add_docs(self, db_str: str, path_to_files: List[str]):
        if len(path_to_files) == 0:
            return
        
        collection = self.get_collection(db_str)

        tasks = [self.load_and_split_file(file_path, chunk_size=512, chunk_overlap=100) for file_path in path_to_files]
        results = await asyncio.gather(*tasks)

        all_docs = [doc for result in results for doc in result]

        for index, doc in enumerate(all_docs):
            doc.metadata['chunk_index'] = index

        await collection.aadd_documents(all_docs)
        
    
    def delete_coll(self, db_str: str):
        collection = self.get_collection(db_str)
        collection.delete_collection()
        
        
    def load_all_docs(self, 
                      db_docs_path: str, 
                      chunk_size: int = 512, 
                      chunk_overlap: int = 100
                      )-> List[Document]:
        
        dir_doc_loader = PyPDFDirectoryLoader(db_docs_path)
        docs = dir_doc_loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(docs)
        
        for idx, doc in enumerate(split_docs):
            doc.metadata["chunk_index"] = idx
        
        return split_docs
    
    def get_all_docs(self, db_str: str):
        collection = self.get_collection(db_str)
        return collection.get()
