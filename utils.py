import re
from fastapi import UploadFile
from langchain.schema import Document
from typing import List
    
def get_surrounding_chunks(retrieved_docs: List[Document], 
                           all_docs: List[str], 
                           window_size: int = 1) -> List[List[str]]:
    
    
    windows = []
    for doc in retrieved_docs:
        chunk_index = doc.metadata["chunk_index"]

        start_idx = max(chunk_index - window_size, 0)
        end_idx = min(chunk_index + window_size + 1, len(all_docs))
        surrounding_chunks = all_docs[start_idx:end_idx]

        windows.append(surrounding_chunks)

    return windows


async def save_file(file: UploadFile, file_path):
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return file_path


def preprocess_text(text):
    # Remove excessive newlines and whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Normalize other irregularities (e.g., broken words)
    text = re.sub(r'-\s+', '', text)
    
    return text



def process_table(table, title=""):
    """
    Convert a table (list of lists) into a formatted string.
    Handles cases where the table contains None values.
    """
    if not table:
        return ""  # Return empty string if the table is empty

    # Ensure the header is a list of strings, replacing None with empty string
    header = table[0] if table[0] is not None else []
    header = [str(item) if item is not None else "" for item in header]

    # Ensure rows are lists of strings, replacing None with empty string
    rows = table[1:] if len(table) > 1 else []
    rows = [[str(item) if item is not None else "" for item in row] for row in rows]

    # Build the table text
    table_text = f"{title}\n" if title else ""
    table_text += "\t".join(header) + "\n"
    table_text += "\n".join(["\t".join(row) for row in rows])

    return table_text


def parse_memory_history(memory_history: str) -> str:
    parsed_history = ""
    interactions = memory_history.strip().split("\n")
    
    for i in range(0, len(interactions), 2):
        if i + 1 >= len(interactions):
            break
        
        user_input = interactions[i].replace("Human: ", "").strip()
        ai_response = interactions[i + 1].replace("AI: ", "").strip()
        
        parsed_history += (
            f"<|im_start|>user\n{user_input}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{ai_response}\n<|im_end|>\n"
        )
    
    return parsed_history.strip()



