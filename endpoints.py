import asyncio
import io
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.common.authentication import protected_route
from app.common.dependencies import AuthCredentialDepend
import logging
from app.common.enums import SystemRole
from typing import List
from pydub import AudioSegment

from rag.schemas import QuestionSchema
from rag.rag import rag_chain
from rag.chroma_db import ChromaDB
from rag.utils import save_file
from rag.pho_whisper import transcriber
from dotenv import load_dotenv
import os

load_dotenv()


router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_UPLOAD_DIR = Path(f"{os.environ.get('ROOT_PATH')}/uploads")
BASE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/query", status_code=200)
@protected_route([SystemRole.HOTEL_OWNER, SystemRole.DATA_SCIENTIST])
async def query(
    question: QuestionSchema, 
    CREDENTIALS: AuthCredentialDepend, 
    CURRENT_USER=None
):
    global rag_chain
    
    question = question.model_dump()
    text = question.get("text")
    
    db_str, cur_user_id = CURRENT_USER.get("db"), CURRENT_USER.get("_id")
    chroma_db = ChromaDB()

    contexts = chroma_db.get_relevant_context(
        text, 
        db_str,
        k=3
    )
#     response = rag_chain.invoke({
#         "question": text,
#         "context": contexts,
#    })
    # Asynchronous wrapper for sync LLM call
    try:
        response = await asyncio.to_thread(
            rag_chain.invoke,
            {"question": text, "context": contexts}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
    
    return response


@router.post("/get-all-docs", status_code=200)
@protected_route([SystemRole.HOTEL_OWNER, SystemRole.DATA_SCIENTIST])
async def get_all_docs(
    CREDENTIALS: AuthCredentialDepend, 
    CURRENT_USER=None
):
    db_str, cur_user_id = CURRENT_USER.get("db"), CURRENT_USER.get("_id")
    chroma_db = ChromaDB()
    
    return chroma_db.get_all_docs(db_str)


@router.post("/upload", status_code=200)
@protected_route([SystemRole.HOTEL_OWNER, SystemRole.DATA_SCIENTIST])
async def upload_files(
    CREDENTIALS: AuthCredentialDepend,
    files: List[UploadFile] = File(...), 
    CURRENT_USER=None
):
    db_str, cur_user_id = CURRENT_USER.get("db"), CURRENT_USER.get("_id")

    user_folder = BASE_UPLOAD_DIR / db_str
    user_folder.mkdir(parents=True, exist_ok=True)

    tasks = []
    for file in files:
        file_path = user_folder / file.filename
        tasks.append(save_file(file, file_path))

    uploaded_files = await asyncio.gather(*tasks)
    
    chroma_db = ChromaDB()
    await chroma_db.add_docs(db_str, path_to_files=uploaded_files)

    return uploaded_files


@router.get("/files", status_code=200)
@protected_route([SystemRole.HOTEL_OWNER, SystemRole.DATA_SCIENTIST])
async def get_files(CREDENTIALS: AuthCredentialDepend, CURRENT_USER=None):
    db_str = CURRENT_USER.get("db")
    user_folder = BASE_UPLOAD_DIR / db_str

    if not user_folder.exists():
        return {"files": []}

    files = [file.name for file in user_folder.iterdir() if file.is_file()]
    return {"files": files}


@router.delete("/files/{filename}", status_code=200)
@protected_route([SystemRole.HOTEL_OWNER, SystemRole.DATA_SCIENTIST])
async def delete_file(
    filename: str,
    CREDENTIALS: AuthCredentialDepend,
    CURRENT_USER=None
):
    db_str = CURRENT_USER.get("db")
    user_folder = BASE_UPLOAD_DIR / db_str
    file_path = user_folder / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Delete the file
        file_path.unlink()

        # Re-initialized all chunks in ChromaDB
        chroma_db = ChromaDB()
        chroma_db.delete_coll(db_str)
        
        file_names = [file.name for file in user_folder.iterdir() if file.is_file()]
        
        path = [user_folder / file_name for file_name in file_names]
        
        await chroma_db.add_docs(db_str, path_to_files=path)

        return {"detail": "File and related data deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@router.post("/speech-to-text", status_code=200)
@protected_route([SystemRole.HOTEL_OWNER, SystemRole.DATA_SCIENTIST])
async def speech_to_text(
    audio: UploadFile, 
    CREDENTIALS: AuthCredentialDepend,
    CURRENT_USER=None
):
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    try:
        # Convert uploaded audio to WAV format using pydub
        audio_bytes = await audio.read()
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        wav_bytes = io.BytesIO()
        audio_segment.export(wav_bytes, format="wav")
        wav_bytes.seek(0)

        transcription = transcriber(wav_bytes.read())
        return {"text": transcription["text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {e}")
