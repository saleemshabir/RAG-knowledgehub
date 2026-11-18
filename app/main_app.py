from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
from pypdf import PdfReader
from io import BytesIO
from sentence_transformers import SentenceTransformer
import subprocess
import sys
import requests

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

# Make project root accessible for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import existing RAG functions
from retrieve import retrieve_chunks, generate_answer

app = FastAPI(
    title="Knowledge Hub API",
    description="PDF-based Retrieval Augmented Generation application",
    version="1.0"
)

DATA_FOLDER.mkdir(parents=True, exist_ok=True)

# Load embedding model only when needed
embedding_model = None


def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    return embedding_model


# Request model for asking questions
class AskRequest(BaseModel):
    question: str


@app.get("/health")
def read_health():
    return {"status": "healthy"}


# PDF Upload API
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    try:
        content = await file.read()

        safe_filename = Path(file.filename).name
        file_path = DATA_FOLDER / safe_filename

        file_path.write_bytes(content)

        # Extract text from PDF
        pdf_reader = PdfReader(BytesIO(content))
        extracted_text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            extracted_text += page_text + "\n"

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No readable text found in this PDF."
            )

        # Save extracted text
        text_filename = f"{Path(safe_filename).stem}.txt"
        text_file_path = DATA_FOLDER / text_filename

        text_file_path.write_text(
            extracted_text,
            encoding="utf-8"
        )

        # Run chunking
        chunk_result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "make_chunks.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        if chunk_result.returncode != 0:
            raise RuntimeError(
                f"Chunking failed: {chunk_result.stderr}"
            )

        # Generate embeddings
        embedding_result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "make_embeddings.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        if embedding_result.returncode != 0:
            raise RuntimeError(
                f"Embedding generation failed: {embedding_result.stderr}"
            )

        return {
            "filename": safe_filename,
            "saved_text_file": str(text_file_path),
            "message": "PDF uploaded, text extracted, chunks created, and embeddings generated successfully."
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )


# Question Answering API
@app.post("/ask")
def ask_question(request: AskRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    try:
        # Load embedding model
        model = get_embedding_model()

        # Retrieve relevant chunks
        relevant_chunks = retrieve_chunks(
            question,
            model
        )

        # Handle questions outside the PDF
        if not relevant_chunks:
            return {
                "question": question,
                "answer": "I don't know based on the information provided.",
                "sources": []
            }

        # Generate answer using Ollama
        answer = generate_answer(
            question,
            relevant_chunks
        )

        # Collect source information
        sources = [
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in relevant_chunks
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

    except requests.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Make sure Ollama is running."
        )

    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out."
        )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {str(e)}"
        )