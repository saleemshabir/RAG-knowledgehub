import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
WORKSPACE_FOLDER = Path(
    os.environ.get("KNOWLEDGEHUB_DATA_DIR", DATA_FOLDER / "workspace")
).resolve()
DOCUMENTS_FOLDER = WORKSPACE_FOLDER / "documents"
INDEX_FOLDER = WORKSPACE_FOLDER / "index"
DATABASE_PATH = Path(
    os.environ.get(
        "KNOWLEDGEHUB_DATABASE_PATH", WORKSPACE_FOLDER / "knowledgehub.sqlite3"
    )
).resolve()
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json"}
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import (
    TOP_K,
    generate_answer,
    iter_answer_tokens,
    open_answer_stream,
    retrieve_chunks,
)

app = FastAPI(
    title="Knowledge Hub API",
    description="Shared document question-answering API",
    version="1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

DATA_FOLDER.mkdir(parents=True, exist_ok=True)
WORKSPACE_FOLDER.mkdir(parents=True, exist_ok=True)
DOCUMENTS_FOLDER.mkdir(parents=True, exist_ok=True)
embedding_model = None
logger = logging.getLogger(__name__)


@contextmanager
def database():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with database() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                page_count INTEGER,
                pages_processed INTEGER NOT NULL,
                page_chart TEXT NOT NULL DEFAULT '[]',
                chunks_count INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "page_chart" not in columns:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN page_chart TEXT NOT NULL DEFAULT '[]'"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS documents_created_idx ON documents(created_at)"
        )


initialize_database()


class AskRequest(BaseModel):
    question: str = Field(max_length=4000)


def serialize_document(row):
    page_chart = row["page_chart"]
    return {
        "id": row["id"],
        "filename": row["filename"],
        "file_type": row["file_type"],
        "size_bytes": row["size_bytes"],
        "page_count": row["page_count"],
        "pages_processed": row["pages_processed"],
        "page_chart": json.loads(page_chart) if isinstance(page_chart, str) else page_chart,
        "chunks_count": row["chunks_count"],
        "created_at": row["created_at"],
    }


def list_document_rows():
    with database() as connection:
        return connection.execute(
            """
            SELECT id, filename, file_type, size_bytes, page_count,
                   pages_processed, page_chart, chunks_count, created_at
            FROM documents ORDER BY created_at DESC
            """
        ).fetchall()


def rebuild_workspace_index():
    text_files = list(DOCUMENTS_FOLDER.rglob("document.txt"))
    if not text_files:
        for filename in ("chunks.json", "embeddings.npy"):
            index_file = INDEX_FOLDER / filename
            if index_file.exists():
                index_file.unlink()
        return []

    INDEX_FOLDER.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="index-build-", dir=WORKSPACE_FOLDER
    ) as build_dir:
        build_path = Path(build_dir)
        chunks_path = build_path / "chunks.json"
        embeddings_path = build_path / "embeddings.npy"
        chunk_result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "make_chunks.py"),
                "--input-dir",
                str(DOCUMENTS_FOLDER),
                "--output",
                str(chunks_path),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if chunk_result.returncode != 0:
            raise RuntimeError(chunk_result.stderr or "Document chunking failed.")

        embedding_result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "make_embeddings.py"),
                "--chunks",
                str(chunks_path),
                "--output",
                str(embeddings_path),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if embedding_result.returncode != 0:
            raise RuntimeError(
                embedding_result.stderr or "Embedding generation failed."
            )

        with chunks_path.open("r", encoding="utf-8") as chunks_file:
            chunks = json.load(chunks_file)
        os.replace(chunks_path, INDEX_FOLDER / "chunks.json")
        os.replace(embeddings_path, INDEX_FOLDER / "embeddings.npy")
        return chunks


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model


@app.get("/health")
def read_health():
    return {"status": "healthy"}


@app.get("/documents")
def list_documents():
    return {"documents": [serialize_document(row) for row in list_document_rows()]}


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Choose one or more files.")

    new_documents = []
    created_folders = []
    index_rebuilt = False

    try:
        for uploaded_file in files:
            if not uploaded_file.filename:
                raise HTTPException(
                    status_code=400, detail="A selected file has no name."
                )

            filename = Path(uploaded_file.filename).name
            extension = Path(filename).suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail="Supported formats are PDF, TXT, Markdown, CSV, and JSON.",
                )

            content = await uploaded_file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{filename} exceeds the 25 MB per-file limit.",
                )

            document_id = str(uuid.uuid4())
            output_folder = DOCUMENTS_FOLDER / document_id
            output_folder.mkdir(parents=True, exist_ok=False)
            created_folders.append(output_folder)
            (output_folder / f"original{extension}").write_bytes(content)

            page_count = None
            if extension == ".pdf":
                reader = PdfReader(BytesIO(content))
                page_texts = [page.extract_text() or "" for page in reader.pages]
                page_count = len(page_texts)
                pages_processed = sum(bool(text.strip()) for text in page_texts)
                page_chart = [int(bool(text.strip())) for text in page_texts]
                extracted_text = "\n".join(page_texts)
            else:
                try:
                    extracted_text = content.decode("utf-8-sig")
                except UnicodeDecodeError as error:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{filename} must use UTF-8 encoding.",
                    ) from error
                pages_processed = int(bool(extracted_text.strip()))
                page_chart = [pages_processed]

            if not extracted_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"No readable text found in {filename}.",
                )

            (output_folder / "document.txt").write_text(
                extracted_text, encoding="utf-8"
            )
            (output_folder / "metadata.json").write_text(
                json.dumps({"filename": filename}, ensure_ascii=False),
                encoding="utf-8",
            )
            new_documents.append(
                {
                    "id": document_id,
                    "filename": filename,
                    "file_type": extension[1:].upper(),
                    "size_bytes": len(content),
                    "page_count": page_count,
                    "pages_processed": pages_processed,
                    "page_chart": page_chart,
                    "created_at": time.time(),
                }
            )

        chunks = rebuild_workspace_index()
        index_rebuilt = True
        for document in new_documents:
            document["chunks_count"] = sum(
                chunk.get("document_id") == document["id"] for chunk in chunks
            )

        with database() as connection:
            connection.executemany(
                """
                INSERT INTO documents (
                    id, filename, file_type, size_bytes, page_count,
                    pages_processed, page_chart, chunks_count, created_at
                ) VALUES (
                    :id, :filename, :file_type, :size_bytes, :page_count,
                    :pages_processed, :page_chart, :chunks_count, :created_at
                )
                """,
                [
                    {**item, "page_chart": json.dumps(item["page_chart"])}
                    for item in new_documents
                ],
            )

        return {
            "documents": [
                {
                    **serialize_document(item),
                    "vector_status": "ready",
                    "semantic_index_status": "ready",
                    "entity_extraction_status": "not_configured",
                }
                for item in new_documents
            ],
            "message": f"{len(new_documents)} document(s) indexed successfully.",
        }
    except HTTPException as error:
        for folder in created_folders:
            shutil.rmtree(folder, ignore_errors=True)
        if index_rebuilt:
            try:
                rebuild_workspace_index()
            except Exception as restore_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"{error.detail} The previous index could not be restored: {restore_error}",
                ) from restore_error
        raise
    except Exception as error:
        for folder in created_folders:
            shutil.rmtree(folder, ignore_errors=True)
        if index_rebuilt:
            try:
                rebuild_workspace_index()
            except Exception as restore_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Document processing failed: {error}. The previous index could not be restored: {restore_error}",
                ) from restore_error
        raise HTTPException(
            status_code=500, detail=f"Document processing failed: {error}"
        ) from error
    finally:
        for uploaded_file in files:
            await uploaded_file.close()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    with database() as connection:
        row = connection.execute(
            "SELECT id FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    folder = DOCUMENTS_FOLDER / document_id
    try:
        with tempfile.TemporaryDirectory(
            prefix="document-delete-", dir=WORKSPACE_FOLDER
        ) as temp_dir:
            backup = Path(temp_dir) / "removed-document"
            folder.rename(backup)
            try:
                rebuild_workspace_index()
            except Exception:
                backup.rename(folder)
                raise
            with database() as connection:
                connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Could not remove the document: {error}"
        ) from error
    return {"message": "Document removed."}


def retrieve_relevant_chunks(question):
    if not list_document_rows():
        raise HTTPException(
            status_code=400,
            detail="Upload and index a document before asking questions.",
        )

    started_at = time.perf_counter()
    model = get_embedding_model()
    ranked_chunks = retrieve_chunks(
        question,
        model,
        data_directory=INDEX_FOLDER,
        include_scores=True,
        verbose=False,
    )
    ranked_chunks.sort(key=lambda result: result[1], reverse=True)
    logger.info("Question retrieval completed in %.2f seconds", time.perf_counter() - started_at)
    return [chunk for chunk, _ in ranked_chunks[:TOP_K]]


def question_sources(chunks):
    return [
        {
            "source": chunk["source"],
            "document_id": chunk.get("document_id"),
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]


def stream_event(event):
    return json.dumps(event, ensure_ascii=False) + "\n"


def raise_question_error(error):
    if isinstance(error, HTTPException):
        return error
    if isinstance(error, requests.ConnectionError):
        return HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Make sure Ollama is running.",
        )
    if isinstance(error, requests.Timeout):
        return HTTPException(status_code=504, detail="Ollama request timed out.")
    if isinstance(error, requests.RequestException):
        return HTTPException(status_code=502, detail=f"Ollama request failed: {error}")
    return HTTPException(status_code=500, detail=f"Question processing failed: {error}")


@app.post("/ask")
def ask_question(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    try:
        relevant_chunks = retrieve_relevant_chunks(question)
        if not relevant_chunks:
            return {
                "question": question,
                "answer": "I don't know based on the information provided.",
                "sources": [],
            }
        answer = generate_answer(question, relevant_chunks)
        return {
            "question": question,
            "answer": answer,
            "sources": question_sources(relevant_chunks),
        }
    except Exception as error:
        raise raise_question_error(error) from error


@app.post("/ask/stream")
def ask_question_stream(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    try:
        relevant_chunks = retrieve_relevant_chunks(question)
    except Exception as error:
        raise raise_question_error(error) from error

    sources = question_sources(relevant_chunks)
    if not relevant_chunks:
        def no_results():
            yield stream_event(
                {
                    "type": "complete",
                    "answer": "I don't know based on the information provided.",
                    "sources": [],
                }
            )

        return StreamingResponse(no_results(), media_type="application/x-ndjson")

    def generate_events():
        response = None
        generation_started = time.perf_counter()
        try:
            response = open_answer_stream(question, relevant_chunks)
            for token in iter_answer_tokens(response):
                yield stream_event({"type": "token", "content": token})
            logger.info(
                "Answer generation completed in %.2f seconds",
                time.perf_counter() - generation_started,
            )
            yield stream_event({"type": "complete", "sources": sources})
        except requests.RequestException as error:
            logger.warning("Ollama streaming request failed: %s", error)
            yield stream_event(
                {"type": "error", "message": raise_question_error(error).detail}
            )
        except (ValueError, KeyError) as error:
            logger.exception("Could not decode Ollama streaming response")
            yield stream_event(
                {"type": "error", "message": f"Invalid Ollama response: {error}"}
            )
        finally:
            if response is not None:
                response.close()

    return StreamingResponse(
        generate_events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
