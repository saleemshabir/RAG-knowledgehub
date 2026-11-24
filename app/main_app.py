import json

import logging

import os

import shutil

import sqlite3

import sys

import tempfile

import time

import uuid

from contextlib import contextmanager

from pathlib import Path



import numpy as np

import pymupdf

import pytesseract

import requests

from fastapi import FastAPI, File, HTTPException, UploadFile

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import StreamingResponse

from fastapi.staticfiles import StaticFiles

from PIL import Image

from pydantic import BaseModel, Field

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

    generate_answer,

    iter_answer_tokens,

    open_answer_stream,

    retrieve_chunks,

)

from make_chunks import chunk_text, clean_text



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

app.mount("/document-files", StaticFiles(directory=DOCUMENTS_FOLDER), name="document-files")

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





def extract_pdf_page_text(content):

    with pymupdf.open(stream=content, filetype="pdf") as pdf:

        page_texts = [page.get_text("text") or "" for page in pdf]

    pages_needing_ocr = [

        index for index, text in enumerate(page_texts) if not text.strip()

    ]



    if pages_needing_ocr:

        tesseract_path = shutil.which("tesseract")

        if not tesseract_path and os.name == "nt":

            installed_path = (

                Path(os.environ.get("ProgramFiles", r"C:\Program Files"))

                / "Tesseract-OCR"

                / "tesseract.exe"

            )

            if installed_path.is_file():

                tesseract_path = str(installed_path)



        if not tesseract_path:

            raise HTTPException(

                status_code=503,

                detail=(

                    "This PDF contains scanned pages and needs local OCR. "

                    "Install Tesseract OCR, restart the API, and try again."

                ),

            )



        pytesseract.pytesseract.tesseract_cmd = tesseract_path

        with pymupdf.open(stream=content, filetype="pdf") as pdf:

            for page_index in pages_needing_ocr:

                pixmap = pdf[page_index].get_pixmap(

                    dpi=150, colorspace=pymupdf.csGRAY, alpha=False

                )

                image = Image.frombytes(

                    "L", (pixmap.width, pixmap.height), pixmap.samples

                )

                page_texts[page_index] = pytesseract.image_to_string(

                    image, lang="eng"

                )



    return page_texts





def extract_pdf_figures(content, output_folder):

    figures_folder = output_folder / "figures"

    figures_folder.mkdir(parents=True, exist_ok=True)

    visual_pages = []



    with pymupdf.open(stream=content, filetype="pdf") as pdf:

        for page_index, page in enumerate(pdf):

            if len(visual_pages) >= 12:

                break



            drawings = page.get_drawings()

            images = page.get_images(full=True)

            if not drawings and not images:

                continue



            if len(drawings) < 8 and not images:

                continue



            pixmap = page.get_pixmap(

                dpi=120,

                colorspace=pymupdf.csRGB,

                alpha=False,

            )

            filename = f"page-{page_index + 1}.png"

            pixmap.save(figures_folder / filename)

            visual_pages.append({

                "page": page_index + 1,

                "filename": filename,

            })



    return visual_pages





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

    existing_chunks = []

    existing_embeddings = None

    chunks_path = INDEX_FOLDER / "chunks.json"

    embeddings_path = INDEX_FOLDER / "embeddings.npy"

    if chunks_path.exists() and embeddings_path.exists():

        try:

            existing_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

            existing_embeddings = np.load(embeddings_path, allow_pickle=False)

            if (

                not isinstance(existing_chunks, list)

                or existing_embeddings.ndim != 2

                or len(existing_chunks) != existing_embeddings.shape[0]

            ):

                existing_chunks = []

                existing_embeddings = None

        except (OSError, ValueError, json.JSONDecodeError):

            existing_chunks = []

            existing_embeddings = None



    folder_ids = {file_path.parent.name for file_path in text_files}

    kept_chunks = [

        chunk

        for chunk in existing_chunks

        if chunk.get("document_id") in folder_ids

    ]

    kept_indices = [

        index

        for index, chunk in enumerate(existing_chunks)

        if chunk.get("document_id") in folder_ids

    ]

    if existing_embeddings is not None:

        kept_embeddings = existing_embeddings[kept_indices]

    else:

        kept_embeddings = np.empty((0, 384), dtype=np.float32)



    indexed_document_ids = {

        chunk.get("document_id")

        for chunk in existing_chunks

        if chunk.get("document_id")

    }

    all_chunks = list(kept_chunks)

    next_chunk_id = max(

        (chunk.get("chunk_id", 0) for chunk in kept_chunks),

        default=0,

    ) + 1

    new_chunk_texts = []



    for file_path in sorted(text_files):

        document_id = file_path.parent.name

        if document_id in indexed_document_ids:

            continue



        cleaned_text = clean_text(file_path.read_text(encoding="utf-8"))

        if not cleaned_text:

            continue



        document_chunks = chunk_text(

            cleaned_text,

            source=file_path.name[:-4],

            start_id=next_chunk_id,

        )

        metadata_path = file_path.parent / "metadata.json"

        if metadata_path.exists():

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            for chunk in document_chunks:

                chunk["source"] = metadata.get("filename", chunk["source"])

                chunk["document_id"] = document_id



        all_chunks.extend(document_chunks)

        new_chunk_texts.extend(chunk["text"] for chunk in document_chunks)

        next_chunk_id += len(document_chunks)



    if not all_chunks:

        for filename in ("chunks.json", "embeddings.npy"):

            index_file = INDEX_FOLDER / filename

            if index_file.exists():

                index_file.unlink()

        return []



    if new_chunk_texts:

        model = get_embedding_model()

        new_embeddings = model.encode(

            new_chunk_texts,

            batch_size=64,

            convert_to_numpy=True,

            show_progress_bar=False,

            normalize_embeddings=True,

        )

        new_embeddings = np.asarray(new_embeddings, dtype=np.float32)

        embeddings = np.vstack((kept_embeddings, new_embeddings)).astype(

            np.float32, copy=False

        )

    else:

        embeddings = kept_embeddings



    with tempfile.TemporaryDirectory(

        prefix="index-build-", dir=WORKSPACE_FOLDER

    ) as build_dir:

        chunks_path = Path(build_dir) / "chunks.json"

        embeddings_path = Path(build_dir) / "embeddings.npy"

        chunks_path.write_text(

            json.dumps(all_chunks, ensure_ascii=False),

            encoding="utf-8",

        )

        np.save(embeddings_path, embeddings)

        os.replace(chunks_path, INDEX_FOLDER / "chunks.json")

        os.replace(embeddings_path, INDEX_FOLDER / "embeddings.npy")



    return all_chunks





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

            visual_pages = []

            if extension == ".pdf":

                page_texts = extract_pdf_page_text(content)

                page_count = len(page_texts)

                pages_processed = sum(bool(text.strip()) for text in page_texts)

                page_chart = [int(bool(text.strip())) for text in page_texts]

                extracted_text = "\n".join(page_texts)

                visual_pages = extract_pdf_figures(content, output_folder)

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

                json.dumps(

                    {"filename": filename, "visual_pages": visual_pages},

                    ensure_ascii=False,

                ),

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





def retrieve_relevant_chunks(question, detailed=False):

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

        top_k=8 if detailed else 3,

    )

    ranked_chunks.sort(key=lambda result: result[1], reverse=True)

    logger.info("Question retrieval completed in %.2f seconds", time.perf_counter() - started_at)
    return [chunk for chunk, _score in ranked_chunks]





def question_sources(chunks):

    return [

        {

            "source": chunk["source"],

            "document_id": chunk.get("document_id"),

            "chunk_id": chunk["chunk_id"],

        }

        for chunk in chunks

    ]





DIAGRAM_REQUEST_TERMS = (

    "diagram",

    "figure",

    "chart",

    "graph",

    "image",

    "picture",

    "show me",

    "visual",

)



LONG_ANSWER_TERMS = (

    "long answer",

    "detailed",

    "detail",

    "longer",

    "in depth",

    "in-depth",

    "comprehensive",

    "explain fully",

    "explain in detail",

    "describe in detail",

    "descriptive",

    "in detail",

    "comprehensive answer",

    "elaborate",

    "explain thoroughly",

    "full explanation",

    "essay",

    "advantages and disadvantages",

    "pros and cons",

    "uses and benefits",

)





def is_diagram_request(question):

    lowered = question.lower()

    return any(term in lowered for term in DIAGRAM_REQUEST_TERMS)





def is_long_answer_request(question):

    lowered = question.lower()

    return any(term in lowered for term in LONG_ANSWER_TERMS)





def find_requested_figures(relevant_chunks, question):

    if not is_diagram_request(question):

        return []



    figures = []

    seen_document_ids = set()

    for chunk in relevant_chunks:

        document_id = chunk.get("document_id")

        if not document_id or document_id in seen_document_ids:

            continue

        seen_document_ids.add(document_id)



        metadata_path = DOCUMENTS_FOLDER / document_id / "metadata.json"

        if not metadata_path.exists():

            continue

        try:

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        except (OSError, json.JSONDecodeError):

            continue



        for visual_page in metadata.get("visual_pages", [])[:4]:

            filename = visual_page.get("filename")

            if not filename:

                continue

            figures.append({

                "document_id": document_id,

                "page": visual_page.get("page"),

                "url": f"/document-files/{document_id}/figures/{filename}",

                "filename": metadata.get("filename", "Document"),

            })

    return figures





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

        detailed = is_long_answer_request(question)

        relevant_chunks = retrieve_relevant_chunks(question, detailed=detailed)

        if not relevant_chunks:

            return {

                "question": question,

                "answer": "This information is not provided in the uploaded documents.",

                "sources": [],

                "figures": [],

            }

        answer = generate_answer(

            question,

            relevant_chunks,

            detailed=detailed,

        )

        return {

            "question": question,

            "answer": answer,

            "sources": question_sources(relevant_chunks),

            "figures": find_requested_figures(relevant_chunks, question),

        }

    except Exception as error:

        raise raise_question_error(error) from error





@app.post("/ask/stream")

def ask_question_stream(request: AskRequest):

    question = request.question.strip()

    if not question:

        raise HTTPException(status_code=400, detail="Please enter a question.")



    try:

        detailed = is_long_answer_request(question)

        relevant_chunks = retrieve_relevant_chunks(question, detailed=detailed)

    except Exception as error:

        raise raise_question_error(error) from error



    sources = question_sources(relevant_chunks)

    figures = find_requested_figures(relevant_chunks, question)

    if not relevant_chunks:

        def no_results():

            yield stream_event(

                {

                    "type": "complete",

                    "answer": "This information is not provided in the uploaded documents.",

                    "sources": [],

                    "figures": [],

                }

            )



        return StreamingResponse(no_results(), media_type="application/x-ndjson")



    def generate_events():

        response = None

        generation_started = time.perf_counter()

        try:

            response = open_answer_stream(

                question,

                relevant_chunks,

                detailed=detailed,

            )

            for token in iter_answer_tokens(response):

                yield stream_event({"type": "token", "content": token})

            logger.info(

                "Answer generation completed in %.2f seconds",

                time.perf_counter() - generation_started,

            )

            yield stream_event({

                "type": "complete",

                "sources": sources,

                "figures": figures,

            })

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
