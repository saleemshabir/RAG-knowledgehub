from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from pypdf import PdfReader
from io import BytesIO

app = FastAPI()

# Data folder
DATA_FOLDER = Path(__file__).resolve().parent.parent / "data"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def read_health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    try:
        content = await file.read()

        # Save original PDF
        file_path = DATA_FOLDER / Path(file.filename).name
        file_path.write_bytes(content)

        # Extract text from PDF
        pdf_reader = PdfReader(BytesIO(content))

        extracted_text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            extracted_text += page_text + "\n"

        # Save extracted text
        text_file_path = DATA_FOLDER / "extracted_text.txt"
        text_file_path.write_text(
            extracted_text,
            encoding="utf-8"
        )

        return {
            "filename": file.filename,
            "extracted_text": extracted_text,
            "saved_text_file": str(text_file_path),
            "message": "PDF uploaded and text extracted successfully."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )