from pathlib import Path
import json
import sys

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = BASE_DIR / "chunks.json"

# Chunk settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def clean_text(text):
    """Remove unnecessary whitespace."""
    return " ".join(text.split())


def chunk_text(
    text,
    source,
    start_id,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """Split text into overlapping chunks with unique IDs."""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Avoid cutting words when possible
        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "chunk_id": start_id + len(chunks),
                "source": source,
                "text": chunk
            })

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}"
        )

    # Find all text documents
    files = sorted(DATA_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            "No .txt documents found inside the data folder."
        )

    all_chunks = []
    next_chunk_id = 1

    for file_path in files:
        try:
            raw_text = file_path.read_text(encoding="utf-8")

        except (OSError, UnicodeDecodeError) as error:
            print(f"Skipping {file_path.name}: {error}")
            continue

        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            print(f"Skipping empty document: {file_path.name}")
            continue

        print(f"\nProcessing: {file_path.name}")
        print("Total characters:", len(cleaned_text))

        document_chunks = chunk_text(
            cleaned_text,
            source=file_path.name,
            start_id=next_chunk_id
        )

        all_chunks.extend(document_chunks)
        next_chunk_id += len(document_chunks)

        print("Chunks generated:", len(document_chunks))

    if not all_chunks:
        raise ValueError("No chunks were generated.")

    # Save all document chunks
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            all_chunks,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n--- Chunking Summary ---")
    print(
        "Documents processed:",
        len(set(chunk["source"] for chunk in all_chunks))
    )
    print("Total chunks generated:", len(all_chunks))
    print("Chunks saved successfully!")


if __name__ == "__main__":
    try:
        main()

    except (OSError, ValueError) as error:
        print("\nError:", error)
        sys.exit(1)