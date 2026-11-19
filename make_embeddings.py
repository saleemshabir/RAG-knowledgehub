from pathlib import Path
import argparse
import json
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# Project paths
BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "chunks.json"
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npy"

# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=CHUNKS_FILE)
    parser.add_argument("--output", type=Path, default=EMBEDDINGS_FILE)
    args = parser.parse_args()

    # Check whether chunks.json exists
    if not args.chunks.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {args.chunks}"
        )

    # Load chunks
    with args.chunks.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("No chunks available for embedding.")

    print("Total chunks loaded:", len(chunks))

    # Extract text from chunks
    chunk_texts = []
    for chunk in chunks:
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Chunks contain missing or empty text.")
        chunk_texts.append(text)

    if not chunk_texts:
        raise ValueError("No valid text found in chunks.")

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print("\nEmbeddings generated successfully!")
    print("Shape of embeddings:", embeddings.shape)

    # Save embeddings
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)

    print("Embeddings saved successfully!")


if __name__ == "__main__":
    try:
        main()

    except (OSError, ValueError, KeyError) as error:
        print("\nError:", error)
        sys.exit(1)